"""Tests for the interactive text agent and narrative formatting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from labverse.agent.interactive_text import (
    InteractiveTextAgent,
    build_narrative_user_message,
    format_available_actions_as_narrative,
    format_observation_as_narrative,
)
from labverse.agent.interface import Observation
from labverse.agent.prompt import format_available_actions
from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall
from labverse.core.types import ActionCategory, BSLLevel, PPEType, RoomType
from labverse.models.action import ActionSpec, AvailableAction
from labverse.models.space import Room


class FakeInteractiveProvider(LLMProvider):
    """Provider that returns predetermined responses and captures prompts."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.captured_calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        self.captured_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "system": system,
            }
        )
        if not self._responses:
            raise RuntimeError("No fake responses remaining")
        return self._responses.pop(0)


def _make_observation() -> Observation:
    room = Room(
        id="wet_lab",
        name="Wet Lab",
        room_type=RoomType.WET_LAB,
        bsl_level=BSLLevel.BSL2,
    )
    return Observation(
        timestamp=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        tick=2,
        current_room=room,
        current_ppe={PPEType.LAB_COAT},
        active_alerts=["Wear nitrile gloves before BSL-2 work"],
        recent_events=["PCR Setup Pipette entered ready"],
        narrative="You are in the Wet Lab.",
    )


def _make_action_spec() -> ActionSpec:
    return ActionSpec(
        available_actions=[
            AvailableAction(
                action_id="pipette_transfer:pipette_1",
                verb="pipette_transfer",
                category=ActionCategory.HANDLE_SAMPLE,
                description="Transfer from PCR Master Mix to PCR Reaction 1 using pipette_1",
                target_id="pipette_1",
                fixed_parameters={
                    "source_id": "pcr_master_mix",
                    "destination_id": "pcr_reaction_1",
                },
                required_parameters=["volume_ul"],
                parameter_details={
                    "volume_ul": "Transfer volume in uL between the fixed source and destination."
                },
                parameter_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"volume_ul": {"type": "number"}},
                    "required": ["volume_ul"],
                },
            )
        ]
    )


def _tool_call(action_id: str, **parameters: Any) -> LLMResponse:
    """Create a valid plain JSON interactive action response."""
    verb, _, raw_target = action_id.partition(":")
    args: dict[str, Any] = {
        "candidate_ref": action_id,
        "verb": verb,
    }
    if raw_target and raw_target != "self":
        args["target_id"] = raw_target
    if parameters:
        args["parameters"] = parameters
    return _json_action_text(args)


def _provider_tool_call(
    action_id: str,
    *,
    name: str = "take_action",
    **parameters: Any,
) -> LLMResponse:
    verb, _, raw_target = action_id.partition(":")
    args: dict[str, Any] = {
        "candidate_ref": action_id,
        "verb": verb,
    }
    if raw_target and raw_target != "self":
        args["target_id"] = raw_target
    if parameters:
        args["parameters"] = parameters
    return LLMResponse(
        tool_call=ToolCall(name=name, arguments=args),
        usage=TokenUsage(prompt_tokens=11, completion_tokens=4),
    )


def _json_action_text(arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(arguments),
        usage=TokenUsage(prompt_tokens=11, completion_tokens=4),
    )


def test_narrative_formatter_mentions_alerts_and_ppe() -> None:
    text = format_observation_as_narrative(_make_observation())

    assert "biosafety level 2" in text
    assert "You are wearing: lab coat." in text
    assert "Active alerts: Wear nitrile gloves before BSL-2 work." in text


def test_available_actions_narrative_includes_parameter_hints() -> None:
    text = format_available_actions_as_narrative(_make_action_spec().available_actions)

    assert "[pipette_transfer:pipette_1] verb=pipette_transfer" in text
    assert "(requires: volume_ul)" in text
    assert "parameter_hints:" in text
    assert "type=number;" in text
    assert "takes" not in text


def test_available_actions_narrative_includes_estimated_duration() -> None:
    action_spec = _make_action_spec()
    action_spec.available_actions[0].estimated_duration_seconds = 10.0

    text = format_available_actions_as_narrative(action_spec.available_actions)

    assert "takes" in text


def test_narrative_available_actions_match_structured_action_block() -> None:
    action_spec = _make_action_spec()

    assert format_available_actions_as_narrative(
        action_spec.available_actions
    ) == format_available_actions(action_spec.available_actions)


def test_build_narrative_user_message_uses_tagged_layout() -> None:
    user_message = build_narrative_user_message(
        _make_observation(),
        _make_action_spec().available_actions,
    )

    assert "<observation>" in user_message["content"]
    assert "</observation>" in user_message["content"]
    assert "<available_actions>" in user_message["content"]
    assert "</available_actions>" in user_message["content"]


def test_retry_feedback_uses_structured_error_message() -> None:
    provider = FakeInteractiveProvider(
        [
            _tool_call("pipette_transfer:pipette_1"),
            _tool_call("pipette_transfer:pipette_1", volume_ul=25.0),
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    action = agent.act(_make_action_spec())

    assert action.parameters["volume_ul"] == 25.0
    retry_call = provider.captured_calls[1]
    assert "<error>" in retry_call["messages"][-1]["content"]
    assert "exactly one valid JSON object" in retry_call["messages"][-1]["content"]
    assert "valid JSON object" in retry_call["messages"][-1]["content"]
    assert "nested wrapper objects" in retry_call["messages"][-1]["content"]
    assert "matching verb" not in retry_call["messages"][-1]["content"]
    assert "exact target_id" not in retry_call["messages"][-1]["content"]


def test_accepts_plain_json_text_action_output() -> None:
    provider = FakeInteractiveProvider(
        [
            _json_action_text(
                {
                    "candidate_ref": "pipette_transfer:pipette_1",
                    "parameters": {"volume_ul": 25.0},
                }
            )
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    action = agent.act(_make_action_spec())

    assert action.parameters["volume_ul"] == 25.0
    assert agent.audit_steps[0]["provider_tool_call_fallback_count"] == 0


def test_repairs_malformed_plain_json_text_action_output() -> None:
    provider = FakeInteractiveProvider(
        [
            LLMResponse(
                text=(
                    "{candidate_ref:'pipette_transfer:pipette_1',"
                    "parameters:{volume_ul:25.0}}"
                ),
                usage=TokenUsage(prompt_tokens=11, completion_tokens=4),
            )
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    action = agent.act(_make_action_spec())

    assert action.parameters["volume_ul"] == 25.0
    assert agent.audit_steps[0]["json_repair_count"] == 1


def test_rejects_provider_tool_call_output() -> None:
    provider = FakeInteractiveProvider(
        [
            _provider_tool_call("pipette_transfer:pipette_1", parameters='{"volume_ul": 25.0}'),
            _provider_tool_call("pipette_transfer:pipette_1", parameters='{"volume_ul": 25.0}'),
            _provider_tool_call("pipette_transfer:pipette_1", parameters='{"volume_ul": 25.0}'),
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    with pytest.raises(RuntimeError, match="failed after"):
        agent.act(_make_action_spec())


def test_accepts_flattened_top_level_parameters() -> None:
    provider = FakeInteractiveProvider(
        [
            _json_action_text(
                {
                    "candidate_ref": "pipette_transfer:pipette_1",
                    "volume_ul": 25.0,
                }
            )
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    action = agent.act(_make_action_spec())

    assert action.parameters["volume_ul"] == 25.0


def test_provider_tool_call_output_is_not_counted_as_fallback() -> None:
    provider = FakeInteractiveProvider(
        [
            _provider_tool_call("pipette_transfer:pipette_1", volume_ul=25.0),
            _json_action_text(
                {
                    "candidate_ref": "pipette_transfer:pipette_1",
                    "volume_ul": 25.0,
                }
            ),
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    action = agent.act(_make_action_spec())

    assert action.parameters["volume_ul"] == 25.0
    assert agent.audit_steps[0]["provider_tool_call_fallback_count"] == 0
    assert agent.audit_steps[0]["action_output_retry_count"] == 1


def test_raises_after_retry_budget_is_exhausted() -> None:
    provider = FakeInteractiveProvider(
        [
            _tool_call("pipette_transfer:pipette_1"),
            _tool_call("pipette_transfer:pipette_1"),
            _tool_call("pipette_transfer:pipette_1"),
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")
    agent.observe(_make_observation())

    with pytest.raises(RuntimeError, match="failed after"):
        agent.act(_make_action_spec())


def test_followup_turn_does_not_include_assistant_action_transcript() -> None:
    provider = FakeInteractiveProvider(
        [
            _tool_call("pipette_transfer:pipette_1", volume_ul=25.0),
            _tool_call("pipette_transfer:pipette_1", volume_ul=25.0),
        ]
    )
    agent = InteractiveTextAgent("grad_kim", provider, goal="Transfer reagent")

    agent.observe(_make_observation())
    agent.act(_make_action_spec())
    agent.notify_result(True, "Transfer complete.")

    agent.observe(_make_observation())
    agent.act(_make_action_spec())

    second_call = provider.captured_calls[1]
    assert all(message["role"] != "assistant" for message in second_call["messages"])
    assert not any(
        "Action:" in str(message.get("content", "")) for message in second_call["messages"]
    )
