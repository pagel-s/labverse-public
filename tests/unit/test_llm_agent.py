"""Tests for LLMAgent using a FakeLLMProvider."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from labverse.agent.interface import Observation
from labverse.agent.llm import MAX_RETRIES, LLMAgent
from labverse.agent.prompt import TurnStatus
from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall
from labverse.core.types import ActionCategory, BSLLevel, RoomType
from labverse.models.action import VERB_CATEGORIES, ActionSpec, AvailableAction
from labverse.models.space import Room


class FakeLLMProvider(LLMProvider):
    """Test-only provider that returns predetermined responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.captured_calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

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
        if self._call_count >= len(self._responses):
            raise RuntimeError("FakeLLMProvider ran out of responses")
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _make_room() -> Room:
    return Room(
        id="room_1",
        name="Main Wet Lab",
        room_type=RoomType.WET_LAB,
        bsl_level=BSLLevel.BSL2,
    )


def _make_observation(tick: int = 1) -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        tick=tick,
        current_room=_make_room(),
        narrative="You are in the Main Wet Lab.",
    )


def _action_id_for(verb: str, target_id: str | None = None) -> str:
    target_part = target_id if target_id is not None else "self"
    return f"{verb}:{target_part}"


def _make_action_spec(
    *action_defs: tuple[Any, ...],
) -> ActionSpec:
    """Build an ActionSpec from tuples describing available actions."""
    category_map = {
        "don_ppe": ActionCategory.SAFETY,
        "doff_ppe": ActionCategory.SAFETY,
        "move_to": ActionCategory.MOVE,
        "start_equipment": ActionCategory.USE_EQUIPMENT,
        "stop_equipment": ActionCategory.USE_EQUIPMENT,
        "pipette_transfer": ActionCategory.HANDLE_SAMPLE,
        "log_observation": ActionCategory.OBSERVE,
    }
    actions = []
    for action_def in action_defs:
        if len(action_def) == 2:
            verb, target_id = action_def
            fixed_parameters = None
            required_parameters = None
            optional_parameters = None
        else:
            verb, target_id, fixed_parameters, required_parameters, optional_parameters = action_def
        actions.append(
            AvailableAction(
                action_id=_action_id_for(verb, target_id),
                verb=verb,
                category=category_map.get(verb, ActionCategory.ADMINISTRATIVE),
                description=f"Do {verb}",
                target_id=target_id,
                fixed_parameters=fixed_parameters or {},
                required_parameters=required_parameters or [],
                optional_parameters=optional_parameters or [],
            )
        )
    return ActionSpec(available_actions=actions)


def _tool_call_response(
    action_id: str | None = None,
    target_id: str | None = None,
    *,
    verb: str | None = None,
    **params: Any,
) -> LLMResponse:
    """Create an LLMResponse with a valid plain JSON interactive action."""
    if verb is not None:
        candidate_ref = _action_id_for(verb, target_id)
        resolved_verb = verb
    elif action_id is not None and action_id in VERB_CATEGORIES and ":" not in action_id:
        candidate_ref = _action_id_for(action_id, target_id)
        resolved_verb = action_id
    elif action_id is not None:
        candidate_ref = action_id
        resolved_verb = action_id.split(":", 1)[0]
    else:
        raise ValueError("Either action_id or verb must be provided")

    args: dict[str, Any] = {
        "candidate_ref": candidate_ref,
        "verb": resolved_verb,
    }
    resolved_target_id = target_id
    if resolved_target_id is None and ":" in candidate_ref:
        candidate_target = candidate_ref.split(":", 1)[1]
        if candidate_target != "self":
            resolved_target_id = candidate_target
    if resolved_target_id is not None:
        args["target_id"] = resolved_target_id
    if params:
        args["parameters"] = params
    return _json_action_text_response(args)


def _provider_tool_call_response(
    action_id: str | None = None,
    target_id: str | None = None,
    *,
    name: str = "take_action",
    verb: str | None = None,
    **params: Any,
) -> LLMResponse:
    """Create an LLMResponse with a provider-emitted tool call."""
    if verb is not None:
        candidate_ref = _action_id_for(verb, target_id)
        resolved_verb = verb
    elif action_id is not None and action_id in VERB_CATEGORIES and ":" not in action_id:
        candidate_ref = _action_id_for(action_id, target_id)
        resolved_verb = action_id
    elif action_id is not None:
        candidate_ref = action_id
        resolved_verb = action_id.split(":", 1)[0]
    else:
        raise ValueError("Either action_id or verb must be provided")

    args: dict[str, Any] = {
        "candidate_ref": candidate_ref,
        "verb": resolved_verb,
    }
    resolved_target_id = target_id
    if resolved_target_id is None and ":" in candidate_ref:
        candidate_target = candidate_ref.split(":", 1)[1]
        if candidate_target != "self":
            resolved_target_id = candidate_target
    if resolved_target_id is not None:
        args["target_id"] = resolved_target_id
    if params:
        args["parameters"] = params
    return LLMResponse(
        tool_call=ToolCall(name=name, arguments=args),
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
    )


def _text_response(text: str) -> LLMResponse:
    """Create an LLMResponse with only text (no tool call)."""
    return LLMResponse(
        text=text,
        usage=TokenUsage(prompt_tokens=50, completion_tokens=10),
    )


def _json_action_text_response(arguments: dict[str, Any]) -> LLMResponse:
    return _text_response(json.dumps(arguments))


class TestLLMAgentBasics:
    def test_agent_id_and_name(self):
        provider = FakeLLMProvider([])
        agent = LLMAgent("agent_1", provider, goal="Test")
        assert agent.agent_id == "agent_1"
        assert "agent_1" in agent.name

    def test_custom_name(self):
        provider = FakeLLMProvider([])
        agent = LLMAgent("agent_1", provider, goal="Test", name="Dr. Kim")
        assert agent.name == "Dr. Kim"

    def test_observe_stores_observation(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        obs = _make_observation()
        agent.observe(obs)
        # Should not raise — observation is stored
        action_spec = _make_action_spec(("don_ppe", None))
        agent.act(action_spec)

    def test_act_without_observe_raises(self):
        provider = FakeLLMProvider([])
        agent = LLMAgent("agent_1", provider, goal="Test")
        with pytest.raises(RuntimeError, match="observe"):
            agent.act(_make_action_spec(("don_ppe", None)))


class TestLLMAgentAct:
    def test_returns_valid_action(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("don_ppe", None)))

        assert action.verb == "don_ppe"
        assert action.agent_id == "agent_1"
        assert action.category == ActionCategory.SAFETY

    def test_accepts_plain_json_text_action_output(self):
        provider = FakeLLMProvider(
            [
                _json_action_text_response(
                    {
                        "candidate_ref": _action_id_for("don_ppe"),
                    }
                )
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("don_ppe", None)))

        assert action.verb == "don_ppe"
        assert agent.audit_steps[0]["provider_tool_call_fallback_count"] == 0

    def test_accepts_candidate_ref_only_plus_free_parameters(self):
        provider = FakeLLMProvider(
            [
                _json_action_text_response(
                    {
                        "candidate_ref": _action_id_for("pipette_transfer", "pipette_1"),
                        "parameters": {"volume_ul": 25.0},
                    }
                )
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Transfer reagent")
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                ("pipette_transfer", "pipette_1", {"source_id": "src_1"}, ["volume_ul"], None),
            )
        )

        assert action.verb == "pipette_transfer"
        assert action.target_id == "pipette_1"
        assert action.parameters["source_id"] == "src_1"
        assert action.parameters["volume_ul"] == 25.0

    def test_action_with_target(self):
        provider = FakeLLMProvider([_tool_call_response("move_to", "room_2")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("move_to", "room_2")))

        assert action.verb == "move_to"
        assert action.target_id == "room_2"

    def test_action_with_parameters(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response(
                    action_id=_action_id_for("start_equipment", "equip_1"), rpm=1200
                ),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                ("start_equipment", "equip_1", {"command_name": "run_program"}, ["rpm"], None),
            )
        )

        assert action.verb == "start_equipment"
        assert action.parameters["command_name"] == "run_program"
        assert action.parameters["rpm"] == 1200

    def test_passes_system_prompt_to_provider(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Do PCR setup")
        agent.observe(_make_observation())
        agent.act(_make_action_spec(("don_ppe", None)))

        call = provider.captured_calls[0]
        assert "Do PCR setup" in call["system"]

    def test_passes_tool_definition(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())
        agent.act(_make_action_spec(("don_ppe", None), ("move_to", "room_2")))

        call = provider.captured_calls[0]
        assert call["tools"] is None

    def test_action_id_selection_merges_fixed_parameters(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response(action_id=_action_id_for("don_ppe")),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                ("don_ppe", None, {"ppe_type": "lab_coat"}, None, None),
            )
        )

        assert action.verb == "don_ppe"
        assert action.parameters["ppe_type"] == "lab_coat"

    def test_retries_when_required_parameters_are_missing(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response(action_id=_action_id_for("pipette_transfer", "pipette_1")),
                _tool_call_response(
                    action_id=_action_id_for("pipette_transfer", "pipette_1"),
                    volume_ul=25.0,
                ),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Transfer reagent")
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                (
                    "pipette_transfer",
                    "pipette_1",
                    {"source_id": "mix", "destination_id": "sample"},
                    ["volume_ul"],
                    None,
                ),
            )
        )

        assert provider.call_count == 2
        assert action.parameters == {
            "source_id": "mix",
            "destination_id": "sample",
            "volume_ul": 25.0,
        }

    def test_accepts_flattened_top_level_parameters(self):
        provider = FakeLLMProvider(
            [
                _json_action_text_response(
                    {
                        "candidate_ref": _action_id_for("pipette_transfer", "pipette_1"),
                        "verb": "pipette_transfer",
                        "target_id": "pipette_1",
                        "volume_ul": 25.0,
                    }
                ),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Transfer reagent")
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                (
                    "pipette_transfer",
                    "pipette_1",
                    {"source_id": "mix", "destination_id": "sample"},
                    ["volume_ul"],
                    None,
                ),
            )
        )

        assert action.parameters["volume_ul"] == 25.0

    def test_rejects_provider_tool_call_output(self):
        provider = FakeLLMProvider(
            [
                _provider_tool_call_response("don_ppe"),
                _provider_tool_call_response("don_ppe"),
                _provider_tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="failed to produce a valid action"):
            agent.act(_make_action_spec(("don_ppe", None)))


class TestLLMAgentRetries:
    def test_retries_on_no_tool_call(self):
        provider = FakeLLMProvider(
            [
                _text_response("I'll help you with that..."),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("don_ppe", None)))
        assert action.verb == "don_ppe"
        assert provider.call_count == 2

    def test_retries_on_wrong_tool_name(self):
        provider = FakeLLMProvider(
            [
                _provider_tool_call_response("don_ppe", name="wrong_tool"),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("don_ppe", None)))
        assert action.verb == "don_ppe"
        assert provider.call_count == 2

    def test_retries_on_unavailable_verb(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response("move_to", "room_2"),  # not in available actions
                _tool_call_response("don_ppe"),  # correct
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec(("don_ppe", None)))
        assert action.verb == "don_ppe"

    def test_raises_after_max_retries(self):
        bad_responses = [_text_response("text")] * (1 + MAX_RETRIES)
        provider = FakeLLMProvider(bad_responses)
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="failed to produce a valid action"):
            agent.act(_make_action_spec(("don_ppe", None)))

    def test_opaque_provider_protocol_error_is_not_treated_as_invalid_tool_call(self):
        bad_responses = [
            LLMResponse(
                usage=TokenUsage(prompt_tokens=50, completion_tokens=43),
                metadata={
                    "provider_protocol_error": (
                        "Required-tool completion returned nonzero completion tokens but no "
                        "text and no tool call."
                    ),
                    "opaque_response": True,
                },
            )
        ] * (1 + MAX_RETRIES)
        provider = FakeLLMProvider(bad_responses)
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="Provider error"):
            agent.act(_make_action_spec(("don_ppe", None)))

        assert agent.audit_steps[0]["provider_errors"] == 1 + MAX_RETRIES
        assert agent.audit_steps[0]["invalid_tool_calls"] == 0

    def test_retry_includes_error_feedback(self):
        provider = FakeLLMProvider(
            [
                _text_response("oops"),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())
        agent.act(_make_action_spec(("don_ppe", None)))

        retry_call = provider.captured_calls[1]
        last_msg = retry_call["messages"][-1]
        assert "<error>" in last_msg["content"]
        assert "exactly one valid JSON object" in last_msg["content"]
        assert "valid JSON object" in last_msg["content"]
        assert "nested wrapper objects" in last_msg["content"]

    def test_provider_tool_call_output_is_not_counted_as_fallback(self):
        provider = FakeLLMProvider(
            [
                _provider_tool_call_response("don_ppe"),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())

        agent.act(_make_action_spec(("don_ppe", None)))

        assert agent.audit_steps[0]["provider_tool_call_fallback_count"] == 0
        assert agent.audit_steps[0]["action_output_retry_count"] == 1

    def test_retry_reuses_same_turn_tool_schema(self):
        provider = FakeLLMProvider(
            [
                _text_response("oops"),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        action_spec = _make_action_spec(("don_ppe", None), ("move_to", "room_2"))
        agent.observe(_make_observation())

        agent.act(action_spec)

        assert provider.captured_calls[0]["tools"] is None
        assert provider.captured_calls[1]["tools"] is None
        assert (
            agent.audit_steps[0]["attempts"][0]["tool_schema"]
            == agent.audit_steps[0]["attempts"][1]["tool_schema"]
        )


class TestLLMAgentTokenTracking:
    def test_tracks_cumulative_usage(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response("don_ppe"),
                _tool_call_response("move_to", "room_2"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")

        agent.observe(_make_observation(tick=1))
        agent.act(_make_action_spec(("don_ppe", None)))

        agent.observe(_make_observation(tick=2))
        agent.act(_make_action_spec(("move_to", "room_2")))

        assert agent.cumulative_usage.prompt_tokens == 100
        assert agent.cumulative_usage.completion_tokens == 20

    def test_step_history_records_per_step(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())
        agent.act(_make_action_spec(("don_ppe", None)))

        assert len(agent.step_history) == 1
        assert agent.step_history[0].prompt_tokens == 50

    def test_retries_counted_in_stats(self):
        provider = FakeLLMProvider(
            [
                _text_response("oops"),
                _tool_call_response("don_ppe"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.observe(_make_observation())
        agent.act(_make_action_spec(("don_ppe", None)))

        assert agent.total_retries == 1
        assert agent.step_history[0].retries == 1


class TestLLMAgentTrajectory:
    def test_trajectory_records_decisions_up_to_max(self):
        responses = [_tool_call_response("don_ppe") for _ in range(15)]
        provider = FakeLLMProvider(responses)
        agent = LLMAgent("agent_1", provider, goal="Test", max_history_turns=5)

        for i in range(15):
            agent.observe(_make_observation(tick=i))
            agent.act(_make_action_spec(("don_ppe", None)))

        # The compact trajectory buffer caps at max_history_turns entries;
        # older decisions are dropped with a counted elision marker.
        assert len(agent._trajectory) == 5
        assert agent._trajectory.dropped_entry_count == 10

    def test_trajectory_block_appears_in_prompt(self):
        responses = [_tool_call_response("don_ppe") for _ in range(4)]
        provider = FakeLLMProvider(responses)
        agent = LLMAgent("agent_1", provider, goal="Test", max_history_turns=10)

        for i in range(3):
            agent.observe(_make_observation(tick=i))
            agent.act(_make_action_spec(("don_ppe", None)))
            agent.notify_result(success=True, description="donned")

        # The trajectory block must be present in the messages sent to
        # the provider on the 4th call.
        agent.observe(_make_observation(tick=3))
        agent.act(_make_action_spec(("don_ppe", None)))

        last_messages = provider.captured_calls[-1]["messages"]
        assert len(last_messages) == 1  # single user message, no _history list
        content = last_messages[0]["content"]
        assert "<trajectory>" in content
        assert "verb=don_ppe" in content
        assert "→ ok" in content


class TestLLMAgentQuery:
    def test_query_returns_text(self):
        provider = FakeLLMProvider([_text_response("The sample is in storage.")])
        agent = LLMAgent("agent_1", provider, goal="Test")

        result = agent.query("Where is the sample?")
        assert result == "The sample is in storage."

    def test_query_tracks_usage(self):
        provider = FakeLLMProvider([_text_response("answer")])
        agent = LLMAgent("agent_1", provider, goal="Test")
        agent.query("question")

        assert agent.cumulative_usage.prompt_tokens == 50


class TestLLMAgentActionResult:
    def test_notify_result_appears_in_next_turn(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response("don_ppe"),
                _tool_call_response("move_to", "room_2"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")

        agent.observe(_make_observation(tick=1))
        agent.act(_make_action_spec(("don_ppe", None)))
        agent.notify_result(True, "Donned lab coat.")

        agent.observe(_make_observation(tick=2))
        agent.act(_make_action_spec(("move_to", "room_2")))

        # The second call should include the previous result in the
        # compact trajectory block (not a full-state history replay).
        second_call = provider.captured_calls[1]
        last_user_msg = [m for m in second_call["messages"] if m["role"] == "user"][-1]
        assert "→ ok" in last_user_msg["content"]
        assert "<turn_feedback>" in last_user_msg["content"]
        assert "Result: succeeded" in last_user_msg["content"]
        assert "don_ppe" in last_user_msg["content"]
        assert "Donned lab coat" in last_user_msg["content"]

    def test_turn_status_appears_in_action_turn(self):
        provider = FakeLLMProvider([_tool_call_response("don_ppe")])
        agent = LLMAgent("agent_1", provider, goal="Test")

        agent.set_turn_status(
            TurnStatus(
                scenario_name="pcr_setup",
                steps_used=4,
                max_steps=25,
            )
        )
        agent.observe(_make_observation(tick=1))
        agent.act(_make_action_spec(("don_ppe", None)))

        content = provider.captured_calls[0]["messages"][0]["content"]
        assert "<turn_status>" in content
        assert "Steps used: 4/25" in content

    def test_followup_turn_does_not_include_assistant_action_transcript(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response("don_ppe"),
                _tool_call_response("move_to", "room_2"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")

        agent.observe(_make_observation(tick=1))
        agent.act(_make_action_spec(("don_ppe", None)))
        agent.notify_result(True, "Donned lab coat.")

        agent.observe(_make_observation(tick=2))
        agent.act(_make_action_spec(("move_to", "room_2")))

        second_call = provider.captured_calls[1]
        assert all(message["role"] != "assistant" for message in second_call["messages"])
        assert not any(
            "Action:" in str(message.get("content", "")) for message in second_call["messages"]
        )

    def test_action_surface_rebuilds_from_live_turn_state_each_turn(self):
        provider = FakeLLMProvider(
            [
                _tool_call_response("don_ppe"),
                _tool_call_response("move_to", "room_2"),
            ]
        )
        agent = LLMAgent("agent_1", provider, goal="Test")

        first_spec = _make_action_spec(("don_ppe", None))
        second_spec = _make_action_spec(("move_to", "room_2"))

        agent.observe(_make_observation(tick=1))
        agent.act(first_spec)
        agent.notify_result(True, "Donned lab coat.")

        agent.observe(_make_observation(tick=2))
        agent.act(second_spec)

        first_content = provider.captured_calls[0]["messages"][0]["content"]
        second_content = provider.captured_calls[1]["messages"][0]["content"]
        assert "[don_ppe:self]" in first_content
        assert "[move_to:room_2]" in second_content
