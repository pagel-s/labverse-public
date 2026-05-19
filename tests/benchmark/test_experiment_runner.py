"""Tests for the paper experiment runner paths."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from labverse.agent.audit import (
    build_artifact_integrity_step,
    finalize_step_audit,
    start_step_audit,
)
from labverse.agent.global_structured import (
    GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS,
    GlobalStructuredInteractiveAgent,
    format_world_state_summary_json,
)
from labverse.agent.prompt import TurnStatus
from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall
from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
)
from labverse.benchmarks.experiment import (
    OLLAMA_ACTION_MAX_TOKENS,
    ProviderOptions,
    _artifacts_from_capture,
    _create_provider,
    _interactive_team_observation_text,
    _interactive_team_system_prompt,
    _provider_options_for_condition,
    run_experiment,
    run_single,
    run_single_capture,
)
from labverse.benchmarks.paper_agents import (
    MultiAgentInteractiveCoordinator,
    TeamAvailableAction,
    _build_team_system_prompt,
    _build_text_only_multi_agent_system_prompt,
    _format_team_state_message,
    _parse_joint_action_output,
    _parse_joint_tool_call,
    _summarize_team_history,
    collect_team_actions,
)
from labverse.benchmarks.runner import BenchmarkRunner, ScenarioExecutionCapture
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    bsl2_ppe_steps,
    log_observation_step,
    move_to_step,
    retrieve_sample_step,
    start_equipment_step,
    stop_equipment_step,
    store_sample_in_equipment_step,
    store_sample_in_room_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import Event, EventLog, EventType
from labverse.core.types import ActionCategory
from labverse.gamemaster.adjudicator import GameMaster
from labverse.models.action import ActionResult, ActionResultStatus, AvailableAction


class FakeExperimentProvider(LLMProvider):
    """Provider that returns a fixed sequence of experiment responses."""

    def __init__(self, responses: list[Any]) -> None:
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
        call = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "system": system,
        }
        self.captured_calls.append(call)
        if not self._responses:
            raise RuntimeError("FakeExperimentProvider ran out of responses")
        response = self._responses.pop(0)
        if callable(response):
            response = response(call)
        return response


def _shared_schedule_plan() -> list[dict[str, Any]]:
    return [
        {"agent_id": "tech_1", "verb": "don_ppe", "parameters": {"ppe_type": "lab_coat"}},
        {"agent_id": "tech_2", "verb": "don_ppe", "parameters": {"ppe_type": "lab_coat"}},
        {"agent_id": "tech_3", "verb": "don_ppe", "parameters": {"ppe_type": "lab_coat"}},
        {"agent_id": "tech_3", "verb": "don_ppe", "parameters": {"ppe_type": "safety_glasses"}},
        {"agent_id": "tech_1", "verb": "don_ppe", "parameters": {"ppe_type": "safety_glasses"}},
        {"agent_id": "tech_2", "verb": "don_ppe", "parameters": {"ppe_type": "safety_glasses"}},
        {"agent_id": "tech_1", "verb": "don_ppe", "parameters": {"ppe_type": "closed_toe_shoes"}},
        {"agent_id": "tech_2", "verb": "don_ppe", "parameters": {"ppe_type": "closed_toe_shoes"}},
        {"agent_id": "tech_3", "verb": "don_ppe", "parameters": {"ppe_type": "closed_toe_shoes"}},
        {"agent_id": "tech_1", "verb": "move_to", "target_id": "wet_lab"},
        {"agent_id": "tech_2", "verb": "move_to", "target_id": "wet_lab"},
        {"agent_id": "tech_3", "verb": "move_to", "target_id": "wet_lab"},
        {
            "agent_id": "tech_1",
            "verb": "start_equipment",
            "target_id": "centrifuge_1",
            "parameters": {"command_name": "spin", "speed_rpm": 2000, "duration_seconds": 20.0},
        },
        {
            "agent_id": "tech_2",
            "verb": "wait",
        },
        {
            "agent_id": "tech_3",
            "verb": "wait",
        },
        {
            "agent_id": "tech_1",
            "verb": "wait",
        },
        {
            "agent_id": "tech_2",
            "verb": "start_equipment",
            "target_id": "centrifuge_1",
            "parameters": {"command_name": "spin", "speed_rpm": 2500, "duration_seconds": 20.0},
        },
        {
            "agent_id": "tech_3",
            "verb": "wait",
        },
        {
            "agent_id": "tech_1",
            "verb": "wait",
        },
        {
            "agent_id": "tech_2",
            "verb": "wait",
        },
        {
            "agent_id": "tech_3",
            "verb": "start_equipment",
            "target_id": "centrifuge_1",
            "parameters": {"command_name": "spin", "speed_rpm": 3000, "duration_seconds": 20.0},
        },
        {
            "agent_id": "tech_1",
            "verb": "wait",
        },
    ]


def _parse_legacy_team_action(action_id: str) -> tuple[str, str, str | None, dict[str, Any]]:
    agent_id, preview_ref = action_id.split("::", 1)
    if preview_ref.startswith("preview_don_"):
        return agent_id, "don_ppe", None, {"ppe_type": preview_ref.removeprefix("preview_don_")}
    if preview_ref.startswith("preview_doff_"):
        return agent_id, "doff_ppe", None, {"ppe_type": preview_ref.removeprefix("preview_doff_")}
    if preview_ref.startswith("preview_move_"):
        return agent_id, "move_to", preview_ref.removeprefix("preview_move_"), {}
    if preview_ref.startswith("preview_start_"):
        return agent_id, "start_equipment", preview_ref.removeprefix("preview_start_"), {}
    if preview_ref.startswith("preview_stop_"):
        return agent_id, "stop_equipment", preview_ref.removeprefix("preview_stop_"), {}
    if preview_ref == "preview_log_observation":
        return agent_id, "log_observation", None, {}
    raise ValueError(f"Unsupported legacy team action for test helper: {action_id}")


def _find_team_candidate_ref(
    call: dict[str, Any],
    *,
    agent_id: str,
    verb: str,
    target_id: str | None,
    fixed_markers: dict[str, Any],
) -> str:
    content = str(call["messages"][-1]["content"])
    for line in content.splitlines():
        if f"actor={agent_id}" not in line or f"verb={verb}" not in line:
            continue
        if target_id is not None and f"target_id={target_id}" not in line:
            continue
        if any(f"{name}={value}" not in line for name, value in fixed_markers.items()):
            continue
        match = re.search(r"\[(.*?)\]", line)
        if match is not None:
            return match.group(1)
    raise AssertionError(
        "Could not resolve team candidate_ref for "
        f"actor={agent_id} verb={verb} target_id={target_id}"
    )


def _allowed_parameter_keys(call: dict[str, Any], candidate_ref: str) -> set[str]:
    content = str(call["messages"][-1]["content"])
    for line in content.splitlines():
        if f"[{candidate_ref}]" not in line:
            continue
        allowed: set[str] = set()
        requires = re.search(r"\(requires: ([^)]+)\)", line)
        if requires is not None:
            allowed.update(part.strip() for part in requires.group(1).split(","))
        optional = re.search(r"\(optional: ([^)]+)\)", line)
        if optional is not None:
            allowed.update(part.strip() for part in optional.group(1).split(","))
        return {name for name in allowed if name}
    return set()


def _joint_action_response(action_id: str, **parameters: Any):
    agent_id, verb, target_id, fixed_markers = _parse_legacy_team_action(action_id)

    def responder(call: dict[str, Any]) -> LLMResponse:
        candidate_ref = _find_team_candidate_ref(
            call,
            agent_id=agent_id,
            verb=verb,
            target_id=target_id,
            fixed_markers=fixed_markers,
        )
        allowed_parameter_keys = _allowed_parameter_keys(call, candidate_ref)
        provided_parameters = {
            name: value for name, value in parameters.items() if name in allowed_parameter_keys
        }
        arguments: dict[str, Any] = {
            "candidate_ref": candidate_ref,
            "actor_id": agent_id,
            "verb": verb,
        }
        if target_id is not None:
            arguments["target_id"] = target_id
        if provided_parameters:
            arguments["parameters"] = provided_parameters
        return LLMResponse(
            text=json.dumps(arguments),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=4),
        )

    return responder


def _make_team_action(
    *, candidate_ref: str, agent_id: str, action: AvailableAction
) -> TeamAvailableAction:
    return TeamAvailableAction(
        candidate_ref=candidate_ref,
        agent_id=agent_id,
        action=action.model_copy(
            update={"candidate_ref": candidate_ref, "actor_id": agent_id},
            deep=True,
        ),
    )


def _find_single_candidate_ref(
    call: dict[str, Any],
    *,
    verb: str,
    target_id: str | None,
    fixed_markers: dict[str, Any],
) -> str:
    content = str(call["messages"][-1]["content"])
    for line in content.splitlines():
        if f"verb={verb}" not in line:
            continue
        if target_id is not None and f"target_id={target_id}" not in line:
            continue
        if any(f"{name}={value}" not in line for name, value in fixed_markers.items()):
            continue
        match = re.search(r"\[(.*?)\]", line)
        if match is not None:
            return match.group(1)
    raise AssertionError(
        f"Could not resolve single-agent candidate_ref for verb={verb} target_id={target_id}"
    )


def _single_action_response(
    *,
    verb: str,
    target_id: str | None = None,
    fixed_markers: dict[str, Any] | None = None,
    **parameters: Any,
):
    fixed_markers = fixed_markers or {}

    def responder(call: dict[str, Any]) -> LLMResponse:
        candidate_ref = _find_single_candidate_ref(
            call,
            verb=verb,
            target_id=target_id,
            fixed_markers=fixed_markers,
        )
        allowed_parameter_keys = _allowed_parameter_keys(call, candidate_ref)
        provided_parameters = {
            name: value for name, value in parameters.items() if name in allowed_parameter_keys
        }
        arguments: dict[str, Any] = {
            "candidate_ref": candidate_ref,
            "verb": verb,
        }
        if target_id is not None:
            arguments["target_id"] = target_id
        if provided_parameters:
            arguments["parameters"] = provided_parameters
        return LLMResponse(
            text=json.dumps(arguments),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=4),
        )

    return responder


def test_run_single_global_one_shot_shared_equipment_uses_multi_agent_plan(monkeypatch) -> None:
    provider = FakeExperimentProvider(
        [
            LLMResponse(
                text=json.dumps(_shared_schedule_plan()),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=12),
            )
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition="global_one_shot_planner",
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert result.score.success
    assert result.score.blocked_actions == 0
    assert result.prompt_tokens == 24
    prompt = provider.captured_calls[0]["messages"][0]["content"]
    assert "<world_state_json>" in prompt
    assert "<valid_id_inventory_json>" in prompt
    assert "<task_contract>" in prompt
    assert "<run_constraints>" in prompt
    assert "Max steps: 30" in prompt
    assert "<initial_team_state>" in prompt
    assert "<initial_available_actions>" in prompt
    system = provider.captured_calls[0]["system"]
    assert "wait()" in system
    assert "Use log_observation only when the goal explicitly requires documentation." in system
    assert "If the task contract specifies an exact note, report, or finding format" in system
    assert "parameters.notes" in system


def test_run_single_global_one_shot_rejects_empty_team_plan(monkeypatch) -> None:
    provider = FakeExperimentProvider(
        [
            LLMResponse(
                text="[]",
                usage=TokenUsage(prompt_tokens=24, completion_tokens=2),
            ),
            LLMResponse(
                text="[]",
                usage=TokenUsage(prompt_tokens=24, completion_tokens=2),
            ),
            LLMResponse(
                text="[]",
                usage=TokenUsage(prompt_tokens=24, completion_tokens=2),
            ),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition="global_one_shot_planner",
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert not result.score.success
    assert result.score.details["runner_error"] == "Plan must contain at least one step"


def test_run_single_global_one_shot_rejects_team_plan_missing_ppe_before_move(
    monkeypatch,
) -> None:
    invalid_plan = [
        {"agent_id": "tech_1", "verb": "don_ppe", "parameters": {"ppe_type": "lab_coat"}},
        {"agent_id": "tech_1", "verb": "move_to", "target_id": "wet_lab"},
    ]
    provider = FakeExperimentProvider(
        [
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=6),
            ),
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=6),
            ),
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=6),
            ),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition="global_one_shot_planner",
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert not result.score.success
    assert "before donning required PPE" in result.score.details["runner_error"]


def test_run_single_global_one_shot_rejects_repeated_shared_equipment_start(
    monkeypatch,
) -> None:
    invalid_plan = [
        *_shared_schedule_plan(),
        {
            "agent_id": "tech_1",
            "verb": "start_equipment",
            "target_id": "centrifuge_1",
            "parameters": {
                "command_name": "spin",
                "speed_rpm": 2000,
                "duration_seconds": 20.0,
            },
        },
    ]
    provider = FakeExperimentProvider(
        [
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=12),
            ),
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=12),
            ),
            LLMResponse(
                text=json.dumps(invalid_plan),
                usage=TokenUsage(prompt_tokens=24, completion_tokens=12),
            ),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition="global_one_shot_planner",
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert not result.score.success
    assert "exactly once" in result.score.details["runner_error"]


def test_artifacts_from_capture_propagates_last_step_error_into_runner_error() -> None:
    score = BenchmarkScore(
        scenario_name="pcr_setup",
        success=False,
        completion_ratio=0.1,
        steps_taken=0,
        time_elapsed_seconds=0.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=0.0,
        details={"master_mix_transferred": False},
    )
    step = start_step_audit(
        actor_id="tech_1",
        actor_name="planner",
        actor_kind="planner",
        system_prompt=None,
        user_message={"role": "user", "content": "plan"},
        tool_schema=None,
        prompt_surface_kind="global_one_shot_planner",
    )
    step["error"] = "Planner grounding failure: move_to target_id 'wet_lab' is not a valid room ID"
    step["status"] = "failed"

    artifacts = _artifacts_from_capture(
        ScenarioExecutionCapture(
            score=score,
            step_audits=[step],
        ),
        usage=TokenUsage(),
    )

    assert (
        artifacts.score.details["runner_error"]
        == "Planner grounding failure: move_to target_id 'wet_lab' is not a valid room ID"
    )


def test_run_single_local_narrative_shared_equipment_uses_joint_action_tool(monkeypatch) -> None:
    provider = FakeExperimentProvider(
        [
            _joint_action_response("tech_1::preview_don_lab_coat"),
            _joint_action_response("tech_2::preview_don_lab_coat"),
            _joint_action_response("tech_3::preview_don_lab_coat"),
            _joint_action_response("tech_1::preview_don_safety_glasses"),
            _joint_action_response("tech_2::preview_don_safety_glasses"),
            _joint_action_response("tech_3::preview_don_safety_glasses"),
            _joint_action_response("tech_1::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_2::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_3::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_1::preview_move_wet_lab"),
            _joint_action_response("tech_2::preview_move_wet_lab"),
            _joint_action_response("tech_3::preview_move_wet_lab"),
            _joint_action_response(
                "tech_1::preview_start_centrifuge_1",
                speed_rpm=2000,
                duration_seconds=20.0,
            ),
            _joint_action_response(
                "tech_2::preview_log_observation",
                notes="tech_2 waiting for first slot",
            ),
            _joint_action_response(
                "tech_3::preview_log_observation",
                notes="tech_3 waiting for first slot",
            ),
            _joint_action_response(
                "tech_1::preview_log_observation",
                notes="tech_1 yielded centrifuge slot",
            ),
            _joint_action_response(
                "tech_2::preview_start_centrifuge_1",
                speed_rpm=2500,
                duration_seconds=20.0,
            ),
            _joint_action_response(
                "tech_3::preview_log_observation",
                notes="tech_3 waiting for second slot",
            ),
            _joint_action_response(
                "tech_1::preview_log_observation",
                notes="tech_1 waiting for slot rotation",
            ),
            _joint_action_response(
                "tech_2::preview_log_observation",
                notes="tech_2 complete",
            ),
            _joint_action_response(
                "tech_3::preview_start_centrifuge_1",
                speed_rpm=3000,
                duration_seconds=20.0,
            ),
            _joint_action_response(
                "tech_1::preview_log_observation",
                notes="tech_1 complete",
            ),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert result.score.success
    assert result.score.blocked_actions == 0
    assert provider.captured_calls[0]["tools"] is None
    assert "biosafety level" in provider.captured_calls[0]["messages"][0]["content"]


def test_run_single_local_structured_shared_equipment_uses_joint_action_tool(
    monkeypatch,
) -> None:
    provider = FakeExperimentProvider(
        [
            _joint_action_response("tech_1::preview_don_lab_coat"),
            _joint_action_response("tech_2::preview_don_lab_coat"),
            _joint_action_response("tech_3::preview_don_lab_coat"),
            _joint_action_response("tech_1::preview_don_safety_glasses"),
            _joint_action_response("tech_2::preview_don_safety_glasses"),
            _joint_action_response("tech_3::preview_don_safety_glasses"),
            _joint_action_response("tech_1::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_2::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_3::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_1::preview_move_wet_lab"),
            _joint_action_response("tech_2::preview_move_wet_lab"),
            _joint_action_response("tech_3::preview_move_wet_lab"),
            _joint_action_response(
                "tech_1::preview_start_centrifuge_1",
                speed_rpm=2000,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_2::preview_log_observation", notes="wait"),
            _joint_action_response("tech_3::preview_log_observation", notes="wait"),
            _joint_action_response("tech_1::preview_log_observation", notes="yield"),
            _joint_action_response(
                "tech_2::preview_start_centrifuge_1",
                speed_rpm=2500,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_3::preview_log_observation", notes="wait"),
            _joint_action_response("tech_1::preview_log_observation", notes="wait"),
            _joint_action_response("tech_2::preview_log_observation", notes="done"),
            _joint_action_response(
                "tech_3::preview_start_centrifuge_1",
                speed_rpm=3000,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_1::preview_log_observation", notes="done"),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition=CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert result.score.success
    assert result.score.blocked_actions == 0
    assert provider.captured_calls[0]["tools"] is None
    assert "<observation>" in provider.captured_calls[0]["messages"][0]["content"]


def test_run_single_global_structured_interactive_includes_world_state_json(
    monkeypatch,
) -> None:
    provider = FakeExperimentProvider(
        [_single_action_response(verb="don_ppe", fixed_markers={"ppe_type": "lab_coat"})]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    run_single(
        condition=CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="pcr_setup",
        replication=1,
    )

    first_call = provider.captured_calls[0]
    assert first_call["tools"] is None
    assert "<world_state_json>" in first_call["messages"][0]["content"]
    assert '"counts"' in first_call["messages"][0]["content"]
    assert '"equipment"' in first_call["messages"][0]["content"]
    assert '"commands"' not in first_call["messages"][0]["content"]
    assert '"properties"' not in first_call["messages"][0]["content"]


def test_format_world_state_json_compacts_verbose_world_fields() -> None:
    world = BenchmarkRunner().create_scenario("pcr_setup").configured_world()

    payload = format_world_state_summary_json(world)
    parsed = json.loads(payload)

    assert set(parsed) == {
        "doors",
        "equipment",
        "lab",
        "materials",
        "personnel",
        "rooms",
        "samples",
    }
    assert parsed["lab"]["counts"]["rooms"] >= 1
    assert all("adjacent_rooms" in room for room in parsed["rooms"])
    assert all("role" in person for person in parsed["personnel"])
    assert all("expiry_date" in material for material in parsed["materials"])
    assert all("incompatible_with" in material for material in parsed["materials"])
    assert '"commands"' not in payload
    assert '"properties"' not in payload
    assert '"chain_of_custody"' not in payload
    assert '"zones"' not in payload
    assert len(payload) < len(world.model_dump_json())


def test_global_structured_summary_includes_search_task_entities() -> None:
    world = BenchmarkRunner().create_scenario("safety_violation_detection").configured_world()

    payload = format_world_state_summary_json(world)
    parsed = json.loads(payload)

    material_ids = {material["id"] for material in parsed["materials"]}
    personnel_ids = {person["id"] for person in parsed["personnel"]}

    assert "expired_buffer" in material_ids
    assert "naoh_1" in material_ids
    assert "visitor_1" in personnel_ids

    expired_buffer = next(
        material for material in parsed["materials"] if material["id"] == "expired_buffer"
    )
    assert expired_buffer["expiry_date"] == "2025-12-31"

    naoh = next(material for material in parsed["materials"] if material["id"] == "naoh_1")
    assert naoh["incompatible_with"] == ["acids"]

    visitor = next(person for person in parsed["personnel"] if person["id"] == "visitor_1")
    assert visitor["role"] == "undergrad"


def test_multi_step_protocol_setup_overrides_equipment_durations() -> None:
    world = BenchmarkRunner().create_scenario("multi_step_protocol").configured_world()

    bsc = world.get_equipment("bsc_1")
    incubator = world.get_equipment("incubator_1")

    assert bsc.commands["work"].duration_seconds == 20.0
    assert incubator.commands["incubate"].duration_seconds == 20.0


def test_multi_step_protocol_default_incubator_duration_allows_completion() -> None:
    scenario = BenchmarkRunner().create_scenario("multi_step_protocol")
    world = scenario.configured_world()
    game_master = GameMaster(world)
    scripted = ScriptedAgent(
        "grad_kim",
        [
            *bsl2_ppe_steps(),
            move_to_step("tissue_culture"),
            start_equipment_step(
                "bsc_1",
                parameters={"command_name": "work"},
            ),
            retrieve_sample_step("hela_p12"),
            store_sample_in_equipment_step("hela_p12", "incubator_1"),
            start_equipment_step(
                "incubator_1",
                parameters={"command_name": "incubate"},
            ),
            log_observation_step("protocol hold checkpoint"),
            log_observation_step("protocol hold complete"),
        ],
    )

    for _ in range(scenario.get_max_steps()):
        game_master.run_step(scripted)
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            break
    else:
        raise AssertionError("Protocol did not complete within the benchmark step budget")

    assert score.success is True
    assert score.details["timed_completion_fired"] is True


def test_equipment_failure_recovery_accepts_valid_incident_report_and_secured_sample() -> None:
    scenario = BenchmarkRunner().create_scenario("equipment_failure_recovery")
    game_master = GameMaster(scenario.configured_world())
    scripted = ScriptedAgent(
        "tech_1",
        [
            *basic_lab_ppe_steps(),
            move_to_step("wet_lab"),
            stop_equipment_step("centrifuge_1"),
            retrieve_sample_step("pellet_1"),
            store_sample_in_room_step("pellet_1", "wet_lab"),
            log_observation_step("Incident report filed for centrifuge_1 imbalance failure"),
        ],
    )

    for _ in range(8):
        game_master.run_step(scripted)

    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())

    assert score.success is True
    assert score.details["sample_recovered"] is True
    assert score.details["incident_logged"] is True


def test_equipment_failure_recovery_rejects_vague_incident_note() -> None:
    scenario = BenchmarkRunner().create_scenario("equipment_failure_recovery")
    game_master = GameMaster(scenario.configured_world())
    scripted = ScriptedAgent(
        "tech_1",
        [
            *basic_lab_ppe_steps(),
            move_to_step("wet_lab"),
            stop_equipment_step("centrifuge_1"),
            retrieve_sample_step("pellet_1"),
            log_observation_step("handled the problem"),
        ],
    )

    for _ in range(7):
        game_master.run_step(scripted)

    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())

    assert score.success is False
    assert score.details["sample_recovered"] is True
    assert score.details["incident_logged"] is False


def test_shared_equipment_scorer_counts_successful_starts_not_attempts() -> None:
    scenario = BenchmarkRunner().create_scenario("shared_equipment_scheduling")
    world = scenario.configured_world()
    event_log = EventLog()
    now = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    def append_start(agent_id: str, action_id: str, *, success: bool) -> None:
        event_log.append(
            Event(
                event_type=EventType.ACTION_SUBMITTED,
                tick=0,
                timestamp=now,
                source_id=agent_id,
                target_id="centrifuge_1",
                description=f"{agent_id} submitted start_equipment",
                data={"action_id": action_id, "parameters": {"command_name": "spin"}},
            )
        )
        event_log.append(
            Event(
                event_type=EventType.ACTION_RESOLVED,
                tick=0,
                timestamp=now,
                source_id=agent_id,
                target_id="centrifuge_1",
                description="start_equipment resolved successfully"
                if success
                else "start_equipment failed",
                data={"action_id": action_id, "success": success},
            )
        )

    append_start("tech_1", "start_1", success=True)
    append_start("tech_1", "start_2", success=True)
    append_start("tech_2", "start_3", success=True)
    append_start("tech_3", "start_4", success=True)
    for index in range(3):
        event_log.append(
            Event(
                event_type=EventType.TIMED_EVENT,
                tick=index + 1,
                timestamp=now,
                target_id="centrifuge_1",
                description="centrifuge_1 completed spin",
            )
        )

    score = scenario.evaluate(event_log, world)

    assert score.success is False
    assert score.details["tech_1_start_count"] == 2.0
    assert score.details["tech_1_attempted_start_count"] == 2.0
    assert score.details["tech_2_start_count"] == 1.0
    assert score.details["tech_3_start_count"] == 1.0


def test_global_structured_agent_caps_verbatim_history_turns() -> None:
    agent = GlobalStructuredInteractiveAgent(
        agent_id="tech_1",
        provider=FakeExperimentProvider([]),
        goal="Test goal",
        max_history_turns=10,
    )

    assert agent._max_history_turns == GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS


def test_run_experiment_infers_ollama_provider_from_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    def fake_run_single(
        condition: str,
        provider_name: str,
        model: str,
        scenario_name: str,
        replication: int,
        *,
        provider_options=None,
    ) -> BenchmarkScore:
        del provider_options
        captured.append(
            {
                "condition": condition,
                "provider_name": provider_name,
                "model": model,
                "scenario_name": scenario_name,
                "replication": str(replication),
            }
        )
        return type(
            "FakeRunResult",
            (),
            {
                "condition": condition,
                "model": model,
                "provider": provider_name,
                "scenario": scenario_name,
                "replication": replication,
                "score": BenchmarkScore(
                    scenario_name=scenario_name,
                    success=False,
                    completion_ratio=0.0,
                    steps_taken=0,
                    time_elapsed_seconds=0.0,
                    safety_violations=0,
                    blocked_actions=0,
                    efficiency_score=0.0,
                ),
                "wall_time_seconds": 0.01,
            },
        )()

    monkeypatch.setattr("labverse.benchmarks.experiment.run_single", fake_run_single)

    run_experiment(
        conditions=[CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED],
        models=["ollama/qwen3:14b"],
        scenarios=["chemical_spill_response"],
        replications=1,
    )

    assert captured == [
        {
            "condition": CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
            "provider_name": "ollama",
            "model": "ollama/qwen3:14b",
            "scenario_name": "chemical_spill_response",
            "replication": "1",
        }
    ]


def test_team_history_summary_omits_stale_move_action_ids() -> None:
    summary = _summarize_team_history(
        [
            {
                "role": "user",
                "content": (
                    "<team_action_result>\n"
                    "Previous team action failed: tech_1::preview_move_office (verb: move_to). "
                    "Blocked.\n"
                    "</team_action_result>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<team_action_result>\n"
                    "Previous team action succeeded: tech_2::preview_log_observation "
                    "(verb: log_observation). Waited.\n"
                    "</team_action_result>"
                ),
            },
        ]
    )

    assert "preview_move_office" not in summary
    assert "tech_2::preview_log_observation" in summary
    assert "current move candidate list" in summary


def test_shared_prompt_text_is_identical_across_team_modes() -> None:
    prompts = [
        _interactive_team_system_prompt(CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED),
        _interactive_team_system_prompt(CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED),
        _interactive_team_system_prompt(CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED),
    ]

    normalized = [_normalize_team_prompt(prompt) for prompt in prompts]

    assert normalized[0] == normalized[1] == normalized[2]
    assert "<state_representation>\n<STATE_BLOCK>\n</state_representation>" in normalized[0]


def test_multi_agent_prompt_builders_render_successfully() -> None:
    one_shot_prompt = _build_text_only_multi_agent_system_prompt(allowed_verbs={"wait"})
    team_prompt = _build_team_system_prompt(state_representation="<STATE_BLOCK>")

    assert "Output a COMPLETE global plan as a valid JSON array." in one_shot_prompt
    assert "- wait()" in one_shot_prompt
    assert "<response_format>" in team_prompt
    assert "<STATE_BLOCK>" in team_prompt
    assert "If you include actor_id or target_id" not in team_prompt


def test_joint_action_parser_requires_required_parameters() -> None:
    team_actions = [
        _make_team_action(
            candidate_ref="t1_equipment_1",
            agent_id="tech_1",
            action=AvailableAction(
                action_id="preview_start_centrifuge_1",
                candidate_ref="t1_equipment_1",
                actor_id="tech_1",
                verb="start_equipment",
                category=ActionCategory.USE_EQUIPMENT,
                target_id="centrifuge_1",
                fixed_parameters={"command_name": "spin", "duration_seconds": 20.0},
                required_parameters=["speed_rpm"],
                description="Start centrifuge",
            ),
        )
    ]

    try:
        _parse_joint_action_output(
            LLMResponse(
                text=json.dumps(
                    {
                        "candidate_ref": "t1_equipment_1",
                        "actor_id": "tech_1",
                        "verb": "start_equipment",
                        "target_id": "centrifuge_1",
                    }
                )
            ),
            team_actions,
        )
    except ValueError as exc:
        assert "Missing required parameters" in str(exc)
    else:
        raise AssertionError("Expected missing required parameter validation")


def _normalize_team_prompt(prompt: str) -> str:
    return re.sub(
        r"<state_representation>\n.*?\n</state_representation>",
        "<state_representation>\n<STATE_BLOCK>\n</state_representation>",
        prompt,
        flags=re.S,
    )


def test_joint_action_parser_accepts_flattened_top_level_parameters() -> None:
    team_actions = [
        _make_team_action(
            candidate_ref="t1_equipment_1",
            agent_id="tech_1",
            action=AvailableAction(
                action_id="preview_start_centrifuge_1",
                candidate_ref="t1_equipment_1",
                actor_id="tech_1",
                verb="start_equipment",
                category=ActionCategory.USE_EQUIPMENT,
                target_id="centrifuge_1",
                fixed_parameters={"command_name": "spin", "duration_seconds": 20.0},
                required_parameters=["speed_rpm"],
                description="Start centrifuge",
            ),
        )
    ]

    action, selected_action_id, resolved_action_id, normalization_kinds = (
        _parse_joint_action_output(
            LLMResponse(
                text=json.dumps(
                    {
                        "candidate_ref": "t1_equipment_1",
                        "actor_id": "tech_1",
                        "verb": "start_equipment",
                        "target_id": "centrifuge_1",
                        "speed_rpm": 2000,
                    }
                )
            ),
            team_actions,
        )
    )

    assert action.parameters["command_name"] == "spin"
    assert action.parameters["speed_rpm"] == 2000
    assert selected_action_id == "t1_equipment_1"
    assert normalization_kinds == ["flattened_parameters"]


def test_joint_action_parser_rejects_provider_tool_calls() -> None:
    team_actions = [
        _make_team_action(
            candidate_ref="t1_observe_1",
            agent_id="tech_1",
            action=AvailableAction(
                action_id="preview_log_observation",
                candidate_ref="t1_observe_1",
                actor_id="tech_1",
                verb="log_observation",
                category=ActionCategory.OBSERVE,
                description="Wait safely",
                required_parameters=["notes"],
            ),
        )
    ]

    tool_call = ToolCall(
        name="default_api:take_action",
        arguments={
            "candidate_ref": "t1_observe_1",
            "actor_id": "tech_1",
            "verb": "log_observation",
            "parameters": {"notes": "waiting"},
        },
    )

    with pytest.raises(
        ValueError,
        match="Interactive output must be plain JSON text",
    ):
        _parse_joint_tool_call(tool_call, team_actions)


def test_joint_action_parser_rejects_bare_verb() -> None:
    team_actions = [
        _make_team_action(
            candidate_ref="t1_observe_1",
            agent_id="tech_1",
            action=AvailableAction(
                action_id="preview_log_observation",
                candidate_ref="t1_observe_1",
                actor_id="tech_1",
                verb="log_observation",
                category=ActionCategory.OBSERVE,
                description="Wait safely",
                required_parameters=["notes"],
            ),
        )
    ]

    with pytest.raises(ValueError, match="Unknown candidate_ref 'log_observation'"):
        _parse_joint_action_output(
            LLMResponse(
                text=json.dumps(
                    {
                        "candidate_ref": "log_observation",
                        "actor_id": "tech_1",
                        "verb": "log_observation",
                        "parameters": {"notes": "waiting"},
                    }
                )
            ),
            team_actions,
        )


def test_create_provider_caps_ollama_action_turn_tokens(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOllamaProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "labverse.agent.providers.ollama.OllamaProvider",
        FakeOllamaProvider,
    )

    options = _provider_options_for_condition(
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
        "ollama",
        ProviderOptions(max_tokens=1024),
    )
    _create_provider(
        "ollama",
        "ollama/gemma4:26b",
        options=options,
    )

    assert captured["model"] == "ollama/gemma4:26b"
    assert captured["max_tokens"] == OLLAMA_ACTION_MAX_TOKENS


def test_team_state_message_includes_non_oracle_turn_status() -> None:
    message = _format_team_state_message(
        goal="Coordinate centrifuge runs",
        observations={"tech_1": "Tech 1 is ready."},
        team_actions=[
            _make_team_action(
                candidate_ref="t1_movement_1",
                agent_id="tech_1",
                action=AvailableAction(
                    action_id="preview_move_wet_lab",
                    candidate_ref="t1_movement_1",
                    actor_id="tech_1",
                    verb="move_to",
                    category=ActionCategory.MOVE,
                    target_id="wet_lab",
                    description="Move to Wet Lab",
                ),
            )
        ],
        last_result_text=None,
        turn_status=TurnStatus(
            scenario_name="shared_equipment_scheduling",
            steps_used=12,
            max_steps=30,
        ),
    )

    assert "<turn_status>" in message
    assert "Steps used: 12/30" in message
    assert "completion_ratio" not in message
    assert "Movement is destination-specific." in message
    assert "actor=tech_1" in message
    assert "verb=move_to" in message


def test_run_single_local_narrative_preserves_token_usage_on_coordinator_failure(
    monkeypatch,
) -> None:
    provider = FakeExperimentProvider(
        [
            LLMResponse(
                text="not a tool call",
                usage=TokenUsage(prompt_tokens=17, completion_tokens=5),
            ),
            LLMResponse(
                text="still not a tool call",
                usage=TokenUsage(prompt_tokens=19, completion_tokens=6),
            ),
            LLMResponse(
                text="nope",
                usage=TokenUsage(prompt_tokens=23, completion_tokens=7),
            ),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result = run_single(
        condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert not result.score.success
    assert result.prompt_tokens == 59
    assert result.completion_tokens == 18
    assert result.error == "Multi-agent interactive coordinator failed to produce a valid action"


def test_run_single_capture_shared_equipment_records_trace_ids_for_interactive_conditions(
    monkeypatch,
) -> None:
    provider = FakeExperimentProvider(
        [
            _joint_action_response("tech_1::preview_don_lab_coat"),
            _joint_action_response("tech_2::preview_don_lab_coat"),
            _joint_action_response("tech_3::preview_don_lab_coat"),
            _joint_action_response("tech_1::preview_don_safety_glasses"),
            _joint_action_response("tech_2::preview_don_safety_glasses"),
            _joint_action_response("tech_3::preview_don_safety_glasses"),
            _joint_action_response("tech_1::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_2::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_3::preview_don_closed_toe_shoes"),
            _joint_action_response("tech_1::preview_move_wet_lab"),
            _joint_action_response("tech_2::preview_move_wet_lab"),
            _joint_action_response("tech_3::preview_move_wet_lab"),
            _joint_action_response(
                "tech_1::preview_start_centrifuge_1",
                speed_rpm=2000,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_2::preview_log_observation", notes="wait"),
            _joint_action_response("tech_3::preview_log_observation", notes="wait"),
            _joint_action_response("tech_1::preview_log_observation", notes="yield"),
            _joint_action_response(
                "tech_2::preview_start_centrifuge_1",
                speed_rpm=2500,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_3::preview_log_observation", notes="wait"),
            _joint_action_response("tech_1::preview_log_observation", notes="wait"),
            _joint_action_response("tech_2::preview_log_observation", notes="done"),
            _joint_action_response(
                "tech_3::preview_start_centrifuge_1",
                speed_rpm=3000,
                duration_seconds=20.0,
            ),
            _joint_action_response("tech_1::preview_log_observation", notes="done"),
        ]
    )
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: provider,
    )

    result, artifacts = run_single_capture(
        condition=CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert result.score.success
    assert artifacts.trace_ids
    assert set(artifacts.trace_ids_by_agent) == {"tech_1", "tech_2", "tech_3"}
    assert all(trace_ids for trace_ids in artifacts.trace_ids_by_agent.values())


def test_multi_agent_interactive_coordinator_records_failed_audit_on_unexpected_internal_error(
    monkeypatch,
) -> None:
    provider = FakeExperimentProvider(
        [
            _joint_action_response("tech_1::preview_don_lab_coat"),
        ]
    )
    scenario = BenchmarkRunner().create_scenario("shared_equipment_scheduling")
    from labverse.gamemaster.adjudicator import GameMaster

    game_master = GameMaster(scenario.configured_world())
    agent_ids = [agent.agent_id for agent in scenario.build_scripted_agents()]
    observations = {agent_id: game_master.get_observation(agent_id) for agent_id in agent_ids}
    action_specs = {agent_id: game_master.get_available_actions(agent_id) for agent_id in agent_ids}
    team_actions = collect_team_actions(
        {agent_id: action_spec.available_actions for agent_id, action_spec in action_specs.items()}
    )
    coordinator = MultiAgentInteractiveCoordinator(
        provider=provider,
        goal=scenario.get_goal_description(),
        agent_ids=agent_ids,
    )

    def boom(*args: Any, **kwargs: Any) -> None:
        raise TypeError("unexpected audit failure")

    monkeypatch.setattr("labverse.benchmarks.paper_agents.record_attempt", boom)

    with pytest.raises(TypeError, match="unexpected audit failure"):
        coordinator.next_action(
            observations=_interactive_team_observation_text(
                condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
                observations=observations,
            ),
            team_actions=team_actions,
            turn_status=TurnStatus(
                scenario_name=scenario.name,
                steps_used=0,
                max_steps=scenario.get_max_steps(),
            ),
        )

    assert len(coordinator.audit_steps) == 1
    assert coordinator.audit_steps[0]["status"] == "failed"
    assert coordinator.audit_steps[0]["error"] == "unexpected audit failure"


def test_multi_agent_interactive_coordinator_materializes_malformed_audit_payloads() -> None:
    coordinator = MultiAgentInteractiveCoordinator(
        provider=FakeExperimentProvider([]),
        goal="Coordinate centrifuge runs",
        agent_ids=["tech_1"],
    )
    coordinator._audit_steps.append(  # noqa: SLF001 - intentional white-box corruption test
        {"status": "succeeded", "payload": object()}
    )

    materialized = coordinator.audit_steps

    assert len(materialized) == 1
    assert materialized[0]["artifact_integrity_errors"] >= 1
    assert materialized[0]["payload"]["__type__"] == "object"


def test_run_single_capture_propagates_artifact_integrity_steps_without_crashing(
    monkeypatch,
) -> None:
    def boom(actor: object, previous_count: int) -> list[dict[str, Any]]:
        return [
            build_artifact_integrity_step(
                error="synthetic audit corruption",
                actor_id="team_coordinator",
                actor_name="InteractiveText-gpt-test",
            )
        ]

    monkeypatch.setattr("labverse.benchmarks.experiment._new_actor_audit_steps", boom)
    monkeypatch.setattr(
        "labverse.benchmarks.experiment._create_provider",
        lambda provider_name, model, options=None: FakeExperimentProvider(
            [_joint_action_response("tech_1::preview_don_lab_coat")]
        ),
    )

    result, artifacts = run_single_capture(
        condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="shared_equipment_scheduling",
        replication=1,
    )

    assert not result.score.success
    assert artifacts.run_metrics["artifact_integrity_errors"] >= 1


def test_run_single_capture_includes_runner_traceback_on_unhandled_exception(monkeypatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("unexpected runner failure")

    monkeypatch.setattr("labverse.benchmarks.experiment._run_paper_scenario", boom)

    result, artifacts = run_single_capture(
        condition=CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
        provider_name="openai",
        model="gpt-test",
        scenario_name="pcr_setup",
        replication=1,
    )

    assert not result.score.success
    assert result.error == "unexpected runner failure"
    assert "runner_traceback" in artifacts.score.details
    assert "TypeError: unexpected runner failure" in artifacts.score.details["runner_traceback"]


def test_benchmark_runner_stops_after_success_without_post_completion_turn() -> None:
    class DummyScenario:
        name = "dummy"

        def evaluate(self, event_log: Any, final_state: Any) -> BenchmarkScore:
            return BenchmarkScore(
                scenario_name="dummy",
                success=bool(event_log.events),
                completion_ratio=1.0 if event_log.events else 0.0,
                efficiency_score=1.0,
                steps_taken=len(event_log.events),
                safety_violations=0,
                blocked_actions=0,
                time_elapsed_seconds=0.0,
                details={},
            )

    class DummyEventLog:
        def __init__(self) -> None:
            self.events: list[Any] = []

    class DummyWorld:
        def snapshot(self) -> dict[str, str]:
            return {"state": "ok"}

    class DummyTrace:
        def __init__(self, trace_id: str) -> None:
            self.trace_id = trace_id

    class DummyAgent:
        agent_id = "tech_1"
        name = "DummyAgent"
        is_exhausted = False

        def __init__(self) -> None:
            self.audit_steps: list[dict[str, Any]] = []

    class DummyGameMaster:
        def __init__(self) -> None:
            self.event_log = DummyEventLog()
            self.world = DummyWorld()
            self.calls = 0

        def trace_records(self, agent_id: str, limit: int = 10_000) -> list[DummyTrace]:
            return [DummyTrace("trace_1")] if self.calls else []

        def run_step(self, agent: DummyAgent) -> ActionResult:
            self.calls += 1
            step = start_step_audit(
                actor_id=agent.agent_id,
                actor_name=agent.name,
                actor_kind="agent",
                system_prompt="system",
                user_message={"role": "user", "content": "observe"},
                tool_schema={"name": "take_action"},
                prompt_surface_kind="local_structured_interactive_grounded",
            )
            finalize_step_audit(
                step,
                resolved_action_id="preview_log_observation",
                resolved_parameters={},
            )
            agent.audit_steps.append(step)
            self.event_log.events.append(type("Event", (), {"id": "event_1"})())
            return ActionResult(
                action_id="action_1",
                success=True,
                status=ActionResultStatus.SUCCEEDED,
                outcome_description="done",
            )

    runner = BenchmarkRunner()
    scenario = DummyScenario()
    agent = DummyAgent()
    game_master = DummyGameMaster()

    step_audits = runner._run_agents(  # noqa: SLF001 - direct regression coverage
        game_master,
        [agent],
        scenario=scenario,
        max_steps=2,
    )

    assert game_master.calls == 1
    assert len(step_audits) == 1
    assert not step_audits[0].get("post_completion_noop", False)
