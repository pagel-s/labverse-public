"""Tests for LLM prompt formatting."""

import re
from datetime import UTC, datetime

from labverse.agent.interface import EquipmentSummary, Observation, SampleSummary
from labverse.agent.prompt import (
    TurnStatus,
    build_action_tool,
    build_global_structured_system_prompt,
    build_local_structured_user_message,
    build_narrative_system_prompt,
    build_system_prompt,
    format_action_result_message,
    format_available_actions,
    format_move_action_guidance,
    format_observation_message,
    format_timed_process_status,
    format_turn_status,
    summarize_history,
)
from labverse.core.types import (
    ActionCategory,
    BSLLevel,
    EquipmentStatus,
    PPEType,
    RoomType,
)
from labverse.models.action import AvailableAction
from labverse.models.space import Room


def _make_room() -> Room:
    return Room(
        id="room_1",
        name="Main Wet Lab",
        room_type=RoomType.WET_LAB,
        bsl_level=BSLLevel.BSL2,
    )


def _make_observation(**overrides) -> Observation:
    defaults = dict(
        timestamp=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        tick=5,
        current_room=_make_room(),
        narrative="You are in the Main Wet Lab.",
    )
    defaults.update(overrides)
    return Observation(**defaults)


def _make_available_action(**overrides) -> AvailableAction:
    defaults = dict(
        verb="don_ppe",
        category=ActionCategory.SAFETY,
        description="Don lab coat",
    )
    defaults.update(overrides)
    defaults.setdefault(
        "action_id",
        f"{defaults['verb']}:{defaults.get('target_id') or 'self'}",
    )
    return AvailableAction(**defaults)


class TestBuildSystemPrompt:
    def test_includes_goal(self):
        prompt = build_system_prompt(goal="Complete the PCR setup protocol.")
        assert "Complete the PCR setup protocol." in prompt

    def test_includes_environment_section(self):
        prompt = build_system_prompt(goal="test")
        assert "<environment>" in prompt
        assert "</environment>" in prompt
        assert "<task_contract>" in prompt

    def test_includes_rules_section(self):
        prompt = build_system_prompt(goal="test")
        assert "<action_contract>" in prompt
        assert "take_action" in prompt

    def test_includes_explicit_tool_response_format(self):
        prompt = build_system_prompt(goal="test")
        assert "<response_format>" in prompt
        assert "valid JSON object" in prompt
        assert "normal assistant text response" in prompt
        assert "<take_action>" in prompt
        assert '{"candidate_ref":"move_1"}' in prompt
        assert '{"candidate_ref":"move_1","parameters":{"target_id":"<target_id>"}}' in prompt

    def test_includes_negative_parameters_guidance(self):
        prompt = build_system_prompt(goal="test")
        assert (
            "Never place candidate_ref, verb, actor_id, or target_id inside the parameters "
            "object." in prompt
        )
        assert (
            "The parameters object is only for the free parameter fields shown by the selected "
            "candidate." in prompt
        )

    def test_includes_ppe_guidance(self):
        prompt = build_system_prompt(goal="test")
        assert "PPE" in prompt

    def test_shared_prompt_text_is_identical_across_single_agent_modes(self):
        prompts = [
            build_system_prompt(goal="test"),
            build_narrative_system_prompt(goal="test"),
            build_global_structured_system_prompt(goal="test"),
        ]

        normalized = [_normalize_prompt(prompt) for prompt in prompts]

        assert normalized[0] == normalized[1] == normalized[2]
        assert "<state_representation>\n<STATE_BLOCK>\n</state_representation>" in normalized[0]


class TestFormatObservationMessage:
    def test_contains_narrative(self):
        obs = _make_observation()
        msg = format_observation_message(obs)
        assert msg["role"] == "user"
        assert "Main Wet Lab" in msg["content"]

    def test_contains_tick(self):
        obs = _make_observation(tick=42)
        msg = format_observation_message(obs)
        assert "Tick: 42" in msg["content"]

    def test_contains_room_type(self):
        obs = _make_observation()
        msg = format_observation_message(obs)
        assert "wet_lab" in msg["content"]

    def test_includes_ppe_when_present(self):
        obs = _make_observation(current_ppe={PPEType.LAB_COAT, PPEType.SAFETY_GLASSES})
        msg = format_observation_message(obs)
        assert "lab_coat" in msg["content"]
        assert "safety_glasses" in msg["content"]

    def test_includes_carrying_when_present(self):
        obs = _make_observation(carrying=["sample_1", "reagent_2"])
        msg = format_observation_message(obs)
        assert "sample_1" in msg["content"]

    def test_includes_alerts_when_present(self):
        obs = _make_observation(active_alerts=["PPE required for this room"])
        msg = format_observation_message(obs)
        assert "PPE required" in msg["content"]

    def test_includes_observation_tags(self):
        obs = _make_observation()
        msg = format_observation_message(obs)
        assert "<observation>" in msg["content"]
        assert "</observation>" in msg["content"]

    def test_includes_timed_process_status_when_visible(self):
        obs = _make_observation(
            visible_equipment=[
                EquipmentSummary(
                    id="incubator_1",
                    name="Incubator 1",
                    equipment_type="incubator",
                    status=EquipmentStatus.RUNNING,
                    running_command="incubate",
                )
            ],
            visible_samples=[
                SampleSummary(
                    id="hela_p12",
                    name="HeLa P12",
                    sample_type="cell_culture",
                    status="incubating",
                    equipment_id="incubator_1",
                )
            ],
        )

        msg = format_observation_message(obs)

        assert "<timed_process_status>" in msg["content"]
        assert "incubator_1 status=running command=incubate" in msg["content"]
        assert "hela_p12 stored_in=incubator_1" in msg["content"]


class TestTurnStatus:
    def test_format_turn_status_is_non_oracle(self):
        text = format_turn_status(
            TurnStatus(
                scenario_name="pcr_setup",
                steps_used=3,
                max_steps=25,
            )
        )

        assert "<turn_status>" in text
        assert "Steps used: 3/25" in text
        assert "Steps remaining: 22" in text
        assert "completion_ratio" not in text

    def test_local_structured_user_message_includes_turn_status(self):
        message = build_local_structured_user_message(
            _make_observation(),
            [_make_available_action()],
            turn_status=TurnStatus(
                scenario_name="pcr_setup",
                steps_used=0,
                max_steps=25,
            ),
        )

        assert "<turn_status>" in message["content"]
        assert "<observation>" in message["content"]
        assert "<available_actions>" in message["content"]


class TestFormatAvailableActions:
    def test_formats_numbered_list(self):
        actions = [
            _make_available_action(verb="don_ppe", description="Don lab coat"),
            _make_available_action(
                verb="move_to",
                category=ActionCategory.MOVE,
                description="Move to storage room",
                target_id="room_2",
            ),
        ]
        text = format_available_actions(actions)
        assert "1. [don_ppe:self] verb=don_ppe" in text
        assert "2. [move_to:room_2] verb=move_to" in text
        assert "target=room_2" in text

    def test_includes_required_parameters(self):
        actions = [
            _make_available_action(
                verb="start_equipment",
                required_parameters=["command_name", "rpm"],
            ),
        ]
        text = format_available_actions(actions)
        assert "requires: command_name, rpm" in text

    def test_includes_parameter_hints(self):
        actions = [
            _make_available_action(
                verb="start_equipment",
                required_parameters=["annealing_temp"],
                parameter_details={"annealing_temp": "Annealing temperature; unit=C; current=55.0"},
                parameter_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"annealing_temp": {"type": "number"}},
                    "required": ["annealing_temp"],
                },
            ),
        ]
        text = format_available_actions(actions)
        assert "parameter_hints:" in text
        assert "annealing_temp: type=number; Annealing temperature" in text

    def test_includes_estimated_duration(self):
        actions = [
            _make_available_action(
                verb="log_observation",
                estimated_duration_seconds=10.0,
            )
        ]
        text = format_available_actions(actions)

        assert "takes" in text

    def test_highlights_current_move_action_ids(self):
        actions = [
            _make_available_action(
                action_id="preview_move_office",
                verb="move_to",
                category=ActionCategory.MOVE,
                description="Move to office",
                target_id="office",
            ),
            _make_available_action(
                action_id="preview_move_cold_room",
                verb="move_to",
                category=ActionCategory.MOVE,
                description="Move to cold room",
                target_id="cold_room",
            ),
        ]

        text = format_available_actions(actions)

        assert "Movement is destination-specific." in text
        assert "preview_move_office" in text
        assert "preview_move_cold_room" in text
        assert "may no longer be valid" in text

    def test_empty_actions(self):
        text = format_available_actions([])
        assert "No actions available" in text


def _normalize_prompt(prompt: str) -> str:
    return re.sub(
        r"<state_representation>\n.*?\n</state_representation>",
        "<state_representation>\n<STATE_BLOCK>\n</state_representation>",
        prompt,
        flags=re.S,
    )


class TestFormatMoveActionGuidance:
    def test_empty_when_no_move_actions_exist(self):
        assert format_move_action_guidance([_make_available_action()]) == ""

    def test_lists_single_move_action_id(self):
        action = _make_available_action(
            action_id="preview_move_office",
            verb="move_to",
            category=ActionCategory.MOVE,
            description="Move to office",
            target_id="office",
        )

        guidance = format_move_action_guidance([action])

        assert "only valid move candidate" in guidance
        assert "preview_move_office" in guidance
        assert "target_id" not in guidance


class TestTimedProcessStatus:
    def test_returns_empty_when_no_active_equipment_or_stored_samples(self):
        assert format_timed_process_status(_make_observation()) == ""


class TestFormatActionResultMessage:
    def test_success_message(self):
        msg = format_action_result_message(
            verb="don_ppe", success=True, description="Donned lab coat."
        )
        assert msg["role"] == "user"
        assert "succeeded" in msg["content"]
        assert "don_ppe" in msg["content"]

    def test_failure_message(self):
        msg = format_action_result_message(
            verb="move_to", success=False, description="Door is locked."
        )
        assert "failed" in msg["content"]


class TestBuildActionTool:
    def test_tool_has_correct_name(self):
        tool = build_action_tool([_make_available_action()])
        assert tool["name"] == "take_action"

    def test_tool_description_requires_plain_json_object(self):
        tool = build_action_tool([_make_available_action()])
        assert "plain JSON object" in tool["description"]
        assert "nested wrapper objects" in tool["description"]

    def test_tool_schema_uses_action_local_branches(self):
        actions = [
            _make_available_action(verb="don_ppe"),
            _make_available_action(action_id="move_to:room_2", verb="move_to"),
        ]
        tool = build_action_tool(actions)
        schema = tool["input_schema"]
        branch_action_ids = [
            branch["properties"]["candidate_ref"]["const"] for branch in schema["oneOf"]
        ]
        assert branch_action_ids == ["don_ppe:self", "move_to:room_2"]

    def test_candidate_ref_is_the_only_required_const_field(self):
        tool = build_action_tool([_make_available_action()])
        assert "candidate_ref" in tool["input_schema"]["required"]
        assert "verb" not in tool["input_schema"]["required"]
        assert "target_id" not in tool["input_schema"]["required"]
        assert "actor_id" not in tool["input_schema"]["required"]

    def test_preserves_distinct_action_ids(self):
        actions = [
            _make_available_action(
                action_id="don_ppe:coat",
                verb="don_ppe",
                description="Don lab coat",
            ),
            _make_available_action(
                action_id="don_ppe:gloves",
                verb="don_ppe",
                description="Don gloves",
            ),
        ]
        tool = build_action_tool(actions)
        candidate_refs = [
            branch["properties"]["candidate_ref"]["const"]
            for branch in tool["input_schema"]["oneOf"]
        ]
        assert candidate_refs == ["don_ppe:coat", "don_ppe:gloves"]

    def test_parameters_schema_is_action_local(self):
        actions = [
            _make_available_action(
                action_id="pipette_transfer:pipette_1",
                verb="pipette_transfer",
                required_parameters=["volume_ul"],
                parameter_details={"volume_ul": "Transfer volume in uL."},
                parameter_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"volume_ul": {"type": "number"}},
                    "required": ["volume_ul"],
                },
            )
        ]
        tool = build_action_tool(actions)
        schema = tool["input_schema"]
        branch = schema["oneOf"][0] if "oneOf" in schema else schema
        parameters_schema = branch["properties"]["parameters"]

        assert branch["additionalProperties"] is False
        assert "volume_ul" in parameters_schema["properties"]
        assert "parameters" in branch["required"]


class TestSummarizeHistory:
    def test_extracts_action_ids_from_user_side_action_results(self):
        history = [
            {
                "role": "user",
                "content": (
                    "<action_result>\n"
                    "Action 'preview_don_lab_coat' (verb: don_ppe) succeeded: donned coat\n"
                    "</action_result>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<action_result>\n"
                    "Your last action failed: preview_move_office (verb: move_to). blocked\n"
                    "</action_result>"
                ),
            },
        ]
        summary = summarize_history(history)
        assert "preview_don_lab_coat" in summary
        assert "preview_move_office" not in summary
        assert "current move candidate list" in summary

    def test_empty_history(self):
        assert summarize_history([]) == ""

    def test_no_assistant_messages(self):
        history = [{"role": "user", "content": "hello"}]
        assert summarize_history(history) == ""
