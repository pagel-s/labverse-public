"""Tests for the single-shot text planner agent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from labverse.agent.interface import Observation
from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage
from labverse.agent.text_planner import (
    TextPlannerAgent,
    _build_system_prompt,
    normalize_json_text_response,
)
from labverse.core.types import ActionCategory, BSLLevel, PersonRole, PPEType, RoomType
from labverse.models.action import ActionSpec, AvailableAction
from labverse.models.equipment import Equipment, EquipmentCommand, EquipmentParameter
from labverse.models.person import Person
from labverse.models.space import Door, LabLayout, Room
from labverse.world.state import WorldState


class FakePlannerProvider(LLMProvider):
    """Test provider that captures the prompt and returns fixed text."""

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
        del tools, tool_choice
        self.captured_calls.append({"messages": messages, "system": system})
        if not self._responses:
            raise RuntimeError("No fake planner responses left")
        return self._responses.pop(0)


def _make_world() -> WorldState:
    wet_lab = Room(
        id="wet_lab",
        name="Wet Lab",
        room_type=RoomType.WET_LAB,
        bsl_level=BSLLevel.BSL1,
    )
    office = Room(
        id="office",
        name="Office",
        room_type=RoomType.OFFICE,
        bsl_level=BSLLevel.BSL1,
    )
    world = WorldState(
        lab=LabLayout(
            id="test_lab",
            name="Test Lab",
            rooms={"wet_lab": wet_lab, "office": office},
            doors=[Door(id="door_office_wet_lab", connects=("office", "wet_lab"))],
        )
    )
    world.add_person(
        Person(
            id="tech_1",
            name="Test Tech",
            role=PersonRole.TECHNICIAN,
            current_room_id="wet_lab",
            current_ppe={
                PPEType.LAB_COAT,
                PPEType.SAFETY_GLASSES,
                PPEType.CLOSED_TOE_SHOES,
            },
        )
    )
    world.add_equipment(
        Equipment(
            id="pcr_1",
            name="PCR Thermocycler",
            equipment_type="thermocycler",
            room_id="wet_lab",
            commands={
                "run_program": EquipmentCommand(
                    name="run_program",
                    parameters=[
                        EquipmentParameter(
                            name="annealing_temp",
                            description="Annealing temperature",
                            param_type="float",
                        )
                    ],
                )
            },
        )
    )
    return world


def _make_observation() -> Observation:
    room = Room(
        id="wet_lab",
        name="Wet Lab",
        room_type=RoomType.WET_LAB,
        bsl_level=BSLLevel.BSL1,
    )
    return Observation(
        timestamp=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        tick=0,
        current_room=room,
        narrative="You are in the Wet Lab.",
    )


def _make_action_spec(*action_defs: tuple[Any, ...]) -> ActionSpec:
    if not action_defs:
        action_defs = (("move_to", "office"),)

    category_map = {
        "move_to": ActionCategory.MOVE,
        "don_ppe": ActionCategory.SAFETY,
        "start_equipment": ActionCategory.USE_EQUIPMENT,
        "log_observation": ActionCategory.OBSERVE,
    }
    actions: list[AvailableAction] = []
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
                action_id=f"{verb}:{target_id or 'self'}",
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


class TestTextPlannerAgent:
    def test_system_prompt_renders_with_generic_shape_example(self) -> None:
        prompt = _build_system_prompt(allowed_verbs={"wait", "log_observation"})

        assert "Generic shape example only:" in prompt
        assert '{"verb":"<verb>"}' in prompt
        assert "Output ONLY the valid JSON array." in prompt

    def test_normalize_json_text_response_strips_separator_and_code_fence(self) -> None:
        raw = '---\n```json\n[{"verb": "wait"}]\n```'

        normalized = normalize_json_text_response(raw)

        assert normalized == '[{"verb": "wait"}]'

    def test_repairs_malformed_json_plan_text(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text="[{verb:'move_to', target_id:'office'}]",
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        agent = TextPlannerAgent("tech_1", provider, goal="Leave the lab.")
        agent.observe(_make_observation())

        action = agent.act(_make_action_spec())

        assert action.verb == "move_to"
        assert action.target_id == "office"
        assert agent.audit_steps[0]["json_repair_count"] == 1

    def test_repairs_stringified_parameters_in_plan_steps(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text=(
                        '[{"verb":"start_equipment","target_id":"pcr_1",'
                        '"parameters":"{\\"command_name\\": \\"run_program\\", '
                        '\\"annealing_temp\\": 55.0}"}]'
                    ),
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Run the PCR machine.",
            world_state=_make_world(),
        )
        agent.observe(_make_observation())

        action = agent.act(
            _make_action_spec(
                (
                    "start_equipment",
                    "pcr_1",
                    {"command_name": "run_program"},
                    ["annealing_temp"],
                    [],
                ),
            )
        )

        assert action.parameters["annealing_temp"] == 55.0
        assert agent.audit_steps[0]["stringified_parameters_repair_count"] == 1

    def test_includes_full_lab_state_in_initial_prompt(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "office", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Leave the lab.",
            world_state=_make_world(),
            max_steps=10,
        )
        agent.observe(_make_observation())

        agent.act(_make_action_spec())

        first_call = provider.captured_calls[0]
        user_message = first_call["messages"][0]["content"]
        assert "<full_lab_state_json>" in user_message
        assert "<valid_id_inventory_json>" in user_message
        assert "<task_contract>" in user_message
        assert "<run_constraints>" in user_message
        assert "Max steps: 10" in user_message
        assert "<goal>" not in user_message
        assert '"test_lab"' in user_message

    def test_raises_stop_iteration_when_plan_exhausted(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "office", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        agent = TextPlannerAgent("tech_1", provider, goal="Leave the lab.")
        agent.observe(_make_observation())

        first_action = agent.act(_make_action_spec())

        assert first_action.verb == "move_to"
        assert agent.remaining_steps == 0

        agent.observe(_make_observation())
        with pytest.raises(StopIteration):
            agent.act(_make_action_spec())

    def test_initial_context_includes_parameter_hints(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text=(
                        '[{"verb": "start_equipment", "target_id": "pcr_1", '
                        '"parameters": {"command_name": "run_program", "annealing_temp": 55.0}}]'
                    ),
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        action_spec = ActionSpec(
            available_actions=[
                AvailableAction(
                    action_id="start_equipment:pcr_1",
                    verb="start_equipment",
                    category=ActionCategory.USE_EQUIPMENT,
                    description="Start PCR thermocycler",
                    target_id="pcr_1",
                    fixed_parameters={"command_name": "run_program"},
                    required_parameters=["annealing_temp"],
                    parameter_details={
                        "annealing_temp": "Annealing temperature; unit=C; current=55.0"
                    },
                    parameter_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"annealing_temp": {"type": "number"}},
                        "required": ["annealing_temp"],
                    },
                )
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Run PCR.",
            world_state=_make_world(),
        )
        agent.observe(_make_observation())

        agent.act(action_spec)

        user_message = provider.captured_calls[0]["messages"][0]["content"]
        assert "fixed(command_name=run_program)" in user_message
        assert "hints(annealing_temp: type=number; Annealing temperature" in user_message

    def test_rejects_legacy_verbs_in_planner_output(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "equip_ppe", "parameters": {"ppe_items": ["lab_coat"]}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "equip_ppe", "parameters": {"ppe_items": ["lab_coat"]}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "equip_ppe", "parameters": {"ppe_items": ["lab_coat"]}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
            ]
        )
        agent = TextPlannerAgent("tech_1", provider, goal="Leave the lab.")
        agent.observe(_make_observation())

        with pytest.raises(
            RuntimeError,
            match="unsupported verb 'equip_ppe'.*canonical action vocabulary",
        ):
            agent.act(_make_action_spec())

        last_call = provider.captured_calls[0]
        assert "Do NOT output candidate_ref values in the plan" in last_call["system"]
        assert "Invalid legacy aliases include equip_ppe" in last_call["system"]

    def test_rejects_plan_with_unknown_world_id(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "imaginary_room", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "imaginary_room", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "imaginary_room", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Leave the lab.",
            world_state=_make_world(),
        )
        agent.observe(_make_observation())

        with pytest.raises(
            RuntimeError,
            match="target_id 'imaginary_room' is not a valid room ID",
        ):
            agent.act(_make_action_spec())

    def test_rejects_plan_that_moves_into_wet_lab_before_required_ppe(self) -> None:
        world = _make_world()
        person = world.get_person("tech_1")
        person.current_room_id = "office"
        person.current_ppe = set()
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "wet_lab", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "wet_lab", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "wet_lab", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Enter the wet lab.",
            world_state=world,
        )
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="before donning required PPE"):
            agent.act(_make_action_spec(("move_to", "wet_lab")))

    def test_rejects_plan_that_uses_equipment_before_moving_to_target_room(self) -> None:
        world = _make_world()
        person = world.get_person("tech_1")
        person.current_room_id = "office"
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text=(
                        '[{"verb": "start_equipment", "target_id": "pcr_1", '
                        '"parameters": {"command_name": "run_program", "annealing_temp": 55.0}}]'
                    ),
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text=(
                        '[{"verb": "start_equipment", "target_id": "pcr_1", '
                        '"parameters": {"command_name": "run_program", "annealing_temp": 55.0}}]'
                    ),
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text=(
                        '[{"verb": "start_equipment", "target_id": "pcr_1", '
                        '"parameters": {"command_name": "run_program", "annealing_temp": 55.0}}]'
                    ),
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Run PCR.",
            world_state=world,
        )
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="while the agent is in 'office'"):
            agent.act(
                _make_action_spec(
                    (
                        "start_equipment",
                        "pcr_1",
                        {"command_name": "run_program"},
                        ["annealing_temp"],
                        None,
                    )
                )
            )

    def test_rejects_empty_plan(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text="[]",
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=2),
                ),
                LLMResponse(
                    text="[]",
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=2),
                ),
                LLMResponse(
                    text="[]",
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=2),
                ),
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Leave the lab.",
            world_state=_make_world(),
        )
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="Plan must contain at least one step"):
            agent.act(_make_action_spec())

    def test_preserves_last_invalid_reason_when_retries_exhaust(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "imaginary_room", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "imaginary_room", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "wet_lab", "parameters": {}}] trailing',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                ),
            ]
        )
        agent = TextPlannerAgent(
            "tech_1",
            provider,
            goal="Leave the lab.",
            world_state=_make_world(),
        )
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="Planner grounding failure"):
            agent.act(_make_action_spec())

        assert agent.audit_steps[-1]["error"].startswith("Planner grounding failure:")

    def test_replay_fails_when_planned_step_no_longer_matches_surface(self) -> None:
        provider = FakePlannerProvider(
            [
                LLMResponse(
                    text='[{"verb": "move_to", "target_id": "office", "parameters": {}}]',
                    usage=TokenUsage(prompt_tokens=12, completion_tokens=6),
                )
            ]
        )
        agent = TextPlannerAgent("tech_1", provider, goal="Leave the lab.")
        agent.observe(_make_observation())

        with pytest.raises(RuntimeError, match="Planner grounding failure"):
            agent.act(_make_action_spec(("don_ppe", None)))
