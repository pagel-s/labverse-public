"""Tests for the agent-facing action space."""

from __future__ import annotations

from labverse.core.clock import SimulationClock
from labverse.core.types import ActionCategory, PPEType
from labverse.gamemaster import GameMaster
from labverse.models.action import Action, AvailableAction
from labverse.world.state import WorldState


def _equip_basic_lab_ppe() -> set[PPEType]:
    return {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    }


def _find_action(
    actions: list[AvailableAction],
    *,
    verb: str,
    target_id: str | None = None,
    description_fragment: str | None = None,
) -> AvailableAction | None:
    for action in actions:
        if action.verb != verb:
            continue
        if target_id is not None and action.target_id != target_id:
            continue
        if description_fragment is not None and description_fragment not in action.description:
            continue
        return action
    return None


def test_action_space_surfaces_move_affordance_and_blocks_at_attempt_when_ppe_missing(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    """Affordance-only: the move_to action is exposed even when PPE is missing.

    Discovery of the missing-PPE precondition happens at attempt time via the
    GameMaster's BLOCKED result, not by hint-leaking it into the action surface.
    """

    gm = GameMaster(minimal_world, clock=clock)

    spec = gm.get_available_actions("tech_1")

    move_to_wet_lab = _find_action(
        spec.available_actions,
        verb="move_to",
        target_id="wet_lab",
    )
    assert move_to_wet_lab is not None
    assert (
        _find_action(
            spec.available_actions,
            verb="don_ppe",
            description_fragment="lab coat",
        )
        is not None
    )
    assert (
        _find_action(
            spec.available_actions,
            verb="don_ppe",
            description_fragment="safety glasses",
        )
        is not None
    )
    assert (
        _find_action(
            spec.available_actions,
            verb="don_ppe",
            description_fragment="closed toe shoes",
        )
        is not None
    )

    blocked_result = gm.submit_action(
        Action(
            id="attempt_move_no_ppe",
            agent_id="tech_1",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="wet_lab",
        )
    )
    assert not blocked_result.success
    assert blocked_result.status is not None
    assert blocked_result.status.value == "blocked"
    block_reason = blocked_result.blocked_reason or ""
    assert "missing required PPE" in block_reason


def test_action_space_updates_after_state_changes(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(minimal_world, clock=clock)

    for ppe_type in (
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    ):
        result = gm.submit_action(
            Action(
                id=f"don_{ppe_type.value}",
                agent_id="tech_1",
                category=ActionCategory.SAFETY,
                verb="don_ppe",
                parameters={"ppe_type": ppe_type.value},
            )
        )
        assert result.success

    dressed_spec = gm.get_available_actions("tech_1")
    move_action = _find_action(
        dressed_spec.available_actions,
        verb="move_to",
        target_id="wet_lab",
    )

    assert move_action is not None

    move_result = gm.submit_action(
        Action(
            id="move_to_wet_lab",
            agent_id="tech_1",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="wet_lab",
        )
    )
    assert move_result.success

    wet_lab_spec = gm.get_available_actions("tech_1")
    start_action = _find_action(
        wet_lab_spec.available_actions,
        verb="start_equipment",
        target_id="centrifuge_1",
    )

    assert start_action is not None
    assert "speed_rpm" in start_action.required_parameters
    assert "duration_seconds" not in start_action.required_parameters
    assert start_action.fixed_parameters["duration_seconds"] == 600
    assert "speed_rpm" in start_action.parameter_details
    assert "Rotor speed" in start_action.parameter_details["speed_rpm"]
    assert start_action.parameter_schema["properties"]["speed_rpm"]["type"] == "integer"


def test_action_space_surfaces_door_controls_when_a_neighboring_door_changes_state(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(minimal_world, clock=clock)

    for ppe_type in (
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    ):
        result = gm.submit_action(
            Action(
                id=f"prep_{ppe_type.value}",
                agent_id="tech_1",
                category=ActionCategory.SAFETY,
                verb="don_ppe",
                parameters={"ppe_type": ppe_type.value},
            )
        )
        assert result.success

    preview = gm.get_available_actions("tech_1")
    close_action = _find_action(
        preview.available_actions,
        verb="close_door",
        target_id="door_office_wet_lab",
    )
    assert close_action is not None

    close_result = gm.submit_action(
        Action(
            id="close_office_wet",
            agent_id="tech_1",
            category=ActionCategory.MOVE,
            verb="close_door",
            target_id="door_office_wet_lab",
        )
    )
    assert close_result.success

    after_close = gm.get_available_actions("tech_1")
    move_action = _find_action(after_close.available_actions, verb="move_to", target_id="wet_lab")
    open_action = _find_action(
        after_close.available_actions,
        verb="open_door",
        target_id="door_office_wet_lab",
    )

    assert move_action is not None
    assert open_action is not None

    blocked_move = gm.submit_action(
        Action(
            id="attempt_move_closed_door",
            agent_id="tech_1",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="wet_lab",
        )
    )
    assert not blocked_move.success
    block_reason = blocked_move.blocked_reason or blocked_move.failure_reason or ""
    assert "closed" in block_reason


def test_research_action_space_uses_adjacent_moves_only(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    person = research_world.get_person("tech_johnson")
    person.current_room_id = "cold_room"
    person.current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(research_world, clock=clock)

    spec = gm.get_available_actions("tech_johnson")
    move_actions = [action for action in spec.available_actions if action.verb == "move_to"]

    assert [action.target_id for action in move_actions] == ["main_wet_lab"]
    assert [action.description for action in move_actions] == ["Move to Main Wet Lab"]


def test_action_space_humanizes_surface_descriptions(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(minimal_world, clock=clock)

    spec = gm.get_available_actions("tech_1")

    don_descriptions = {
        action.description for action in spec.available_actions if action.verb == "don_ppe"
    }

    assert "Don lab coat" in don_descriptions
    assert "Don safety glasses" in don_descriptions
    assert "Don closed toe shoes" in don_descriptions


def test_action_space_surfaces_wait_and_reporting_as_distinct_actions(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(minimal_world, clock=clock)

    spec = gm.get_available_actions("tech_1")

    wait_action = _find_action(spec.available_actions, verb="wait")
    log_action = _find_action(spec.available_actions, verb="log_observation")

    assert wait_action is not None
    assert wait_action.description == "Wait one turn without changing the world"
    assert log_action is not None
    assert log_action.description == "Record an observation note"


def test_log_observation_notes_hint_mentions_exact_task_format(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(minimal_world, clock=clock)

    spec = gm.get_available_actions("tech_1")
    log_action = _find_action(spec.available_actions, verb="log_observation")

    assert log_action is not None
    assert log_action.parameter_details is not None
    assert "exact note or finding format" in log_action.parameter_details["notes"]
