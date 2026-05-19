"""Tests for the GameMaster action pipeline and built-in handlers."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.core.clock import SimulationClock
from labverse.core.events import EventBus, EventLog, EventType
from labverse.core.types import (
    ActionCategory,
    EquipmentStatus,
    PersonRole,
    PPEType,
    WasteCategory,
)
from labverse.gamemaster import GameMaster
from labverse.models.action import Action
from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import Sample
from labverse.world.state import WorldState


def _equip_basic_lab_ppe(*, include_gloves: bool = False) -> set[PPEType]:
    ppe = {
        PPEType.LAB_COAT,
        PPEType.SAFETY_GLASSES,
        PPEType.CLOSED_TOE_SHOES,
    }
    if include_gloves:
        ppe.add(PPEType.NITRILE_GLOVES)
    return ppe


def test_move_action_updates_world_and_emits_events(
    research_world: WorldState,
    clock: SimulationClock,
    logging_bus: tuple[EventBus, EventLog],
) -> None:
    research_world.get_person("tech_johnson").current_ppe = _equip_basic_lab_ppe()
    event_bus, event_log = logging_bus
    gm = GameMaster(research_world, clock=clock, event_bus=event_bus)

    result = gm.submit_action(
        Action(
            id="move_1",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="main_wet_lab",
        )
    )

    assert result.success
    assert research_world.get_person("tech_johnson").current_room_id == "main_wet_lab"
    assert gm.clock.tick == 1
    assert event_log.query(event_type=EventType.ACTION_SUBMITTED)
    assert event_log.query(event_type=EventType.PERSON_MOVED)
    assert event_log.query(event_type=EventType.ACTION_RESOLVED)


def test_move_action_blocks_without_bsl_clearance_and_reports_safety_violations(
    research_world: WorldState,
    clock: SimulationClock,
    logging_bus: tuple[EventBus, EventLog],
) -> None:
    research_world.add_person(
        Person(
            id="visitor_1",
            name="Visitor",
            role=PersonRole.UNDERGRAD,
            current_room_id="main_wet_lab",
            current_ppe=_equip_basic_lab_ppe(include_gloves=True),
        )
    )
    event_bus, event_log = logging_bus
    gm = GameMaster(research_world, clock=clock, event_bus=event_bus)

    result = gm.submit_action(
        Action(
            id="move_blocked",
            agent_id="visitor_1",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="tissue_culture",
        )
    )

    assert not result.success
    assert result.blocked_reason is not None
    assert "bsl2_training" in result.blocked_reason
    assert result.safety_violations == [result.blocked_reason]
    assert research_world.get_person("visitor_1").current_room_id == "main_wet_lab"
    assert event_log.query(event_type=EventType.SAFETY_VIOLATION)
    blocked_events = event_log.query(event_type=EventType.ACTION_BLOCKED)
    assert blocked_events
    assert blocked_events[-1].data["reason"] == result.blocked_reason
    resolved_events = event_log.query(event_type=EventType.ACTION_RESOLVED)
    assert resolved_events[-1].data["success"] is False


def test_door_actions_gate_movement_and_emit_state_changes(
    research_world: WorldState,
    clock: SimulationClock,
    logging_bus: tuple[EventBus, EventLog],
) -> None:
    door = research_world.get_door("door_main_office")
    door.is_open = False
    research_world.get_person("tech_johnson").current_ppe = _equip_basic_lab_ppe()
    event_bus, event_log = logging_bus
    gm = GameMaster(research_world, clock=clock, event_bus=event_bus)

    blocked_move = gm.submit_action(
        Action(
            id="move_closed_door",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="main_wet_lab",
        )
    )

    assert not blocked_move.success
    assert blocked_move.blocked_reason is not None
    assert "closed" in blocked_move.blocked_reason

    open_result = gm.submit_action(
        Action(
            id="open_office_door",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="open_door",
            target_id="door_main_office",
        )
    )

    assert open_result.success
    assert research_world.get_door("door_main_office").is_open is True

    move_result = gm.submit_action(
        Action(
            id="move_after_open",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="move_to",
            target_id="main_wet_lab",
        )
    )

    assert move_result.success
    state_changes = event_log.query(event_type=EventType.STATE_CHANGE)
    assert state_changes
    assert state_changes[-1].target_id == "door_main_office"


def test_unlock_door_action_clears_lock_before_opening(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    door = research_world.get_door("door_main_office")
    door.locked = True
    door.is_open = False
    gm = GameMaster(research_world, clock=clock)

    unlock_result = gm.submit_action(
        Action(
            id="unlock_office_door",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="unlock_door",
            target_id="door_main_office",
        )
    )
    assert unlock_result.success
    assert door.locked is False
    assert door.is_open is False

    open_result = gm.submit_action(
        Action(
            id="open_unlocked_office_door",
            agent_id="tech_johnson",
            category=ActionCategory.MOVE,
            verb="open_door",
            target_id="door_main_office",
        )
    )
    assert open_result.success
    assert door.is_open is True


def test_start_equipment_schedules_timed_completion(
    game_master: GameMaster,
    research_world: WorldState,
) -> None:
    research_world.get_person("grad_kim").current_ppe = _equip_basic_lab_ppe()

    result = game_master.submit_action(
        Action(
            id="start_1",
            agent_id="grad_kim",
            category=ActionCategory.USE_EQUIPMENT,
            verb="start_equipment",
            target_id="centrifuge_1",
            parameters={"command_name": "spin", "duration_seconds": 25.0},
        )
    )
    equipment = research_world.get_equipment("centrifuge_1")

    assert result.success
    assert equipment.status.value == "running"
    assert equipment.running_command == "spin"
    assert game_master.pending_timed_events == 1
    assert game_master.clock.tick == 1

    emitted = game_master.tick(2)

    assert game_master.pending_timed_events == 0
    assert equipment.status.value == "idle"
    assert any(event.event_type == EventType.TIMED_EVENT for event in emitted)


def test_submit_action_by_id_records_trace_and_working_memory(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    research_world.get_person("tech_johnson").current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(research_world, clock=clock)

    actions = gm.get_available_actions("tech_johnson")
    move_action = next(
        action
        for action in actions.available_actions
        if action.verb == "move_to" and action.target_id == "main_wet_lab"
    )

    result = gm.submit_action_by_id("tech_johnson", move_action.action_id)

    assert result.success
    assert result.status == "succeeded"
    assert result.emitted_event_ids
    assert result.post_observation_id is not None

    traces = gm.trace_records("tech_johnson")
    assert len(traces) == 1
    assert traces[0].observation.agent_id == "tech_johnson"
    assert traces[0].action_spec.agent_id == "tech_johnson"
    assert traces[0].action.id == result.action_id
    assert traces[0].post_observation is not None
    assert traces[0].post_observation.current_room.id == "main_wet_lab"

    snapshot = gm.set_working_memory("tech_johnson", "Entered wet lab after PPE checks.")
    assert snapshot.working_memory.startswith("Entered wet lab")
    assert snapshot.trace_count == 1

    observation = gm.get_observation("tech_johnson")
    assert observation.working_memory.startswith("Entered wet lab")
    assert observation.trace_parent_action_id == result.action_id


def test_available_actions_are_affordance_only_and_block_at_attempt_when_ppe_missing(
    game_master: GameMaster,
    research_world: WorldState,
) -> None:
    """Affordance-only contract: start_equipment for an in-room device is exposed
    structurally even when PPE is missing. Discovery of the missing-PPE precondition
    happens at submit time via a BLOCKED ActionResult, not by hint-leaking it into
    the action surface.
    """

    research_world.get_person("grad_kim").current_ppe = set()
    action_spec = game_master.get_available_actions("grad_kim")
    don_actions = [action for action in action_spec.available_actions if action.verb == "don_ppe"]
    start_actions = [
        action
        for action in action_spec.available_actions
        if action.verb == "start_equipment" and action.target_id == "centrifuge_1"
    ]

    assert don_actions
    assert len(start_actions) == 1
    assert "speed_rpm" in start_actions[0].required_parameters

    blocked = game_master.submit_action(
        Action(
            id="attempt_start_no_ppe",
            agent_id="grad_kim",
            category=ActionCategory.USE_EQUIPMENT,
            verb="start_equipment",
            target_id="centrifuge_1",
            parameters={"command_name": "spin", "speed_rpm": 5000, "duration_seconds": 60.0},
        )
    )
    assert not blocked.success
    assert blocked.blocked_reason is not None

    research_world.get_person("grad_kim").current_ppe = _equip_basic_lab_ppe()
    refreshed = game_master.get_available_actions("grad_kim")
    refreshed_start = [
        action
        for action in refreshed.available_actions
        if action.verb == "start_equipment" and action.target_id == "centrifuge_1"
    ]
    assert len(refreshed_start) == 1
    assert "speed_rpm" in refreshed_start[0].required_parameters


def test_stop_equipment_cancels_pending_completion(
    game_master: GameMaster,
    research_world: WorldState,
) -> None:
    research_world.get_person("grad_kim").current_ppe = _equip_basic_lab_ppe()

    game_master.submit_action(
        Action(
            id="start_2",
            agent_id="grad_kim",
            category=ActionCategory.USE_EQUIPMENT,
            verb="start_equipment",
            target_id="centrifuge_1",
            parameters={"command_name": "spin", "duration_seconds": 100.0},
        )
    )

    stop_result = game_master.submit_action(
        Action(
            id="stop_1",
            agent_id="grad_kim",
            category=ActionCategory.USE_EQUIPMENT,
            verb="stop_equipment",
            target_id="centrifuge_1",
        )
    )
    equipment = research_world.get_equipment("centrifuge_1")

    assert stop_result.success
    assert equipment.status.value == "idle"
    assert equipment.running_command is None
    assert game_master.pending_timed_events == 0

    game_master.tick(20)
    assert equipment.status.value == "idle"


def test_don_and_doff_ppe_handlers_mutate_person_state(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(research_world, clock=clock)
    person = research_world.get_person("grad_kim")

    don_result = gm.submit_action(
        Action(
            id="don_gloves",
            agent_id="grad_kim",
            category=ActionCategory.SAFETY,
            verb="don_ppe",
            parameters={"ppe_type": PPEType.NITRILE_GLOVES.value},
        )
    )

    assert don_result.success
    assert PPEType.NITRILE_GLOVES in person.current_ppe

    doff_result = gm.submit_action(
        Action(
            id="doff_gloves",
            agent_id="grad_kim",
            category=ActionCategory.SAFETY,
            verb="doff_ppe",
            parameters={"ppe_type": PPEType.NITRILE_GLOVES.value},
        )
    )

    assert doff_result.success
    assert PPEType.NITRILE_GLOVES not in person.current_ppe


def test_dispose_material_handler_removes_material_with_correct_waste_stream(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    research_world.add_material(
        Material(
            id="water_1",
            name="Water",
            room_id="main_wet_lab",
            quantity=250.0,
            waste_category=WasteCategory.GENERAL,
        )
    )
    person = research_world.get_person("grad_kim")
    person.current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(research_world, clock=clock)

    result = gm.submit_action(
        Action(
            id="dispose_ok",
            agent_id="grad_kim",
            category=ActionCategory.HANDLE_MATERIAL,
            verb="dispose_material",
            target_id="water_1",
            parameters={"waste_category": WasteCategory.GENERAL.value},
        )
    )

    assert result.success
    assert "water_1" not in research_world.materials


def test_dispose_material_handler_is_blocked_by_wrong_waste_stream(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    research_world.add_material(
        Material(
            id="water_2",
            name="Water",
            room_id="main_wet_lab",
            quantity=100.0,
            waste_category=WasteCategory.GENERAL,
        )
    )
    person = research_world.get_person("grad_kim")
    person.current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(research_world, clock=clock)

    result = gm.submit_action(
        Action(
            id="dispose_blocked",
            agent_id="grad_kim",
            category=ActionCategory.HANDLE_MATERIAL,
            verb="dispose_material",
            target_id="water_2",
            parameters={"waste_category": WasteCategory.CHEMICAL.value},
        )
    )

    assert not result.success
    assert result.blocked_reason is not None
    assert "general" in result.blocked_reason
    assert "water_2" in research_world.materials


def test_retrieve_and_store_sample_handlers_update_location_and_custody(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    person = research_world.get_person("grad_kim")
    person.current_room_id = "tissue_culture"
    person.current_ppe = _equip_basic_lab_ppe(include_gloves=True)
    gm = GameMaster(research_world, clock=clock)
    sample = research_world.get_sample("hela_p12")

    retrieve_result = gm.submit_action(
        Action(
            id="retrieve_sample",
            agent_id="grad_kim",
            category=ActionCategory.HANDLE_SAMPLE,
            verb="retrieve_sample",
            target_id="hela_p12",
        )
    )
    store_result = gm.submit_action(
        Action(
            id="store_sample",
            agent_id="grad_kim",
            category=ActionCategory.HANDLE_SAMPLE,
            verb="store_sample",
            target_id="hela_p12",
            parameters={"destination_equipment_id": "incubator_1"},
        )
    )

    assert retrieve_result.success
    assert store_result.success
    assert "hela_p12" not in person.carrying
    assert sample.equipment_id == "incubator_1"
    assert sample.status == "stored"
    assert [record.action for record in sample.chain_of_custody[-2:]] == ["retrieved", "stored"]


def test_pipette_transfer_consumes_material_and_updates_sample(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    minimal_world.add_equipment(
        Equipment(
            id="pipette_1",
            name="P20 Pipette",
            equipment_type="pipette",
            room_id="wet_lab",
            zone_id="bench_1",
            status=EquipmentStatus.READY,
        )
    )
    minimal_world.add_sample(
        Sample(
            id="dna_extract_1",
            name="DNA Extract",
            sample_type="dna_extract",
            room_id="wet_lab",
            volume_ul=10.0,
            created_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            created_by="tech_1",
        )
    )
    person = minimal_world.get_person("tech_1")
    person.current_room_id = "wet_lab"
    person.current_zone_id = "bench_1"
    person.current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(minimal_world, clock=clock)
    sample = minimal_world.get_sample("dna_extract_1")
    material = minimal_world.get_material("pbs_1x")

    result = gm.submit_action(
        Action(
            id="pipette_transfer",
            agent_id="tech_1",
            category=ActionCategory.HANDLE_SAMPLE,
            verb="pipette_transfer",
            target_id="pipette_1",
            parameters={
                "source_id": "pbs_1x",
                "destination_id": "dna_extract_1",
                "volume_ul": 50.0,
            },
        )
    )

    assert result.success
    assert material.quantity == 499.95
    assert sample.volume_ul == 60.0


def test_pipette_transfer_refuses_sample_source_with_unknown_volume(
    minimal_world: WorldState,
    clock: SimulationClock,
) -> None:
    """volume_ul is None is a contract violation: refuse rather than fabricate zero.

    The affordance-only surface still lists the action structurally; the refusal
    must surface at attempt time via a BLOCKED ActionResult with a clear reason,
    and the source sample's volume must remain untouched.
    """

    minimal_world.add_equipment(
        Equipment(
            id="pipette_1",
            name="P20 Pipette",
            equipment_type="pipette",
            room_id="wet_lab",
            zone_id="bench_1",
            status=EquipmentStatus.READY,
        )
    )
    minimal_world.add_sample(
        Sample(
            id="unknown_volume_sample",
            name="Unlabeled Sample",
            sample_type="dna_extract",
            room_id="wet_lab",
            volume_ul=None,
            created_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            created_by="tech_1",
        )
    )
    minimal_world.add_sample(
        Sample(
            id="destination_sample",
            name="Destination",
            sample_type="dna_extract",
            room_id="wet_lab",
            volume_ul=5.0,
            created_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            created_by="tech_1",
        )
    )
    person = minimal_world.get_person("tech_1")
    person.current_room_id = "wet_lab"
    person.current_zone_id = "bench_1"
    person.current_ppe = _equip_basic_lab_ppe()
    gm = GameMaster(minimal_world, clock=clock)
    source = minimal_world.get_sample("unknown_volume_sample")
    destination = minimal_world.get_sample("destination_sample")

    result = gm.submit_action(
        Action(
            id="pipette_transfer_unknown_source",
            agent_id="tech_1",
            category=ActionCategory.HANDLE_SAMPLE,
            verb="pipette_transfer",
            target_id="pipette_1",
            parameters={
                "source_id": "unknown_volume_sample",
                "destination_id": "destination_sample",
                "volume_ul": 10.0,
            },
        )
    )

    assert not result.success
    assert result.blocked_reason is not None
    assert "unknown volume" in result.blocked_reason
    assert source.volume_ul is None
    assert destination.volume_ul == 5.0


def test_log_observation_handler_succeeds_without_target(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(research_world, clock=clock)

    result = gm.submit_action(
        Action(
            id="observe_1",
            agent_id="grad_kim",
            category=ActionCategory.OBSERVE,
            verb="log_observation",
            parameters={"notes": "Bench looks clear"},
        )
    )

    assert result.success
    assert result.outcome_description == "Bench looks clear"


def test_wait_handler_succeeds_without_creating_a_report(
    research_world: WorldState,
    clock: SimulationClock,
) -> None:
    gm = GameMaster(research_world, clock=clock)

    result = gm.submit_action(
        Action(
            id="wait_1",
            agent_id="grad_kim",
            category=ActionCategory.OBSERVE,
            verb="wait",
        )
    )

    assert result.success
    assert result.outcome_description == "Waited one turn"
