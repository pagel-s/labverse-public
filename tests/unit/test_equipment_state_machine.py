"""Tests for equipment state machines, device specializations, and backend wiring."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from labverse.core.types import EquipmentStatus
from labverse.equipment import (
    EquipmentStateMachine,
    MachineError,
    SimulatedBackend,
    create_state_machine,
)
from labverse.equipment.devices import (
    AutoclaveStateMachine,
    BiosafetyCabinetStateMachine,
    CentrifugeStateMachine,
    IncubatorStateMachine,
    PcrThermocyclerStateMachine,
    PipetteStateMachine,
)
from labverse.models.equipment import Equipment
from labverse.world.state import WorldState


def _equipment(equipment_type: str, status: EquipmentStatus = EquipmentStatus.OFF) -> Equipment:
    return Equipment(
        id=f"{equipment_type}_1",
        name=equipment_type,
        equipment_type=equipment_type,
        room_id="wet_lab",
        status=status,
    )


def test_base_state_machine_syncs_status() -> None:
    equipment = _equipment("spectrophotometer", EquipmentStatus.OFF)
    machine = EquipmentStateMachine(equipment)

    machine.prepare()
    machine.begin_command("measure")

    assert equipment.status.value == "running"
    machine.complete_command("measure")
    assert equipment.status.value == "idle"


def test_base_state_machine_invalid_transition_raises() -> None:
    machine = EquipmentStateMachine(_equipment("spectrophotometer", EquipmentStatus.OFF))

    with pytest.raises(MachineError):
        machine.trigger("start")


def test_factory_returns_device_specific_state_machine() -> None:
    assert isinstance(create_state_machine(_equipment("centrifuge")), CentrifugeStateMachine)
    assert isinstance(
        create_state_machine(_equipment("pcr_thermocycler")),
        PcrThermocyclerStateMachine,
    )
    assert isinstance(create_state_machine(_equipment("incubator")), IncubatorStateMachine)
    assert isinstance(
        create_state_machine(_equipment("biosafety_cabinet")),
        BiosafetyCabinetStateMachine,
    )
    assert isinstance(create_state_machine(_equipment("autoclave")), AutoclaveStateMachine)
    assert isinstance(create_state_machine(_equipment("pipette")), PipetteStateMachine)


def test_centrifuge_machine_supports_loading_workflow() -> None:
    machine = create_state_machine(_equipment("centrifuge", EquipmentStatus.IDLE))

    machine.trigger("open_lid")
    assert machine.state_name == "lid_open"
    machine.trigger("load_sample")
    assert machine.state_name == "loading"
    machine.trigger("close_lid")
    assert machine.state_name == "balancing"
    machine.trigger("balance_load")
    assert machine.state_name == "ready"
    machine.begin_command("spin")
    assert machine.state_name == "running"
    machine.complete_command("spin")
    assert machine.state_name == "idle"


def test_pcr_machine_tracks_program_phases() -> None:
    machine = create_state_machine(_equipment("pcr_thermocycler"))

    machine.prepare_for_command("run_program")
    assert machine.state_name == "ready"
    machine.begin_command("run_program")
    assert machine.state_name == "cycling"
    machine.trigger("begin_denaturation")
    assert machine.state_name == "denaturing"
    machine.trigger("begin_annealing")
    assert machine.state_name == "annealing"
    machine.trigger("begin_extension")
    assert machine.state_name == "extending"
    machine.complete_command("run_program")
    assert machine.state_name == "idle"


def test_biosafety_cabinet_requires_airflow_before_work() -> None:
    machine = create_state_machine(_equipment("biosafety_cabinet"))

    machine.prepare_for_command("work")
    assert machine.state_name == "ready"
    machine.begin_command("work")
    assert machine.state_name == "working"
    machine.complete_command("work")
    assert machine.state_name == "idle"

    machine.begin_command("sterilize_uv")
    assert machine.state_name == "uv_sterilizing"
    machine.complete_command("sterilize_uv")
    assert machine.state_name == "idle"


def test_autoclave_completes_full_cycle() -> None:
    machine = create_state_machine(_equipment("autoclave"))

    machine.begin_command("sterilize")
    assert machine.state_name == "pressurizing"
    machine.complete_command("sterilize")
    assert machine.state_name == "idle"


def test_pipette_is_stateless_and_always_ready() -> None:
    equipment = _equipment("pipette", EquipmentStatus.OFF)
    machine = create_state_machine(equipment)

    assert machine.state_name == "ready"
    assert equipment.status.value == "ready"
    machine.begin_command("aspirate_dispense")
    assert machine.state_name == "ready"
    machine.complete_command("aspirate_dispense")
    assert machine.state_name == "ready"


def test_simulated_backend_uses_device_specific_machine(minimal_world: WorldState) -> None:
    backend = SimulatedBackend(minimal_world)
    started_at = datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC)

    backend.start_command(
        "centrifuge_1",
        "tech_1",
        "spin",
        started_at=started_at,
        duration_seconds=30.0,
    )
    equipment = minimal_world.get_equipment("centrifuge_1")

    assert equipment.status.value == "running"
    assert equipment.running_command == "spin"
    assert equipment.running_command_started_at == started_at
    assert equipment.current_users == ["tech_1"]

    backend.complete_command("centrifuge_1")

    assert equipment.status.value == "idle"
    assert equipment.running_command is None
    assert equipment.current_users == []
