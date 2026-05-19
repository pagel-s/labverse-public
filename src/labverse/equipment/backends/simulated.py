"""Pure in-memory equipment backend for the semantic simulation."""

from __future__ import annotations

from datetime import datetime, timedelta

from labverse.core.types import EquipmentStatus
from labverse.equipment.backends.base import EquipmentBackend
from labverse.equipment.devices import create_state_machine
from labverse.equipment.state_machine import EquipmentStateMachine, MachineError
from labverse.world.state import WorldState


class SimulatedBackend(EquipmentBackend):
    """Runtime backend that owns equipment state machines for a `WorldState`."""

    def __init__(self, world: WorldState) -> None:
        self._world = world
        self._state_machines: dict[str, EquipmentStateMachine] = {}

    def get_state_machine(self, equipment_id: str) -> EquipmentStateMachine:
        if equipment_id not in self._state_machines:
            equipment = self._world.get_equipment(equipment_id)
            self._state_machines[equipment_id] = create_state_machine(equipment)
        return self._state_machines[equipment_id]

    def start_command(
        self,
        equipment_id: str,
        agent_id: str,
        command_name: str,
        *,
        started_at: datetime,
        duration_seconds: float | None = None,
    ) -> float | None:
        equipment = self._world.get_equipment(equipment_id)
        machine = self.get_state_machine(equipment_id)

        if equipment.status == EquipmentStatus.RESERVED:
            machine.trigger("release")

        if equipment.status in {
            EquipmentStatus.RUNNING,
            EquipmentStatus.COOLING_DOWN,
            EquipmentStatus.MAINTENANCE,
            EquipmentStatus.ERROR,
        }:
            raise MachineError(
                f"Cannot start equipment {equipment.id} from status {equipment.status.value}"
            )

        machine.prepare_for_command(command_name)
        machine.begin_command(command_name)

        if agent_id not in equipment.current_users:
            equipment.current_users.append(agent_id)
        equipment.running_command = command_name
        equipment.running_command_started_at = started_at
        equipment.running_command_duration = (
            timedelta(seconds=duration_seconds) if duration_seconds is not None else None
        )
        return duration_seconds

    def stop_command(self, equipment_id: str, *, agent_id: str | None = None) -> None:
        equipment = self._world.get_equipment(equipment_id)
        if equipment.status != EquipmentStatus.RUNNING:
            raise MachineError(
                f"Cannot stop equipment {equipment.id} from status {equipment.status.value}"
            )
        self.complete_command(equipment_id)
        if agent_id is not None and agent_id in equipment.current_users:
            equipment.current_users.remove(agent_id)

    def complete_command(self, equipment_id: str) -> None:
        equipment = self._world.get_equipment(equipment_id)
        machine = self.get_state_machine(equipment_id)
        command_name = equipment.running_command or "run"
        machine.complete_command(command_name)
        equipment.running_command = None
        equipment.running_command_started_at = None
        equipment.running_command_duration = None
        equipment.current_users.clear()
