"""Pipette-specific runtime state machine."""

from __future__ import annotations

from labverse.core.types import EquipmentStatus
from labverse.equipment.state_machine import EquipmentStateMachine


class PipetteStateMachine(EquipmentStateMachine):
    """Pipettes are effectively stateless and always ready."""

    DEFAULT_STATUS = EquipmentStatus.READY

    def prepare(self) -> None:
        if self.state_name != "ready":
            self.equipment.status = EquipmentStatus.READY
            self._sync_equipment_status()

    def prepare_for_command(self, command_name: str) -> None:
        del command_name
        self.prepare()

    def begin_command(self, command_name: str) -> None:
        del command_name
        self.prepare()

    def complete_command(self, command_name: str) -> None:
        del command_name
        self.prepare()
