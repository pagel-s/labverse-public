"""Incubator-specific runtime state machine."""

from __future__ import annotations

from labverse.equipment.state_machine import EquipmentStateMachine


class IncubatorStateMachine(EquipmentStateMachine):
    """Long-running incubators mostly use the universal lifecycle."""

    def begin_command(self, command_name: str) -> None:
        if command_name != "incubate":
            super().begin_command(command_name)
            return
        self.prepare_for_command(command_name)
        if self.state_name == "ready":
            self.trigger("start")
