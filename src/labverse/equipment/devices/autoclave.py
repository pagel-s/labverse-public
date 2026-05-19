"""Autoclave-specific runtime state machine."""

from __future__ import annotations

from labverse.core.types import EquipmentStatus
from labverse.equipment.state_machine import EquipmentStateMachine


class AutoclaveStateMachine(EquipmentStateMachine):
    """Models the pressurization and drying phases of an autoclave cycle."""

    EXTRA_STATES = ["pressurizing", "sterilizing", "depressurizing", "drying"]
    STATUS_MAP = {
        "pressurizing": EquipmentStatus.RUNNING,
        "sterilizing": EquipmentStatus.RUNNING,
        "depressurizing": EquipmentStatus.RUNNING,
        "drying": EquipmentStatus.RUNNING,
    }
    EXTRA_TRANSITIONS = [
        {"trigger": "start_cycle", "source": "ready", "dest": "pressurizing"},
        {"trigger": "reach_sterilizing", "source": "pressurizing", "dest": "sterilizing"},
        {"trigger": "release_pressure", "source": "sterilizing", "dest": "depressurizing"},
        {"trigger": "start_drying", "source": "depressurizing", "dest": "drying"},
        {"trigger": "finish_drying", "source": "drying", "dest": "cooling_down"},
    ]

    def begin_command(self, command_name: str) -> None:
        if command_name != "sterilize":
            super().begin_command(command_name)
            return
        self.prepare_for_command(command_name)
        if self.state_name == "ready":
            self.trigger("start_cycle")

    def complete_command(self, command_name: str) -> None:
        if command_name != "sterilize":
            super().complete_command(command_name)
            return

        transition_map = {
            "pressurizing": "reach_sterilizing",
            "sterilizing": "release_pressure",
            "depressurizing": "start_drying",
            "drying": "finish_drying",
        }
        while self.state_name in transition_map:
            self.trigger(transition_map[self.state_name])
        if self.state_name == "cooling_down":
            self.trigger("cooled")
