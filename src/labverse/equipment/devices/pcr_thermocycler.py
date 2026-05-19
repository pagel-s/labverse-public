"""PCR thermocycler-specific runtime state machine."""

from __future__ import annotations

from labverse.core.types import EquipmentStatus
from labverse.equipment.state_machine import EquipmentStateMachine


class PcrThermocyclerStateMachine(EquipmentStateMachine):
    """Adds semantic PCR cycling phases beyond the universal lifecycle."""

    EXTRA_STATES = ["cycling", "denaturing", "annealing", "extending"]
    STATUS_MAP = {
        "cycling": EquipmentStatus.RUNNING,
        "denaturing": EquipmentStatus.RUNNING,
        "annealing": EquipmentStatus.RUNNING,
        "extending": EquipmentStatus.RUNNING,
    }
    EXTRA_TRANSITIONS = [
        {"trigger": "start_program", "source": "ready", "dest": "cycling"},
        {"trigger": "begin_denaturation", "source": "cycling", "dest": "denaturing"},
        {"trigger": "begin_annealing", "source": "denaturing", "dest": "annealing"},
        {"trigger": "begin_extension", "source": "annealing", "dest": "extending"},
        {"trigger": "next_cycle", "source": "extending", "dest": "cycling"},
        {
            "trigger": "complete_program",
            "source": ["cycling", "denaturing", "annealing", "extending"],
            "dest": "cooling_down",
        },
    ]

    def begin_command(self, command_name: str) -> None:
        if command_name != "run_program":
            super().begin_command(command_name)
            return
        self.prepare_for_command(command_name)
        if self.state_name == "ready":
            self.trigger("start_program")

    def complete_command(self, command_name: str) -> None:
        if command_name == "run_program" and self.state_name in {
            "cycling",
            "denaturing",
            "annealing",
            "extending",
        }:
            self.trigger("complete_program")
        if self.state_name == "cooling_down":
            self.trigger("cooled")
