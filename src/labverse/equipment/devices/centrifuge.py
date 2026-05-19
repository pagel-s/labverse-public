"""Centrifuge-specific runtime state machine."""

from __future__ import annotations

from labverse.core.types import EquipmentStatus
from labverse.equipment.state_machine import EquipmentStateMachine


class CentrifugeStateMachine(EquipmentStateMachine):
    """Adds lid/loading/balancing phases before a centrifuge run."""

    EXTRA_STATES = ["lid_open", "loading", "balancing"]
    STATUS_MAP = {
        "lid_open": EquipmentStatus.IDLE,
        "loading": EquipmentStatus.IDLE,
        "balancing": EquipmentStatus.WARMING_UP,
    }
    EXTRA_TRANSITIONS = [
        {"trigger": "open_lid", "source": "idle", "dest": "lid_open"},
        {"trigger": "load_sample", "source": "lid_open", "dest": "loading"},
        {"trigger": "close_lid", "source": "loading", "dest": "balancing"},
        {"trigger": "balance_load", "source": "balancing", "dest": "ready"},
    ]

    def prepare_for_command(self, command_name: str) -> None:
        if command_name != "spin":
            super().prepare_for_command(command_name)
            return
        if self.state_name == "off":
            self.trigger("power_on")
        if self.state_name == "idle":
            self.trigger("open_lid")
        if self.state_name == "lid_open":
            self.trigger("load_sample")
        if self.state_name == "loading":
            self.trigger("close_lid")
        if self.state_name == "balancing":
            self.trigger("balance_load")

    def begin_command(self, command_name: str) -> None:
        self.prepare_for_command(command_name)
        if self.state_name == "ready":
            self.trigger("start")
