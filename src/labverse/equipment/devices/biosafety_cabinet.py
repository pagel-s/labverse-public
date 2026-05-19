"""Biosafety cabinet-specific runtime state machine."""

from __future__ import annotations

from labverse.core.types import EquipmentStatus
from labverse.equipment.state_machine import EquipmentStateMachine


class BiosafetyCabinetStateMachine(EquipmentStateMachine):
    """Models airflow stabilization, working mode, and UV sterilization."""

    EXTRA_STATES = ["uv_sterilizing", "airflow_stabilizing", "working"]
    STATUS_MAP = {
        "uv_sterilizing": EquipmentStatus.RUNNING,
        "airflow_stabilizing": EquipmentStatus.WARMING_UP,
        "working": EquipmentStatus.RUNNING,
    }
    EXTRA_TRANSITIONS = [
        {"trigger": "stabilize_airflow", "source": "idle", "dest": "airflow_stabilizing"},
        {"trigger": "airflow_ready", "source": "airflow_stabilizing", "dest": "ready"},
        {"trigger": "begin_work", "source": "ready", "dest": "working"},
        {"trigger": "stop_work", "source": "working", "dest": "idle"},
        {"trigger": "start_uv", "source": ["idle", "ready"], "dest": "uv_sterilizing"},
        {"trigger": "stop_uv", "source": "uv_sterilizing", "dest": "idle"},
    ]

    def prepare_for_command(self, command_name: str) -> None:
        if self.state_name == "off":
            self.trigger("power_on")
        if command_name == "work":
            if self.state_name == "idle":
                self.trigger("stabilize_airflow")
            if self.state_name == "airflow_stabilizing":
                self.trigger("airflow_ready")

    def begin_command(self, command_name: str) -> None:
        self.prepare_for_command(command_name)
        if command_name == "work" and self.state_name == "ready":
            self.trigger("begin_work")
        elif command_name == "sterilize_uv" and self.state_name in {"idle", "ready"}:
            self.trigger("start_uv")
        else:
            super().begin_command(command_name)

    def complete_command(self, command_name: str) -> None:
        if command_name == "work" and self.state_name == "working":
            self.trigger("stop_work")
            return
        if command_name == "sterilize_uv" and self.state_name == "uv_sterilizing":
            self.trigger("stop_uv")
            return
        super().complete_command(command_name)
