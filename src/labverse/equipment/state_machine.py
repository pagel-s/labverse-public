"""Base runtime equipment state machine infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from transitions import Machine, MachineError

from labverse.core.types import EquipmentStatus
from labverse.models.equipment import Equipment


class EquipmentStateMachine:
    """Runtime state machine that keeps an `Equipment` model in sync."""

    EXTRA_STATES: ClassVar[list[str]] = []
    EXTRA_TRANSITIONS: ClassVar[list[dict[str, Any]]] = []
    DEFAULT_STATUS: ClassVar[EquipmentStatus | None] = None
    STATUS_MAP: ClassVar[dict[str, EquipmentStatus]] = {}

    UNIVERSAL_STATES: ClassVar[list[str]] = [status.value for status in EquipmentStatus]
    UNIVERSAL_TRANSITIONS: ClassVar[list[dict[str, Any]]] = [
        {
            "trigger": "power_on",
            "source": EquipmentStatus.OFF.value,
            "dest": EquipmentStatus.IDLE.value,
        },
        {
            "trigger": "power_off",
            "source": EquipmentStatus.IDLE.value,
            "dest": EquipmentStatus.OFF.value,
        },
        {
            "trigger": "warm_up",
            "source": EquipmentStatus.IDLE.value,
            "dest": EquipmentStatus.WARMING_UP.value,
        },
        {
            "trigger": "ready",
            "source": EquipmentStatus.WARMING_UP.value,
            "dest": EquipmentStatus.READY.value,
        },
        {
            "trigger": "start",
            "source": EquipmentStatus.READY.value,
            "dest": EquipmentStatus.RUNNING.value,
        },
        {
            "trigger": "complete",
            "source": EquipmentStatus.RUNNING.value,
            "dest": EquipmentStatus.COOLING_DOWN.value,
        },
        {
            "trigger": "cooled",
            "source": EquipmentStatus.COOLING_DOWN.value,
            "dest": EquipmentStatus.IDLE.value,
        },
        {"trigger": "fault", "source": "*", "dest": EquipmentStatus.ERROR.value},
        {
            "trigger": "reset_error",
            "source": EquipmentStatus.ERROR.value,
            "dest": EquipmentStatus.IDLE.value,
        },
        {
            "trigger": "begin_maintenance",
            "source": "*",
            "dest": EquipmentStatus.MAINTENANCE.value,
        },
        {
            "trigger": "end_maintenance",
            "source": EquipmentStatus.MAINTENANCE.value,
            "dest": EquipmentStatus.OFF.value,
        },
        {
            "trigger": "reserve",
            "source": [EquipmentStatus.IDLE.value, EquipmentStatus.READY.value],
            "dest": EquipmentStatus.RESERVED.value,
        },
        {
            "trigger": "release",
            "source": EquipmentStatus.RESERVED.value,
            "dest": EquipmentStatus.IDLE.value,
        },
    ]

    if TYPE_CHECKING:
        state: str

        def trigger(self, trigger_name: str) -> bool: ...

    def __init__(self, equipment: Equipment) -> None:
        self.equipment = equipment
        if self.DEFAULT_STATUS is not None and equipment.status == EquipmentStatus.OFF:
            equipment.status = self.DEFAULT_STATUS

        self.machine = Machine(
            model=self,
            states=self._all_states(),
            transitions=cast(Any, self._all_transitions()),
            initial=equipment.status.value,
            after_state_change="_sync_equipment_status",
            queued=True,
        )
        self._sync_equipment_status()

    @property
    def state_name(self) -> str:
        return str(self.state)

    def _all_states(self) -> list[str]:
        return list(dict.fromkeys(self.UNIVERSAL_STATES + self.EXTRA_STATES))

    def _all_transitions(self) -> list[dict[str, Any]]:
        return self.UNIVERSAL_TRANSITIONS + self.EXTRA_TRANSITIONS

    def _sync_equipment_status(self) -> None:
        if self.state_name in self.STATUS_MAP:
            self.equipment.status = self.STATUS_MAP[self.state_name]
            return
        self.equipment.status = EquipmentStatus(self.state_name)

    def prepare(self) -> None:
        """Drive the machine into `READY` from the standard startup states."""
        if self.state_name == EquipmentStatus.OFF.value:
            self.trigger("power_on")
        if self.state_name == EquipmentStatus.IDLE.value:
            self.trigger("warm_up")
        if self.state_name == EquipmentStatus.WARMING_UP.value:
            self.trigger("ready")

    def finish_cycle(self) -> None:
        """Return a running device back to `IDLE` through the cooling path."""
        if self.state_name == EquipmentStatus.RUNNING.value:
            self.trigger("complete")
        if self.state_name == EquipmentStatus.COOLING_DOWN.value:
            self.trigger("cooled")

    def prepare_for_command(self, command_name: str) -> None:
        """Prepare the device for a command by moving it to `READY`."""
        del command_name
        self.prepare()

    def begin_command(self, command_name: str) -> None:
        """Begin executing a command from the ready state."""
        del command_name
        if self.state_name != EquipmentStatus.READY.value:
            self.prepare()
        if self.state_name == EquipmentStatus.READY.value:
            self.trigger("start")

    def complete_command(self, command_name: str) -> None:
        """Mark a command as finished and return to the idle state."""
        del command_name
        self.finish_cycle()


__all__ = ["EquipmentStateMachine", "MachineError"]
