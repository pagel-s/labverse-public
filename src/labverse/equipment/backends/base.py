"""Base classes for equipment backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from labverse.equipment.state_machine import EquipmentStateMachine


class EquipmentBackend(ABC):
    """Abstract backend for runtime equipment control."""

    @abstractmethod
    def get_state_machine(self, equipment_id: str) -> EquipmentStateMachine: ...

    @abstractmethod
    def start_command(
        self,
        equipment_id: str,
        agent_id: str,
        command_name: str,
        *,
        started_at: datetime,
        duration_seconds: float | None = None,
    ) -> float | None: ...

    @abstractmethod
    def stop_command(self, equipment_id: str, *, agent_id: str | None = None) -> None: ...

    @abstractmethod
    def complete_command(self, equipment_id: str) -> None: ...
