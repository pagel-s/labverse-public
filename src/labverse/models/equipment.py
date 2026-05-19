"""Equipment models: parameters, commands, and device instances."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from labverse.core.types import CertificationType, EquipmentStatus
from labverse.models.space import Position


class EquipmentParameter(BaseModel):
    """A configurable or readable parameter on a device (SiLA 2 Property analog)."""

    name: str
    description: str = ""
    param_type: str = "float"
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    default_value: float | str | bool | None = None
    current_value: float | str | bool | None = None


class EquipmentCommand(BaseModel):
    """A command the equipment can execute (SiLA 2 Command analog)."""

    name: str
    description: str = ""
    parameters: list[EquipmentParameter] = Field(default_factory=list)
    duration_seconds: float | None = None
    preconditions: list[str] = Field(default_factory=list)


class Equipment(BaseModel):
    """A lab device or instrument."""

    id: str
    name: str
    equipment_type: str
    manufacturer: str | None = None
    model_name: str | None = None
    status: EquipmentStatus = EquipmentStatus.OFF
    room_id: str
    zone_id: str | None = None
    position: Position | None = None
    commands: dict[str, EquipmentCommand] = Field(default_factory=dict)
    properties: dict[str, EquipmentParameter] = Field(default_factory=dict)
    required_certifications: list[CertificationType] = Field(default_factory=list)
    max_concurrent_users: int = 1
    current_users: list[str] = Field(default_factory=list)
    running_command: str | None = None
    running_command_started_at: datetime | None = None
    running_command_duration: timedelta | None = None
    error_message: str | None = None
    last_maintenance: datetime | None = None
    maintenance_interval_days: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)

    @property
    def is_available(self) -> bool:
        """Whether the equipment can accept a new user/command."""
        return (
            self.status in (EquipmentStatus.IDLE, EquipmentStatus.READY)
            and len(self.current_users) < self.max_concurrent_users
        )

    @property
    def is_running(self) -> bool:
        return self.status == EquipmentStatus.RUNNING
