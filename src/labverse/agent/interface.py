"""Agent interface definitions and observation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from labverse.core.ids import generate_id
from labverse.core.types import BSLLevel, CertificationType, EquipmentStatus, PersonRole, PPEType
from labverse.models.action import Action, ActionSpec
from labverse.models.space import Room


class PersonSummary(BaseModel):
    """A lightweight view of another visible person."""

    id: str
    name: str
    role: PersonRole
    current_zone_id: str | None = None
    current_activity: str | None = None


class EquipmentSummary(BaseModel):
    """A lightweight view of visible equipment."""

    id: str
    name: str
    equipment_type: str
    status: EquipmentStatus
    zone_id: str | None = None
    running_command: str | None = None


class MaterialSummary(BaseModel):
    """A lightweight view of visible materials."""

    id: str
    name: str
    quantity: float
    quantity_unit: str
    zone_id: str | None = None
    is_hazardous: bool = False


class SampleSummary(BaseModel):
    """A lightweight view of visible samples."""

    id: str
    name: str
    sample_type: str
    status: str
    zone_id: str | None = None
    equipment_id: str | None = None
    bsl_level: BSLLevel | None = None
    volume_ul: float | None = None


class DoorSummary(BaseModel):
    """A lightweight view of visible doors from the current room."""

    id: str
    connects: tuple[str, str]
    is_open: bool
    locked: bool
    requires_keycard: bool = False
    requires_bsl_clearance: BSLLevel | None = None


class SelfState(BaseModel):
    """The agent's own embodied state."""

    id: str = ""
    name: str = ""
    role: PersonRole = PersonRole.GRAD_STUDENT
    current_room_id: str | None = None
    current_zone_id: str | None = None
    current_ppe: set[PPEType] = Field(default_factory=set)
    carrying: list[str] = Field(default_factory=list)
    current_activity: str | None = None
    certifications: list[CertificationType] = Field(default_factory=list)


class Observation(BaseModel):
    """What an agent perceives at a point in time."""

    observation_id: str = Field(default_factory=lambda: generate_id("obs"))
    agent_id: str = ""
    timestamp: datetime
    tick: int
    current_room: Room
    self_state: SelfState = Field(default_factory=SelfState)
    visible_persons: list[PersonSummary] = Field(default_factory=list)
    visible_equipment: list[EquipmentSummary] = Field(default_factory=list)
    visible_materials: list[MaterialSummary] = Field(default_factory=list)
    visible_samples: list[SampleSummary] = Field(default_factory=list)
    visible_doors: list[DoorSummary] = Field(default_factory=list)
    carrying: list[str] = Field(default_factory=list)
    current_ppe: set[PPEType] = Field(default_factory=set)
    active_alerts: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    working_memory: str = ""
    trace_parent_action_id: str | None = None
    narrative: str = ""


class AgentInterface(ABC):
    """Public interface implemented by all agent types."""

    @abstractmethod
    def observe(self, observation: Observation) -> None: ...

    @abstractmethod
    def act(self, action_spec: ActionSpec) -> Action: ...

    @abstractmethod
    def query(self, question: str) -> str: ...

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


__all__ = [
    "AgentInterface",
    "DoorSummary",
    "EquipmentSummary",
    "MaterialSummary",
    "Observation",
    "PersonSummary",
    "SampleSummary",
    "SelfState",
]
