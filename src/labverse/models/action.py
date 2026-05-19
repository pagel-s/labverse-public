"""Action models: agent actions, results, and available action specs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from labverse.core.types import ActionCategory

_ACTION_GROUP_BY_VERB: dict[str, str] = {
    "move_to": "movement",
    "open_door": "doors",
    "close_door": "doors",
    "unlock_door": "doors",
    "don_ppe": "ppe",
    "doff_ppe": "ppe",
    "start_equipment": "equipment",
    "stop_equipment": "equipment",
    "dispose_material": "materials",
    "pipette_transfer": "samples",
    "retrieve_sample": "samples",
    "store_sample": "samples",
    "wait": "wait",
    "log_observation": "observe",
}
_ACTION_GROUP_LABELS: dict[str, str] = {
    "movement": "Movement",
    "doors": "Doors",
    "ppe": "PPE",
    "equipment": "Equipment",
    "materials": "Materials",
    "samples": "Samples",
    "wait": "Wait",
    "observe": "Observe",
}


def _group_key_for_verb(verb: str) -> str:
    return _ACTION_GROUP_BY_VERB.get(verb, "observe")


class Action(BaseModel):
    """An action submitted by an agent to the Game Master."""

    id: str
    agent_id: str
    category: ActionCategory
    verb: str
    target_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    protocol_step_id: str | None = None


class ActionResultStatus(StrEnum):
    """Machine-readable action outcome classification."""

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


class ChangedEntity(BaseModel):
    """A summarized entity touched by an action."""

    entity_type: str
    entity_id: str
    summary: str = ""
    room_id: str | None = None
    zone_id: str | None = None


class ActionResult(BaseModel):
    """The outcome of an action after Game Master adjudication."""

    action_id: str
    success: bool
    status: ActionResultStatus | None = None
    outcome_description: str = ""
    state_changes: list[dict[str, Any]] = Field(default_factory=list)
    changed_entities: list[ChangedEntity] = Field(default_factory=list)
    time_elapsed_seconds: float = 0.0
    safety_violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None
    failure_reason: str | None = None
    emitted_event_ids: list[str] = Field(default_factory=list)
    post_observation_id: str | None = None


class AvailableAction(BaseModel):
    """A single executable action candidate available on the current turn."""

    action_id: str = Field(exclude=True)
    candidate_ref: str = ""
    group_key: str = ""
    group_label: str = ""
    actor_id: str | None = None
    label: str = ""
    verb: str
    category: ActionCategory
    description: str = ""
    target_id: str | None = None
    target_type: str | None = None
    target_label: str | None = None
    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    required_parameters: list[str] = Field(default_factory=list)
    optional_parameters: list[str] = Field(default_factory=list)
    parameter_details: dict[str, str] = Field(default_factory=dict)
    parameter_schema: dict[str, Any] = Field(default_factory=dict, exclude=True)
    free_parameter_schema: dict[str, Any] = Field(default_factory=dict)
    estimated_duration_seconds: float | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.candidate_ref:
            self.candidate_ref = self.action_id
        if not self.group_key:
            self.group_key = _group_key_for_verb(self.verb)
        if not self.group_label:
            self.group_label = _ACTION_GROUP_LABELS.get(self.group_key, self.group_key.title())
        if not self.label:
            self.label = self.description or self.verb
        if not self.free_parameter_schema and self.parameter_schema:
            self.free_parameter_schema = dict(self.parameter_schema)


class ActionGroup(BaseModel):
    """A grouped presentation slice of the current action surface."""

    group_key: str
    label: str
    candidates: list[AvailableAction] = Field(default_factory=list)


class ActionSpec(BaseModel):
    """The full grouped action surface for an agent at a point in time."""

    agent_id: str | None = None
    tick: int | None = None
    groups: list[ActionGroup] = Field(default_factory=list)
    available_actions: list[AvailableAction] = Field(default_factory=list, exclude=True)
    context: str = ""


VERB_CATEGORIES: dict[str, ActionCategory] = {
    "move_to": ActionCategory.MOVE,
    "open_door": ActionCategory.MOVE,
    "close_door": ActionCategory.MOVE,
    "unlock_door": ActionCategory.MOVE,
    "start_equipment": ActionCategory.USE_EQUIPMENT,
    "stop_equipment": ActionCategory.USE_EQUIPMENT,
    "pipette_transfer": ActionCategory.HANDLE_SAMPLE,
    "retrieve_sample": ActionCategory.HANDLE_SAMPLE,
    "store_sample": ActionCategory.HANDLE_SAMPLE,
    "don_ppe": ActionCategory.SAFETY,
    "doff_ppe": ActionCategory.SAFETY,
    "dispose_material": ActionCategory.HANDLE_MATERIAL,
    "wait": ActionCategory.OBSERVE,
    "log_observation": ActionCategory.OBSERVE,
}
