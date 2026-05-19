"""Shared benchmark scripted-step and event-log helpers."""

from __future__ import annotations

from typing import Any

from labverse.agent.interface import Observation
from labverse.agent.scripted import ScriptedStep
from labverse.core.events import Event, EventLog, EventType
from labverse.core.types import PPEType, WasteCategory
from labverse.models.action import Action, ActionSpec, AvailableAction


def action_step(
    *,
    verb: str,
    target_id: str | None = None,
    description_fragment: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> ScriptedStep:
    """Create a scripted step that selects a currently available action."""

    def build_action(observation: Observation, action_spec: ActionSpec) -> Action:
        del observation
        available = _matching_action(
            action_spec,
            verb=verb,
            target_id=target_id,
            description_fragment=description_fragment,
        )
        return Action(
            id="",
            agent_id="",
            category=available.category,
            verb=available.verb,
            target_id=available.target_id,
            parameters=dict(parameters or {}),
        )

    return build_action


def don_ppe_step(ppe_type: PPEType) -> ScriptedStep:
    return action_step(
        verb="don_ppe",
        description_fragment=ppe_type.value,
        parameters={"ppe_type": ppe_type.value},
    )


def move_to_step(room_id: str) -> ScriptedStep:
    return action_step(verb="move_to", target_id=room_id)


def start_equipment_step(
    equipment_id: str,
    *,
    parameters: dict[str, Any],
) -> ScriptedStep:
    return action_step(
        verb="start_equipment",
        target_id=equipment_id,
        parameters=parameters,
    )


def stop_equipment_step(equipment_id: str) -> ScriptedStep:
    return action_step(verb="stop_equipment", target_id=equipment_id)


def pipette_transfer_step(
    equipment_id: str,
    *,
    source_id: str,
    destination_id: str,
    volume_ul: float,
    extra_parameters: dict[str, Any] | None = None,
) -> ScriptedStep:
    parameters: dict[str, Any] = {
        "source_id": source_id,
        "destination_id": destination_id,
        "volume_ul": volume_ul,
    }
    parameters.update(extra_parameters or {})
    return action_step(
        verb="pipette_transfer",
        target_id=equipment_id,
        parameters=parameters,
    )


def retrieve_sample_step(sample_id: str) -> ScriptedStep:
    return action_step(verb="retrieve_sample", target_id=sample_id)


def store_sample_in_equipment_step(sample_id: str, equipment_id: str) -> ScriptedStep:
    return action_step(
        verb="store_sample",
        target_id=sample_id,
        parameters={"destination_equipment_id": equipment_id},
    )


def store_sample_in_room_step(sample_id: str, room_id: str) -> ScriptedStep:
    return action_step(
        verb="store_sample",
        target_id=sample_id,
        parameters={"destination_room_id": room_id},
    )


def dispose_material_step(material_id: str, waste_category: WasteCategory) -> ScriptedStep:
    return dispose_material_step_with_parameters(
        material_id,
        parameters={"waste_category": waste_category.value},
    )


def dispose_material_step_with_parameters(
    material_id: str,
    *,
    parameters: dict[str, Any],
) -> ScriptedStep:
    return action_step(
        verb="dispose_material",
        target_id=material_id,
        parameters=parameters,
    )


def log_observation_step(notes: str) -> ScriptedStep:
    return action_step(
        verb="log_observation",
        parameters={"notes": notes},
    )


def wait_step() -> ScriptedStep:
    return action_step(verb="wait")


def structured_finding_note(
    *,
    entity_id: str,
    issue_code: str,
    room_id: str,
    related_entity_id: str | None = None,
) -> str:
    """Build a compact structured finding note for audit-style scenarios."""

    tokens = [
        "finding",
        f"entity={entity_id}",
        f"issue={issue_code}",
        f"room={room_id}",
    ]
    if related_entity_id is not None:
        tokens.append(f"related={related_entity_id}")
    return " ".join(tokens)


def basic_lab_ppe_steps() -> list[ScriptedStep]:
    return [
        don_ppe_step(PPEType.LAB_COAT),
        don_ppe_step(PPEType.SAFETY_GLASSES),
        don_ppe_step(PPEType.CLOSED_TOE_SHOES),
    ]


def bsl2_ppe_steps() -> list[ScriptedStep]:
    return [*basic_lab_ppe_steps(), don_ppe_step(PPEType.NITRILE_GLOVES)]


def submitted_actions(
    event_log: EventLog,
    verb: str,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
) -> list[Event]:
    events = event_log.query(event_type=EventType.ACTION_SUBMITTED)
    result: list[Event] = []
    for event in events:
        if _submitted_verb(event) != verb:
            continue
        if source_id is not None and event.source_id != source_id:
            continue
        if target_id is not None and event.target_id != target_id:
            continue
        result.append(event)
    return result


def successful_actions(
    event_log: EventLog,
    verb: str,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
) -> list[Event]:
    """Return submitted actions whose matching resolution succeeded."""

    resolved_success_by_action_id = {
        event.data.get("action_id"): event.data.get("success") is True
        for event in event_log.query(event_type=EventType.ACTION_RESOLVED)
        if isinstance(event.data.get("action_id"), str)
    }
    return [
        event
        for event in submitted_actions(
            event_log,
            verb,
            source_id=source_id,
            target_id=target_id,
        )
        if resolved_success_by_action_id.get(event.data.get("action_id")) is True
    ]


def successful_action_indices(
    event_log: EventLog,
    verb: str,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
) -> list[int]:
    """Return event-log indices for submitted actions whose resolution succeeded."""

    successful_ids = {
        event.data.get("action_id")
        for event in successful_actions(
            event_log,
            verb,
            source_id=source_id,
            target_id=target_id,
        )
        if isinstance(event.data.get("action_id"), str)
    }
    indices: list[int] = []
    for index, event in enumerate(event_log.events):
        if event.event_type != EventType.ACTION_SUBMITTED:
            continue
        if event.data.get("action_id") in successful_ids:
            indices.append(index)
    return indices


def latest_action_parameters(
    event_log: EventLog,
    verb: str,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    actions = submitted_actions(
        event_log,
        verb,
        source_id=source_id,
        target_id=target_id,
    )
    if not actions:
        return {}

    raw_parameters = actions[-1].data.get("parameters", {})
    return raw_parameters if isinstance(raw_parameters, dict) else {}


def logged_notes(event_log: EventLog, *, source_id: str | None = None) -> list[str]:
    notes: list[str] = []
    for event in submitted_actions(event_log, "log_observation", source_id=source_id):
        raw_parameters = event.data.get("parameters", {})
        if not isinstance(raw_parameters, dict):
            continue
        note = raw_parameters.get("notes")
        if isinstance(note, str) and note:
            notes.append(note)
    return notes


def parse_structured_finding_note(note: str) -> dict[str, str] | None:
    """Parse a structured_finding_note payload into normalized lowercase fields."""

    tokens = note.strip().split()
    if not tokens or tokens[0].casefold() != "finding":
        return None
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key.casefold()] = value.casefold()
    if not {"entity", "issue", "room"} <= set(fields):
        return None
    return fields


def blocked_action_count(event_log: EventLog) -> int:
    return len(event_log.query(event_type=EventType.ACTION_BLOCKED))


def _matching_action(
    action_spec: ActionSpec,
    *,
    verb: str,
    target_id: str | None,
    description_fragment: str | None,
) -> AvailableAction:
    normalized_fragment = (
        _normalize_description_fragment(description_fragment)
        if description_fragment is not None
        else None
    )
    for action in action_spec.available_actions:
        if action.verb != verb:
            continue
        if target_id is not None and action.target_id != target_id:
            continue
        if (
            normalized_fragment is not None
            and normalized_fragment not in _normalize_description_fragment(action.description)
        ):
            continue
        return action

    descriptions = [candidate.description for candidate in action_spec.available_actions]
    raise AssertionError(
        f"Action {verb!r} with target {target_id!r} not available. Visible options: {descriptions}"
    )


def _normalize_description_fragment(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


def _submitted_verb(event: Event) -> str | None:
    prefix = f"{event.source_id} submitted "
    if not event.description.startswith(prefix):
        return None
    return event.description[len(prefix) :]


__all__ = [
    "action_step",
    "basic_lab_ppe_steps",
    "blocked_action_count",
    "bsl2_ppe_steps",
    "dispose_material_step",
    "dispose_material_step_with_parameters",
    "don_ppe_step",
    "latest_action_parameters",
    "log_observation_step",
    "logged_notes",
    "move_to_step",
    "parse_structured_finding_note",
    "pipette_transfer_step",
    "retrieve_sample_step",
    "start_equipment_step",
    "stop_equipment_step",
    "store_sample_in_equipment_step",
    "store_sample_in_room_step",
    "structured_finding_note",
    "submitted_actions",
    "successful_actions",
    "successful_action_indices",
    "wait_step",
]
