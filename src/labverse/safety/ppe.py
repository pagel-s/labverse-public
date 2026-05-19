"""PPE requirement resolution helpers."""

from __future__ import annotations

from typing import Any

from labverse.core.types import PPEType, RoomType
from labverse.models.action import Action
from labverse.models.person import Person
from labverse.models.space import Room
from labverse.safety.bsl_rules import minimum_ppe_for_bsl
from labverse.world.state import WorldState


def resolve_required_ppe(action: Action, world: WorldState) -> set[PPEType]:
    """Compute the PPE required for a specific action in the current world."""
    required: set[PPEType] = set()

    room = action_room(action, world)
    if room is not None:
        required.update(room_required_ppe(room))

    for entity_id in _candidate_entity_ids(action):
        if entity_id in world.materials:
            required.update(world.get_material(entity_id).required_ppe)
        if entity_id in world.samples:
            sample = world.get_sample(entity_id)
            required.update(minimum_ppe_for_bsl(sample.bsl_level))

    explicit_required_ppe = action.parameters.get("required_ppe")
    required.update(_coerce_ppe_set(explicit_required_ppe))
    return required


def missing_ppe(person: Person, required_ppe: set[PPEType]) -> set[PPEType]:
    """Return the subset of required PPE the person is not wearing."""
    return required_ppe - person.current_ppe


def action_room(action: Action, world: WorldState) -> Room | None:
    """Resolve the most relevant room for an action."""
    destination_room_id = action.parameters.get("destination_room_id")
    if isinstance(destination_room_id, str) and destination_room_id in world.rooms:
        return world.get_room(destination_room_id)

    if (
        action.verb == "move_to"
        and action.target_id is not None
        and action.target_id in world.rooms
    ):
        return world.get_room(action.target_id)

    person = world.persons.get(action.agent_id)
    if person is not None and person.current_room_id is not None:
        return world.get_room(person.current_room_id)

    if action.target_id is None:
        return None
    if action.target_id in world.equipment:
        return world.get_room(world.get_equipment(action.target_id).room_id)
    if action.target_id in world.materials:
        return world.get_room(world.get_material(action.target_id).room_id)
    if action.target_id in world.samples:
        return world.get_room(world.get_sample(action.target_id).room_id)
    if action.target_id in world.rooms:
        return world.get_room(action.target_id)
    return None


def action_zone_type(action: Action, world: WorldState) -> str | None:
    """Resolve the active zone type for an action when one is known."""
    room = action_room(action, world)
    if room is None:
        return None

    zone_id = action.parameters.get("zone_id")
    if isinstance(zone_id, str):
        zone = next((candidate for candidate in room.zones if candidate.id == zone_id), None)
        return zone.zone_type if zone is not None else None

    person = world.persons.get(action.agent_id)
    if (
        person is not None
        and person.current_room_id == room.id
        and person.current_zone_id is not None
    ):
        zone = next(
            (candidate for candidate in room.zones if candidate.id == person.current_zone_id),
            None,
        )
        return zone.zone_type if zone is not None else None

    if action.target_id is not None and action.target_id in world.materials:
        material = world.get_material(action.target_id)
        if material.zone_id is not None:
            zone = next(
                (candidate for candidate in room.zones if candidate.id == material.zone_id), None
            )
            return zone.zone_type if zone is not None else None

    return None


def room_required_ppe(room: Room) -> set[PPEType]:
    """Resolve baseline PPE expectations for a room."""
    if room.room_type in {RoomType.OFFICE, RoomType.SERVER_ROOM, RoomType.HALLWAY}:
        return set()
    return set(minimum_ppe_for_bsl(room.bsl_level))


def _coerce_ppe_set(raw_value: Any) -> set[PPEType]:
    values: list[str] = []
    if isinstance(raw_value, PPEType):
        values = [raw_value.value]
    elif isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = [value for value in raw_value if isinstance(value, str)]

    result: set[PPEType] = set()
    for value in values:
        try:
            result.add(PPEType(value))
        except ValueError:
            continue
    return result


def _candidate_entity_ids(action: Action) -> set[str]:
    ids: set[str] = set()
    if action.target_id is not None:
        ids.add(action.target_id)

    for value in action.parameters.values():
        if isinstance(value, str):
            ids.add(value)
        elif isinstance(value, list):
            ids.update(item for item in value if isinstance(item, str))
    return ids


__all__ = [
    "action_room",
    "action_zone_type",
    "missing_ppe",
    "room_required_ppe",
    "resolve_required_ppe",
]
