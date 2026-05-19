"""EntityRegistry: indexed lookups across all entity types."""

from __future__ import annotations

from datetime import date
from typing import Any, TypedDict

from labverse.core.types import EquipmentStatus, PersonRole, RoomType
from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import Sample
from labverse.models.space import Room
from labverse.world.state import WorldState


class RoomContents(TypedDict):
    """Typed aggregate of everything physically located in a room."""

    equipment: list[Equipment]
    materials: list[Material]
    samples: list[Sample]
    persons: list[Person]


class EntityRegistry:
    """Provides efficient querying across all entities in a WorldState.

    Wraps a WorldState and adds filtered queries beyond basic ID lookup.
    """

    def __init__(self, world: WorldState) -> None:
        self._world = world

    @property
    def world(self) -> WorldState:
        return self._world

    # ── By-ID lookup (delegates to WorldState) ─────────────────────────

    def get(self, entity_type: str, entity_id: str) -> Any:
        """Generic get by type string and ID."""
        accessors = {
            "room": self._world.get_room,
            "equipment": self._world.get_equipment,
            "material": self._world.get_material,
            "sample": self._world.get_sample,
            "person": self._world.get_person,
        }
        if entity_type not in accessors:
            raise ValueError(f"Unknown entity type: {entity_type}")
        return accessors[entity_type](entity_id)

    # ── Room queries ───────────────────────────────────────────────────

    def rooms_by_type(self, room_type: RoomType) -> list[Room]:
        return [r for r in self._world.rooms.values() if r.room_type == room_type]

    def rooms_by_bsl(self, min_bsl: int) -> list[Room]:
        return [r for r in self._world.rooms.values() if r.bsl_level >= min_bsl]

    # ── Equipment queries ──────────────────────────────────────────────

    def equipment_by_type(self, eq_type: str) -> list[Equipment]:
        return [eq for eq in self._world.equipment.values() if eq.equipment_type == eq_type]

    def equipment_by_status(self, status: EquipmentStatus) -> list[Equipment]:
        return [eq for eq in self._world.equipment.values() if eq.status == status]

    def available_equipment(self, eq_type: str | None = None) -> list[Equipment]:
        result = [eq for eq in self._world.equipment.values() if eq.is_available]
        if eq_type is not None:
            result = [eq for eq in result if eq.equipment_type == eq_type]
        return result

    # ── Material queries ──────────────────────────────────────���────────

    def hazardous_materials(self) -> list[Material]:
        return [m for m in self._world.materials.values() if m.is_hazardous]

    def expired_materials(self, current_date: date | None = None) -> list[Material]:
        effective_date = current_date or date.today()
        return [m for m in self._world.materials.values() if m.is_expired_on(effective_date)]

    def materials_by_room(self, room_id: str) -> list[Material]:
        return self._world.materials_in_room(room_id)

    # ── Person queries ─────────────────────────────────────────────────

    def persons_by_role(self, role: PersonRole) -> list[Person]:
        return [p for p in self._world.persons.values() if p.role == role]

    def persons_in_room(self, room_id: str) -> list[Person]:
        return self._world.persons_in_room(room_id)

    def active_persons(self) -> list[Person]:
        return [p for p in self._world.persons.values() if p.is_active]

    # ── Sample queries ─────────────────────────────────────────────────

    def active_samples(self) -> list[Sample]:
        return [s for s in self._world.samples.values() if s.is_active]

    def samples_by_type(self, sample_type: str) -> list[Sample]:
        return [s for s in self._world.samples.values() if s.sample_type == sample_type]

    # ── Aggregate queries ──────────────────────────────────────────────

    def all_in_room(self, room_id: str) -> RoomContents:
        """Get everything located in a specific room."""
        return {
            "equipment": self._world.equipment_in_room(room_id),
            "materials": self._world.materials_in_room(room_id),
            "samples": self._world.samples_in_room(room_id),
            "persons": self._world.persons_in_room(room_id),
        }
