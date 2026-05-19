"""WorldState: the single source of truth for the entire simulation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import Sample
from labverse.models.space import Door, LabLayout, Room


class WorldState(BaseModel):
    """Central container holding all simulation entities.

    All queries and mutations go through WorldState.
    This is the single source of truth — no entity exists outside it.
    """

    lab: LabLayout = Field(default_factory=lambda: LabLayout(id="default", name="Default Lab"))
    equipment: dict[str, Equipment] = Field(default_factory=dict)
    materials: dict[str, Material] = Field(default_factory=dict)
    samples: dict[str, Sample] = Field(default_factory=dict)
    persons: dict[str, Person] = Field(default_factory=dict)

    # ── Room accessors ─────────────────────────────────────────────────

    @property
    def rooms(self) -> dict[str, Room]:
        return self.lab.rooms

    @property
    def doors(self) -> list[Door]:
        return self.lab.doors

    def get_room(self, room_id: str) -> Room:
        if room_id not in self.lab.rooms:
            raise KeyError(f"Room not found: {room_id}")
        return self.lab.rooms[room_id]

    def get_door(self, door_id: str) -> Door:
        for door in self.lab.doors:
            if door.id == door_id:
                return door
        raise KeyError(f"Door not found: {door_id}")

    # ── Generic entity accessors ───────────────────────────────────────

    def get_equipment(self, eq_id: str) -> Equipment:
        if eq_id not in self.equipment:
            raise KeyError(f"Equipment not found: {eq_id}")
        return self.equipment[eq_id]

    def get_material(self, mat_id: str) -> Material:
        if mat_id not in self.materials:
            raise KeyError(f"Material not found: {mat_id}")
        return self.materials[mat_id]

    def get_sample(self, sample_id: str) -> Sample:
        if sample_id not in self.samples:
            raise KeyError(f"Sample not found: {sample_id}")
        return self.samples[sample_id]

    def get_person(self, person_id: str) -> Person:
        if person_id not in self.persons:
            raise KeyError(f"Person not found: {person_id}")
        return self.persons[person_id]

    def _validate_zone(self, room: Room, zone_id: str | None) -> None:
        if zone_id is None:
            return
        if zone_id not in {zone.id for zone in room.zones}:
            raise KeyError(f"Zone not found in room {room.id}: {zone_id}")

    # ── Spatial queries ────────────────────────────────────────────────

    def equipment_in_room(self, room_id: str) -> list[Equipment]:
        return [eq for eq in self.equipment.values() if eq.room_id == room_id]

    def materials_in_room(self, room_id: str) -> list[Material]:
        return [m for m in self.materials.values() if m.room_id == room_id]

    def samples_in_room(self, room_id: str) -> list[Sample]:
        return [s for s in self.samples.values() if s.room_id == room_id]

    def persons_in_room(self, room_id: str) -> list[Person]:
        return [p for p in self.persons.values() if p.current_room_id == room_id]

    # ── Add / remove entities ──────────────────────────────────────────

    def add_room(self, room: Room) -> None:
        self.lab.rooms[room.id] = room

    def add_door(self, door: Door) -> None:
        self.lab.doors.append(door)

    def add_equipment(self, eq: Equipment) -> None:
        room = self.get_room(eq.room_id)
        self._validate_zone(room, eq.zone_id)
        existing = self.equipment.get(eq.id)
        if existing is not None:
            old_room = self.lab.rooms.get(existing.room_id)
            if old_room is not None and eq.id in old_room.equipment_ids:
                old_room.equipment_ids.remove(eq.id)
        self.equipment[eq.id] = eq
        if eq.id not in room.equipment_ids:
            room.equipment_ids.append(eq.id)

    def add_material(self, mat: Material) -> None:
        room = self.get_room(mat.room_id)
        self._validate_zone(room, mat.zone_id)
        existing = self.materials.get(mat.id)
        if existing is not None:
            old_room = self.lab.rooms.get(existing.room_id)
            if old_room is not None and mat.id in old_room.material_ids:
                old_room.material_ids.remove(mat.id)
        self.materials[mat.id] = mat
        if mat.id not in room.material_ids:
            room.material_ids.append(mat.id)

    def add_sample(self, sample: Sample) -> None:
        room = self.get_room(sample.room_id)
        self._validate_zone(room, sample.zone_id)
        if sample.equipment_id is not None:
            equipment = self.get_equipment(sample.equipment_id)
            if equipment.room_id != sample.room_id:
                raise ValueError(
                    f"Sample {sample.id} is in room {sample.room_id} but equipment "
                    f"{equipment.id} is in room {equipment.room_id}"
                )
        self.samples[sample.id] = sample

    def add_person(self, person: Person) -> None:
        if person.current_room_id is None:
            if person.current_zone_id is not None:
                raise ValueError(
                    f"Person {person.id} has zone {person.current_zone_id} but no current room"
                )
        else:
            room = self.get_room(person.current_room_id)
            self._validate_zone(room, person.current_zone_id)
        self.persons[person.id] = person

    def remove_equipment(self, eq_id: str) -> Equipment:
        if eq_id not in self.equipment:
            raise KeyError(f"Equipment not found: {eq_id}")
        equipment = self.equipment.pop(eq_id)
        room = self.lab.rooms.get(equipment.room_id)
        if room is not None and eq_id in room.equipment_ids:
            room.equipment_ids.remove(eq_id)
        return equipment

    def remove_material(self, mat_id: str) -> Material:
        if mat_id not in self.materials:
            raise KeyError(f"Material not found: {mat_id}")
        material = self.materials.pop(mat_id)
        room = self.lab.rooms.get(material.room_id)
        if room is not None and mat_id in room.material_ids:
            room.material_ids.remove(mat_id)
        return material

    def remove_sample(self, sample_id: str) -> Sample:
        if sample_id not in self.samples:
            raise KeyError(f"Sample not found: {sample_id}")
        return self.samples.pop(sample_id)

    def remove_person(self, person_id: str) -> Person:
        if person_id not in self.persons:
            raise KeyError(f"Person not found: {person_id}")
        return self.persons.pop(person_id)

    # ── Snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> WorldState:
        """Create a deep copy of the current state."""
        return type(self).model_validate(
            self.model_dump(mode="python", round_trip=True),
        )

    # ── Summary ────────────────────────────────────────────────────────

    @property
    def entity_counts(self) -> dict[str, int]:
        return {
            "rooms": len(self.lab.rooms),
            "doors": len(self.lab.doors),
            "equipment": len(self.equipment),
            "materials": len(self.materials),
            "samples": len(self.samples),
            "persons": len(self.persons),
        }

    def __repr__(self) -> str:
        counts = self.entity_counts
        return (
            f"WorldState(lab={self.lab.name!r}, "
            f"rooms={counts['rooms']}, eq={counts['equipment']}, "
            f"mat={counts['materials']}, persons={counts['persons']})"
        )
