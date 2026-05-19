"""Pydantic schema models for YAML lab configuration files."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field, model_validator

from labverse.core.types import CertificationType, EquipmentStatus
from labverse.models.equipment import EquipmentCommand
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import Sample
from labverse.models.space import Door, Position, Room


class ConfigError(ValueError):
    """Raised when a lab configuration file is invalid."""


class LabMetadata(BaseModel):
    """Top-level metadata for a lab configuration."""

    id: str
    name: str
    description: str = ""
    default_entry_room: str = ""


class EquipmentConfig(BaseModel):
    """Config representation of an equipment instance.

    `properties` intentionally accepts raw YAML values. The builder normalizes
    them into `EquipmentParameter` objects when constructing the runtime world.
    """

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
    properties: dict[str, object] = Field(default_factory=dict)
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


class LabConfig(BaseModel):
    """Validated YAML structure for building a `WorldState`."""

    lab: LabMetadata
    rooms: list[Room] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    equipment: list[EquipmentConfig] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list)
    persons: list[Person] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> LabConfig:
        room_ids = _ensure_unique_ids("room", [room.id for room in self.rooms])
        equipment_ids = _ensure_unique_ids(
            "equipment",
            [equipment.id for equipment in self.equipment],
        )
        _ensure_unique_ids("material", [material.id for material in self.materials])
        _ensure_unique_ids("person", [person.id for person in self.persons])
        _ensure_unique_ids("sample", [sample.id for sample in self.samples])

        room_by_id = {room.id: room for room in self.rooms}
        if not self.lab.default_entry_room and self.rooms:
            self.lab.default_entry_room = self.rooms[0].id
        if self.lab.default_entry_room and self.lab.default_entry_room not in room_by_id:
            raise ValueError(
                f"default_entry_room references unknown room {self.lab.default_entry_room!r}"
            )

        for door in self.doors:
            room_a, room_b = door.connects
            if room_a == room_b:
                raise ValueError(f"door {door.id!r} must connect two different rooms")
            _ensure_room_reference(room_ids, room_a, f"door {door.id!r}")
            _ensure_room_reference(room_ids, room_b, f"door {door.id!r}")

        for equipment in self.equipment:
            room = _validate_room_zone_reference(
                room_by_id,
                equipment.room_id,
                equipment.zone_id,
                f"equipment {equipment.id!r}",
            )
            if equipment.max_concurrent_users < 1:
                raise ValueError(f"equipment {equipment.id!r} must allow at least one user")
            if len(equipment.current_users) > equipment.max_concurrent_users:
                raise ValueError(
                    f"equipment {equipment.id!r} has more current users than max_concurrent_users"
                )
            if room.id != equipment.room_id:
                raise AssertionError("validated room mismatch")

        for material in self.materials:
            _validate_room_zone_reference(
                room_by_id,
                material.room_id,
                material.zone_id,
                f"material {material.id!r}",
            )

        for person in self.persons:
            if person.current_zone_id is not None and person.current_room_id is None:
                raise ValueError(f"person {person.id!r} has current_zone_id but no current_room_id")
            if person.current_room_id is not None:
                _validate_room_zone_reference(
                    room_by_id,
                    person.current_room_id,
                    person.current_zone_id,
                    f"person {person.id!r}",
                )

        for sample in self.samples:
            _validate_room_zone_reference(
                room_by_id,
                sample.room_id,
                sample.zone_id,
                f"sample {sample.id!r}",
            )
            if sample.equipment_id is not None:
                if sample.equipment_id not in equipment_ids:
                    raise ValueError(
                        f"sample {sample.id!r} references unknown equipment {sample.equipment_id!r}"
                    )
                equipment_room_id = next(
                    equipment.room_id
                    for equipment in self.equipment
                    if equipment.id == sample.equipment_id
                )
                if equipment_room_id != sample.room_id:
                    raise ValueError(
                        f"sample {sample.id!r} room {sample.room_id!r} does not match "
                        f"equipment {sample.equipment_id!r} room {equipment_room_id!r}"
                    )

        return self


def _ensure_unique_ids(entity_type: str, ids: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entity_id in ids:
        if entity_id in seen:
            duplicates.add(entity_id)
        seen.add(entity_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {entity_type} ids: {duplicate_list}")
    return seen


def _ensure_room_reference(room_ids: set[str], room_id: str, context: str) -> None:
    if room_id not in room_ids:
        raise ValueError(f"{context} references unknown room {room_id!r}")


def _validate_room_zone_reference(
    room_by_id: dict[str, Room],
    room_id: str,
    zone_id: str | None,
    context: str,
) -> Room:
    if room_id not in room_by_id:
        raise ValueError(f"{context} references unknown room {room_id!r}")
    room = room_by_id[room_id]
    if zone_id is None:
        return room
    valid_zone_ids = {zone.id for zone in room.zones}
    if zone_id not in valid_zone_ids:
        raise ValueError(f"{context} references unknown zone {zone_id!r} in room {room_id!r}")
    return room
