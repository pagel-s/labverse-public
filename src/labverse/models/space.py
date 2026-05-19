"""Spatial models: positions, doors, zones, rooms, and lab layouts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labverse.core.types import BSLLevel, RoomType


class Position(BaseModel):
    """A 2D coordinate within a room or floor plan."""

    x: float = 0.0
    y: float = 0.0


class Door(BaseModel):
    """A connection between two rooms."""

    id: str
    connects: tuple[str, str]
    is_open: bool = True
    locked: bool = False
    requires_keycard: bool = False
    requires_bsl_clearance: BSLLevel | None = None


class Zone(BaseModel):
    """A sub-area within a room (bench, fume hood area, storage shelf, etc.)."""

    id: str
    name: str
    position: Position = Field(default_factory=Position)
    width: float = 1.0
    height: float = 1.0
    zone_type: str = "bench"
    capacity: int = 1


class Room(BaseModel):
    """A discrete space in the laboratory."""

    id: str
    name: str
    room_type: RoomType
    bsl_level: BSLLevel = BSLLevel.BSL1
    temperature_celsius: float = 22.0
    humidity_percent: float = 45.0
    has_ventilation: bool = True
    has_emergency_shower: bool = False
    has_eyewash: bool = False
    max_occupancy: int = 10
    position: Position = Field(default_factory=Position)
    width: float = 6.0
    height: float = 4.0
    zones: list[Zone] = Field(default_factory=list)
    equipment_ids: list[str] = Field(default_factory=list)
    material_ids: list[str] = Field(default_factory=list)


class LabLayout(BaseModel):
    """Top-level spatial model describing the entire lab."""

    id: str
    name: str
    description: str = ""
    rooms: dict[str, Room] = Field(default_factory=dict)
    doors: list[Door] = Field(default_factory=list)
    default_entry_room: str = ""
