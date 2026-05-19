"""Sample models: tracked specimens with chain of custody."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from labverse.core.types import BSLLevel


class CustodyRecord(BaseModel):
    """A single entry in the chain of custody."""

    timestamp: datetime
    person_id: str
    action: str
    location_room_id: str
    location_zone_id: str | None = None
    location_equipment_id: str | None = None
    notes: str | None = None


class Sample(BaseModel):
    """A tracked biological or chemical sample."""

    id: str
    name: str
    sample_type: str
    source: str | None = None
    container_type: str = "microcentrifuge_tube"
    volume_ul: float | None = None
    concentration: str | None = None
    room_id: str
    zone_id: str | None = None
    equipment_id: str | None = None
    status: str = "active"
    bsl_level: BSLLevel = BSLLevel.BSL1
    created_at: datetime
    created_by: str
    chain_of_custody: list[CustodyRecord] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    derived_from: list[str] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def last_custody_record(self) -> CustodyRecord | None:
        return self.chain_of_custody[-1] if self.chain_of_custody else None
