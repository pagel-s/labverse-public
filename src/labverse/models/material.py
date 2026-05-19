"""Material models: chemicals, reagents, and biological materials with GHS data."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from labverse.core.types import (
    HazardPictogram,
    MaterialPhase,
    PPEType,
    SignalWord,
    WasteCategory,
)


class GHSData(BaseModel):
    """GHS classification for a chemical substance."""

    pictograms: list[HazardPictogram] = Field(default_factory=list)
    signal_word: SignalWord = SignalWord.NONE
    h_statements: list[str] = Field(default_factory=list)
    p_statements: list[str] = Field(default_factory=list)
    storage_codes: list[str] = Field(default_factory=list)


class Material(BaseModel):
    """A reagent, chemical, or biological material."""

    id: str
    name: str
    cas_number: str | None = None
    formula: str | None = None
    phase: MaterialPhase = MaterialPhase.LIQUID
    ghs: GHSData = Field(default_factory=GHSData)
    concentration: str | None = None
    quantity: float = 0.0
    quantity_unit: str = "mL"
    room_id: str
    zone_id: str | None = None
    storage_temperature_celsius: float | None = None
    expiry_date: date | None = None
    lot_number: str | None = None
    supplier: str | None = None
    incompatible_with: list[str] = Field(default_factory=list)
    required_ppe: list[PPEType] = Field(default_factory=list)
    waste_category: WasteCategory = WasteCategory.GENERAL
    is_controlled: bool = False

    def is_expired_on(self, current_date: date) -> bool:
        """Check expiry against an explicit date for deterministic simulation logic."""
        if self.expiry_date is None:
            return False
        return current_date > self.expiry_date

    @property
    def is_expired(self) -> bool:
        return self.is_expired_on(date.today())

    @property
    def is_hazardous(self) -> bool:
        return len(self.ghs.pictograms) > 0
