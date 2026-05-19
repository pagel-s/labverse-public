"""Protocol models: structured experimental procedures."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labverse.core.types import BSLLevel, CertificationType


class Precondition(BaseModel):
    """A condition that must be true before a protocol step can execute."""

    condition_type: str
    target_id: str | None = None
    expected_value: str | None = None
    description: str = ""


class ProtocolStep(BaseModel):
    """A single step in a protocol."""

    id: str
    order: int
    description: str
    action_type: str
    equipment_type: str | None = None
    equipment_id: str | None = None
    parameters: dict[str, float | str | bool] = Field(default_factory=dict)
    duration_seconds: float | None = None
    materials_consumed: dict[str, float] = Field(default_factory=dict)
    preconditions: list[Precondition] = Field(default_factory=list)
    substeps: list[ProtocolStep] = Field(default_factory=list)
    produces_sample: bool = False
    produces_waste: bool = False
    notes: str | None = None


class Protocol(BaseModel):
    """A complete lab protocol."""

    id: str
    name: str
    description: str = ""
    version: str = "1.0"
    bsl_level_required: BSLLevel = BSLLevel.BSL1
    estimated_duration_minutes: float = 0.0
    steps: list[ProtocolStep] = Field(default_factory=list)
    required_equipment_types: list[str] = Field(default_factory=list)
    required_materials: list[str] = Field(default_factory=list)
    required_certifications: list[CertificationType] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)

    @property
    def total_steps(self) -> int:
        """Count all steps including substeps recursively."""

        def _count(steps: list[ProtocolStep]) -> int:
            total = len(steps)
            for step in steps:
                total += _count(step.substeps)
            return total

        return _count(self.steps)
