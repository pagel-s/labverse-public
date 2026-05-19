"""Safety models: constraints, PPE requirements, and compliance rules."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from labverse.core.types import BSLLevel, CertificationType, PPEType


class SafetyVerdict(StrEnum):
    """Outcome of a safety constraint check."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class SafetyCheckResult(BaseModel):
    """Result of evaluating a single safety constraint."""

    verdict: SafetyVerdict
    constraint_id: str = ""
    constraint_name: str = ""
    message: str = ""
    remediation: str | None = None


class SafetyConstraint(BaseModel):
    """A declarative safety rule definition."""

    id: str
    name: str
    description: str = ""
    severity: SafetyVerdict = SafetyVerdict.BLOCK
    applies_to: list[str] = Field(default_factory=list)
    condition: str = ""
    remediation: str = ""
    regulation_ref: str | None = None


class BSLRequirements(BaseModel):
    """Requirements for a specific biosafety level."""

    level: BSLLevel
    min_ppe: set[PPEType] = Field(default_factory=set)
    requires_biosafety_cabinet: bool = False
    required_certifications: list[CertificationType] = Field(default_factory=list)
    requires_access_logging: bool = False
    requires_directional_airflow: bool = False
    requires_double_door_entry: bool = False
    decontamination_required_on_exit: bool = False
