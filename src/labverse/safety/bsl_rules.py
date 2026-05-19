"""Per-BSL safety requirements and convenience helpers."""

from __future__ import annotations

from labverse.core.types import BSLLevel, CertificationType, PPEType
from labverse.models.safety import BSLRequirements

BSL_REQUIREMENTS: dict[BSLLevel, BSLRequirements] = {
    BSLLevel.BSL1: BSLRequirements(
        level=BSLLevel.BSL1,
        min_ppe={
            PPEType.LAB_COAT,
            PPEType.SAFETY_GLASSES,
            PPEType.CLOSED_TOE_SHOES,
        },
    ),
    BSLLevel.BSL2: BSLRequirements(
        level=BSLLevel.BSL2,
        min_ppe={
            PPEType.LAB_COAT,
            PPEType.SAFETY_GLASSES,
            PPEType.NITRILE_GLOVES,
            PPEType.CLOSED_TOE_SHOES,
        },
        required_certifications=[CertificationType.BSL2_TRAINING],
    ),
    BSLLevel.BSL3: BSLRequirements(
        level=BSLLevel.BSL3,
        min_ppe={
            PPEType.LAB_COAT,
            PPEType.SAFETY_GLASSES,
            PPEType.NITRILE_GLOVES,
            PPEType.RESPIRATOR,
            PPEType.CLOSED_TOE_SHOES,
        },
        requires_biosafety_cabinet=True,
        required_certifications=[CertificationType.BSL3_TRAINING],
        requires_access_logging=True,
        requires_directional_airflow=True,
        requires_double_door_entry=True,
        decontamination_required_on_exit=True,
    ),
    BSLLevel.BSL4: BSLRequirements(
        level=BSLLevel.BSL4,
        min_ppe={PPEType.FULL_BODY_SUIT},
        requires_biosafety_cabinet=True,
        required_certifications=[CertificationType.BSL4_TRAINING],
        requires_access_logging=True,
        requires_directional_airflow=True,
        requires_double_door_entry=True,
        decontamination_required_on_exit=True,
    ),
}


def requirements_for_bsl(level: BSLLevel) -> BSLRequirements:
    """Return the modeled safety requirements for a biosafety level."""
    return BSL_REQUIREMENTS[level]


def minimum_ppe_for_bsl(level: BSLLevel) -> set[PPEType]:
    """Return the minimum PPE set expected for a BSL level."""
    return set(requirements_for_bsl(level).min_ppe)


def required_certifications_for_bsl(level: BSLLevel) -> set[CertificationType]:
    """Return certification requirements for a BSL level."""
    return set(requirements_for_bsl(level).required_certifications)


def bsl_requires_biosafety_cabinet(
    level: BSLLevel,
    *,
    aerosol_generating: bool = False,
) -> bool:
    """Return whether a biosafety cabinet is required for this work."""
    if level >= BSLLevel.BSL3:
        return True
    if level == BSLLevel.BSL2:
        return aerosol_generating
    return False


__all__ = [
    "BSL_REQUIREMENTS",
    "bsl_requires_biosafety_cabinet",
    "minimum_ppe_for_bsl",
    "requirements_for_bsl",
    "required_certifications_for_bsl",
]
