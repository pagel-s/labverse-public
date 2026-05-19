"""Chemical hazard compatibility heuristics for safety checks."""

from __future__ import annotations

from labverse.core.types import HazardPictogram
from labverse.models.material import Material
from labverse.models.safety import SafetyVerdict

INCOMPATIBLE_PAIRS: dict[frozenset[HazardPictogram], str] = {
    frozenset({HazardPictogram.GHS02_FLAMMABLE, HazardPictogram.GHS03_OXIDIZER}): (
        "fire/explosion risk"
    ),
    frozenset({HazardPictogram.GHS05_CORROSIVE, HazardPictogram.GHS02_FLAMMABLE}): (
        "toxic gas generation risk"
    ),
    frozenset({HazardPictogram.GHS03_OXIDIZER, HazardPictogram.GHS06_TOXIC}): (
        "toxic oxidation products"
    ),
}

SUBOPTIMAL_PAIRS: dict[frozenset[HazardPictogram], str] = {
    frozenset({HazardPictogram.GHS02_FLAMMABLE, HazardPictogram.GHS07_IRRITANT}): (
        "heightened ventilation and storage separation recommended"
    ),
    frozenset({HazardPictogram.GHS05_CORROSIVE, HazardPictogram.GHS07_IRRITANT}): (
        "segregated handling recommended to reduce exposure risk"
    ),
}

_PICTOGRAM_KEYWORDS: dict[HazardPictogram, set[str]] = {
    HazardPictogram.GHS02_FLAMMABLE: {"flammable", "flammables"},
    HazardPictogram.GHS03_OXIDIZER: {"oxidizer", "oxidizers"},
    HazardPictogram.GHS05_CORROSIVE: {"corrosive", "corrosives", "acid", "acids", "base", "bases"},
    HazardPictogram.GHS06_TOXIC: {"toxic", "toxics"},
    HazardPictogram.GHS07_IRRITANT: {"irritant", "irritants"},
    HazardPictogram.GHS08_HEALTH_HAZARD: {"health_hazard", "health hazards"},
}


def assess_chemical_compatibility(
    left: Material,
    right: Material,
) -> tuple[SafetyVerdict, str] | None:
    """Assess whether two materials are safely compatible."""
    explicit_reason = _explicit_incompatibility_reason(left, right)
    if explicit_reason is not None:
        return SafetyVerdict.BLOCK, explicit_reason

    warning_message: str | None = None
    for left_pictogram in left.ghs.pictograms:
        for right_pictogram in right.ghs.pictograms:
            pair = frozenset({left_pictogram, right_pictogram})
            if pair in INCOMPATIBLE_PAIRS:
                return SafetyVerdict.BLOCK, INCOMPATIBLE_PAIRS[pair]
            if warning_message is None and pair in SUBOPTIMAL_PAIRS:
                warning_message = SUBOPTIMAL_PAIRS[pair]

    if warning_message is not None:
        return SafetyVerdict.WARN, warning_message
    return None


def material_compatibility_tokens(material: Material) -> set[str]:
    """Expand a material into compatibility keywords used by the rule engine."""
    tokens = {material.id.casefold(), material.name.casefold()}
    if material.formula is not None:
        tokens.add(material.formula.casefold())
    for pictogram in material.ghs.pictograms:
        tokens.update(_PICTOGRAM_KEYWORDS.get(pictogram, set()))
    return tokens


def _explicit_incompatibility_reason(left: Material, right: Material) -> str | None:
    left_tokens = {token.casefold() for token in left.incompatible_with}
    right_tokens = {token.casefold() for token in right.incompatible_with}

    right_refs = material_compatibility_tokens(right)
    left_refs = material_compatibility_tokens(left)

    if left_tokens & right_refs:
        return f"{left.name} is explicitly marked incompatible with {right.name}"
    if right_tokens & left_refs:
        return f"{right.name} is explicitly marked incompatible with {left.name}"
    return None


__all__ = [
    "INCOMPATIBLE_PAIRS",
    "SUBOPTIMAL_PAIRS",
    "assess_chemical_compatibility",
    "material_compatibility_tokens",
]
