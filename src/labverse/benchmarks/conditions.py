"""Canonical condition definitions for benchmark and paper runs."""

from __future__ import annotations

CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED = "local_structured_interactive_grounded"
CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED = "local_narrative_interactive_grounded"
CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED = "global_structured_interactive_grounded"
CONDITION_GLOBAL_ONE_SHOT_PLANNER = "global_one_shot_planner"
CONDITION_SCRIPTED_ORACLE = "scripted_oracle"

PRIMARY_INTERACTIVE_CONDITIONS = [
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
]
"""The three interactive-loop primary conditions.

Kept as a distinct constant because some call sites (grounding-failure
classification, per-turn prompt documentation, paper-probe) are genuinely
specific to the closed-enum per-turn surface and do not apply to the
one-shot planner, whose failure modes and prompting are structurally
different.
"""

PRIMARY_CONDITIONS = [
    *PRIMARY_INTERACTIVE_CONDITIONS,
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
]
"""All primary paper conditions — the three interactive + one-shot planner.

The current paper promotes ``global_one_shot_planner`` from appendix to a primary
condition because it is the cleanest experimental contrast for the paper's
central hypothesis (interactive grounded simulation vs one-shot text
planning). Use this list anywhere the semantics are "all four primaries"
(manifests, analysis comparisons, writeup tables). Use
:data:`PRIMARY_INTERACTIVE_CONDITIONS` only where interactive-specific
behavior is genuinely required.
"""

APPENDIX_PAPER_CONDITIONS = [
    CONDITION_SCRIPTED_ORACLE,
]

ALL_CONDITIONS = [
    *PRIMARY_CONDITIONS,
    *APPENDIX_PAPER_CONDITIONS,
]

LEGACY_CONDITION_ALIASES = {
    "embodied": CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    "text_feedback": CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    "text_only": CONDITION_GLOBAL_ONE_SHOT_PLANNER,
}


def normalize_condition_name(condition: str) -> str:
    """Map legacy paper condition labels onto the canonical names."""
    return LEGACY_CONDITION_ALIASES.get(condition, condition)


def normalize_condition_names(conditions: list[str] | None) -> list[str] | None:
    """Normalize a list of condition labels while preserving order."""
    if conditions is None:
        return None
    normalized: list[str] = []
    for condition in conditions:
        mapped = normalize_condition_name(condition)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


__all__ = [
    "ALL_CONDITIONS",
    "APPENDIX_PAPER_CONDITIONS",
    "CONDITION_GLOBAL_ONE_SHOT_PLANNER",
    "CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED",
    "CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED",
    "CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED",
    "CONDITION_SCRIPTED_ORACLE",
    "LEGACY_CONDITION_ALIASES",
    "PRIMARY_CONDITIONS",
    "PRIMARY_INTERACTIVE_CONDITIONS",
    "normalize_condition_name",
    "normalize_condition_names",
]
