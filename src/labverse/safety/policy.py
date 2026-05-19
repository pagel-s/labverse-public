"""Per-scenario constraint enforcement policy parsing.

Scenarios can promote policy-tunable constraints (e.g. material expiry,
sample contamination) from their default ``WARN`` verdict to ``BLOCK`` by
declaring a ``constraint_policy`` block. This module provides a small,
reusable parser that converts a raw mapping — typically loaded from YAML —
into a typed ``dict[constraint_id, SafetyVerdict]`` suitable for passing to
:class:`labverse.safety.engine.SafetyEngine`.

Expected YAML form::

    constraint_policy:
      material_expiry: block
      sample_contamination: block

Unknown verdict strings or non-string keys raise :class:`ConstraintPolicyError`
so that a typo in scenario YAML fails loudly rather than silently reverting to
the default enforcement level.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from labverse.models.safety import SafetyVerdict


class ConstraintPolicyError(ValueError):
    """Raised when a constraint policy block fails validation."""


_VERDICT_ALIASES: dict[str, SafetyVerdict] = {
    "allow": SafetyVerdict.ALLOW,
    "warn": SafetyVerdict.WARN,
    "block": SafetyVerdict.BLOCK,
}


def parse_constraint_policy(data: Mapping[str, Any] | None) -> dict[str, SafetyVerdict]:
    """Parse a ``constraint_policy`` mapping into typed verdict overrides.

    ``data`` is the raw mapping as it would appear in scenario / lab YAML.
    ``None`` and empty mappings produce an empty override dict, which leaves
    the default constraint pipeline unchanged.
    """
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConstraintPolicyError(
            "constraint_policy must be a mapping of constraint_id -> verdict"
        )

    overrides: dict[str, SafetyVerdict] = {}
    for raw_key, raw_value in data.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ConstraintPolicyError(
                f"constraint_policy keys must be non-empty strings, got {raw_key!r}"
            )
        if isinstance(raw_value, SafetyVerdict):
            overrides[raw_key] = raw_value
            continue
        if not isinstance(raw_value, str):
            raise ConstraintPolicyError(
                f"constraint_policy[{raw_key!r}] must be a verdict string, got {raw_value!r}"
            )
        try:
            overrides[raw_key] = _VERDICT_ALIASES[raw_value.strip().lower()]
        except KeyError as exc:
            allowed = ", ".join(sorted(_VERDICT_ALIASES))
            raise ConstraintPolicyError(
                f"constraint_policy[{raw_key!r}] has unknown verdict "
                f"{raw_value!r}; expected one of: {allowed}"
            ) from exc
    return overrides


__all__ = ["ConstraintPolicyError", "parse_constraint_policy"]
