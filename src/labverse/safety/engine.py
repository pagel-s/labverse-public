"""Safety engine orchestration over the built-in constraint pipeline."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping

from labverse.models.action import Action
from labverse.models.safety import SafetyCheckResult, SafetyVerdict
from labverse.safety.constraints import BUILTIN_CONSTRAINTS, ConstraintEvaluator
from labverse.world.state import WorldState


class SafetyEngine:
    """Evaluate a sequence of safety constraints against an action.

    A per-scenario ``enforcement_overrides`` map can promote policy-tunable
    constraints from their default verdict (usually ``WARN``) to ``BLOCK`` for
    high-stakes scenarios. The map is keyed by ``constraint_id`` and its values
    are the desired :class:`SafetyVerdict` for that constraint's violations.
    Constraints that do not declare an ``enforcement_level`` keyword argument on
    their constructor are silently skipped — overrides apply only where the
    constraint has opted in.
    """

    def __init__(
        self,
        constraints: Iterable[ConstraintEvaluator] | None = None,
        *,
        enforcement_overrides: Mapping[str, SafetyVerdict] | None = None,
    ) -> None:
        if constraints is not None:
            self._constraints = list(constraints)
        else:
            self._constraints = self.default_constraints(
                enforcement_overrides=enforcement_overrides,
            )

    @classmethod
    def default_constraints(
        cls,
        *,
        enforcement_overrides: Mapping[str, SafetyVerdict] | None = None,
    ) -> list[ConstraintEvaluator]:
        """Build the default built-in constraint pipeline.

        ``enforcement_overrides`` is consulted per constraint type: if the type
        advertises an ``enforcement_level`` constructor kwarg and its
        ``constraint_id`` is present in the map, the override is passed in.
        """

        overrides = dict(enforcement_overrides or {})
        constraints: list[ConstraintEvaluator] = []
        for constraint_type in BUILTIN_CONSTRAINTS:
            override = overrides.get(constraint_type.constraint_id)
            if override is not None and _accepts_enforcement_level(constraint_type):
                # Feature-detected above: the constraint type advertises an
                # enforcement_level kwarg. mypy can't see through the base-class
                # ABC signature, so suppress the call-arg check here.
                constraints.append(
                    constraint_type(enforcement_level=override)  # type: ignore[call-arg]
                )
            else:
                constraints.append(constraint_type())
        return constraints

    @property
    def constraints(self) -> list[ConstraintEvaluator]:
        """Return the configured constraints in evaluation order."""
        return list(self._constraints)

    def evaluate(self, action: Action, world: WorldState) -> list[SafetyCheckResult]:
        """Evaluate all applicable constraints and return their results."""
        results: list[SafetyCheckResult] = []
        for constraint in self._constraints:
            if not constraint.applies_to(action):
                continue
            result = constraint.check(action, world)
            if result is not None:
                results.append(result)
        return results


def _accepts_enforcement_level(constraint_type: type[ConstraintEvaluator]) -> bool:
    try:
        signature = inspect.signature(constraint_type.__init__)
    except (TypeError, ValueError):
        return False
    return "enforcement_level" in signature.parameters


__all__ = ["SafetyEngine"]
