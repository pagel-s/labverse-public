"""Base abstractions for benchmark scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scorer import BenchmarkDetailValue, BenchmarkScore, BenchmarkScorer
from labverse.config import load_world
from labverse.core.events import EventLog
from labverse.models.action import VERB_CATEGORIES
from labverse.models.safety import SafetyVerdict
from labverse.safety.engine import SafetyEngine
from labverse.world.state import WorldState

CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs" / "labs"
DEFAULT_ONE_SHOT_ALLOWED_VERBS = frozenset(VERB_CATEGORIES)


@dataclass(frozen=True)
class ScenarioTaskSpec:
    """Canonical paper-facing task information for one benchmark scenario."""

    objective: str
    known_requirements: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    safety_constraints: tuple[str, ...] = ()
    intentional_unknowns: tuple[str, ...] = ()
    referenced_ids: tuple[str, ...] = ()

    def render(self) -> str:
        """Render the task contract as stable, model-agnostic prompt text."""

        lines = [f"Objective: {self.objective}"]
        _append_section(lines, "Known requirements", self.known_requirements)
        _append_section(lines, "Required outputs", self.required_outputs)
        _append_section(lines, "Safety constraints", self.safety_constraints)
        _append_section(lines, "Intentional unknowns", self.intentional_unknowns)
        return "\n".join(lines)


def _append_section(lines: list[str], title: str, items: tuple[str, ...]) -> None:
    if not items:
        return
    lines.append(f"{title}:")
    lines.extend(f"- {item}" for item in items)


class BenchmarkScenario(ABC):
    """A deterministic, evaluatable benchmark scenario.

    Subclasses that need stricter-than-default safety enforcement can set
    :attr:`constraint_policy` — a mapping of ``constraint_id`` -> desired
    :class:`SafetyVerdict`. At construction time the scenario's safety engine
    promotes the matching builtins to that verdict (typically to ``BLOCK``
    for high-stakes scenarios where a ``WARN`` is not strong enough).
    Constraints that do not advertise an ``enforcement_level`` kwarg are
    silently skipped, so adding a new tunable constraint only requires
    extending the constraint class — not this base.
    """

    name: ClassVar[str]
    lab_config_name: ClassVar[str]
    max_steps: ClassVar[int]
    optimal_steps: ClassVar[int]
    constraint_policy: ClassVar[dict[str, SafetyVerdict]] = {}
    task: ClassVar[ScenarioTaskSpec | None] = None

    def build_base_world(self) -> WorldState:
        return load_world(CONFIGS_DIR / self.lab_config_name)

    def configured_world(self) -> WorldState:
        return self.setup(self.build_base_world())

    @abstractmethod
    def setup(self, world: WorldState) -> WorldState:
        """Configure the initial world state for this scenario."""

    @abstractmethod
    def build_scripted_agents(self) -> list[ScriptedAgent]:
        """Return the known-good scripted agents for this scenario."""

    def task_spec(self) -> ScenarioTaskSpec:
        """Return the canonical task contract shown to paper-facing agents."""
        if self.task is None:
            raise NotImplementedError(f"{type(self).__name__} does not define task")
        return self.task

    def get_goal_description(self) -> str:
        """Render the canonical task contract for prompt builders."""
        return self.task_spec().render()

    @abstractmethod
    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        """Evaluate the run and return a scenario score."""

    def get_primary_agent_id(self) -> str:
        """Return the agent ID of the primary actor for this scenario."""
        agents = self.build_scripted_agents()
        if not agents:
            raise RuntimeError(f"Scenario {self.name} has no scripted agents")
        return agents[0].agent_id

    def get_max_steps(self) -> int:
        return self.max_steps

    def get_optimal_steps(self) -> int:
        return self.optimal_steps

    def one_shot_allowed_verbs(self) -> set[str]:
        """Return the canonical verb subset allowed for one-shot planning."""

        return set(DEFAULT_ONE_SHOT_ALLOWED_VERBS)

    def build_safety_engine(self) -> SafetyEngine:
        """Build the scenario's safety engine with its enforcement policy.

        The default implementation returns a pipeline built from
        :data:`labverse.safety.constraints.BUILTIN_CONSTRAINTS`, with any
        policy-tunable constraints promoted per :attr:`constraint_policy`.
        Scenarios that need bespoke constraint instances can override this
        method and return a fully-configured engine instead.
        """
        return SafetyEngine(enforcement_overrides=self.constraint_policy)

    def _score(
        self,
        event_log: EventLog,
        *,
        success: bool,
        completion_ratio: float,
        details: dict[str, BenchmarkDetailValue],
    ) -> BenchmarkScore:
        return BenchmarkScorer.score(
            self.name,
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            optimal_steps=self.get_optimal_steps(),
            details=details,
        )


__all__ = [
    "BenchmarkScenario",
    "CONFIGS_DIR",
    "DEFAULT_ONE_SHOT_ALLOWED_VERBS",
    "ScenarioTaskSpec",
]
