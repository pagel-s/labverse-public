"""Benchmark score models and aggregation helpers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from labverse.core.events import EventLog, EventType

BenchmarkDetailValue = float | str | bool


class BenchmarkScore(BaseModel):
    """The score for a single benchmark scenario run."""

    scenario_name: str
    success: bool
    completion_ratio: float
    steps_taken: int
    time_elapsed_seconds: float
    safety_violations: int
    blocked_actions: int
    efficiency_score: float
    details: dict[str, BenchmarkDetailValue] = Field(default_factory=dict)


class AggregateScore(BaseModel):
    """Aggregate score across multiple benchmark scenarios."""

    total_scenarios: int
    passed: int
    mean_completion: float
    mean_efficiency: float
    total_safety_violations: int
    total_blocked_actions: int
    per_scenario: list[BenchmarkScore] = Field(default_factory=list)


class BenchmarkScorer:
    """Helpers for turning scenario outcomes into benchmark scores."""

    @staticmethod
    def score(
        scenario_name: str,
        event_log: EventLog,
        *,
        success: bool,
        completion_ratio: float,
        optimal_steps: int,
        details: Mapping[str, BenchmarkDetailValue] | None = None,
    ) -> BenchmarkScore:
        events = event_log.events
        steps_taken = len(event_log.query(event_type=EventType.ACTION_SUBMITTED))
        safety_violations = len(event_log.query(event_type=EventType.SAFETY_VIOLATION))
        blocked_actions = len(event_log.query(event_type=EventType.ACTION_BLOCKED))
        time_elapsed_seconds = 0.0
        if events:
            time_elapsed_seconds = (events[-1].timestamp - events[0].timestamp).total_seconds()

        clamped_completion = max(0.0, min(1.0, completion_ratio))
        if abs(clamped_completion - 1.0) < 1e-9:
            clamped_completion = 1.0
        if steps_taken == 0:
            efficiency_score = 1.0 if optimal_steps == 0 else 0.0
        else:
            efficiency_score = min(1.0, optimal_steps / steps_taken)

        return BenchmarkScore(
            scenario_name=scenario_name,
            success=success,
            completion_ratio=clamped_completion,
            steps_taken=steps_taken,
            time_elapsed_seconds=time_elapsed_seconds,
            safety_violations=safety_violations,
            blocked_actions=blocked_actions,
            efficiency_score=efficiency_score,
            details=dict(details or {}),
        )

    @staticmethod
    def failure_score(
        scenario_name: str,
        *,
        error: str,
    ) -> BenchmarkScore:
        return BenchmarkScore(
            scenario_name=scenario_name,
            success=False,
            completion_ratio=0.0,
            steps_taken=0,
            time_elapsed_seconds=0.0,
            safety_violations=0,
            blocked_actions=0,
            efficiency_score=0.0,
            details={"runner_error": error},
        )

    @staticmethod
    def aggregate(scores: list[BenchmarkScore]) -> AggregateScore:
        total_scenarios = len(scores)
        if not scores:
            return AggregateScore(
                total_scenarios=0,
                passed=0,
                mean_completion=0.0,
                mean_efficiency=0.0,
                total_safety_violations=0,
                total_blocked_actions=0,
                per_scenario=[],
            )

        return AggregateScore(
            total_scenarios=total_scenarios,
            passed=sum(1 for score in scores if score.success),
            mean_completion=sum(score.completion_ratio for score in scores) / total_scenarios,
            mean_efficiency=sum(score.efficiency_score for score in scores) / total_scenarios,
            total_safety_violations=sum(score.safety_violations for score in scores),
            total_blocked_actions=sum(score.blocked_actions for score in scores),
            per_scenario=scores,
        )


__all__ = [
    "AggregateScore",
    "BenchmarkDetailValue",
    "BenchmarkScore",
    "BenchmarkScorer",
]
