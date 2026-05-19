"""Lightweight serialization helpers for isolated paper worker subprocesses."""

from __future__ import annotations

from typing import Any

from labverse.agent.providers.base import TokenUsage
from labverse.benchmarks.experiment import RunResult, ScenarioRunArtifacts
from labverse.benchmarks.scorer import BenchmarkScore


def serialize_run_result(result: RunResult) -> dict[str, Any]:
    return {
        "condition": result.condition,
        "model": result.model,
        "provider": result.provider,
        "scenario": result.scenario,
        "replication": result.replication,
        "score": result.score.model_dump(mode="json"),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "wall_time_seconds": result.wall_time_seconds,
        "error": result.error,
        "timestamp": result.timestamp,
    }


def deserialize_run_result(payload: dict[str, Any]) -> RunResult:
    return RunResult(
        condition=str(payload["condition"]),
        model=str(payload["model"]),
        provider=str(payload["provider"]),
        scenario=str(payload["scenario"]),
        replication=int(payload["replication"]),
        score=BenchmarkScore.model_validate(payload["score"]),
        prompt_tokens=int(payload.get("prompt_tokens", 0)),
        completion_tokens=int(payload.get("completion_tokens", 0)),
        wall_time_seconds=float(payload.get("wall_time_seconds", 0.0)),
        error=payload.get("error"),
        timestamp=str(payload.get("timestamp", "")),
    )


def serialize_artifacts(artifacts: ScenarioRunArtifacts) -> dict[str, Any]:
    return {
        "score": artifacts.score.model_dump(mode="json"),
        "usage": {
            "prompt_tokens": artifacts.usage.prompt_tokens,
            "completion_tokens": artifacts.usage.completion_tokens,
        },
        "event_ids": list(artifacts.event_ids),
        "trace_ids": list(artifacts.trace_ids),
        "trace_ids_by_agent": {
            agent_id: list(trace_ids)
            for agent_id, trace_ids in artifacts.trace_ids_by_agent.items()
        },
        "step_audits": [dict(step) for step in artifacts.step_audits],
        "replay_frames": [dict(frame) for frame in artifacts.replay_frames],
        "run_metrics": dict(artifacts.run_metrics),
    }


def deserialize_artifacts(payload: dict[str, Any]) -> ScenarioRunArtifacts:
    return ScenarioRunArtifacts(
        score=BenchmarkScore.model_validate(payload["score"]),
        usage=TokenUsage(**payload.get("usage", {})),
        event_ids=list(payload.get("event_ids", [])),
        trace_ids=list(payload.get("trace_ids", [])),
        trace_ids_by_agent={
            str(agent_id): list(trace_ids)
            for agent_id, trace_ids in payload.get("trace_ids_by_agent", {}).items()
        },
        step_audits=[dict(step) for step in payload.get("step_audits", [])],
        replay_frames=[dict(frame) for frame in payload.get("replay_frames", [])],
        run_metrics={str(key): int(value) for key, value in payload.get("run_metrics", {}).items()},
    )


__all__ = [
    "deserialize_artifacts",
    "deserialize_run_result",
    "serialize_artifacts",
    "serialize_run_result",
]
