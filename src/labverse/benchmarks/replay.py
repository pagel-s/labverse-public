"""Helpers for building paper-facing replay frames from recorded steps."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from labverse.models.action import ActionResult
from labverse.world.state import WorldState


def _serialize_world_snapshot(world: WorldState | dict[str, Any]) -> dict[str, Any]:
    if isinstance(world, WorldState):
        return world.model_dump(mode="json")
    return dict(world)


def _serialize_time(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _change_summary_for_step(
    step: dict[str, Any],
    *,
    result: ActionResult | None,
    is_final_step: bool,
) -> list[str]:
    summary: list[str] = []
    if is_final_step and result is not None:
        for entity in result.changed_entities:
            if entity.summary:
                summary.append(entity.summary)
        if not summary:
            outcome = result.outcome_description or result.blocked_reason or result.failure_reason
            if outcome:
                summary.append(outcome)
    if not summary:
        action_result = step.get("action_result")
        if isinstance(action_result, dict):
            description = action_result.get("description")
            if isinstance(description, str) and description:
                summary.append(description)
        error = step.get("error")
        if isinstance(error, str) and error:
            summary.append(error)
    return summary


def build_replay_frames_for_steps(
    *,
    steps: list[dict[str, Any]],
    starting_step_index: int,
    world_before: WorldState,
    world_after: WorldState,
    tick_before: int,
    tick_after: int,
    current_time_before: datetime | str | None,
    current_time_after: datetime | str | None,
    result: ActionResult | None = None,
) -> list[dict[str, Any]]:
    """Build 1:1 replay frames for recorded audit steps."""

    serialized_world_before = _serialize_world_snapshot(world_before)
    serialized_world_after = _serialize_world_snapshot(world_after)
    frames: list[dict[str, Any]] = []
    final_index = len(steps) - 1

    for index, step in enumerate(steps):
        is_final_step = index == final_index
        frames.append(
            {
                "step_index": starting_step_index + index,
                "actor_id": step.get("actor_id"),
                "resolved_candidate_ref": step.get("resolved_candidate_ref"),
                "resolved_action_id": step.get("resolved_action_id"),
                "resolved_verb": step.get("resolved_verb"),
                "resolved_target_id": step.get("resolved_target_id"),
                "resolved_action_parameters": dict(
                    step.get("resolved_action_parameters", {}) or {}
                ),
                "action_result": step.get("action_result"),
                "world_before": serialized_world_before,
                "world_after": serialized_world_after,
                "tick_before": tick_before,
                "tick_after": tick_after,
                "current_time_before": _serialize_time(current_time_before),
                "current_time_after": _serialize_time(current_time_after),
                "change_summary": _change_summary_for_step(
                    step,
                    result=result if is_final_step else None,
                    is_final_step=is_final_step,
                ),
                "trace_ids": list(step.get("trace_ids", []) or []),
                "event_ids": list(step.get("event_ids", []) or []),
            }
        )
    return frames


__all__ = ["build_replay_frames_for_steps"]
