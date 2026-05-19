"""Condition-independent executed-action traces for paper bundles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

NORMALIZED_TRACE_SCHEMA_VERSION = 1
NORMALIZED_TRACE_FILENAME = "normalized_trace.jsonl"


def build_normalized_trace(step_audits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one normalized row per simulator-submitted action."""

    rows: list[dict[str, Any]] = []
    for step in step_audits:
        action_result = step.get("action_result")
        if not isinstance(action_result, Mapping):
            continue
        rows.append(
            {
                "schema_version": NORMALIZED_TRACE_SCHEMA_VERSION,
                "step_index": len(rows) + 1,
                "source_step_index": step.get("step_index"),
                "actor_id": step.get("actor_id"),
                "actor_kind": step.get("actor_kind"),
                "candidate_ref": step.get("resolved_candidate_ref"),
                "resolved_action_id": step.get("resolved_action_id"),
                "verb": step.get("resolved_verb"),
                "target_id": step.get("resolved_target_id"),
                "parameters": dict(step.get("resolved_action_parameters", {}) or {}),
                "success": action_result.get("success"),
                "status": action_result.get("status"),
                "description": action_result.get("description") or "",
                "blocked_reason": action_result.get("blocked_reason"),
                "failure_reason": action_result.get("failure_reason"),
                "safety_violations": list(action_result.get("safety_violations", []) or []),
                "warnings": list(action_result.get("warnings", []) or []),
                "trace_ids": list(step.get("trace_ids", []) or []),
                "event_ids": list(step.get("event_ids", []) or []),
                "error": step.get("error"),
            }
        )
    return rows


def write_normalized_trace_jsonl(
    path: str | Path,
    step_audits: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Write normalized trace rows and return the rows that were written."""

    rows = build_normalized_trace(step_audits)
    trace_path = Path(path)
    with trace_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


__all__ = [
    "NORMALIZED_TRACE_FILENAME",
    "NORMALIZED_TRACE_SCHEMA_VERSION",
    "build_normalized_trace",
    "write_normalized_trace_jsonl",
]
