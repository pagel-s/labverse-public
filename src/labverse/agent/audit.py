"""Shared audit helpers for experiment-time prompt and tool-call tracing."""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from labverse.agent.providers.base import LLMResponse, ToolCall
from labverse.models.action import ActionResult


def start_step_audit(
    *,
    actor_id: str,
    actor_name: str,
    actor_kind: str,
    system_prompt: str | None,
    user_message: dict[str, Any] | None,
    tool_schema: dict[str, Any] | None,
    prompt_surface_kind: str,
) -> dict[str, Any]:
    """Create a mutable audit record for one decision turn."""

    return {
        "actor_id": actor_id,
        "actor_name": actor_name,
        "actor_kind": actor_kind,
        "prompt_surface_kind": prompt_surface_kind,
        "system_prompt": system_prompt,
        "user_message": deepcopy(user_message),
        "tool_schema": deepcopy(tool_schema),
        "attempts": [],
        "resolved_candidate_ref": None,
        "resolved_action_id": None,
        "resolved_verb": None,
        "resolved_target_id": None,
        "resolved_action_parameters": {},
        "fallback_kind": None,
        "normalization_kinds": [],
        "used_legacy_tool_format": False,
        "status": "pending",
        "invalid_action_outputs": 0,
        "action_output_retry_count": 0,
        "provider_tool_call_fallback_count": 0,
        "invalid_tool_calls": 0,
        "tool_retry_count": 0,
        "invalid_model_outputs": 0,
        "model_output_retry_count": 0,
        "model_output_format_errors": 0,
        "grounding_error_count": 0,
        "action_id_fallback_count": 0,
        "bare_verb_fallback_count": 0,
        "transport_normalization_count": 0,
        "tool_name_alias_count": 0,
        "flattened_parameter_count": 0,
        "json_repair_count": 0,
        "stringified_parameters_repair_count": 0,
        "semantic_fallback_count": 0,
        "ambiguous_action_choice_errors": 0,
        "provider_errors": 0,
        "artifact_integrity_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "max_prompt_tokens_single_attempt": 0,
        "max_completion_tokens_single_attempt": 0,
        "trace_ids": [],
        "event_ids": [],
        "post_completion_noop": False,
    }


def record_attempt(
    step_audit: dict[str, Any],
    *,
    attempt_index: int,
    messages: list[dict[str, Any]],
    tool_schema: dict[str, Any] | None,
    tool_choice: str | None,
    response: LLMResponse | None,
    provider_error: str | None = None,
    provider_metadata: dict[str, Any] | None = None,
    invalid_reason: str | None = None,
) -> None:
    """Append one provider call attempt to a step audit."""

    attempt_record = {
        "attempt_index": attempt_index,
        "messages": deepcopy(messages),
        "tool_schema": deepcopy(tool_schema),
        "tool_choice": tool_choice,
        "provider_error": provider_error,
        "invalid_reason": invalid_reason,
    }
    if response is not None:
        attempt_record.update(
            {
                "response_tool_call": _tool_call_payload(response.tool_call),
                "response_text": response.text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                "metadata": deepcopy(response.metadata),
            }
        )
        step_audit["prompt_tokens"] += response.usage.prompt_tokens
        step_audit["completion_tokens"] += response.usage.completion_tokens
        step_audit["max_prompt_tokens_single_attempt"] = max(
            int(step_audit["max_prompt_tokens_single_attempt"]),
            response.usage.prompt_tokens,
        )
        step_audit["max_completion_tokens_single_attempt"] = max(
            int(step_audit["max_completion_tokens_single_attempt"]),
            response.usage.completion_tokens,
        )
    else:
        attempt_record["usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
        attempt_record["metadata"] = deepcopy(provider_metadata or {})

    if invalid_reason:
        step_audit["invalid_action_outputs"] += 1
        step_audit["invalid_tool_calls"] += 1
        step_audit["invalid_model_outputs"] += 1
        if _looks_like_grounding_error(invalid_reason):
            step_audit["grounding_error_count"] += 1
        else:
            step_audit["model_output_format_errors"] += 1
        if "ambiguous" in invalid_reason.casefold():
            step_audit["ambiguous_action_choice_errors"] += 1
    if provider_error:
        step_audit["provider_errors"] += 1

    step_audit["attempts"].append(attempt_record)


def finalize_step_audit(
    step_audit: dict[str, Any],
    *,
    resolved_action_id: str,
    resolved_parameters: dict[str, Any],
    resolved_candidate_ref: str | None = None,
    resolved_verb: str | None = None,
    resolved_target_id: str | None = None,
    fallback_kind: str | None = None,
    normalization_kinds: list[str] | None = None,
    used_legacy_tool_format: bool = False,
    status: str = "succeeded",
) -> None:
    """Finalize a successful action-selection audit record."""

    step_audit["resolved_candidate_ref"] = resolved_candidate_ref or resolved_action_id
    step_audit["resolved_action_id"] = resolved_action_id
    step_audit["resolved_verb"] = resolved_verb
    step_audit["resolved_target_id"] = resolved_target_id
    step_audit["resolved_action_parameters"] = deepcopy(resolved_parameters)
    step_audit["fallback_kind"] = fallback_kind
    effective_normalization_kinds = normalization_kinds or (
        [] if fallback_kind is None else [fallback_kind]
    )
    step_audit["normalization_kinds"] = list(effective_normalization_kinds)
    step_audit["used_legacy_tool_format"] = used_legacy_tool_format
    step_audit["status"] = status
    semantic_kinds = {"bare_verb", "non_namespaced_action_id", "legacy_tool_call"}
    transport_kinds = {
        "tool_name_alias",
        "flattened_parameters",
        "omitted_const_fields",
        "content_tool_call",
        "legacy_function_call",
        "leading_dashes_removed",
        "code_fence_removed",
        "provider_tool_call_fallback",
        "json_repair",
        "stringified_parameters",
        "stringified_number",
        "integral_float_to_integer",
        "dropped_superfluous_target_id",
    }

    if fallback_kind == "bare_verb":
        step_audit["bare_verb_fallback_count"] += 1
    if used_legacy_tool_format:
        step_audit["action_id_fallback_count"] += 1
        step_audit["semantic_fallback_count"] += 1
    for kind in step_audit["normalization_kinds"]:
        if kind in semantic_kinds:
            step_audit["action_id_fallback_count"] += 1
            step_audit["semantic_fallback_count"] += 1
        if kind in transport_kinds:
            step_audit["transport_normalization_count"] += 1
        if kind == "provider_tool_call_fallback":
            step_audit["provider_tool_call_fallback_count"] += 1
        if kind == "tool_name_alias":
            step_audit["tool_name_alias_count"] += 1
        if kind == "flattened_parameters":
            step_audit["flattened_parameter_count"] += 1
        if kind == "json_repair":
            step_audit["json_repair_count"] += 1
        if kind == "stringified_parameters":
            step_audit["stringified_parameters_repair_count"] += 1
    step_audit["action_output_retry_count"] = max(0, len(step_audit["attempts"]) - 1)
    step_audit["tool_retry_count"] = step_audit["action_output_retry_count"]
    step_audit["model_output_retry_count"] = step_audit["action_output_retry_count"]


def fail_step_audit(
    step_audit: dict[str, Any],
    *,
    error: str,
    status: str = "failed",
) -> None:
    """Mark an audit step as failed after exhausting retries or execution."""

    step_audit["status"] = status
    step_audit["error"] = error
    step_audit["action_output_retry_count"] = max(0, len(step_audit["attempts"]) - 1)
    step_audit["tool_retry_count"] = step_audit["action_output_retry_count"]
    step_audit["model_output_retry_count"] = step_audit["action_output_retry_count"]
    if not step_audit["attempts"] and _looks_like_grounding_error(error):
        step_audit["invalid_action_outputs"] += 1
        step_audit["invalid_model_outputs"] += 1
        step_audit["grounding_error_count"] += 1


def attach_action_result(
    step_audit: dict[str, Any],
    *,
    result: ActionResult | None = None,
    success: bool | None = None,
    description: str = "",
    trace_ids: list[str] | None = None,
    event_ids: list[str] | None = None,
) -> None:
    """Attach simulator-side outcome information to a completed audit step."""

    if result is not None:
        step_audit["action_result"] = {
            "success": result.success,
            "status": result.status.value if result.status is not None else None,
            "description": (
                result.outcome_description or result.blocked_reason or result.failure_reason or ""
            ),
            "time_elapsed_seconds": result.time_elapsed_seconds,
            "warnings": list(result.warnings),
            "safety_violations": list(result.safety_violations),
            "blocked_reason": result.blocked_reason,
            "failure_reason": result.failure_reason,
        }
    else:
        step_audit["action_result"] = {
            "success": success,
            "description": description,
        }
    step_audit["trace_ids"] = list(trace_ids or [])
    step_audit["event_ids"] = list(event_ids or [])


def compute_step_metrics(step_audits: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize retry and grounding metrics over one run cell."""

    counted_steps = [
        step for step in step_audits if not bool(step.get("post_completion_noop", False))
    ]
    invalid_action_outputs = sum(
        int(step.get("invalid_action_outputs", 0)) for step in counted_steps
    )
    action_output_retry_count = sum(
        int(step.get("action_output_retry_count", 0)) for step in counted_steps
    )
    return {
        "invalid_action_outputs": invalid_action_outputs,
        "action_output_retry_count": action_output_retry_count,
        "provider_tool_call_fallback_count": sum(
            int(step.get("provider_tool_call_fallback_count", 0)) for step in counted_steps
        ),
        "invalid_tool_calls": invalid_action_outputs,
        "tool_retry_count": action_output_retry_count,
        "invalid_model_outputs": sum(
            int(step.get("invalid_model_outputs", 0)) for step in counted_steps
        ),
        "model_output_retry_count": sum(
            int(step.get("model_output_retry_count", 0)) for step in counted_steps
        ),
        "model_output_format_errors": sum(
            int(step.get("model_output_format_errors", 0)) for step in counted_steps
        ),
        "grounding_error_count": sum(
            int(step.get("grounding_error_count", 0)) for step in counted_steps
        ),
        "action_id_fallback_count": sum(
            int(step.get("action_id_fallback_count", 0)) for step in counted_steps
        ),
        "bare_verb_fallback_count": sum(
            int(step.get("bare_verb_fallback_count", 0)) for step in counted_steps
        ),
        "transport_normalization_count": sum(
            int(step.get("transport_normalization_count", 0)) for step in counted_steps
        ),
        "tool_name_alias_count": sum(
            int(step.get("tool_name_alias_count", 0)) for step in counted_steps
        ),
        "flattened_parameter_count": sum(
            int(step.get("flattened_parameter_count", 0)) for step in counted_steps
        ),
        "json_repair_count": sum(int(step.get("json_repair_count", 0)) for step in counted_steps),
        "stringified_parameters_repair_count": sum(
            int(step.get("stringified_parameters_repair_count", 0)) for step in counted_steps
        ),
        "semantic_fallback_count": sum(
            int(step.get("semantic_fallback_count", 0)) for step in counted_steps
        ),
        "ambiguous_action_choice_errors": sum(
            int(step.get("ambiguous_action_choice_errors", 0)) for step in counted_steps
        ),
        "provider_errors": sum(int(step.get("provider_errors", 0)) for step in counted_steps),
        "artifact_integrity_errors": sum(
            int(step.get("artifact_integrity_errors", 0)) for step in step_audits
        ),
        "post_completion_turns": sum(
            1 for step in step_audits if bool(step.get("post_completion_noop", False))
        ),
        "max_prompt_tokens_single_turn": max(
            (int(step.get("prompt_tokens", 0)) for step in step_audits),
            default=0,
        ),
        "max_completion_tokens_single_turn": max(
            (int(step.get("completion_tokens", 0)) for step in step_audits),
            default=0,
        ),
    }


def _tool_call_payload(tool_call: ToolCall | None) -> dict[str, Any] | None:
    if tool_call is None:
        return None
    return {
        "name": tool_call.name,
        "arguments": deepcopy(tool_call.arguments),
    }


def _looks_like_grounding_error(text: str) -> bool:
    lowered = text.casefold()
    markers = (
        "unknown action_id",
        "available action_ids",
        "unknown candidate_ref",
        "available candidate_refs",
        "missing or invalid 'candidate_ref' in action output",
        "action output verb mismatch",
        "action output actor_id mismatch",
        "action output target_id mismatch",
        "failed to produce a valid action after",
        "did not call the required tool",
        "no valid tool call",
        "invalid tool call",
        "bare verb",
        "planner grounding failure",
        "does not match the current action surface",
        "not a valid room id",
        "not a valid door id",
        "not a valid equipment id",
        "not a valid sample id",
        "not a valid material id",
        "invalid agent_id",
    )
    return any(marker in lowered for marker in markers)


def build_artifact_integrity_step(
    *,
    error: str,
    actor_id: str | None = None,
    actor_name: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    """Build a synthetic audit step for artifact-integrity failures."""

    step = start_step_audit(
        actor_id=actor_id or "unknown_actor",
        actor_name=actor_name or "unknown_actor",
        actor_kind="artifact_integrity",
        system_prompt=None,
        user_message={"role": "system", "content": "Audit artifact integrity failure."},
        tool_schema=None,
        prompt_surface_kind="artifact_integrity",
    )
    fail_step_audit(step, error=error, status="artifact_integrity_failure")
    step["artifact_integrity_errors"] = 1
    if traceback_text is not None:
        step["artifact_integrity_traceback"] = traceback_text
    return step


def materialize_audit_steps(
    raw_steps: Any,
    *,
    actor_id: str | None = None,
    actor_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return a JSON-safe view of audit steps without raising on malformed payloads."""

    if not isinstance(raw_steps, list):
        return [
            build_artifact_integrity_step(
                error=(f"Audit steps payload was not a list: {type(raw_steps).__name__}"),
                actor_id=actor_id,
                actor_name=actor_name,
            )
        ]

    materialized: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps, start=1):
        issues: list[str] = []
        try:
            if not isinstance(step, Mapping):
                materialized_step = build_artifact_integrity_step(
                    error=(f"Audit step {index} was not a mapping: {type(step).__name__}"),
                    actor_id=actor_id,
                    actor_name=actor_name,
                )
            else:
                materialized_step = _materialize_mapping(
                    step,
                    path=f"step[{index}]",
                    issues=issues,
                    seen=set(),
                )
                materialized_step.setdefault("artifact_integrity_errors", 0)
                materialized_step.setdefault("post_completion_noop", False)
                if issues:
                    materialized_step["artifact_integrity_errors"] = int(
                        materialized_step.get("artifact_integrity_errors", 0)
                    ) + len(issues)
                    materialized_step["artifact_integrity_messages"] = issues
        except Exception:
            materialized_step = build_artifact_integrity_step(
                error=f"Audit step {index} could not be materialized.",
                actor_id=actor_id,
                actor_name=actor_name,
                traceback_text=traceback.format_exc(),
            )
        materialized.append(materialized_step)
    return materialized


def read_actor_audit_steps(actor: object) -> list[dict[str, Any]]:
    """Read an actor's audit steps without letting malformed payloads crash callers."""

    actor_id = getattr(actor, "agent_id", None)
    actor_name = getattr(actor, "name", type(actor).__name__)
    try:
        raw_steps = getattr(actor, "audit_steps", None)
    except Exception:
        return [
            build_artifact_integrity_step(
                error="Failed while reading actor audit steps.",
                actor_id=str(actor_id) if actor_id is not None else None,
                actor_name=str(actor_name),
                traceback_text=traceback.format_exc(),
            )
        ]
    return materialize_audit_steps(
        raw_steps,
        actor_id=str(actor_id) if actor_id is not None else None,
        actor_name=str(actor_name),
    )


def mark_post_completion_noop(step_audit: dict[str, Any]) -> None:
    """Flag a step as a post-completion no-op that should not count against grounding health."""

    step_audit["post_completion_noop"] = True


def _materialize_mapping(
    payload: Mapping[str, Any],
    *,
    path: str,
    issues: list[str],
    seen: set[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if key_text != key:
            issues.append(f"{path}.{key_text}: coerced non-string key to string")
        result[key_text] = _materialize_value(
            value,
            path=f"{path}.{key_text}",
            issues=issues,
            seen=seen,
        )
    return result


def _materialize_value(
    value: Any,
    *,
    path: str,
    issues: list[str],
    seen: set[int],
) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    object_id = id(value)
    if object_id in seen:
        issues.append(f"{path}: replaced recursive reference with placeholder")
        return "<recursive-reference>"

    if isinstance(value, Mapping):
        seen.add(object_id)
        try:
            return _materialize_mapping(value, path=path, issues=issues, seen=seen)
        finally:
            seen.discard(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(object_id)
        try:
            return [
                _materialize_value(
                    item,
                    path=f"{path}[{index}]",
                    issues=issues,
                    seen=seen,
                )
                for index, item in enumerate(value)
            ]
        finally:
            seen.discard(object_id)

    if hasattr(value, "model_dump") and callable(value.model_dump):
        seen.add(object_id)
        try:
            dumped = value.model_dump(mode="json")
            issues.append(f"{path}: coerced {type(value).__name__} via model_dump()")
            return _materialize_value(dumped, path=path, issues=issues, seen=seen)
        except Exception as exc:
            issues.append(f"{path}: model_dump() failed for {type(value).__name__}: {exc}")
        finally:
            seen.discard(object_id)

    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive
        issues.append(f"{path}: repr() failed for {type(value).__name__}: {exc}")
        return {"__type__": type(value).__name__, "__repr__": "<unrepresentable>"}

    issues.append(f"{path}: coerced {type(value).__name__} to repr()")
    return {"__type__": type(value).__name__, "__repr__": rendered}


__all__ = [
    "attach_action_result",
    "build_artifact_integrity_step",
    "compute_step_metrics",
    "fail_step_audit",
    "finalize_step_audit",
    "mark_post_completion_noop",
    "materialize_audit_steps",
    "record_attempt",
    "read_actor_audit_steps",
    "start_step_audit",
]
