"""Tests for audit-materialization and metric helpers."""

from __future__ import annotations

from labverse.agent.audit import compute_step_metrics, materialize_audit_steps


class _UncopyableAuditPayload:
    def __repr__(self) -> str:
        return "<uncopyable-audit-payload>"


def test_materialize_audit_steps_coerces_unsupported_values_without_raising() -> None:
    steps = [
        {
            "status": "succeeded",
            "resolved_action_id": "preview_log_observation",
            "payload": _UncopyableAuditPayload(),
        }
    ]

    materialized = materialize_audit_steps(
        steps,
        actor_id="tech_1",
        actor_name="AuditTester",
    )

    assert len(materialized) == 1
    assert materialized[0]["resolved_action_id"] == "preview_log_observation"
    assert materialized[0]["artifact_integrity_errors"] >= 1
    assert materialized[0]["payload"]["__type__"] == "_UncopyableAuditPayload"


def test_compute_step_metrics_excludes_post_completion_noops_from_grounding_counts() -> None:
    metrics = compute_step_metrics(
        [
            {
                "invalid_action_outputs": 2,
                "action_output_retry_count": 1,
                "invalid_tool_calls": 2,
                "tool_retry_count": 1,
                "action_id_fallback_count": 1,
                "bare_verb_fallback_count": 0,
                "ambiguous_action_choice_errors": 1,
                "provider_errors": 0,
                "artifact_integrity_errors": 0,
                "post_completion_noop": True,
                "prompt_tokens": 150,
                "completion_tokens": 12,
            },
            {
                "invalid_action_outputs": 1,
                "action_output_retry_count": 1,
                "invalid_tool_calls": 1,
                "tool_retry_count": 1,
                "action_id_fallback_count": 0,
                "bare_verb_fallback_count": 0,
                "ambiguous_action_choice_errors": 0,
                "provider_errors": 0,
                "artifact_integrity_errors": 2,
                "post_completion_noop": False,
                "prompt_tokens": 200,
                "completion_tokens": 15,
            },
        ]
    )

    assert metrics["invalid_action_outputs"] == 1
    assert metrics["action_output_retry_count"] == 1
    assert metrics["invalid_tool_calls"] == 1
    assert metrics["tool_retry_count"] == 1
    assert metrics["action_id_fallback_count"] == 0
    assert metrics["artifact_integrity_errors"] == 2
    assert metrics["post_completion_turns"] == 1
    assert metrics["max_prompt_tokens_single_turn"] == 200
