"""Pipeline-health reports for frozen paper bundles."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from labverse.benchmarks.conditions import CONDITION_GLOBAL_ONE_SHOT_PLANNER
from labverse.paper.runner import PaperBundleMetadata, PaperRunRecord, load_paper_run_records
from labverse.paper.validate import validate_paper_bundle

HEALTH_PROMPT_TOKEN_WARNING = 200_000
HEALTH_PROMPT_TOKEN_BLOCKING = 800_000
HEALTH_SINGLE_TURN_PROMPT_TOKEN_BLOCKING = 35_000
HEALTH_CONDITION_MEAN_PROMPT_TOKEN_WARNING = 400_000
HEALTH_COMPLETION_RATIO_FLOOR = 0.2
HEALTH_MEAN_INVALID_TOOL_CALLS_SCALE_LIMIT = 1.0
HEALTH_MEAN_TOOL_RETRY_SCALE_LIMIT = 1.0

# Per-condition token-budget thresholds. The interactive conditions produce
# small per-turn prompts, so the default bands from v1 still apply. The
# one-shot planner is structurally different: it serialises the full world
# state and action surface into a single prompt, so both its single-turn
# maximum and its per-run mean are legitimately much larger than interactive
# budgets. Flagging one-shot runs under the interactive thresholds would
# produce false "scale-blocked" warnings for the condition that the current paper
# is specifically trying to evaluate.
CONDITION_SINGLE_TURN_PROMPT_TOKEN_BLOCKING: dict[str, int] = {
    CONDITION_GLOBAL_ONE_SHOT_PLANNER: 200_000,
}
CONDITION_MEAN_PROMPT_TOKEN_WARNING: dict[str, int] = {
    CONDITION_GLOBAL_ONE_SHOT_PLANNER: 120_000,
}


def _single_turn_prompt_limit(condition: str) -> int:
    return CONDITION_SINGLE_TURN_PROMPT_TOKEN_BLOCKING.get(
        condition, HEALTH_SINGLE_TURN_PROMPT_TOKEN_BLOCKING
    )


def _mean_prompt_warning(condition: str) -> int:
    return CONDITION_MEAN_PROMPT_TOKEN_WARNING.get(
        condition, HEALTH_CONDITION_MEAN_PROMPT_TOKEN_WARNING
    )


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _looks_like_grounding_contract_failure(record: PaperRunRecord) -> bool:
    text_parts = [record.error or ""]
    runner_error = record.details.get("runner_error")
    if isinstance(runner_error, str):
        text_parts.append(runner_error)
    combined = " ".join(part.casefold() for part in text_parts if part)
    if not combined:
        return False
    grounding_markers = (
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
    )
    return any(marker in combined for marker in grounding_markers)


def _effective_error_taxonomy(record: PaperRunRecord) -> str:
    if record.error_taxonomy == "runner_failure" and _looks_like_grounding_contract_failure(record):
        return "grounding_failure"
    return record.error_taxonomy


class PipelineHealthReport(BaseModel):
    """Health summary for one experiment bundle."""

    bundle_dir: str
    bundle_valid: bool
    total_expected_runs: int
    completed_runs: int
    running_runs: list[str] = Field(default_factory=list)
    total_runs: int
    trace_coverage_rate: float
    mean_invalid_action_outputs: float
    mean_action_output_retry_count: float
    mean_provider_tool_call_fallback_count: float
    mean_invalid_tool_calls: float
    mean_tool_retry_count: float
    mean_invalid_model_outputs: float
    mean_model_output_retry_count: float
    mean_model_output_format_errors: float
    mean_grounding_error_count: float
    mean_prompt_tokens: float
    mean_artifact_integrity_errors: float
    mean_post_completion_turns: float
    max_prompt_tokens_single_turn: int
    max_completion_tokens_single_turn: int
    provider_error_rows: int
    unexpected_error_rows: int
    ready_for_scale: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    low_completion_rows: list[str] = Field(default_factory=list)
    high_token_rows: list[str] = Field(default_factory=list)
    blocking_high_token_rows: list[str] = Field(default_factory=list)
    high_single_turn_prompt_rows: list[str] = Field(default_factory=list)
    missing_trace_rows: list[str] = Field(default_factory=list)
    artifact_integrity_rows: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    by_condition: dict[str, dict[str, float | int]] = Field(default_factory=dict)


def build_pipeline_health_report(bundle_dir: str | Path) -> PipelineHealthReport:
    bundle_path = Path(bundle_dir)
    validation = validate_paper_bundle(bundle_path)
    metadata = PaperBundleMetadata.model_validate(
        json.loads((bundle_path / "bundle_metadata.json").read_text())
    )
    records = load_paper_run_records(bundle_path / "results.jsonl")

    expected_runs = 0
    for condition in metadata.conditions:
        model_values = ["scripted_oracle"] if condition == "scripted_oracle" else metadata.models
        expected_runs += len(model_values) * len(metadata.scenarios) * metadata.replications

    running_runs: list[str] = []
    runs_root = bundle_path / "runs"
    if runs_root.exists():
        for run_meta_path in runs_root.rglob("run_meta.json"):
            payload = json.loads(run_meta_path.read_text())
            if payload.get("status") == "running" and (
                _pid_is_alive(payload.get("worker_pid"))
                or _pid_is_alive(payload.get("launcher_pid"))
                or ("worker_pid" not in payload and "launcher_pid" not in payload)
            ):
                running_runs.append(str(run_meta_path.parent.relative_to(bundle_path)))

    if not records:
        return PipelineHealthReport(
            bundle_dir=str(bundle_path),
            bundle_valid=validation.valid,
            total_expected_runs=expected_runs,
            completed_runs=0,
            running_runs=sorted(running_runs),
            total_runs=0,
            trace_coverage_rate=0.0,
            mean_invalid_action_outputs=0.0,
            mean_action_output_retry_count=0.0,
            mean_provider_tool_call_fallback_count=0.0,
            mean_invalid_tool_calls=0.0,
            mean_tool_retry_count=0.0,
            mean_invalid_model_outputs=0.0,
            mean_model_output_retry_count=0.0,
            mean_model_output_format_errors=0.0,
            mean_grounding_error_count=0.0,
            mean_prompt_tokens=0.0,
            mean_artifact_integrity_errors=0.0,
            mean_post_completion_turns=0.0,
            max_prompt_tokens_single_turn=0,
            max_completion_tokens_single_turn=0,
            provider_error_rows=0,
            unexpected_error_rows=0,
            ready_for_scale=False,
            blocking_reasons=[
                *list(validation.issues),
                "No results rows were found in the bundle.",
            ],
            issues=list(validation.issues),
        )

    with_traces = sum(1 for record in records if not record.steps_taken or record.trace_ids)
    low_completion_rows: list[str] = []
    high_token_rows: list[str] = []
    blocking_high_token_rows: list[str] = []
    high_single_turn_prompt_rows: list[str] = []
    missing_trace_rows: list[str] = []
    artifact_integrity_rows: list[str] = []
    warnings: list[str] = []

    by_condition: dict[str, list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        by_condition[record.condition].append(record)
        row_id = f"{record.condition}/{record.model}/{record.scenario}/rep_{record.replication}"
        if record.steps_taken > 0 and not record.trace_ids:
            missing_trace_rows.append(row_id)
        if record.prompt_tokens > HEALTH_PROMPT_TOKEN_WARNING:
            high_token_rows.append(row_id)
        if record.prompt_tokens > HEALTH_PROMPT_TOKEN_BLOCKING:
            blocking_high_token_rows.append(row_id)
        if record.max_prompt_tokens_single_turn > _single_turn_prompt_limit(record.condition):
            high_single_turn_prompt_rows.append(row_id)
        if record.artifact_integrity_errors > 0:
            artifact_integrity_rows.append(row_id)
        if record.completion_ratio < HEALTH_COMPLETION_RATIO_FLOOR and _effective_error_taxonomy(
            record
        ) not in {"provider_failure", "artifact_integrity_failure"}:
            low_completion_rows.append(row_id)

    if high_token_rows:
        warnings.append(
            f"{len(high_token_rows)} runs exceeded the prompt-token warning band "
            f"({HEALTH_PROMPT_TOKEN_WARNING})."
        )
    if low_completion_rows:
        warnings.append(
            f"{len(low_completion_rows)} runs fell below the completion ratio floor "
            f"({HEALTH_COMPLETION_RATIO_FLOOR:.1f})."
        )
    if missing_trace_rows:
        warnings.append(f"{len(missing_trace_rows)} runs are missing trace IDs.")
    if high_single_turn_prompt_rows:
        warnings.append(
            f"{len(high_single_turn_prompt_rows)} runs exceeded their condition-specific "
            "single-turn prompt-token scale limit."
        )
    if artifact_integrity_rows:
        warnings.append(f"{len(artifact_integrity_rows)} runs recorded artifact-integrity issues.")
    by_condition_summary: dict[str, dict[str, float | int]] = {}
    high_condition_mean_prompt_tokens: list[str] = []
    for condition, items in sorted(by_condition.items()):
        n = len(items)
        summary = {
            "n": n,
            "mean_invalid_action_outputs": (sum(item.invalid_action_outputs for item in items) / n),
            "mean_action_output_retry_count": (
                sum(item.action_output_retry_count for item in items) / n
            ),
            "mean_provider_tool_call_fallback_count": (
                sum(item.provider_tool_call_fallback_count for item in items) / n
            ),
            "mean_invalid_tool_calls": sum(item.invalid_tool_calls for item in items) / n,
            "mean_tool_retry_count": sum(item.tool_retry_count for item in items) / n,
            "mean_invalid_model_outputs": (sum(item.invalid_model_outputs for item in items) / n),
            "mean_model_output_retry_count": (
                sum(item.model_output_retry_count for item in items) / n
            ),
            "mean_model_output_format_errors": (
                sum(item.model_output_format_errors for item in items) / n
            ),
            "mean_grounding_error_count": sum(item.grounding_error_count for item in items) / n,
            "mean_prompt_tokens": sum(item.prompt_tokens for item in items) / n,
            "mean_completion_ratio": sum(item.completion_ratio for item in items) / n,
            "provider_error_rows": sum(1 for item in items if item.provider_errors > 0),
            "mean_artifact_integrity_errors": (
                sum(item.artifact_integrity_errors for item in items) / n
            ),
            "mean_post_completion_turns": sum(item.post_completion_turns for item in items) / n,
        }
        by_condition_summary[condition] = summary
        if float(summary["mean_prompt_tokens"]) > _mean_prompt_warning(condition):
            high_condition_mean_prompt_tokens.append(condition)

    if high_condition_mean_prompt_tokens:
        warnings.append(
            "Mean prompt tokens exceeded the condition-specific warning band for: "
            + ", ".join(high_condition_mean_prompt_tokens)
        )

    blocking_reasons: list[str] = []
    if validation.issues:
        blocking_reasons.extend(list(validation.issues))
    if len(records) != expected_runs:
        blocking_reasons.append(
            f"Bundle is incomplete: completed {len(records)} of {expected_runs} expected runs."
        )
    if running_runs:
        blocking_reasons.append(f"{len(running_runs)} run(s) are still marked running.")
    if missing_trace_rows:
        blocking_reasons.append(f"{len(missing_trace_rows)} run(s) are missing trace IDs.")
    if artifact_integrity_rows:
        blocking_reasons.append(
            f"{len(artifact_integrity_rows)} run(s) recorded artifact-integrity errors."
        )
    if any(record.provider_errors > 0 for record in records):
        blocking_reasons.append("At least one run recorded provider errors.")
    if any(_effective_error_taxonomy(record) == "runner_failure" for record in records):
        blocking_reasons.append("At least one run recorded a runtime error.")
        blocking_reasons.append("At least one run ended in runner_failure.")
    if any(record.semantic_fallback_count > 0 for record in records):
        blocking_reasons.append(
            "At least one run used a semantic fallback instead of the exact paper tool contract."
        )
    if blocking_high_token_rows:
        blocking_reasons.append(
            f"{len(blocking_high_token_rows)} run(s) exceeded the per-run prompt-token scale "
            f"limit ({HEALTH_PROMPT_TOKEN_BLOCKING})."
        )
    if high_single_turn_prompt_rows:
        blocking_reasons.append(
            f"{len(high_single_turn_prompt_rows)} run(s) exceeded their condition-specific "
            "single-turn prompt-token scale limit."
        )
    mean_invalid_action_outputs = sum(record.invalid_action_outputs for record in records) / len(
        records
    )
    mean_action_output_retry_count = sum(
        record.action_output_retry_count for record in records
    ) / len(records)
    mean_provider_tool_call_fallback_count = sum(
        record.provider_tool_call_fallback_count for record in records
    ) / len(records)
    mean_invalid_tool_calls = sum(record.invalid_tool_calls for record in records) / len(records)
    mean_tool_retry_count = sum(record.tool_retry_count for record in records) / len(records)
    mean_invalid_model_outputs = sum(record.invalid_model_outputs for record in records) / len(
        records
    )
    mean_model_output_retry_count = sum(
        record.model_output_retry_count for record in records
    ) / len(records)
    mean_model_output_format_errors = sum(
        record.model_output_format_errors for record in records
    ) / len(records)
    mean_grounding_error_count = sum(record.grounding_error_count for record in records) / len(
        records
    )
    if mean_invalid_action_outputs > HEALTH_MEAN_INVALID_TOOL_CALLS_SCALE_LIMIT:
        blocking_reasons.append(
            "Mean invalid action outputs exceeded the scale limit "
            f"({HEALTH_MEAN_INVALID_TOOL_CALLS_SCALE_LIMIT:.1f})."
        )
    if mean_action_output_retry_count > HEALTH_MEAN_TOOL_RETRY_SCALE_LIMIT:
        blocking_reasons.append(
            "Mean action-output retries exceeded the scale limit "
            f"({HEALTH_MEAN_TOOL_RETRY_SCALE_LIMIT:.1f})."
        )

    return PipelineHealthReport(
        bundle_dir=str(bundle_path),
        bundle_valid=validation.valid,
        total_expected_runs=expected_runs,
        completed_runs=len(records),
        running_runs=sorted(running_runs),
        total_runs=len(records),
        trace_coverage_rate=with_traces / len(records),
        mean_invalid_action_outputs=mean_invalid_action_outputs,
        mean_action_output_retry_count=mean_action_output_retry_count,
        mean_provider_tool_call_fallback_count=mean_provider_tool_call_fallback_count,
        mean_invalid_tool_calls=mean_invalid_tool_calls,
        mean_tool_retry_count=mean_tool_retry_count,
        mean_invalid_model_outputs=mean_invalid_model_outputs,
        mean_model_output_retry_count=mean_model_output_retry_count,
        mean_model_output_format_errors=mean_model_output_format_errors,
        mean_grounding_error_count=mean_grounding_error_count,
        mean_prompt_tokens=sum(record.prompt_tokens for record in records) / len(records),
        mean_artifact_integrity_errors=(
            sum(record.artifact_integrity_errors for record in records) / len(records)
        ),
        mean_post_completion_turns=(
            sum(record.post_completion_turns for record in records) / len(records)
        ),
        max_prompt_tokens_single_turn=max(
            record.max_prompt_tokens_single_turn for record in records
        ),
        max_completion_tokens_single_turn=max(
            record.max_completion_tokens_single_turn for record in records
        ),
        provider_error_rows=sum(1 for record in records if record.provider_errors > 0),
        unexpected_error_rows=sum(
            1
            for record in records
            if _effective_error_taxonomy(record)
            in {"provider_failure", "artifact_integrity_failure", "runner_failure"}
        ),
        ready_for_scale=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        low_completion_rows=low_completion_rows,
        high_token_rows=high_token_rows,
        blocking_high_token_rows=blocking_high_token_rows,
        high_single_turn_prompt_rows=high_single_turn_prompt_rows,
        missing_trace_rows=missing_trace_rows,
        artifact_integrity_rows=artifact_integrity_rows,
        issues=list(validation.issues),
        warnings=warnings,
        by_condition=by_condition_summary,
    )


def write_pipeline_health_report(bundle_dir: str | Path) -> Path:
    """Write machine-readable and markdown health reports into a bundle."""

    bundle_path = Path(bundle_dir)
    health_dir = bundle_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    report = build_pipeline_health_report(bundle_path)
    (health_dir / "pipeline_health.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Pipeline Health",
        "",
        f"- Bundle valid: `{report.bundle_valid}`",
        f"- Ready for scale: `{report.ready_for_scale}`",
        f"- Completed runs: `{report.completed_runs}/{report.total_expected_runs}`",
        f"- Running runs: `{len(report.running_runs)}`",
        f"- Total runs: `{report.total_runs}`",
        f"- Trace coverage: `{report.trace_coverage_rate:.1%}`",
        f"- Mean invalid action outputs: `{report.mean_invalid_action_outputs:.2f}`",
        f"- Mean action-output retries: `{report.mean_action_output_retry_count:.2f}`",
        (
            "- Mean provider tool-call fallbacks: "
            f"`{report.mean_provider_tool_call_fallback_count:.2f}`"
        ),
        f"- Mean invalid model outputs: `{report.mean_invalid_model_outputs:.2f}`",
        f"- Mean model-output retries: `{report.mean_model_output_retry_count:.2f}`",
        f"- Mean model-output format errors: `{report.mean_model_output_format_errors:.2f}`",
        f"- Mean grounding errors: `{report.mean_grounding_error_count:.2f}`",
        f"- Mean prompt tokens: `{report.mean_prompt_tokens:.1f}`",
        f"- Mean artifact-integrity errors: `{report.mean_artifact_integrity_errors:.2f}`",
        f"- Mean post-completion turns: `{report.mean_post_completion_turns:.2f}`",
        f"- Max single-turn prompt tokens: `{report.max_prompt_tokens_single_turn}`",
        f"- Max single-turn completion tokens: `{report.max_completion_tokens_single_turn}`",
        "",
    ]
    if report.blocking_reasons:
        lines.extend(["## Blocking Reasons", ""])
        lines.extend(f"- {reason}" for reason in report.blocking_reasons)
        lines.append("")
    if report.issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in report.issues)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    if report.by_condition:
        lines.extend(["## By Condition", ""])
        for condition, stats in report.by_condition.items():
            lines.append(
                f"- `{condition}`: "
                f"invalid_action_outputs={stats['mean_invalid_action_outputs']:.2f}, "
                f"retries={stats['mean_action_output_retry_count']:.2f}, "
                "provider_tool_call_fallbacks="
                f"{stats['mean_provider_tool_call_fallback_count']:.2f}, "
                f"invalid_model_outputs={stats['mean_invalid_model_outputs']:.2f}, "
                f"grounding_errors={stats['mean_grounding_error_count']:.2f}, "
                f"prompt_tokens={stats['mean_prompt_tokens']:.1f}, "
                f"completion={stats['mean_completion_ratio']:.2f}"
            )
    (health_dir / "pipeline_health.md").write_text("\n".join(lines) + "\n")
    return health_dir


__all__ = [
    "PipelineHealthReport",
    "build_pipeline_health_report",
    "write_pipeline_health_report",
]
