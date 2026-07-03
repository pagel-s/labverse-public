"""Operational helpers for inspecting and comparing paper bundles."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labverse.paper.runner import PaperBundleMetadata, load_paper_run_records
from labverse.paper.validate import validate_paper_bundle


class BundleStatusReport(BaseModel):
    """Current progress snapshot for a bundle directory."""

    bundle_dir: str
    total_expected_runs: int
    completed_runs: int
    running_runs: list[str] = Field(default_factory=list)
    latest_run: str | None = None
    validation_ok: bool = False
    validation_issues: list[str] = Field(default_factory=list)
    diagnostics_clear: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)


class BundleCompareReport(BaseModel):
    """High-level comparison between two bundles."""

    left_bundle: str
    right_bundle: str
    left_valid: bool
    right_valid: bool
    condition_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)
    integrity_differences: list[str] = Field(default_factory=list)


class BundleRunSummary(BaseModel):
    """Summary of one run directory inside a paper bundle."""

    run_path: str
    condition: str | None = None
    model: str | None = None
    scenario: str | None = None
    replication: int | None = None
    status: str | None = None
    step_count: int = 0
    replay_available: bool = False
    replay_frame_count: int = 0
    error: str | None = None


class BundleChatViewReport(BaseModel):
    """Rendered transcript view for one run or step selection."""

    run_path: str | None = None
    recorded_steps: int = 0
    transcript: str


class BundleStepSummary(BaseModel):
    """Summary row for one recorded step inside a run."""

    run_path: str
    step_index: int
    actor_id: str | None = None
    actor_name: str | None = None
    prompt_surface_kind: str | None = None
    status: str | None = None
    resolved_candidate_ref: str | None = None
    resolved_action_id: str | None = None
    resolved_verb: str | None = None
    resolved_target_id: str | None = None
    action_status: str | None = None
    error: str | None = None
    replay_available: bool = False


class BundleReplayFrame(BaseModel):
    """Historical world-state replay frame for one recorded step."""

    step_index: int
    actor_id: str | None = None
    resolved_candidate_ref: str | None = None
    resolved_action_id: str | None = None
    resolved_verb: str | None = None
    resolved_target_id: str | None = None
    resolved_action_parameters: dict[str, Any] = Field(default_factory=dict)
    action_result: dict[str, Any] | None = None
    world_before: dict[str, Any]
    world_after: dict[str, Any]
    tick_before: int
    tick_after: int
    current_time_before: str | None = None
    current_time_after: str | None = None
    change_summary: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)


class BundleStepDetail(BaseModel):
    """Structured step detail for the Paper Studio frontend."""

    run_path: str
    recorded_steps: int
    step_index: int
    step: dict[str, Any]
    replay_available: bool = False
    compatibility_notice: str | None = None
    replay_frame: BundleReplayFrame | None = None


def _load_run_steps(run_dir: Path) -> list[dict[str, Any]]:
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        return []
    rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    for index, row in enumerate(rows, start=1):
        row.setdefault("step_index", index)
    return rows


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as file:
        return sum(1 for line in file if line.strip())


def _load_replay_frame_indices(run_dir: Path) -> set[int]:
    replay_path = run_dir / "replay.jsonl"
    if not replay_path.exists():
        return set()
    indices: set[int] = set()
    with replay_path.open() as file:
        for index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            indices.add(int(payload.get("step_index", index)))
    return indices


def _load_replay_frame(run_dir: Path, step: int) -> BundleReplayFrame | None:
    replay_path = run_dir / "replay.jsonl"
    if not replay_path.exists():
        return None
    with replay_path.open() as file:
        for index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            step_index = int(payload.get("step_index", index))
            if step_index != step:
                continue
            payload.setdefault("step_index", step_index)
            return BundleReplayFrame.model_validate(payload)
    return None


def _select_target_run(
    bundle_dir: str | Path,
    *,
    run: str | None = None,
) -> tuple[Path, str] | tuple[None, None]:
    """Resolve the target run directory for inspection commands."""

    bundle_path = Path(bundle_dir)
    if run is not None:
        return bundle_path / run, run

    status = bundle_status(bundle_dir)
    target_run = status.running_runs[0] if status.running_runs else status.latest_run
    if target_run is None:
        return None, None
    return bundle_path / target_run, target_run


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


def bundle_status(bundle_dir: str | Path) -> BundleStatusReport:
    bundle_path = Path(bundle_dir)
    metadata = PaperBundleMetadata.model_validate(
        json.loads((bundle_path / "bundle_metadata.json").read_text())
    )
    records = load_paper_run_records(bundle_path / "results.jsonl")
    validation = validate_paper_bundle(bundle_path)
    from labverse.paper.health import build_pipeline_health_report

    expected_runs = 0
    for condition in metadata.conditions:
        model_values = ["scripted_oracle"] if condition == "scripted_oracle" else metadata.models
        expected_runs += len(model_values) * len(metadata.scenarios) * metadata.replications

    running_runs: list[str] = []
    latest_run: tuple[float, str] | None = None
    runs_root = bundle_path / "runs"
    if runs_root.exists():
        for run_meta_path in runs_root.rglob("run_meta.json"):
            payload = json.loads(run_meta_path.read_text())
            relative = str(run_meta_path.parent.relative_to(bundle_path))
            if payload.get("status") == "running" and (
                _pid_is_alive(payload.get("worker_pid"))
                or _pid_is_alive(payload.get("launcher_pid"))
                or ("worker_pid" not in payload and "launcher_pid" not in payload)
            ):
                running_runs.append(relative)
            mtime = run_meta_path.stat().st_mtime
            if latest_run is None or mtime > latest_run[0]:
                latest_run = (mtime, relative)

    health = build_pipeline_health_report(bundle_path)

    return BundleStatusReport(
        bundle_dir=str(bundle_path),
        total_expected_runs=expected_runs,
        completed_runs=len(records),
        running_runs=sorted(running_runs),
        latest_run=latest_run[1] if latest_run is not None else None,
        validation_ok=validation.valid,
        validation_issues=list(validation.issues),
        diagnostics_clear=health.diagnostics_clear,
        blocking_reasons=list(health.blocking_reasons),
    )


def inspect_bundle_steps(
    bundle_dir: str | Path,
    *,
    tail: int = 10,
    run: str | None = None,
) -> list[dict[str, Any]]:
    """Return the latest step records for the newest or currently running run."""

    run_dir, _ = _select_target_run(bundle_dir, run=run)
    if run_dir is None:
        return []
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        return []
    lines = [line for line in steps_path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-tail:]]


def list_bundle_runs(bundle_dir: str | Path) -> list[BundleRunSummary]:
    """List all run directories in a bundle with light metadata."""

    bundle_path = Path(bundle_dir)
    runs_root = bundle_path / "runs"
    if not runs_root.exists():
        return []

    summaries: list[BundleRunSummary] = []
    for run_meta_path in sorted(runs_root.rglob("run_meta.json")):
        run_dir = run_meta_path.parent
        relative = str(run_dir.relative_to(bundle_path))
        payload = json.loads(run_meta_path.read_text())
        parts = run_dir.relative_to(runs_root).parts
        condition = parts[0] if len(parts) > 0 else None
        model = parts[1] if len(parts) > 1 else None
        scenario = parts[2] if len(parts) > 2 else None
        replication = None
        if len(parts) > 3 and parts[3].startswith("rep_"):
            try:
                replication = int(parts[3].split("_", 1)[1])
            except ValueError:
                replication = None
        step_count = len(_load_run_steps(run_dir))
        error = payload.get("error")
        details = payload.get("details")
        if error is None and isinstance(details, dict):
            error = details.get("runner_error") or details.get("provider_error")
        replay_frame_count = int(
            payload.get("replay_frame_count", _count_jsonl_rows(run_dir / "replay.jsonl"))
        )
        summaries.append(
            BundleRunSummary(
                run_path=relative,
                condition=condition,
                model=model,
                scenario=scenario,
                replication=replication,
                status=payload.get("status"),
                step_count=step_count,
                replay_available=bool(payload.get("replay_available", False) or replay_frame_count),
                replay_frame_count=replay_frame_count,
                error=error,
            )
        )
    return summaries


def list_bundle_steps(
    bundle_dir: str | Path,
    *,
    run: str | None = None,
) -> list[BundleStepSummary]:
    """List recorded steps for a selected run."""

    run_dir, relative_run = _select_target_run(bundle_dir, run=run)
    if run_dir is None or relative_run is None:
        return []

    rows = _load_run_steps(run_dir)
    replay_frame_indices = _load_replay_frame_indices(run_dir)
    return [
        BundleStepSummary(
            run_path=relative_run,
            step_index=int(row.get("step_index", index)),
            actor_id=row.get("actor_id"),
            actor_name=row.get("actor_name"),
            prompt_surface_kind=row.get("prompt_surface_kind"),
            status=row.get("status"),
            resolved_candidate_ref=row.get("resolved_candidate_ref"),
            resolved_action_id=row.get("resolved_action_id"),
            resolved_verb=row.get("resolved_verb"),
            resolved_target_id=row.get("resolved_target_id"),
            action_status=(row.get("action_result") or {}).get("status")
            if isinstance(row.get("action_result"), dict)
            else None,
            error=row.get("error"),
            replay_available=int(row.get("step_index", index)) in replay_frame_indices,
        )
        for index, row in enumerate(rows, start=1)
    ]


def get_bundle_step_detail(
    bundle_dir: str | Path,
    *,
    run: str,
    step: int,
) -> BundleStepDetail:
    """Return the structured detail for one run step."""

    run_dir, relative_run = _select_target_run(bundle_dir, run=run)
    if run_dir is None or relative_run is None:
        raise ValueError("No runs available.")

    rows = _load_run_steps(run_dir)
    if step < 1 or step > len(rows):
        raise ValueError(f"Step {step} is out of range for run {relative_run} (1-{len(rows)}).")

    replay_frame = _load_replay_frame(run_dir, step)
    return BundleStepDetail(
        run_path=relative_run,
        recorded_steps=len(rows),
        step_index=step,
        step=rows[step - 1],
        replay_available=replay_frame is not None,
        compatibility_notice=(
            None if replay_frame is not None else "Historical replay unavailable for this bundle."
        ),
        replay_frame=replay_frame,
    )


def build_bundle_chat_view(
    bundle_dir: str | Path,
    *,
    run: str | None = None,
    step: int | None = None,
    tail: int = 1,
    show_tool_schema: bool = False,
) -> BundleChatViewReport:
    """Render step audits in a chat-style transcript for human inspection."""

    run_dir, relative_run = _select_target_run(bundle_dir, run=run)
    if run_dir is None or relative_run is None:
        return BundleChatViewReport(transcript="No runs available.")

    if not (run_dir / "steps.jsonl").exists():
        return BundleChatViewReport(
            run_path=relative_run,
            transcript=f"No steps found for run {relative_run}.",
        )

    rows = _load_run_steps(run_dir)
    if not rows:
        return BundleChatViewReport(
            run_path=relative_run,
            transcript=f"No recorded steps for run {relative_run}.",
        )

    if step is not None:
        if step < 1 or step > len(rows):
            raise ValueError(f"Step {step} is out of range for run {relative_run} (1-{len(rows)}).")
        selected = [(step, rows[step - 1])]
    else:
        start = max(1, len(rows) - tail + 1)
        selected = list(enumerate(rows[start - 1 :], start=start))

    parts = [f"Run: {relative_run}", f"Recorded steps: {len(rows)}"]
    if step is None:
        parts.append(f"Showing last {len(selected)} step(s)")

    for step_index, row in selected:
        parts.append("")
        parts.append(f"######## STEP {step_index} ########")
        parts.append("")
        parts.append("[SYSTEM]")
        parts.append(str(row.get("system_prompt", "")))
        parts.append("")
        parts.append("[USER]")
        user_message = row.get("user_message", {})
        parts.append(str(user_message.get("content", "")))
        if show_tool_schema:
            parts.append("")
            parts.append("[ACTION SCHEMA]")
            parts.append(json.dumps(row.get("tool_schema", {}), indent=2, sort_keys=True))

        for attempt in row.get("attempts", []):
            parts.append("")
            parts.append(f"[ASSISTANT attempt {attempt.get('attempt_index')}]")
            tool_call = attempt.get("response_tool_call")
            response_text = attempt.get("response_text")
            if tool_call is not None:
                parts.append(json.dumps(tool_call, indent=2, sort_keys=True))
            elif response_text:
                parts.append(str(response_text))
            else:
                parts.append("(no text/action output)")

            finish_reason = attempt.get("metadata", {}).get("finish_reason")
            invalid_reason = attempt.get("invalid_reason")
            provider_error = attempt.get("provider_error")
            parts.append(
                "[META] "
                f"finish_reason={finish_reason!r} "
                f"invalid_reason={invalid_reason!r} "
                f"provider_error={provider_error!r}"
            )

        action_result = row.get("action_result")
        if action_result:
            parts.append("")
            parts.append("[RESULT]")
            parts.append(json.dumps(action_result, indent=2, sort_keys=True))

        resolved_candidate_ref = row.get("resolved_candidate_ref")
        resolved_action_id = row.get("resolved_action_id")
        resolved_verb = row.get("resolved_verb")
        resolved_target_id = row.get("resolved_target_id")
        if resolved_candidate_ref or resolved_action_id:
            parts.append("")
            if resolved_candidate_ref:
                parts.append(f"[SELECTED CANDIDATE] {resolved_candidate_ref}")
            if resolved_verb:
                parts.append(f"[VERB] {resolved_verb}")
            if resolved_target_id:
                parts.append(f"[TARGET] {resolved_target_id}")
            if resolved_action_id:
                parts.append(f"[RESOLVED ACTION] {resolved_action_id}")

        if row.get("error"):
            parts.append("")
            parts.append(f"[STEP ERROR] {row['error']}")

    return BundleChatViewReport(
        run_path=relative_run,
        recorded_steps=len(rows),
        transcript="\n".join(parts),
    )


def render_bundle_chat_view(
    bundle_dir: str | Path,
    *,
    run: str | None = None,
    step: int | None = None,
    tail: int = 1,
    show_tool_schema: bool = False,
) -> str:
    """Render step audits as a plain string for CLI output."""

    return build_bundle_chat_view(
        bundle_dir,
        run=run,
        step=step,
        tail=tail,
        show_tool_schema=show_tool_schema,
    ).transcript


def compare_bundles(left_bundle: str | Path, right_bundle: str | Path) -> BundleCompareReport:
    """Compare two bundles at the condition-summary level."""

    left_path = Path(left_bundle)
    right_path = Path(right_bundle)
    left_validation = validate_paper_bundle(left_path)
    right_validation = validate_paper_bundle(right_path)
    left_summary = json.loads((left_path / "summary.json").read_text())
    right_summary = json.loads((right_path / "summary.json").read_text())

    condition_deltas: dict[str, dict[str, float]] = {}
    left_conditions = left_summary.get("by_condition", {})
    right_conditions = right_summary.get("by_condition", {})
    for condition in sorted(set(left_conditions) & set(right_conditions)):
        left_stats = left_conditions[condition]
        right_stats = right_conditions[condition]
        condition_deltas[condition] = {
            "success_rate_delta": float(left_stats.get("success_rate", 0.0))
            - float(right_stats.get("success_rate", 0.0)),
            "completion_ratio_delta": float(left_stats.get("mean_completion_ratio", 0.0))
            - float(right_stats.get("mean_completion_ratio", 0.0)),
            "prompt_tokens_delta": float(left_stats.get("mean_prompt_tokens", 0.0))
            - float(right_stats.get("mean_prompt_tokens", 0.0)),
            "retry_delta": float(left_stats.get("mean_tool_retry_count", 0.0))
            - float(right_stats.get("mean_tool_retry_count", 0.0)),
        }

    integrity_differences: list[str] = []
    if left_validation.valid != right_validation.valid:
        integrity_differences.append(
            f"bundle validity differs: left={left_validation.valid} right={right_validation.valid}"
        )
    if left_validation.issues != right_validation.issues:
        integrity_differences.append("validation issues differ between bundles")

    return BundleCompareReport(
        left_bundle=str(left_path),
        right_bundle=str(right_path),
        left_valid=left_validation.valid,
        right_valid=right_validation.valid,
        condition_deltas=condition_deltas,
        integrity_differences=integrity_differences,
    )


def watch_bundle(bundle_dir: str | Path, *, interval_seconds: float = 5.0) -> None:
    """Print bundle status repeatedly until interrupted."""

    while True:
        status = bundle_status(bundle_dir)
        print(
            json.dumps(
                status.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
        )
        print()
        time.sleep(interval_seconds)


__all__ = [
    "BundleChatViewReport",
    "BundleCompareReport",
    "BundleReplayFrame",
    "BundleRunSummary",
    "BundleStepDetail",
    "BundleStepSummary",
    "BundleStatusReport",
    "build_bundle_chat_view",
    "bundle_status",
    "compare_bundles",
    "get_bundle_step_detail",
    "inspect_bundle_steps",
    "list_bundle_runs",
    "list_bundle_steps",
    "render_bundle_chat_view",
    "watch_bundle",
]
