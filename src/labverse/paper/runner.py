"""Dedicated paper runner and artifact bundle generation."""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
import queue
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_SCRIPTED_ORACLE,
    PRIMARY_INTERACTIVE_CONDITIONS,
    normalize_condition_name,
    normalize_condition_names,
)
from labverse.benchmarks.experiment import (
    ProviderOptions,
    RunResult,
    ScenarioRunArtifacts,
    run_single_capture,
)
from labverse.benchmarks.initial_variants import (
    BASELINE_INITIAL_VARIANT_ID,
    initial_condition_variant_hash,
    load_initial_condition_variants,
)
from labverse.benchmarks.runner import BenchmarkRunner
from labverse.benchmarks.scorer import BenchmarkScorer
from labverse.paper.integrity import scientific_artifact_hashes, validate_scientific_artifacts
from labverse.paper.manifest import PaperManifest, load_paper_manifest, manifest_sha256
from labverse.paper.trace import NORMALIZED_TRACE_FILENAME, write_normalized_trace_jsonl
from labverse.paper_worker_payload import (
    deserialize_artifacts,
    deserialize_run_result,
    serialize_artifacts,
    serialize_run_result,
)

SCRIPTED_ORACLE_CONDITION = CONDITION_SCRIPTED_ORACLE
ALL_PAPER_CONDITIONS = [
    *PRIMARY_INTERACTIVE_CONDITIONS,
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    SCRIPTED_ORACLE_CONDITION,
]
REPLAY_SCHEMA_VERSION = 1
STDIO_TAIL_CHARS = 4000


class PaperBundleMetadata(BaseModel):
    """Metadata for an immutable paper run bundle."""

    bundle_id: str
    created_at: str
    manifest_id: str
    manifest_version: str
    manifest_hash: str
    manifest_path: str
    git_commit_sha: str
    git_dirty: bool
    profile_name: str
    temperature: float
    max_tokens: int
    replications: int
    random_seed: int
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID
    initial_variant_hash: str = ""
    conditions: list[str]
    models: list[str]
    scenarios: list[str]
    allow_dirty: bool = False
    preflight: dict[str, Any] = Field(default_factory=dict)
    scientific_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class PaperPreflightReport(BaseModel):
    """Preflight checks for a paper run before spending API budget."""

    checked_at: str
    bundle_dir: str
    manifest_id: str
    manifest_version: str
    profile_name: str
    git_commit_sha: str
    git_dirty: bool
    allow_dirty: bool
    resume: bool
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID
    initial_variant_hash: str = ""
    conditions: list[str]
    models: list[str]
    scenarios: list[str]
    provider_credentials: dict[str, bool] = Field(default_factory=dict)
    prompt_family_hashes: dict[str, str] = Field(default_factory=dict)
    scientific_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


class PaperPreflightState(BaseModel):
    """Persisted hash snapshot recorded at preflight to detect drift at run time.

    Written to ``<bundle_dir>/preflight_state.json`` when ``preflight_paper_experiment``
    is called with an explicit ``bundle_dir``. Reloaded by ``run_paper_experiment``
    before the first API call; any hash drift (e.g. from a ``git pull`` or a stray
    prompt edit between preflight and launch) aborts the run so the 1600-run
    frontier-model budget is not wasted on a bundle that would fail post-hoc
    validation.
    """

    schema_version: int = 1
    checked_at: str
    manifest_id: str
    manifest_version: str
    manifest_hash: str
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID
    initial_variant_hash: str = ""
    conditions: list[str]
    scenarios: list[str]
    prompt_family_hashes: dict[str, str] = Field(default_factory=dict)
    scenario_source_hashes: dict[str, str] = Field(default_factory=dict)
    config_hashes: dict[str, str] = Field(default_factory=dict)
    scientific_artifact_hashes: dict[str, str] = Field(default_factory=dict)


class PaperRunRecord(BaseModel):
    """One frozen paper run row with provenance metadata."""

    bundle_id: str
    manifest_id: str
    manifest_version: str
    manifest_hash: str
    git_commit_sha: str
    git_dirty: bool
    profile_name: str
    temperature: float
    max_tokens: int
    prompt_family_hash: str
    scenario_source_hash: str
    config_hash: str
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID
    initial_variant_hash: str = ""
    condition: str
    model: str
    provider: str
    scenario: str
    replication: int
    order_index: int
    success: bool
    completion_ratio: float
    steps_taken: int
    time_elapsed_seconds: float
    safety_violations: int
    blocked_actions: int
    efficiency_score: float
    prompt_tokens: int
    completion_tokens: int
    wall_time_seconds: float
    error: str | None = None
    error_taxonomy: str = "none"
    timestamp: str
    trace_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    invalid_action_outputs: int = 0
    action_output_retry_count: int = 0
    provider_tool_call_fallback_count: int = 0
    invalid_tool_calls: int = 0
    tool_retry_count: int = 0
    invalid_model_outputs: int = 0
    model_output_retry_count: int = 0
    model_output_format_errors: int = 0
    grounding_error_count: int = 0
    action_id_fallback_count: int = 0
    bare_verb_fallback_count: int = 0
    transport_normalization_count: int = 0
    tool_name_alias_count: int = 0
    flattened_parameter_count: int = 0
    json_repair_count: int = 0
    stringified_parameters_repair_count: int = 0
    semantic_fallback_count: int = 0
    ambiguous_action_choice_errors: int = 0
    provider_errors: int = 0
    artifact_integrity_errors: int = 0
    post_completion_turns: int = 0
    max_prompt_tokens_single_turn: int = 0
    max_completion_tokens_single_turn: int = 0
    run_artifact_dir: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class PaperRunBundle(BaseModel):
    """Resolved bundle paths for a paper run."""

    bundle_dir: str
    manifest_path: str
    metadata_path: str
    results_path: str
    summary_path: str


def _infer_provider(model: str) -> str:
    lowered = model.casefold()
    if lowered.startswith("ollama/"):
        return "ollama"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    if "gpt" in lowered or lowered.startswith("o1") or lowered.startswith("o3"):
        return "openai"
    raise ValueError(
        f"Cannot infer provider for model {model!r}; use a recognized OpenAI/Anthropic model ID."
    )


def _git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prompt_family_hash_for_condition(condition: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    condition = normalize_condition_name(condition)
    condition_files = {
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED: [
            repo_root / "src/labverse/agent/prompt.py",
            repo_root / "src/labverse/agent/llm.py",
            repo_root / "src/labverse/agent/interactive_output.py",
            repo_root / "src/labverse/agent/action_binding.py",
            repo_root / "src/labverse/benchmarks/paper_agents.py",
        ],
        CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED: [
            repo_root / "src/labverse/agent/prompt.py",
            repo_root / "src/labverse/agent/interactive_text.py",
            repo_root / "src/labverse/agent/interactive_output.py",
            repo_root / "src/labverse/benchmarks/paper_agents.py",
            repo_root / "src/labverse/agent/action_binding.py",
        ],
        CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED: [
            repo_root / "src/labverse/agent/global_structured.py",
            repo_root / "src/labverse/agent/llm.py",
            repo_root / "src/labverse/agent/prompt.py",
            repo_root / "src/labverse/agent/interactive_output.py",
            repo_root / "src/labverse/benchmarks/paper_agents.py",
            repo_root / "src/labverse/agent/action_binding.py",
        ],
        CONDITION_GLOBAL_ONE_SHOT_PLANNER: [
            repo_root / "src/labverse/agent/text_planner.py",
            repo_root / "src/labverse/agent/interactive_output.py",
            repo_root / "src/labverse/benchmarks/paper_agents.py",
            repo_root / "src/labverse/agent/action_binding.py",
        ],
        SCRIPTED_ORACLE_CONDITION: [
            repo_root / "src/labverse/agent/scripted.py",
        ],
    }
    digest = hashlib.sha256()
    for path in condition_files[condition]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scenario_source_hash_for_name(scenario_name: str) -> str:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    scenario_path = Path(inspect.getfile(type(scenario)))
    return _sha256_bytes(scenario_path.read_bytes())


def config_hash_for_manifest_scenario(manifest: PaperManifest, scenario_name: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    scenario_manifest = manifest.scenario_map()[scenario_name]
    return _sha256_bytes((repo_root / scenario_manifest.config_path).read_bytes())


PREFLIGHT_STATE_FILENAME = "preflight_state.json"


def _compute_preflight_state(
    *,
    manifest: PaperManifest,
    conditions: list[str],
    scenarios: list[str],
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
) -> PaperPreflightState:
    """Compute the full hash snapshot that anchors a frozen paper run."""

    prompt_family_hashes = {
        condition: prompt_family_hash_for_condition(condition) for condition in conditions
    }
    scenario_source_hashes = {
        scenario: scenario_source_hash_for_name(scenario) for scenario in scenarios
    }
    config_hashes = {
        scenario: config_hash_for_manifest_scenario(manifest, scenario) for scenario in scenarios
    }
    return PaperPreflightState(
        checked_at=datetime.now(UTC).isoformat(),
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.version,
        manifest_hash=manifest_sha256(manifest),
        initial_variant_id=initial_variant_id,
        initial_variant_hash=initial_condition_variant_hash(initial_variant_id),
        conditions=list(conditions),
        scenarios=list(scenarios),
        prompt_family_hashes=prompt_family_hashes,
        scenario_source_hashes=scenario_source_hashes,
        config_hashes=config_hashes,
        scientific_artifact_hashes=scientific_artifact_hashes(),
    )


def _write_preflight_state(bundle_dir: Path, state: PaperPreflightState) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / PREFLIGHT_STATE_FILENAME).write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )


def _verify_preflight_state_no_drift(
    bundle_dir: Path,
    *,
    manifest: PaperManifest,
    conditions: list[str],
    scenarios: list[str],
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
) -> None:
    """Raise :class:`ValueError` if the persisted preflight state has drifted.

    Called by :func:`run_paper_experiment` before spending any API budget. If the
    preflight wrote a ``preflight_state.json`` snapshot, recompute every hash and
    refuse to launch if the repo state no longer matches — e.g. a stray ``git
    pull`` or a prompt edit happened between preflight and launch.
    """

    state_path = bundle_dir / PREFLIGHT_STATE_FILENAME
    if not state_path.exists():
        return

    recorded_payload = json.loads(state_path.read_text())
    recorded = PaperPreflightState.model_validate(recorded_payload)
    has_recorded_initial_variant = (
        "initial_variant_id" in recorded_payload and "initial_variant_hash" in recorded_payload
    )
    current = _compute_preflight_state(
        manifest=manifest,
        conditions=conditions,
        scenarios=scenarios,
        initial_variant_id=initial_variant_id,
    )

    drifted: list[str] = []
    if recorded.manifest_hash != current.manifest_hash:
        drifted.append(
            f"manifest_hash drift: preflight={recorded.manifest_hash[:12]}, "
            f"run={current.manifest_hash[:12]}"
        )
    if recorded.manifest_id != current.manifest_id:
        drifted.append(
            f"manifest_id drift: preflight={recorded.manifest_id}, run={current.manifest_id}"
        )
    if recorded.manifest_version != current.manifest_version:
        drifted.append(
            "manifest_version drift: "
            f"preflight={recorded.manifest_version}, run={current.manifest_version}"
        )
    if has_recorded_initial_variant and recorded.initial_variant_id != current.initial_variant_id:
        drifted.append(
            "initial_variant_id drift: "
            f"preflight={recorded.initial_variant_id}, run={current.initial_variant_id}"
        )
    if (
        has_recorded_initial_variant
        and recorded.initial_variant_hash != current.initial_variant_hash
    ):
        drifted.append(
            "initial_variant_hash drift: "
            f"preflight={recorded.initial_variant_hash[:12]}, "
            f"run={current.initial_variant_hash[:12]}"
        )
    # Only compare entries that were present at the original preflight. New
    # conditions/scenarios introduced on resume are treated as not-yet-preflighted
    # and are silently allowed through — the user is explicitly expanding the
    # bundle's scope, and the drift check for those keys will attach on the next
    # preflight.
    for condition in conditions:
        recorded_hash = recorded.prompt_family_hashes.get(condition)
        if recorded_hash is None:
            continue
        current_hash = current.prompt_family_hashes[condition]
        if recorded_hash != current_hash:
            drifted.append(
                f"prompt_family_hash drift for {condition}: "
                f"preflight={recorded_hash[:12]}, run={current_hash[:12]}"
            )
    for scenario in scenarios:
        recorded_source = recorded.scenario_source_hashes.get(scenario)
        if recorded_source is not None:
            current_source = current.scenario_source_hashes[scenario]
            if recorded_source != current_source:
                drifted.append(
                    f"scenario_source_hash drift for {scenario}: "
                    f"preflight={recorded_source[:12]}, run={current_source[:12]}"
                )
        recorded_config = recorded.config_hashes.get(scenario)
        if recorded_config is not None:
            current_config = current.config_hashes[scenario]
            if recorded_config != current_config:
                drifted.append(
                    f"config_hash drift for {scenario}: "
                    f"preflight={recorded_config[:12]}, run={current_config[:12]}"
                )
    if recorded.scientific_artifact_hashes != current.scientific_artifact_hashes:
        drifted.append(
            "scientific_artifact_hashes drift between preflight and run-start "
            "(scenario integrity, observability, audit rubric/template, "
            "or initial variants changed)"
        )

    if drifted:
        raise ValueError(
            "Paper preflight state drifted before run-start. Re-run "
            "`labverse paper-preflight` against a clean tree, or delete the stale "
            f"{PREFLIGHT_STATE_FILENAME} if the drift is intentional:\n- " + "\n- ".join(drifted)
        )


def _error_taxonomy(
    *,
    result: RunResult,
    condition: str,
    scenario: str,
    run_metrics: dict[str, int] | None = None,
) -> str:
    error_text = (result.error or "").casefold()
    details = {key: value for key, value in result.score.details.items()}
    runner_error = details.get("runner_error")
    metrics = run_metrics or {}
    runner_error_text = runner_error.casefold() if isinstance(runner_error, str) else ""

    def _looks_like_grounding_contract_failure(text: str) -> bool:
        if not text:
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
        return any(marker in text for marker in grounding_markers)

    if result.score.success:
        return "none"
    if metrics.get("artifact_integrity_errors", 0) > 0:
        return "artifact_integrity_failure"
    if metrics.get("provider_errors", 0) > 0 or "provider error" in error_text:
        return "provider_failure"
    if condition in PRIMARY_INTERACTIVE_CONDITIONS and (
        _looks_like_grounding_contract_failure(error_text)
        or _looks_like_grounding_contract_failure(runner_error_text)
    ):
        return "grounding_failure"
    if condition == CONDITION_GLOBAL_ONE_SHOT_PLANNER and (
        _looks_like_grounding_contract_failure(error_text)
        or _looks_like_grounding_contract_failure(runner_error_text)
    ):
        return "grounding_failure"
    if isinstance(runner_error, str) and runner_error:
        return "runner_failure"
    if result.error:
        return "runner_failure"
    if result.score.safety_violations > 0:
        return "safety_failure"
    if scenario == "shared_equipment_scheduling" and result.score.blocked_actions > 0:
        return "resource_contention_failure"
    if "parameter" in error_text or any("parameter" in key for key in details):
        return "parameter_failure"
    if "invalid action" in error_text or "valid action" in error_text or "tool" in error_text:
        return "grounding_failure"
    if "blocked" in error_text or result.score.blocked_actions > 0:
        return "sequencing_failure"
    if (
        result.score.completion_ratio < 1.0
        and result.score.steps_taken > 0
        and isinstance(result.score.details.get("max_steps_used"), float)
        and result.score.steps_taken >= int(result.score.details["max_steps_used"])
    ):
        return "timeout_or_no_progress"
    if condition in PRIMARY_INTERACTIVE_CONDITIONS and "failed after retries" in error_text:
        return "grounding_failure"
    if result.score.steps_taken > 0:
        return "scenario_behavior_failure"
    return "sequencing_failure"


def _default_bundle_dir(profile_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "paper_runs" / f"{timestamp}-{profile_name}"


def _safe_path_part(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _run_artifact_dir(
    bundle_dir: Path,
    *,
    condition: str,
    model: str,
    scenario: str,
    replication: int,
) -> Path:
    return (
        bundle_dir
        / "runs"
        / _safe_path_part(condition)
        / _safe_path_part(model)
        / _safe_path_part(scenario)
        / f"rep_{replication}"
    )


def _provider_env_var(provider: str) -> str | None:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "ollama":
        return None
    return None


def _existing_completed_keys(results_path: Path) -> set[tuple[str, str, str, int]]:
    if not results_path.exists():
        return set()
    completed: set[tuple[str, str, str, int]] = set()
    for record in load_paper_run_records(results_path):
        completed.add((record.condition, record.model, record.scenario, record.replication))
    return completed


def _cleanup_stale_running_run_metas(
    bundle_dir: Path,
    *,
    completed_keys: set[tuple[str, str, str, int]],
) -> None:
    runs_root = bundle_dir / "runs"
    if not runs_root.exists():
        return

    for run_meta_path in runs_root.rglob("run_meta.json"):
        payload = json.loads(run_meta_path.read_text())
        if payload.get("status") != "running":
            continue
        key = (
            str(payload.get("condition", "")),
            str(payload.get("model", "")),
            str(payload.get("scenario", "")),
            int(payload.get("replication", 0)),
        )
        if key in completed_keys:
            payload["status"] = "completed"
            payload.setdefault("recovered_at", datetime.now(UTC).isoformat())
            payload.setdefault(
                "recovered_reason",
                "A matching results row already exists for this run.",
            )
        else:
            payload["status"] = "stale"
            payload["stale_marked_at"] = datetime.now(UTC).isoformat()
            payload["stale_reason"] = "Run was left marked running without a matching results row."
        run_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_paper_bundle_summary(records: list[PaperRunRecord]) -> dict[str, Any]:
    by_condition: dict[str, list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        by_condition[record.condition].append(record)

    def _stats(items: list[PaperRunRecord]) -> dict[str, float | int]:
        n = len(items)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "success_rate": sum(1 for item in items if item.success) / n,
            "mean_completion_ratio": sum(item.completion_ratio for item in items) / n,
            "mean_safety_violations": sum(item.safety_violations for item in items) / n,
            "mean_blocked_actions": sum(item.blocked_actions for item in items) / n,
            "mean_steps_taken": sum(item.steps_taken for item in items) / n,
            "mean_time_elapsed_seconds": sum(item.time_elapsed_seconds for item in items) / n,
            "mean_prompt_tokens": sum(item.prompt_tokens for item in items) / n,
            "mean_completion_tokens": sum(item.completion_tokens for item in items) / n,
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
            "mean_action_id_fallback_count": (
                sum(item.action_id_fallback_count for item in items) / n
            ),
            "mean_bare_verb_fallback_count": (
                sum(item.bare_verb_fallback_count for item in items) / n
            ),
            "mean_transport_normalization_count": (
                sum(item.transport_normalization_count for item in items) / n
            ),
            "mean_tool_name_alias_count": (sum(item.tool_name_alias_count for item in items) / n),
            "mean_flattened_parameter_count": (
                sum(item.flattened_parameter_count for item in items) / n
            ),
            "mean_json_repair_count": sum(item.json_repair_count for item in items) / n,
            "mean_stringified_parameters_repair_count": (
                sum(item.stringified_parameters_repair_count for item in items) / n
            ),
            "mean_semantic_fallback_count": (
                sum(item.semantic_fallback_count for item in items) / n
            ),
            "mean_ambiguous_action_choice_errors": (
                sum(item.ambiguous_action_choice_errors for item in items) / n
            ),
            "mean_provider_errors": sum(item.provider_errors for item in items) / n,
            "mean_artifact_integrity_errors": (
                sum(item.artifact_integrity_errors for item in items) / n
            ),
            "mean_post_completion_turns": sum(item.post_completion_turns for item in items) / n,
            "max_prompt_tokens_single_turn": max(
                item.max_prompt_tokens_single_turn for item in items
            ),
            "max_completion_tokens_single_turn": max(
                item.max_completion_tokens_single_turn for item in items
            ),
        }

    return {
        "total_runs": len(records),
        "by_condition": {
            condition: _stats(items) for condition, items in sorted(by_condition.items())
        },
    }


def load_paper_run_records(path: str | Path) -> list[PaperRunRecord]:
    records: list[PaperRunRecord] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(PaperRunRecord.model_validate_json(line))
    return records


def preflight_paper_experiment(
    *,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    scenarios: list[str] | None = None,
    manifest_path: str | Path | None = None,
    profile_name: str = "deterministic",
    replications: int | None = None,
    random_seed: int = 7,
    bundle_dir: str | Path | None = None,
    resume: bool = False,
    allow_dirty: bool = False,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
) -> PaperPreflightReport:
    """Validate a frozen paper run before creating or resuming a bundle."""

    manifest = load_paper_manifest(manifest_path)
    issues: list[str] = []
    warnings: list[str] = []
    if profile_name not in manifest.profiles:
        issues.append(f"Unknown paper profile: {profile_name}")

    selected_conditions = (
        normalize_condition_names(list(conditions or manifest.primary_conditions)) or []
    )
    for condition in selected_conditions:
        if condition not in ALL_PAPER_CONDITIONS:
            issues.append(f"Unsupported paper condition: {condition}")

    selected_scenarios = list(scenarios or manifest.scenario_names())
    frozen_scenarios = set(manifest.scenario_names())
    missing_scenarios = [name for name in selected_scenarios if name not in frozen_scenarios]
    if missing_scenarios:
        issues.append(
            "Scenarios are not in the frozen paper manifest: " + ", ".join(missing_scenarios)
        )

    needs_models = any(condition != SCRIPTED_ORACLE_CONDITION for condition in selected_conditions)
    resolved_models = list(models or [])
    if needs_models and not resolved_models:
        issues.append(
            "Paper runs for model-backed interactive or planning conditions "
            "require at least one model."
        )

    provider_credentials: dict[str, bool] = {}
    if needs_models:
        for model in resolved_models:
            provider = _infer_provider(model)
            env_var = _provider_env_var(provider)
            if env_var is None:
                continue
            has_credential = bool(os.environ.get(env_var))
            provider_credentials[f"{provider}:{model}"] = has_credential
            if not has_credential:
                issues.append(f"Missing provider credential {env_var} for model {model!r}.")

    effective_bundle_dir = (
        Path(bundle_dir) if bundle_dir is not None else _default_bundle_dir(profile_name)
    )
    metadata_path = effective_bundle_dir / "bundle_metadata.json"
    if metadata_path.exists() and not resume:
        issues.append(
            "Bundle directory already exists: "
            f"{effective_bundle_dir}. Use resume=True to continue it."
        )
    if not metadata_path.exists() and resume:
        issues.append(
            f"Cannot resume bundle {effective_bundle_dir}: bundle_metadata.json is missing."
        )
    if metadata_path.exists() and resume:
        try:
            metadata = PaperBundleMetadata.model_validate_json(metadata_path.read_text())
            if metadata.initial_variant_id != initial_variant_id:
                issues.append(
                    "Cannot resume bundle with a different initial-condition variant: "
                    f"bundle={metadata.initial_variant_id}, requested={initial_variant_id}"
                )
        except Exception as exc:
            issues.append(f"Cannot read existing bundle metadata: {exc}")

    git_commit_sha = _git_commit_sha()
    git_dirty = _git_dirty()
    if git_dirty and not allow_dirty:
        issues.append(
            "Git worktree is dirty. Commit or stash changes, or pass allow_dirty=True "
            "for an audit-only run."
        )
    if git_dirty and allow_dirty:
        warnings.append("Git worktree is dirty; bundle will be marked non-reproducible.")

    initial_variant_hash = ""
    try:
        registry = load_initial_condition_variants()
        if initial_variant_id not in registry.variants:
            available = ", ".join(sorted(registry.variants))
            issues.append(
                f"Unknown initial-condition variant {initial_variant_id!r}; "
                f"available: {available}"
            )
        else:
            initial_variant_hash = initial_condition_variant_hash(initial_variant_id)
    except Exception as exc:
        issues.append(f"Initial-condition variants are invalid: {exc}")

    prompt_family_hashes: dict[str, str] = {}
    for condition in selected_conditions:
        prompt_family_hashes[condition] = prompt_family_hash_for_condition(condition)
    issues.extend(validate_scientific_artifacts(manifest=manifest))
    artifact_hashes = scientific_artifact_hashes()

    report = PaperPreflightReport(
        checked_at=datetime.now(UTC).isoformat(),
        bundle_dir=str(effective_bundle_dir),
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.version,
        profile_name=profile_name,
        git_commit_sha=git_commit_sha,
        git_dirty=git_dirty,
        allow_dirty=allow_dirty,
        resume=resume,
        initial_variant_id=initial_variant_id,
        initial_variant_hash=initial_variant_hash,
        conditions=selected_conditions,
        models=resolved_models,
        scenarios=selected_scenarios,
        provider_credentials=provider_credentials,
        prompt_family_hashes=prompt_family_hashes,
        scientific_artifact_hashes=artifact_hashes,
        issues=issues,
        warnings=warnings,
    )

    # Persist a full hash snapshot beside the target bundle so
    # ``run_paper_experiment`` can re-verify it at launch time and abort on
    # any drift before spending API budget. We only persist when the caller
    # passed an explicit bundle_dir; default-dir preflight calls (e.g. from
    # unit tests) are non-destructive by design. If a preflight state file
    # already exists we never overwrite it: the standalone
    # ``labverse paper-preflight`` CLI writes it first, and a subsequent
    # ``labverse paper-run`` invocation must compare against the original
    # snapshot to detect intra-window drift.
    if report.valid and bundle_dir is not None:
        state_path = effective_bundle_dir / PREFLIGHT_STATE_FILENAME
        if not state_path.exists():
            try:
                state = _compute_preflight_state(
                    manifest=manifest,
                    conditions=selected_conditions,
                    scenarios=selected_scenarios,
                    initial_variant_id=initial_variant_id,
                )
                _write_preflight_state(effective_bundle_dir, state)
            except OSError as exc:
                report.warnings.append(f"Could not persist {PREFLIGHT_STATE_FILENAME}: {exc}")

    return report


def _prompt_surface_snapshot(step_audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_audits:
        return {}
    first_step = step_audits[0]
    return {
        "system_prompt": first_step.get("system_prompt"),
        "first_turn_user_message": first_step.get("user_message"),
        "first_turn_tool_schema": first_step.get("tool_schema"),
        "prompt_surface_kind": first_step.get("prompt_surface_kind"),
    }


def _record_details(
    *,
    result: RunResult,
    run_metrics: dict[str, int],
    step_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    details: dict[str, Any] = dict(result.score.details)
    if int(run_metrics.get("artifact_integrity_errors", 0)) > 0:
        details["artifact_integrity_errors"] = int(run_metrics.get("artifact_integrity_errors", 0))
        messages = [
            message
            for step in step_audits
            for message in step.get("artifact_integrity_messages", [])
            if isinstance(message, str)
        ]
        if messages:
            details["artifact_integrity_messages"] = messages
        for step in step_audits:
            traceback_text = step.get("artifact_integrity_traceback")
            if isinstance(traceback_text, str) and traceback_text:
                details["artifact_integrity_traceback"] = traceback_text
                break
    if int(run_metrics.get("post_completion_turns", 0)) > 0:
        details["post_completion_turns"] = int(run_metrics.get("post_completion_turns", 0))
    return details


def _write_run_artifacts(
    *,
    bundle_dir: Path,
    record: PaperRunRecord,
    step_audits: list[dict[str, Any]],
    replay_frames: list[dict[str, Any]],
    status: str,
    started_at: str,
) -> None:
    run_dir = _run_artifact_dir(
        bundle_dir,
        condition=record.condition,
        model=record.model,
        scenario=record.scenario,
        replication=record.replication,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    steps_path = run_dir / "steps.jsonl"
    with steps_path.open("w") as handle:
        for index, step in enumerate(step_audits, start=1):
            payload = dict(step)
            payload.setdefault("step_index", index)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    replay_path = run_dir / "replay.jsonl"
    with replay_path.open("w") as handle:
        for index, frame in enumerate(replay_frames, start=1):
            payload = dict(frame)
            payload.setdefault("step_index", index)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    normalized_trace_rows = write_normalized_trace_jsonl(
        run_dir / NORMALIZED_TRACE_FILENAME,
        step_audits,
    )

    run_meta = {
        "bundle_id": record.bundle_id,
        "condition": record.condition,
        "initial_variant_id": record.initial_variant_id,
        "initial_variant_hash": record.initial_variant_hash,
        "model": record.model,
        "provider": record.provider,
        "scenario": record.scenario,
        "replication": record.replication,
        "status": status,
        "started_at": started_at,
        "completed_at": record.timestamp,
        "prompt_family_hash": record.prompt_family_hash,
        "run_metrics": {
            "invalid_tool_calls": record.invalid_tool_calls,
            "tool_retry_count": record.tool_retry_count,
            "invalid_model_outputs": record.invalid_model_outputs,
            "model_output_retry_count": record.model_output_retry_count,
            "model_output_format_errors": record.model_output_format_errors,
            "grounding_error_count": record.grounding_error_count,
            "action_id_fallback_count": record.action_id_fallback_count,
            "bare_verb_fallback_count": record.bare_verb_fallback_count,
            "transport_normalization_count": record.transport_normalization_count,
            "tool_name_alias_count": record.tool_name_alias_count,
            "flattened_parameter_count": record.flattened_parameter_count,
            "json_repair_count": record.json_repair_count,
            "stringified_parameters_repair_count": record.stringified_parameters_repair_count,
            "semantic_fallback_count": record.semantic_fallback_count,
            "ambiguous_action_choice_errors": record.ambiguous_action_choice_errors,
            "provider_errors": record.provider_errors,
            "artifact_integrity_errors": record.artifact_integrity_errors,
            "post_completion_turns": record.post_completion_turns,
            "max_prompt_tokens_single_turn": record.max_prompt_tokens_single_turn,
            "max_completion_tokens_single_turn": record.max_completion_tokens_single_turn,
        },
        "score": {
            "success": record.success,
            "completion_ratio": record.completion_ratio,
            "steps_taken": record.steps_taken,
            "time_elapsed_seconds": record.time_elapsed_seconds,
            "safety_violations": record.safety_violations,
            "blocked_actions": record.blocked_actions,
            "error": record.error,
            "error_taxonomy": record.error_taxonomy,
        },
        "replay_available": bool(replay_frames),
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "replay_frame_count": len(replay_frames),
        "normalized_trace_available": bool(normalized_trace_rows),
        "normalized_trace_file": NORMALIZED_TRACE_FILENAME,
        "normalized_trace_row_count": len(normalized_trace_rows),
        "prompt_surface": _prompt_surface_snapshot(step_audits),
        "trace_ids": record.trace_ids,
        "event_ids": record.event_ids,
        "details": record.details,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")


def _write_pending_run_meta(
    *,
    bundle_dir: Path,
    condition: str,
    model: str,
    provider: str,
    scenario: str,
    replication: int,
    prompt_family_hash: str,
    started_at: str,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    initial_variant_hash: str = "",
) -> Path:
    run_dir = _run_artifact_dir(
        bundle_dir,
        condition=condition,
        model=model,
        scenario=scenario,
        replication=replication,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "steps.jsonl").write_text("")
    (run_dir / "replay.jsonl").write_text("")
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "initial_variant_id": initial_variant_id,
                "initial_variant_hash": initial_variant_hash,
                "launcher_pid": os.getpid(),
                "model": model,
                "provider": provider,
                "scenario": scenario,
                "replication": replication,
                "status": "running",
                "started_at": started_at,
                "prompt_family_hash": prompt_family_hash,
                "replay_available": False,
                "replay_schema_version": REPLAY_SCHEMA_VERSION,
                "replay_frame_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return run_dir


def _update_run_meta_with_worker_pid(run_meta_path: Path, *, worker_pid: int) -> None:
    if not run_meta_path.exists():
        return
    payload = json.loads(run_meta_path.read_text())
    payload["worker_pid"] = worker_pid
    payload["worker_started_at"] = datetime.now(UTC).isoformat()
    run_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_single_capture_worker(
    result_queue: Any,
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    initial_variant_id: str,
    provider_options: ProviderOptions | None,
) -> None:
    result, artifacts = run_single_capture(
        condition=condition,
        provider_name=provider_name,
        model=model,
        scenario_name=scenario_name,
        replication=replication,
        initial_variant_id=initial_variant_id,
        provider_options=provider_options,
    )
    result_queue.put(
        {
            "result": serialize_run_result(result),
            "artifacts": serialize_artifacts(artifacts),
        }
    )


def _run_single_capture_inline(
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None,
) -> tuple[RunResult, ScenarioRunArtifacts]:
    return run_single_capture(
        condition=condition,
        provider_name=provider_name,
        model=model,
        scenario_name=scenario_name,
        replication=replication,
        initial_variant_id=initial_variant_id,
        provider_options=provider_options,
    )


def _run_single_capture_subprocess(
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None,
    run_meta_path: Path | None = None,
) -> tuple[RunResult, ScenarioRunArtifacts]:
    started = time.monotonic()
    effective_options = provider_options or ProviderOptions()
    with tempfile.TemporaryDirectory(prefix="labverse-paper-cell-") as tmp_dir:
        output_path = Path(tmp_dir) / "cell_result.json"
        cmd = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "labverse.paper_worker",
            "--condition",
            condition,
            "--provider",
            provider_name,
            "--model",
            model,
            "--scenario",
            scenario_name,
            "--replication",
            str(replication),
            "--temperature",
            str(effective_options.temperature),
            "--max-tokens",
            str(effective_options.max_tokens),
            "--request-timeout-seconds",
            str(effective_options.request_timeout_seconds),
            "--client-max-retries",
            str(effective_options.client_max_retries),
            "--worker-timeout-seconds",
            str(effective_options.worker_timeout_seconds),
            "--initial-variant",
            initial_variant_id,
            "--output",
            str(output_path),
        ]
        env = dict(os.environ)
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
            env=env,
        )
        if run_meta_path is not None:
            _update_run_meta_with_worker_pid(run_meta_path, worker_pid=process.pid)

        try:
            stdout_text, stderr_text = process.communicate(
                timeout=effective_options.worker_timeout_seconds
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_text, stderr_text = process.communicate()
            wall_time_seconds = time.monotonic() - started
            error = (
                "Isolated subprocess timed out for "
                f"{condition}/{model}/{scenario_name} rep={replication} after "
                f"{effective_options.worker_timeout_seconds:.0f}s"
            )
            if stderr_text.strip():
                error = f"{error}; stderr={stderr_text.strip()[:500]}"
            return _crash_failure_capture(
                condition=condition,
                provider_name=provider_name,
                model=model,
                scenario_name=scenario_name,
                replication=replication,
                error=error,
                wall_time_seconds=wall_time_seconds,
                diagnostics=_subprocess_diagnostics(
                    process=process,
                    output_path=output_path,
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    timed_out=True,
                ),
            )

        wall_time_seconds = time.monotonic() - started
        if process.returncode == 0 and output_path.exists():
            payload = json.loads(output_path.read_text())
            return (
                deserialize_run_result(payload["result"]),
                deserialize_artifacts(payload["artifacts"]),
            )

        error = (
            "Isolated subprocess crashed for "
            f"{condition}/{model}/{scenario_name} rep={replication}; "
            f"exit_code={process.returncode}"
        )
        stderr_clean = stderr_text.strip()
        if stderr_clean:
            error = f"{error}; stderr={stderr_clean[:500]}"
        return _crash_failure_capture(
            condition=condition,
            provider_name=provider_name,
            model=model,
            scenario_name=scenario_name,
            replication=replication,
            error=error,
            wall_time_seconds=wall_time_seconds,
            diagnostics=_subprocess_diagnostics(
                process=process,
                output_path=output_path,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                timed_out=False,
            ),
        )


def _tail_text(text: str, *, limit: int = STDIO_TAIL_CHARS) -> str:
    """Return a bounded diagnostic tail without hiding that output was truncated."""

    if len(text) <= limit:
        return text
    return f"... <truncated {len(text) - limit} chars>\n{text[-limit:]}"


def _exit_signal_name(exit_code: int | None) -> str | None:
    if exit_code is None or exit_code >= 0:
        return None
    try:
        return signal.Signals(-exit_code).name
    except ValueError:
        return None


def _subprocess_diagnostics(
    *,
    process: subprocess.Popen[str],
    output_path: Path,
    stdout_text: str,
    stderr_text: str,
    timed_out: bool,
) -> dict[str, Any]:
    """Build transparent diagnostics for an isolated paper cell failure."""

    output_exists = output_path.exists()
    return {
        "worker_pid": getattr(process, "pid", None),
        "exit_code": process.returncode,
        "exit_signal": _exit_signal_name(process.returncode),
        "timed_out": timed_out,
        "stdout_tail": _tail_text(stdout_text),
        "stderr_tail": _tail_text(stderr_text),
        "output_file_exists": output_exists,
        "output_file_bytes": output_path.stat().st_size if output_exists else 0,
        "runner_traceback_available": bool(stderr_text.strip()),
    }


def _crash_failure_capture(
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    error: str,
    wall_time_seconds: float,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[RunResult, ScenarioRunArtifacts]:
    details: dict[str, Any] = {
        "error": error,
        "runner_traceback": (
            "unavailable; isolated worker crashed before Python could report a traceback"
        ),
    }
    if diagnostics:
        details.update(diagnostics)
    score = BenchmarkScorer.failure_score(scenario_name, error=error).model_copy(
        update={
            "details": details,
        },
        deep=True,
    )
    result = RunResult(
        condition=condition,
        model=model,
        provider=provider_name,
        scenario=scenario_name,
        replication=replication,
        score=score,
        prompt_tokens=0,
        completion_tokens=0,
        wall_time_seconds=round(wall_time_seconds, 2),
        error=error,
        timestamp=datetime.now().isoformat(),
    )
    return result, ScenarioRunArtifacts(score=score)


def _run_single_capture_isolated(
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None,
    run_meta_path: Path | None = None,
) -> tuple[RunResult, ScenarioRunArtifacts]:
    started = time.monotonic()
    effective_options = provider_options or ProviderOptions()
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_run_single_capture_worker,
        kwargs={
            "result_queue": result_queue,
            "condition": condition,
            "provider_name": provider_name,
            "model": model,
            "scenario_name": scenario_name,
            "replication": replication,
            "initial_variant_id": initial_variant_id,
            "provider_options": effective_options,
        },
    )
    process.start()
    if run_meta_path is not None and process.pid is not None:
        _update_run_meta_with_worker_pid(run_meta_path, worker_pid=process.pid)
    process.join(effective_options.worker_timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5.0)

    payload: dict[str, Any] | None = None
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = None
    finally:
        result_queue.close()
        result_queue.join_thread()

    wall_time_seconds = time.monotonic() - started
    if process.exitcode is None:
        error = (
            "Isolated run process timed out for "
            f"{condition}/{model}/{scenario_name} rep={replication} after "
            f"{effective_options.worker_timeout_seconds:.0f}s"
        )
        return _crash_failure_capture(
            condition=condition,
            provider_name=provider_name,
            model=model,
            scenario_name=scenario_name,
            replication=replication,
            error=error,
            wall_time_seconds=wall_time_seconds,
            diagnostics={
                "worker_pid": getattr(process, "pid", None),
                "exit_code": process.exitcode,
                "exit_signal": None,
                "timed_out": True,
                "runner_traceback_available": False,
            },
        )

    if process.exitcode == 0 and payload is not None:
        return (
            deserialize_run_result(payload["result"]),
            deserialize_artifacts(payload["artifacts"]),
        )

    error = (
        f"Isolated run process crashed for {condition}/{model}/{scenario_name} rep={replication}; "
        f"exit_code={process.exitcode}"
    )
    return _crash_failure_capture(
        condition=condition,
        provider_name=provider_name,
        model=model,
        scenario_name=scenario_name,
        replication=replication,
        error=error,
        wall_time_seconds=wall_time_seconds,
        diagnostics={
            "worker_pid": getattr(process, "pid", None),
            "exit_code": process.exitcode,
            "exit_signal": _exit_signal_name(process.exitcode),
            "timed_out": False,
            "runner_traceback_available": False,
        },
    )


def run_paper_experiment(
    *,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    scenarios: list[str] | None = None,
    manifest_path: str | Path | None = None,
    profile_name: str = "deterministic",
    replications: int | None = None,
    random_seed: int = 7,
    bundle_dir: str | Path | None = None,
    resume: bool = False,
    allow_dirty: bool = False,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
) -> PaperRunBundle:
    """Execute the frozen paper benchmark into an immutable bundle directory."""

    manifest = load_paper_manifest(manifest_path)
    preflight = preflight_paper_experiment(
        models=models,
        conditions=conditions,
        scenarios=scenarios,
        manifest_path=manifest_path,
        profile_name=profile_name,
        replications=replications,
        random_seed=random_seed,
        bundle_dir=bundle_dir,
        resume=resume,
        allow_dirty=allow_dirty,
        initial_variant_id=initial_variant_id,
    )
    if not preflight.valid:
        raise ValueError("Paper preflight failed:\n- " + "\n- ".join(preflight.issues))

    profile = manifest.profiles[profile_name]
    selected_conditions = list(preflight.conditions)
    selected_scenarios = list(preflight.scenarios)
    resolved_models = list(preflight.models)
    provider_map = {model: _infer_provider(model) for model in resolved_models}
    effective_reps = replications if replications is not None else profile.replications
    effective_bundle_dir = Path(preflight.bundle_dir)
    initial_variant_hash = preflight.initial_variant_hash
    effective_bundle_dir.mkdir(parents=True, exist_ok=True)

    # Hash-drift guard: compare any persisted preflight snapshot against the
    # current repo state before spending API budget. Catches the case where a
    # ``git pull`` or a stray prompt edit happened between a standalone
    # ``labverse paper-preflight`` invocation and this run.
    _verify_preflight_state_no_drift(
        effective_bundle_dir,
        manifest=manifest,
        conditions=selected_conditions,
        scenarios=selected_scenarios,
        initial_variant_id=initial_variant_id,
    )

    metadata_path = effective_bundle_dir / "bundle_metadata.json"
    manifest_copy_path = effective_bundle_dir / "manifest.json"
    results_path = effective_bundle_dir / "results.jsonl"
    summary_path = effective_bundle_dir / "summary.json"

    if metadata_path.exists() and not resume:
        raise ValueError(
            "Bundle directory already exists: "
            f"{effective_bundle_dir}. Use resume=True to continue it."
        )

    resolved_manifest_path = (
        str(Path(manifest_path)) if manifest_path is not None else str(manifest_copy_path)
    )

    if metadata_path.exists() and resume:
        bundle_metadata = PaperBundleMetadata.model_validate_json(metadata_path.read_text())
        missing_conditions = sorted(set(selected_conditions) - set(bundle_metadata.conditions))
        missing_models = sorted(set(resolved_models) - set(bundle_metadata.models))
        missing_scenarios = sorted(set(selected_scenarios) - set(bundle_metadata.scenarios))
        if missing_conditions or missing_models or missing_scenarios:
            issues = []
            if missing_conditions:
                issues.append("conditions: " + ", ".join(missing_conditions))
            if missing_models:
                issues.append("models: " + ", ".join(missing_models))
            if missing_scenarios:
                issues.append("scenarios: " + ", ".join(missing_scenarios))
            raise ValueError(
                "Cannot resume bundle with selections outside existing metadata: "
                + "; ".join(issues)
            )
        initial_variant_hash = bundle_metadata.initial_variant_hash
    else:
        bundle_metadata = PaperBundleMetadata(
            bundle_id=effective_bundle_dir.name,
            created_at=datetime.now(UTC).isoformat(),
            manifest_id=manifest.manifest_id,
            manifest_version=manifest.version,
            manifest_hash=manifest_sha256(manifest),
            manifest_path=resolved_manifest_path,
            git_commit_sha=_git_commit_sha(),
            git_dirty=_git_dirty(),
            profile_name=profile_name,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            replications=effective_reps,
            random_seed=random_seed,
            initial_variant_id=initial_variant_id,
            initial_variant_hash=initial_variant_hash,
            conditions=selected_conditions,
            models=resolved_models,
            scenarios=selected_scenarios,
            allow_dirty=allow_dirty,
            preflight=preflight.model_dump(mode="json"),
            scientific_artifact_hashes=preflight.scientific_artifact_hashes,
        )
        manifest_copy_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        metadata_path.write_text(
            json.dumps(bundle_metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )

    tasks_by_model: dict[str, list[tuple[str, str, str, int]]] = {}
    for condition in selected_conditions:
        model_values = (
            ["scripted_oracle"] if condition == SCRIPTED_ORACLE_CONDITION else resolved_models
        )
        for model in model_values:
            for scenario_name in selected_scenarios:
                for replication in range(1, effective_reps + 1):
                    tasks_by_model.setdefault(model, []).append(
                        (condition, model, scenario_name, replication)
                    )

    rng = random.Random(random_seed)
    model_order = list(tasks_by_model)
    tasks: list[tuple[str, str, str, int]] = []
    for model in model_order:
        model_tasks = list(tasks_by_model[model])
        rng.shuffle(model_tasks)
        tasks.extend(model_tasks)
    completed_keys = _existing_completed_keys(results_path) if resume else set()
    if resume:
        _cleanup_stale_running_run_metas(
            effective_bundle_dir,
            completed_keys=completed_keys,
        )
    provider_options = ProviderOptions(
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
    )

    runner = BenchmarkRunner(initial_variant_id=initial_variant_id)
    with results_path.open("a") as handle:
        for order_index, (condition, model, scenario_name, replication) in enumerate(
            tasks,
            start=1,
        ):
            key = (condition, model, scenario_name, replication)
            if key in completed_keys:
                continue

            if condition == SCRIPTED_ORACLE_CONDITION:
                started_at = datetime.now(UTC).isoformat()
                capture = runner.run_scenario_capture(scenario_name)
                result = RunResult(
                    condition=condition,
                    model=model,
                    provider="scripted",
                    scenario=scenario_name,
                    replication=replication,
                    score=capture.score,
                    prompt_tokens=0,
                    completion_tokens=0,
                    wall_time_seconds=0.0,
                    error=None,
                    timestamp=datetime.now(UTC).isoformat(),
                )
                artifacts = ScenarioRunArtifacts(
                    score=capture.score,
                    event_ids=list(capture.event_ids),
                    trace_ids=list(capture.trace_ids),
                    trace_ids_by_agent={
                        agent_id: list(ids) for agent_id, ids in capture.trace_ids_by_agent.items()
                    },
                    step_audits=[dict(step) for step in capture.step_audits],
                    run_metrics={},
                )
                provider = "scripted"
            else:
                provider = provider_map[model]
                started_at = datetime.now(UTC).isoformat()
                run_artifact_dir = _write_pending_run_meta(
                    bundle_dir=effective_bundle_dir,
                    condition=condition,
                    model=model,
                    provider=provider,
                    scenario=scenario_name,
                    replication=replication,
                    prompt_family_hash=prompt_family_hash_for_condition(condition),
                    started_at=started_at,
                    initial_variant_id=initial_variant_id,
                    initial_variant_hash=initial_variant_hash,
                )
                result, artifacts = _run_single_capture_subprocess(
                    condition=condition,
                    provider_name=provider,
                    model=model,
                    scenario_name=scenario_name,
                    replication=replication,
                    initial_variant_id=initial_variant_id,
                    provider_options=provider_options,
                    run_meta_path=run_artifact_dir / "run_meta.json",
                )

            run_metrics = dict(artifacts.run_metrics)
            run_artifact_dir = _run_artifact_dir(
                effective_bundle_dir,
                condition=condition,
                model=model,
                scenario=scenario_name,
                replication=replication,
            )
            record = PaperRunRecord(
                bundle_id=bundle_metadata.bundle_id,
                manifest_id=manifest.manifest_id,
                manifest_version=manifest.version,
                manifest_hash=bundle_metadata.manifest_hash,
                git_commit_sha=bundle_metadata.git_commit_sha,
                git_dirty=bundle_metadata.git_dirty,
                profile_name=profile_name,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                prompt_family_hash=prompt_family_hash_for_condition(condition),
                scenario_source_hash=scenario_source_hash_for_name(scenario_name),
                config_hash=config_hash_for_manifest_scenario(manifest, scenario_name),
                initial_variant_id=initial_variant_id,
                initial_variant_hash=initial_variant_hash,
                condition=condition,
                model=model,
                provider=provider,
                scenario=scenario_name,
                replication=replication,
                order_index=order_index,
                success=result.score.success,
                completion_ratio=result.score.completion_ratio,
                steps_taken=result.score.steps_taken,
                time_elapsed_seconds=result.score.time_elapsed_seconds,
                safety_violations=result.score.safety_violations,
                blocked_actions=result.score.blocked_actions,
                efficiency_score=result.score.efficiency_score,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_time_seconds=result.wall_time_seconds,
                error=result.error,
                error_taxonomy=_error_taxonomy(
                    result=result,
                    condition=condition,
                    scenario=scenario_name,
                    run_metrics=run_metrics,
                ),
                timestamp=result.timestamp,
                trace_ids=list(artifacts.trace_ids),
                event_ids=list(artifacts.event_ids),
                invalid_action_outputs=int(run_metrics.get("invalid_action_outputs", 0)),
                action_output_retry_count=int(run_metrics.get("action_output_retry_count", 0)),
                provider_tool_call_fallback_count=int(
                    run_metrics.get("provider_tool_call_fallback_count", 0)
                ),
                invalid_tool_calls=int(
                    run_metrics.get(
                        "invalid_tool_calls", run_metrics.get("invalid_action_outputs", 0)
                    )
                ),
                tool_retry_count=int(
                    run_metrics.get(
                        "tool_retry_count", run_metrics.get("action_output_retry_count", 0)
                    )
                ),
                invalid_model_outputs=int(run_metrics.get("invalid_model_outputs", 0)),
                model_output_retry_count=int(run_metrics.get("model_output_retry_count", 0)),
                model_output_format_errors=int(run_metrics.get("model_output_format_errors", 0)),
                grounding_error_count=int(run_metrics.get("grounding_error_count", 0)),
                action_id_fallback_count=int(run_metrics.get("action_id_fallback_count", 0)),
                bare_verb_fallback_count=int(run_metrics.get("bare_verb_fallback_count", 0)),
                transport_normalization_count=int(
                    run_metrics.get("transport_normalization_count", 0)
                ),
                tool_name_alias_count=int(run_metrics.get("tool_name_alias_count", 0)),
                flattened_parameter_count=int(run_metrics.get("flattened_parameter_count", 0)),
                json_repair_count=int(run_metrics.get("json_repair_count", 0)),
                stringified_parameters_repair_count=int(
                    run_metrics.get("stringified_parameters_repair_count", 0)
                ),
                semantic_fallback_count=int(run_metrics.get("semantic_fallback_count", 0)),
                ambiguous_action_choice_errors=int(
                    run_metrics.get("ambiguous_action_choice_errors", 0)
                ),
                provider_errors=int(run_metrics.get("provider_errors", 0)),
                artifact_integrity_errors=int(run_metrics.get("artifact_integrity_errors", 0)),
                post_completion_turns=int(run_metrics.get("post_completion_turns", 0)),
                max_prompt_tokens_single_turn=int(
                    run_metrics.get("max_prompt_tokens_single_turn", 0)
                ),
                max_completion_tokens_single_turn=int(
                    run_metrics.get("max_completion_tokens_single_turn", 0)
                ),
                run_artifact_dir=str(run_artifact_dir.relative_to(effective_bundle_dir)),
                details=_record_details(
                    result=result,
                    run_metrics=run_metrics,
                    step_audits=list(artifacts.step_audits),
                ),
            )
            _write_run_artifacts(
                bundle_dir=effective_bundle_dir,
                record=record,
                step_audits=list(artifacts.step_audits),
                replay_frames=list(artifacts.replay_frames),
                status="completed" if record.error is None else "failed",
                started_at=started_at,
            )
            handle.write(record.model_dump_json() + "\n")
            handle.flush()

    summary = build_paper_bundle_summary(load_paper_run_records(results_path))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    from labverse.paper.health import write_pipeline_health_report

    write_pipeline_health_report(effective_bundle_dir)

    return PaperRunBundle(
        bundle_dir=str(effective_bundle_dir),
        manifest_path=str(manifest_copy_path),
        metadata_path=str(metadata_path),
        results_path=str(results_path),
        summary_path=str(summary_path),
    )


__all__ = [
    "ALL_PAPER_CONDITIONS",
    "PaperBundleMetadata",
    "PaperPreflightReport",
    "PaperRunBundle",
    "PaperRunRecord",
    "SCRIPTED_ORACLE_CONDITION",
    "build_paper_bundle_summary",
    "config_hash_for_manifest_scenario",
    "load_paper_run_records",
    "preflight_paper_experiment",
    "prompt_family_hash_for_condition",
    "run_paper_experiment",
    "scenario_source_hash_for_name",
]
