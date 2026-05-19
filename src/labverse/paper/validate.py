"""Offline validation helpers for frozen paper bundles."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from labverse.benchmarks.conditions import CONDITION_GLOBAL_ONE_SHOT_PLANNER
from labverse.paper.fairness import condition_fairness_issues
from labverse.paper.manifest import PaperManifest, manifest_sha256
from labverse.paper.runner import (
    PaperBundleMetadata,
    build_paper_bundle_summary,
    config_hash_for_manifest_scenario,
    load_paper_run_records,
    prompt_family_hash_for_condition,
    scenario_source_hash_for_name,
)
from labverse.paper.trace import NORMALIZED_TRACE_FILENAME

# Conditions that intentionally use an open-verb prompt (no closed-enum tool
# schema). These must not trip the tool_schema validation check — the paper
# contract explicitly documents this asymmetry between interactive and one-shot.
_OPEN_VERB_CONDITIONS: frozenset[str] = frozenset({CONDITION_GLOBAL_ONE_SHOT_PLANNER})


class PaperBundleValidationReport(BaseModel):
    """Validation result for one paper bundle directory."""

    bundle_dir: str
    valid: bool
    record_count: int
    issues: list[str] = Field(default_factory=list)


def validate_paper_bundle(bundle_dir: str | Path) -> PaperBundleValidationReport:
    """Validate a frozen paper bundle for internal consistency."""

    bundle_path = Path(bundle_dir)
    issues: list[str] = []

    manifest_path = bundle_path / "manifest.json"
    metadata_path = bundle_path / "bundle_metadata.json"
    results_path = bundle_path / "results.jsonl"
    summary_path = bundle_path / "summary.json"

    for required in (manifest_path, metadata_path, results_path):
        if not required.exists():
            issues.append(f"Missing required bundle file: {required.name}")

    if issues:
        return PaperBundleValidationReport(
            bundle_dir=str(bundle_path),
            valid=False,
            record_count=0,
            issues=issues,
        )

    manifest = PaperManifest.model_validate(json.loads(manifest_path.read_text()))
    metadata = PaperBundleMetadata.model_validate(json.loads(metadata_path.read_text()))
    records = load_paper_run_records(results_path)
    expected_manifest_hash = manifest_sha256(manifest)
    manifest_scenarios = set(manifest.scenario_names())
    expected_summary = build_paper_bundle_summary(records)

    if metadata.manifest_hash != expected_manifest_hash:
        issues.append("Bundle metadata manifest hash does not match manifest.json")
    if metadata.manifest_id != manifest.manifest_id:
        issues.append("Bundle metadata manifest ID does not match manifest.json")
    if metadata.manifest_version != manifest.version:
        issues.append("Bundle metadata manifest version does not match manifest.json")
    for fairness_issue in condition_fairness_issues(manifest):
        issues.append(f"Condition fairness audit failed: {fairness_issue}")

    for index, record in enumerate(records, start=1):
        prefix = f"results row {index}"
        if record.bundle_id != metadata.bundle_id:
            issues.append(f"{prefix}: bundle_id does not match bundle metadata")
        if record.manifest_id != manifest.manifest_id:
            issues.append(f"{prefix}: manifest_id does not match manifest.json")
        if record.manifest_version != manifest.version:
            issues.append(f"{prefix}: manifest_version does not match manifest.json")
        if record.manifest_hash != expected_manifest_hash:
            issues.append(f"{prefix}: manifest_hash does not match manifest.json")
        if record.git_commit_sha != metadata.git_commit_sha:
            issues.append(f"{prefix}: git_commit_sha does not match bundle metadata")
        if record.git_dirty != metadata.git_dirty:
            issues.append(f"{prefix}: git_dirty does not match bundle metadata")
        if record.profile_name != metadata.profile_name:
            issues.append(f"{prefix}: profile_name does not match bundle metadata")
        if record.temperature != metadata.temperature:
            issues.append(f"{prefix}: temperature does not match bundle metadata")
        if record.max_tokens != metadata.max_tokens:
            issues.append(f"{prefix}: max_tokens does not match bundle metadata")
        if record.scenario not in manifest_scenarios:
            issues.append(f"{prefix}: scenario is not in the frozen manifest")
        if record.condition not in metadata.conditions:
            issues.append(f"{prefix}: condition is not listed in bundle metadata")
        if record.condition != "scripted_oracle" and record.model not in metadata.models:
            issues.append(f"{prefix}: model is not listed in bundle metadata")
        if record.scenario not in metadata.scenarios:
            issues.append(f"{prefix}: scenario is not listed in bundle metadata")
        if record.prompt_family_hash != prompt_family_hash_for_condition(record.condition):
            issues.append(f"{prefix}: prompt_family_hash does not match repo prompt files")
        if record.scenario_source_hash != scenario_source_hash_for_name(record.scenario):
            issues.append(f"{prefix}: scenario_source_hash does not match repo scenario source")
        if record.config_hash != config_hash_for_manifest_scenario(manifest, record.scenario):
            issues.append(f"{prefix}: config_hash does not match manifest scenario config")
        if record.steps_taken > 0 and not record.event_ids:
            issues.append(f"{prefix}: nonzero steps_taken but event_ids is empty")
        if record.steps_taken > 0 and not record.trace_ids:
            issues.append(f"{prefix}: nonzero steps_taken but trace_ids is empty")
        if not record.run_artifact_dir:
            issues.append(f"{prefix}: run_artifact_dir is missing")
            continue

        run_dir = bundle_path / record.run_artifact_dir
        run_meta_path = run_dir / "run_meta.json"
        steps_path = run_dir / "steps.jsonl"
        if not run_meta_path.exists():
            issues.append(f"{prefix}: run_meta.json is missing from run_artifact_dir")
            continue
        if not steps_path.exists():
            issues.append(f"{prefix}: steps.jsonl is missing from run_artifact_dir")
            continue
        normalized_trace_path = run_dir / NORMALIZED_TRACE_FILENAME

        run_meta = json.loads(run_meta_path.read_text())
        if run_meta.get("condition") != record.condition:
            issues.append(f"{prefix}: run_meta condition does not match result row")
        if run_meta.get("model") != record.model:
            issues.append(f"{prefix}: run_meta model does not match result row")
        if run_meta.get("scenario") != record.scenario:
            issues.append(f"{prefix}: run_meta scenario does not match result row")
        if int(run_meta.get("replication", -1)) != record.replication:
            issues.append(f"{prefix}: run_meta replication does not match result row")
        if run_meta.get("prompt_family_hash") != record.prompt_family_hash:
            issues.append(f"{prefix}: run_meta prompt_family_hash does not match result row")
        prompt_surface = run_meta.get("prompt_surface", {})
        if record.condition != "scripted_oracle" and record.prompt_tokens > 0:
            if not isinstance(prompt_surface, dict):
                issues.append(f"{prefix}: prompt_surface snapshot is missing or invalid")
            elif not prompt_surface.get("system_prompt"):
                issues.append(f"{prefix}: prompt_surface.system_prompt is missing")
            elif prompt_surface.get("first_turn_user_message") is None:
                issues.append(f"{prefix}: prompt_surface.first_turn_user_message is missing")
            elif (
                record.steps_taken > 0
                and record.condition not in _OPEN_VERB_CONDITIONS
                and prompt_surface.get("first_turn_tool_schema") is None
            ):
                issues.append(f"{prefix}: prompt_surface.first_turn_tool_schema is missing")

        step_lines = [line for line in steps_path.read_text().splitlines() if line.strip()]
        if record.prompt_tokens > 0 and not step_lines:
            issues.append(f"{prefix}: prompt-bearing run has empty steps.jsonl")
        if (
            record.condition != "scripted_oracle"
            and record.steps_taken > 0
            and record.prompt_tokens > 0
            and (
                run_meta.get("normalized_trace_file") is not None or normalized_trace_path.exists()
            )
        ):
            if not normalized_trace_path.exists():
                issues.append(f"{prefix}: {NORMALIZED_TRACE_FILENAME} is missing")
            else:
                normalized_trace_lines = [
                    line for line in normalized_trace_path.read_text().splitlines() if line.strip()
                ]
                if len(normalized_trace_lines) != record.steps_taken:
                    issues.append(
                        f"{prefix}: {NORMALIZED_TRACE_FILENAME} has "
                        f"{len(normalized_trace_lines)} rows but steps_taken is "
                        f"{record.steps_taken}"
                    )
                if run_meta.get("normalized_trace_row_count") != len(normalized_trace_lines):
                    issues.append(
                        f"{prefix}: run_meta normalized_trace_row_count does not match "
                        f"{NORMALIZED_TRACE_FILENAME}"
                    )

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary != expected_summary:
            issues.append("summary.json does not match the raw results.jsonl contents")

    return PaperBundleValidationReport(
        bundle_dir=str(bundle_path),
        valid=not issues,
        record_count=len(records),
        issues=issues,
    )


def assert_valid_paper_bundle(bundle_dir: str | Path) -> None:
    """Raise ValueError if a paper bundle is inconsistent."""

    report = validate_paper_bundle(bundle_dir)
    if not report.valid:
        raise ValueError("Invalid paper bundle:\n- " + "\n- ".join(report.issues))


__all__ = [
    "PaperBundleValidationReport",
    "assert_valid_paper_bundle",
    "validate_paper_bundle",
]
