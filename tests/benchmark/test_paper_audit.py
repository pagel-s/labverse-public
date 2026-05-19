"""Tests for the launch-blocking paper scientific audit."""

from __future__ import annotations

from labverse.paper.audit import (
    BundleFailureExplanation,
    ScenarioScientificAudit,
    _audit_scenario,
    _build_scenario_failure_dossiers,
)
from labverse.paper.manifest import load_paper_manifest
from labverse.paper.observability import load_scenario_observability_matrix
from labverse.paper.runner import PaperRunRecord


def test_scenario_observability_matrix_matches_frozen_manifest() -> None:
    manifest = load_paper_manifest()
    matrix = load_scenario_observability_matrix(manifest=manifest)

    assert matrix.manifest_id == manifest.manifest_id
    assert [entry.scenario_name for entry in matrix.scenarios] == manifest.scenario_names()


def test_scientific_audit_scenario_passes_for_chemical_spill_response() -> None:
    matrix = load_scenario_observability_matrix()
    report = _audit_scenario(
        scenario_name="chemical_spill_response",
        observability_entry=matrix.scenario_map()["chemical_spill_response"],
    )

    assert report.oracle_success is True
    assert report.observability_contract_ok is True
    assert report.interactive_prompt_parity_ok is True
    assert report.one_shot_prompt_contract_ok is True
    assert report.timed_process_contract_ok is True
    assert report.issues == []


def test_scientific_audit_scenario_passes_for_shared_equipment_scheduling() -> None:
    matrix = load_scenario_observability_matrix()
    report = _audit_scenario(
        scenario_name="shared_equipment_scheduling",
        observability_entry=matrix.scenario_map()["shared_equipment_scheduling"],
    )

    labels = {milestone.label for milestone in report.milestones}

    assert report.oracle_success is True
    assert report.observability_contract_ok is True
    assert report.interactive_prompt_parity_ok is True
    assert report.one_shot_prompt_contract_ok is True
    assert report.timed_process_contract_ok is True
    assert report.issues == []
    assert "turn_0" in labels
    assert "after_first_timed_start" in labels
    assert "after_first_timed_completion" in labels


def test_failure_dossier_prefers_successful_record_for_representative_success(
    monkeypatch,
    tmp_path,
) -> None:
    success_record = PaperRunRecord(
        bundle_id="bundle",
        manifest_id="labverse-embodiment-freeze-v3",
        manifest_version="3.0.0",
        manifest_hash="manifest-hash",
        git_commit_sha="abc123",
        git_dirty=False,
        profile_name="deterministic",
        temperature=0.0,
        max_tokens=4096,
        prompt_family_hash="prompt-hash",
        scenario_source_hash="scenario-hash",
        config_hash="config-hash",
        condition="local_narrative_interactive_grounded",
        model="ollama/gemma4:31b",
        provider="ollama",
        scenario="branch_selection_under_uncertainty",
        replication=1,
        order_index=1,
        success=True,
        completion_ratio=1.0,
        steps_taken=9,
        time_elapsed_seconds=1.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=1.0,
        prompt_tokens=100,
        completion_tokens=20,
        wall_time_seconds=1.0,
        timestamp="2026-04-21T00:00:00Z",
        run_artifact_dir="runs/local_narrative_interactive_grounded/ollama__gemma4_31b/branch_selection_under_uncertainty/rep_1",
    )
    failed_record = success_record.model_copy(
        update={
            "condition": "global_one_shot_planner",
            "success": False,
            "completion_ratio": 0.85,
            "run_artifact_dir": (
                "runs/global_one_shot_planner/ollama__gemma4_31b/"
                "branch_selection_under_uncertainty/rep_1"
            ),
        }
    )

    monkeypatch.setattr(
        "labverse.paper.audit.load_paper_run_records",
        lambda _path: [failed_record, success_record],
    )

    dossiers = _build_scenario_failure_dossiers(
        bundle_path=tmp_path,
        failure_explanations=[
            BundleFailureExplanation(
                row_id="global_one_shot_planner/ollama/gemma4:31b/branch_selection_under_uncertainty/rep_1",
                scenario="branch_selection_under_uncertainty",
                condition="global_one_shot_planner",
                model="ollama/gemma4:31b",
                classification="model_behavior_failure",
                explanation=(
                    "The run remained within the visible contract but did not satisfy "
                    "the scenario success criteria."
                ),
                run_artifact_dir=failed_record.run_artifact_dir,
            )
        ],
        scenario_reports=[
            ScenarioScientificAudit(
                scenario_name="branch_selection_under_uncertainty",
                oracle_success=True,
                observability_contract_ok=True,
                interactive_prompt_parity_ok=True,
                one_shot_prompt_contract_ok=True,
                timed_process_contract_ok=True,
            )
        ],
    )

    assert dossiers[0].representative_success_run == success_record.run_artifact_dir
    assert dossiers[0].representative_failure_run == failed_record.run_artifact_dir
