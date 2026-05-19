"""Tests for the frozen paper manifest."""

from __future__ import annotations

from labverse.paper import load_paper_manifest, manifest_sha256


def test_paper_manifest_loads_and_matches_frozen_v3_scenarios() -> None:
    manifest = load_paper_manifest()

    assert manifest.manifest_id == "labverse-embodiment-freeze-v3"
    assert manifest.primary_conditions == [
        "local_structured_interactive_grounded",
        "local_narrative_interactive_grounded",
        "global_structured_interactive_grounded",
        "global_one_shot_planner",
    ]
    assert manifest.appendix_conditions == ["scripted_oracle"]
    assert manifest.action_surface_semantics == "affordance_only"
    # High-stakes scenarios opt into BLOCK-level enforcement for the
    # constraints that define them.
    assert (
        manifest.constraint_policy_per_scenario["inventory_management"]["material_expiry"]
        == "block"
    )
    assert (
        manifest.constraint_policy_per_scenario["cell_culture_passage"]["sample_contamination"]
        == "block"
    )
    assert len(manifest.scenarios) == 14
    assert manifest.scenario_names() == [
        "pcr_setup",
        "reagent_preparation",
        "cell_culture_passage",
        "chemical_spill_response",
        "equipment_failure_recovery",
        "safety_violation_detection",
        "inventory_management",
        "sample_tracking",
        "multi_step_protocol",
        "shared_equipment_scheduling",
        "branch_selection_under_uncertainty",
        "contamination_source_tracing",
        "cold_chain_time_critical_transfer",
        "emergency_triage_response",
    ]
    assert manifest.scenario_names()[0] == "pcr_setup"


def test_manifest_hash_is_stable_for_loaded_manifest() -> None:
    manifest = load_paper_manifest()

    digest = manifest_sha256(manifest)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
