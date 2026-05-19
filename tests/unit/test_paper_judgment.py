"""Tests for the canonical paper scenario judgment matrix."""

from __future__ import annotations

from labverse.paper import load_paper_manifest, load_scenario_judgment_matrix


def test_scenario_judgment_matrix_matches_frozen_manifest() -> None:
    manifest = load_paper_manifest()
    matrix = load_scenario_judgment_matrix(manifest=manifest)

    assert matrix.manifest_id == manifest.manifest_id
    assert matrix.paper_scope == "interactive_grounded_simulation_beats_one_shot_text_planning"
    assert [scenario.scenario_name for scenario in matrix.scenarios] == manifest.scenario_names()


def test_scenario_judgment_matrix_contains_expected_flagships_and_boundary_case() -> None:
    matrix = load_scenario_judgment_matrix()
    judgments = matrix.scenario_map()

    assert judgments["chemical_spill_response"].paper_role == "flagship"
    assert judgments["shared_equipment_scheduling"].paper_role == "flagship"
    assert judgments["safety_violation_detection"].paper_role == "boundary_case"
    assert judgments["pcr_setup"].paper_role == "anchor"
