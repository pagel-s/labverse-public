"""Tests for manuscript writeup packet generation."""

from __future__ import annotations

from pathlib import Path

from labverse.paper import build_writeup_packet, generate_paper_writeup, run_paper_experiment


def test_writeup_packet_flags_oracle_bundle_as_not_ready_for_main_results(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    packet = build_writeup_packet(bundle_dir)

    assert not packet.ready_for_main_results
    assert any("Missing primary paper conditions" in note for note in packet.readiness_notes)


def test_generate_paper_writeup_emits_expected_assets(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    run_paper_experiment(
        conditions=["scripted_oracle"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    writeup_dir = generate_paper_writeup(bundle_dir)

    assert (writeup_dir / "writeup_packet.json").exists()
    assert (writeup_dir / "tables" / "overall_condition_results.csv").exists()
    assert (writeup_dir / "tables" / "per_scenario_results.csv").exists()
    assert (writeup_dir / "tables" / "per_model_results.csv").exists()
    assert (writeup_dir / "tables" / "task_family_results.csv").exists()
    assert (writeup_dir / "tables" / "paper_role_results.csv").exists()
    assert (writeup_dir / "tables" / "scenario_judgment_matrix.csv").exists()
    assert (writeup_dir / "tables" / "pairwise_condition_deltas.csv").exists()
    assert (writeup_dir / "figures" / "taxonomy_results.csv").exists()
