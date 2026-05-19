"""Tests for offline paper bundle validation."""

from __future__ import annotations

import json
from pathlib import Path

from labverse.paper import run_paper_experiment, validate_paper_bundle


def test_validate_paper_bundle_accepts_fresh_scripted_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    report = validate_paper_bundle(bundle_dir)

    assert report.valid
    assert report.record_count == 1
    assert report.issues == []


def test_validate_paper_bundle_detects_summary_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["chemical_spill_response"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )
    (bundle_dir / "summary.json").write_text(json.dumps({"total_runs": 999}) + "\n")

    report = validate_paper_bundle(bundle_dir)

    assert not report.valid
    assert any("summary.json does not match" in issue for issue in report.issues)
