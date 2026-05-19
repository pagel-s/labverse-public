"""Tests for frozen paper scientific-integrity artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from labverse.paper.integrity import (
    load_scenario_integrity_matrix,
    scientific_artifact_hashes,
    validate_scientific_artifacts,
)
from labverse.paper.manifest import load_paper_manifest


def test_scenario_integrity_matrix_matches_frozen_manifest() -> None:
    manifest = load_paper_manifest()
    matrix = load_scenario_integrity_matrix(manifest=manifest)

    assert matrix.manifest_id == manifest.manifest_id
    assert list(matrix.scenario_map()) == manifest.scenario_names()


def test_scientific_artifacts_exist_and_validate() -> None:
    assert validate_scientific_artifacts() == []
    hashes = scientific_artifact_hashes()

    assert "paper/scenario_integrity_matrix.yaml" in hashes
    assert "paper/scenario_observability_matrix.yaml" in hashes
    assert "paper/initial_condition_variants.yaml" in hashes


def test_scientific_artifact_hashes_raises_on_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: ``scientific_artifact_hashes`` must fail loudly if any tracked
    artifact disappears. Silently eliding missing files would let preflight and
    post-hoc validate agree on a shorter dict, masking real drift."""

    missing_path = tmp_path / "does-not-exist.yaml"
    monkeypatch.setattr(
        "labverse.paper.integrity.DEFAULT_REALISM_AUDIT_RUBRIC_PATH",
        missing_path,
    )

    with pytest.raises(FileNotFoundError):
        scientific_artifact_hashes()
