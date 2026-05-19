"""Canonical scenario judgment metadata for the paper narrative."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from labverse.paper.manifest import REPO_ROOT, PaperManifest, load_paper_manifest

DEFAULT_JUDGMENT_MATRIX_PATH = REPO_ROOT / "paper" / "scenario_judgment_matrix.json"


class ScenarioJudgment(BaseModel):
    """Narrative role metadata for one frozen paper scenario."""

    scenario_name: str
    task_family: str
    paper_role: str
    expected_best_condition: str
    why_this_task_exists: str
    main_claim_support_level: str


class ScenarioJudgmentMatrix(BaseModel):
    """Canonical scenario judgment matrix for the paper."""

    manifest_id: str
    paper_scope: str
    scenarios: list[ScenarioJudgment] = Field(default_factory=list)

    def scenario_map(self) -> dict[str, ScenarioJudgment]:
        return {scenario.scenario_name: scenario for scenario in self.scenarios}

    def task_families(self) -> list[str]:
        return sorted({scenario.task_family for scenario in self.scenarios})

    def paper_roles(self) -> list[str]:
        return sorted({scenario.paper_role for scenario in self.scenarios})


def default_judgment_matrix_path() -> Path:
    return DEFAULT_JUDGMENT_MATRIX_PATH


def load_scenario_judgment_matrix(
    path: str | Path | None = None,
    *,
    manifest: PaperManifest | None = None,
) -> ScenarioJudgmentMatrix:
    """Load the canonical paper scenario judgment matrix and validate it."""

    manifest = manifest or load_paper_manifest()
    matrix_path = Path(path) if path is not None else default_judgment_matrix_path()
    matrix = ScenarioJudgmentMatrix.model_validate(json.loads(matrix_path.read_text()))
    validate_scenario_judgment_matrix(matrix, manifest=manifest)
    return matrix


def validate_scenario_judgment_matrix(
    matrix: ScenarioJudgmentMatrix,
    *,
    manifest: PaperManifest,
) -> None:
    """Ensure the judgment matrix stays in lock-step with the frozen manifest."""

    if matrix.manifest_id != manifest.manifest_id:
        raise ValueError(
            "Scenario judgment matrix manifest mismatch: "
            f"matrix={matrix.manifest_id}, manifest={manifest.manifest_id}"
        )

    manifest_scenarios = manifest.scenario_names()
    matrix_scenarios = [scenario.scenario_name for scenario in matrix.scenarios]
    if matrix_scenarios != manifest_scenarios:
        raise ValueError(
            "Scenario judgment matrix drifted from the frozen manifest: "
            f"matrix={matrix_scenarios}, manifest={manifest_scenarios}"
        )


__all__ = [
    "DEFAULT_JUDGMENT_MATRIX_PATH",
    "ScenarioJudgment",
    "ScenarioJudgmentMatrix",
    "default_judgment_matrix_path",
    "load_scenario_judgment_matrix",
    "validate_scenario_judgment_matrix",
]
