"""Observable-success metadata for the frozen paper benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from labverse.benchmarks.runner import BenchmarkRunner
from labverse.paper.manifest import REPO_ROOT, PaperManifest, load_paper_manifest

DEFAULT_OBSERVABILITY_MATRIX_PATH = REPO_ROOT / "paper" / "scenario_observability_matrix.yaml"
_NOTE_OUTPUT_MARKERS = ("log", "note", "report", "finding", "document")

ObservableEvidenceSource = Literal[
    "event_log",
    "world_state",
    "timed_event",
    "semantic_note",
    "derived_metric",
]


class ScenarioObservableCriterion(BaseModel):
    """One scorer criterion that must be observable from the paper contract."""

    detail_key: str
    expected_value: bool | int | float | str
    evidence_source: ObservableEvidenceSource
    rationale: str = ""


class ScenarioTimedProcessExpectation(BaseModel):
    """Timed command that must be exposed truthfully to the model."""

    equipment_id: str
    command_name: str
    expected_duration_seconds: float
    completion_event_required: bool = True


class ScenarioObservabilityEntry(BaseModel):
    """Observable-success contract for one frozen scenario."""

    scenario_name: str
    canonical_success_path: list[str] = Field(default_factory=list)
    full_success_criteria: list[ScenarioObservableCriterion] = Field(default_factory=list)
    timed_processes: list[ScenarioTimedProcessExpectation] = Field(default_factory=list)
    plausible_wrong_branches: list[str] = Field(default_factory=list)
    condition_visibility_note: str = ""


class ScenarioObservabilityMatrix(BaseModel):
    """Canonical observability matrix for the frozen paper benchmark."""

    manifest_id: str
    paper_scope: str
    scenarios: list[ScenarioObservabilityEntry] = Field(default_factory=list)

    def scenario_map(self) -> dict[str, ScenarioObservabilityEntry]:
        return {scenario.scenario_name: scenario for scenario in self.scenarios}


def default_observability_matrix_path() -> Path:
    return DEFAULT_OBSERVABILITY_MATRIX_PATH


def load_scenario_observability_matrix(
    path: str | Path | None = None,
    *,
    manifest: PaperManifest | None = None,
) -> ScenarioObservabilityMatrix:
    """Load and validate the canonical observability matrix."""

    manifest = manifest or load_paper_manifest()
    matrix_path = Path(path) if path is not None else default_observability_matrix_path()
    payload = yaml.safe_load(matrix_path.read_text()) or {}
    matrix = ScenarioObservabilityMatrix.model_validate(payload)
    validate_scenario_observability_matrix(matrix, manifest=manifest)
    return matrix


def validate_scenario_observability_matrix(
    matrix: ScenarioObservabilityMatrix,
    *,
    manifest: PaperManifest,
) -> None:
    """Ensure the observability matrix stays aligned with the frozen manifest and scorers."""

    if matrix.manifest_id != manifest.manifest_id:
        raise ValueError(
            "Scenario observability matrix manifest mismatch: "
            f"matrix={matrix.manifest_id}, manifest={manifest.manifest_id}"
        )

    manifest_scenarios = manifest.scenario_names()
    matrix_scenarios = [scenario.scenario_name for scenario in matrix.scenarios]
    if matrix_scenarios != manifest_scenarios:
        raise ValueError(
            "Scenario observability matrix drifted from the frozen manifest: "
            f"matrix={matrix_scenarios}, manifest={manifest_scenarios}"
        )

    runner = BenchmarkRunner()
    for entry in matrix.scenarios:
        if not entry.canonical_success_path:
            raise ValueError(
                f"Scenario observability entry {entry.scenario_name} is missing "
                "canonical_success_path"
            )
        if not entry.full_success_criteria:
            raise ValueError(
                f"Scenario observability entry {entry.scenario_name} is missing "
                "full_success_criteria"
            )

        scenario = runner.create_scenario(entry.scenario_name)
        capture = runner.run_scenario_capture(scenario)
        if not capture.score.success:
            raise ValueError(
                f"Scenario observability oracle failed for {entry.scenario_name}: "
                f"{capture.score.details.get('runner_error') or 'unknown failure'}"
            )
        details = capture.score.details
        missing_detail_keys = [
            criterion.detail_key
            for criterion in entry.full_success_criteria
            if criterion.detail_key not in details
        ]
        if missing_detail_keys:
            raise ValueError(
                f"Scenario observability entry {entry.scenario_name} references unknown "
                f"detail keys: {', '.join(sorted(missing_detail_keys))}"
            )
        mismatched = [
            criterion.detail_key
            for criterion in entry.full_success_criteria
            if details[criterion.detail_key] != criterion.expected_value
        ]
        if mismatched:
            raise ValueError(
                f"Scenario observability entry {entry.scenario_name} does not match oracle "
                f"success details: {', '.join(sorted(mismatched))}"
            )

        task_spec = scenario.task_spec()
        lower_required_text = "\n".join(
            [*task_spec.known_requirements, *task_spec.required_outputs]
        ).casefold()
        for criterion in entry.full_success_criteria:
            if criterion.evidence_source != "semantic_note":
                continue
            if not any(marker in lower_required_text for marker in _NOTE_OUTPUT_MARKERS):
                raise ValueError(
                    f"Scenario observability entry {entry.scenario_name} marks "
                    f"{criterion.detail_key} as semantic_note, but the task contract does not "
                    "explicitly require a note/report/finding."
                )

        world = scenario.configured_world()
        for timed_process in entry.timed_processes:
            if timed_process.equipment_id not in world.equipment:
                raise ValueError(
                    f"Scenario observability entry {entry.scenario_name} references missing "
                    f"equipment {timed_process.equipment_id}"
                )
            equipment = world.get_equipment(timed_process.equipment_id)
            if timed_process.command_name not in equipment.commands:
                raise ValueError(
                    f"Scenario observability entry {entry.scenario_name} references missing "
                    f"command {timed_process.command_name} on {timed_process.equipment_id}"
                )
            duration = equipment.commands[timed_process.command_name].duration_seconds
            if duration is None:
                raise ValueError(
                    f"Scenario observability entry {entry.scenario_name} expects fixed duration "
                    f"for {timed_process.equipment_id}.{timed_process.command_name}, got None"
                )
            if abs(duration - timed_process.expected_duration_seconds) > 1e-9:
                raise ValueError(
                    f"Scenario observability entry {entry.scenario_name} expects "
                    f"{timed_process.equipment_id}.{timed_process.command_name} duration "
                    f"{timed_process.expected_duration_seconds}, got {duration}"
                )


__all__ = [
    "DEFAULT_OBSERVABILITY_MATRIX_PATH",
    "ObservableEvidenceSource",
    "ScenarioObservableCriterion",
    "ScenarioObservabilityEntry",
    "ScenarioObservabilityMatrix",
    "ScenarioTimedProcessExpectation",
    "default_observability_matrix_path",
    "load_scenario_observability_matrix",
    "validate_scenario_observability_matrix",
]
