"""Scientific-integrity artifacts for the frozen paper benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from labverse.benchmarks.initial_variants import (
    BASELINE_INITIAL_VARIANT_ID,
    DEFAULT_INITIAL_VARIANTS_PATH,
    load_initial_condition_variants,
)
from labverse.paper.manifest import REPO_ROOT, PaperManifest, load_paper_manifest
from labverse.paper.observability import (
    DEFAULT_OBSERVABILITY_MATRIX_PATH,
    load_scenario_observability_matrix,
)
from labverse.world.state import WorldState

DEFAULT_SCENARIO_INTEGRITY_MATRIX_PATH = REPO_ROOT / "paper" / "scenario_integrity_matrix.yaml"
DEFAULT_OBSERVABILITY_MATRIX_ARTIFACT_PATH = DEFAULT_OBSERVABILITY_MATRIX_PATH
DEFAULT_REALISM_AUDIT_RUBRIC_PATH = REPO_ROOT / "paper" / "expert_audit_rubric.yaml"
DEFAULT_REALISM_AUDIT_TEMPLATE_PATH = REPO_ROOT / "paper" / "expert_audit_template.csv"
DEFAULT_REALISM_AUDIT_SUMMARY_TEMPLATE_PATH = (
    REPO_ROOT / "paper" / "expert_audit_summary_template.md"
)
DEFAULT_INITIAL_CONDITION_VARIANTS_PATH = DEFAULT_INITIAL_VARIANTS_PATH


class ScenarioIntegrityEntry(BaseModel):
    """Scientific invariants for one frozen scenario."""

    scenario_name: str
    config_path: str
    task_family: str
    paper_role: str
    expected_best_condition: str
    critical_rooms: list[str] = Field(default_factory=list)
    critical_entities: list[str] = Field(default_factory=list)
    required_affordances: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    scorer_expectations: list[str] = Field(default_factory=list)
    notes: str = ""


class ScenarioIntegrityMatrix(BaseModel):
    """Canonical scientific-integrity matrix for the frozen paper benchmark."""

    manifest_id: str
    paper_scope: str
    scenarios: list[ScenarioIntegrityEntry] = Field(default_factory=list)

    def scenario_map(self) -> dict[str, ScenarioIntegrityEntry]:
        return {scenario.scenario_name: scenario for scenario in self.scenarios}


def default_scenario_integrity_matrix_path() -> Path:
    return DEFAULT_SCENARIO_INTEGRITY_MATRIX_PATH


def load_scenario_integrity_matrix(
    path: str | Path | None = None,
    *,
    manifest: PaperManifest | None = None,
) -> ScenarioIntegrityMatrix:
    manifest = manifest or load_paper_manifest()
    matrix_path = Path(path) if path is not None else default_scenario_integrity_matrix_path()
    payload = yaml.safe_load(matrix_path.read_text()) or {}
    matrix = ScenarioIntegrityMatrix.model_validate(payload)
    validate_scenario_integrity_matrix(matrix, manifest=manifest)
    return matrix


def validate_scenario_integrity_matrix(
    matrix: ScenarioIntegrityMatrix,
    *,
    manifest: PaperManifest,
) -> None:
    if matrix.manifest_id != manifest.manifest_id:
        raise ValueError(
            "Scenario integrity matrix manifest mismatch: "
            f"matrix={matrix.manifest_id}, manifest={manifest.manifest_id}"
        )

    manifest_scenarios = manifest.scenario_names()
    matrix_scenarios = [scenario.scenario_name for scenario in matrix.scenarios]
    if matrix_scenarios != manifest_scenarios:
        raise ValueError(
            "Scenario integrity matrix drifted from the frozen manifest: "
            f"matrix={matrix_scenarios}, manifest={manifest_scenarios}"
        )

    manifest_map = manifest.scenario_map()
    from labverse.benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner()
    for scenario in matrix.scenarios:
        manifest_scenario = manifest_map[scenario.scenario_name]
        benchmark_scenario = runner.create_scenario(scenario.scenario_name)
        world = benchmark_scenario.configured_world()
        valid_ids = _world_ids(world)
        if scenario.config_path != manifest_scenario.config_path:
            raise ValueError(
                "Scenario integrity config drift for "
                f"{scenario.scenario_name}: matrix={scenario.config_path}, "
                f"manifest={manifest_scenario.config_path}"
            )
        missing_rooms = [
            room_id for room_id in scenario.critical_rooms if room_id not in world.rooms
        ]
        if missing_rooms:
            raise ValueError(
                f"Scenario integrity entry {scenario.scenario_name} references missing rooms: "
                f"{', '.join(sorted(missing_rooms))}"
            )
        missing_entities = [
            entity_id for entity_id in scenario.critical_entities if entity_id not in valid_ids
        ]
        if missing_entities:
            raise ValueError(
                f"Scenario integrity entry {scenario.scenario_name} references missing entities: "
                f"{', '.join(sorted(missing_entities))}"
            )
        if not scenario.required_affordances:
            raise ValueError(
                f"Scenario integrity entry {scenario.scenario_name} is missing required_affordances"
            )
        if not scenario.scorer_expectations:
            raise ValueError(
                f"Scenario integrity entry {scenario.scenario_name} is missing scorer_expectations"
            )


def _world_ids(world: WorldState) -> set[str]:
    zone_ids = {zone.id for room in world.rooms.values() for zone in room.zones}
    return (
        set(world.rooms)
        | set(world.equipment)
        | set(world.materials)
        | set(world.samples)
        | set(world.persons)
        | {door.id for door in world.doors}
        | zone_ids
    )


def scientific_artifact_hashes() -> dict[str, str]:
    """Hash paper-facing reproduction artifacts tracked in the public repo.

    Raises :class:`FileNotFoundError` if any artifact is missing. Silently
    skipping missing files would make hash-drift comparisons useless: both
    sides would agree on a shorter dict and real drift would be invisible to
    downstream validators like :func:`labverse.paper.validate.validate_paper_bundle`.
    """

    artifact_paths = [
        DEFAULT_SCENARIO_INTEGRITY_MATRIX_PATH,
        DEFAULT_OBSERVABILITY_MATRIX_ARTIFACT_PATH,
        DEFAULT_REALISM_AUDIT_RUBRIC_PATH,
        DEFAULT_REALISM_AUDIT_TEMPLATE_PATH,
        DEFAULT_REALISM_AUDIT_SUMMARY_TEMPLATE_PATH,
        DEFAULT_INITIAL_CONDITION_VARIANTS_PATH,
    ]
    hashes: dict[str, str] = {}
    for path in artifact_paths:
        display_name = (
            str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        )
        if not path.exists():
            raise FileNotFoundError(f"Scientific artifact missing: {display_name}")
        hashes[display_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_scientific_artifacts(
    *,
    manifest: PaperManifest | None = None,
) -> list[str]:
    """Return scientific-artifact issues that should block cited paper runs."""

    manifest = manifest or load_paper_manifest()
    issues: list[str] = []
    required_paths = [
        DEFAULT_SCENARIO_INTEGRITY_MATRIX_PATH,
        DEFAULT_OBSERVABILITY_MATRIX_ARTIFACT_PATH,
        DEFAULT_REALISM_AUDIT_RUBRIC_PATH,
        DEFAULT_REALISM_AUDIT_TEMPLATE_PATH,
        DEFAULT_REALISM_AUDIT_SUMMARY_TEMPLATE_PATH,
        DEFAULT_INITIAL_CONDITION_VARIANTS_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            issues.append(f"Missing scientific artifact: {path.relative_to(REPO_ROOT)}")

    if issues:
        return issues

    try:
        load_scenario_integrity_matrix(manifest=manifest)
    except Exception as exc:
        issues.append(f"Scenario integrity matrix is invalid: {exc}")
    try:
        load_scenario_observability_matrix(manifest=manifest)
    except Exception as exc:
        issues.append(f"Scenario observability matrix is invalid: {exc}")
    try:
        registry = load_initial_condition_variants()
        if BASELINE_INITIAL_VARIANT_ID not in registry.variants:
            issues.append("Initial-condition variants are missing the baseline variant.")
    except Exception as exc:
        issues.append(f"Initial-condition variants are invalid: {exc}")
    return issues


__all__ = [
    "DEFAULT_OBSERVABILITY_MATRIX_ARTIFACT_PATH",
    "DEFAULT_REALISM_AUDIT_RUBRIC_PATH",
    "DEFAULT_REALISM_AUDIT_SUMMARY_TEMPLATE_PATH",
    "DEFAULT_REALISM_AUDIT_TEMPLATE_PATH",
    "DEFAULT_SCENARIO_INTEGRITY_MATRIX_PATH",
    "DEFAULT_INITIAL_CONDITION_VARIANTS_PATH",
    "ScenarioIntegrityEntry",
    "ScenarioIntegrityMatrix",
    "default_scenario_integrity_matrix_path",
    "load_scenario_integrity_matrix",
    "scientific_artifact_hashes",
    "validate_scenario_integrity_matrix",
    "validate_scientific_artifacts",
]
