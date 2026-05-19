"""Frozen paper manifest loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from labverse.benchmarks.runner import BenchmarkRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "paper" / "embodiment_freeze_v3.json"


class PaperScenarioManifest(BaseModel):
    """Frozen scenario metadata for the embodiment benchmark."""

    name: str
    config_path: str
    max_steps: int
    optimal_steps: int
    spatial_complexity: str
    safety_complexity: str
    category: str
    why_embodiment_matters: str


class PaperProfile(BaseModel):
    """Execution profile for one paper evaluation mode."""

    replications: int
    temperature: float
    max_tokens: int
    retry_limit: int
    description: str


class PaperManifest(BaseModel):
    """Top-level frozen paper manifest."""

    manifest_id: str
    version: str
    track: str
    primary_claim: str
    description: str
    primary_conditions: list[str]
    appendix_conditions: list[str] = Field(default_factory=list)
    primary_metrics: list[str]
    secondary_metrics: list[str] = Field(default_factory=list)
    profiles: dict[str, PaperProfile]
    scenarios: list[PaperScenarioManifest]
    # Manifest additions after v1 all have defaults so archived JSON still validates.
    action_surface_semantics: str = "preview_filtered"
    constraint_policy_per_scenario: dict[str, dict[str, str]] = Field(default_factory=dict)
    tool_schema_parity_note: str = ""
    token_budget_warning_thresholds: dict[str, dict[str, int]] = Field(default_factory=dict)

    def scenario_names(self) -> list[str]:
        return [scenario.name for scenario in self.scenarios]

    def scenario_map(self) -> dict[str, PaperScenarioManifest]:
        return {scenario.name: scenario for scenario in self.scenarios}


def default_manifest_path() -> Path:
    return DEFAULT_MANIFEST_PATH


def load_paper_manifest(path: str | Path | None = None) -> PaperManifest:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    payload = json.loads(manifest_path.read_text())
    manifest = PaperManifest.model_validate(payload)
    validate_manifest_against_repo(manifest)
    return manifest


def manifest_sha256(manifest: PaperManifest) -> str:
    canonical = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_manifest_against_repo(manifest: PaperManifest) -> None:
    """Verify the frozen manifest still matches the live scenario registry."""

    runner = BenchmarkRunner()
    registered = set(runner.list_scenarios())
    missing = [name for name in manifest.scenario_names() if name not in registered]
    if missing:
        raise ValueError(
            f"Frozen paper scenarios are missing from the registry: {', '.join(sorted(missing))}"
        )

    for scenario_manifest in manifest.scenarios:
        scenario = runner.create_scenario(scenario_manifest.name)
        expected_config = f"configs/labs/{scenario.lab_config_name}"
        if scenario_manifest.config_path != expected_config:
            raise ValueError(
                f"Scenario {scenario_manifest.name} config mismatch: "
                f"manifest={scenario_manifest.config_path}, repo={expected_config}"
            )
        if scenario_manifest.max_steps != scenario.get_max_steps():
            raise ValueError(
                f"Scenario {scenario_manifest.name} max_steps mismatch: "
                f"manifest={scenario_manifest.max_steps}, repo={scenario.get_max_steps()}"
            )
        if scenario_manifest.optimal_steps != scenario.get_optimal_steps():
            raise ValueError(
                f"Scenario {scenario_manifest.name} optimal_steps mismatch: "
                f"manifest={scenario_manifest.optimal_steps}, repo={scenario.get_optimal_steps()}"
            )
        manifest_policy = manifest.constraint_policy_per_scenario.get(
            scenario_manifest.name,
            {},
        )
        scenario_policy = {
            constraint_id: verdict.value
            for constraint_id, verdict in scenario.constraint_policy.items()
        }
        if manifest_policy != scenario_policy:
            raise ValueError(
                f"Scenario {scenario_manifest.name} constraint policy mismatch: "
                f"manifest={manifest_policy}, repo={scenario_policy}"
            )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "PaperManifest",
    "PaperProfile",
    "PaperScenarioManifest",
    "default_manifest_path",
    "load_paper_manifest",
    "manifest_sha256",
    "validate_manifest_against_repo",
]
