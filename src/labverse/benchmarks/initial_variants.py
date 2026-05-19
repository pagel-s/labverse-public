"""Initial-condition overlays for benchmark sensitivity runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from labverse.benchmarks.scenarios.base import BenchmarkScenario
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import Sample
from labverse.world.state import WorldState

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INITIAL_VARIANTS_PATH = REPO_ROOT / "paper" / "initial_condition_variants.yaml"
BASELINE_INITIAL_VARIANT_ID = "baseline"


class LabInitialOverlay(BaseModel):
    """Nuisance-state additions for one lab config."""

    add_materials: list[dict[str, Any]] = Field(default_factory=list)
    add_samples: list[dict[str, Any]] = Field(default_factory=list)
    add_persons: list[dict[str, Any]] = Field(default_factory=list)


class InitialConditionVariant(BaseModel):
    """One named initial-condition overlay."""

    description: str = ""
    lab_overlays: dict[str, LabInitialOverlay] = Field(default_factory=dict)


class InitialConditionVariantRegistry(BaseModel):
    """YAML-backed registry of initial-condition overlays."""

    schema_version: int
    variants: dict[str, InitialConditionVariant]


def load_initial_condition_variants(
    path: str | Path | None = None,
) -> InitialConditionVariantRegistry:
    """Load the initial-condition variant registry."""

    variant_path = Path(path) if path is not None else DEFAULT_INITIAL_VARIANTS_PATH
    payload = yaml.safe_load(variant_path.read_text()) or {}
    return InitialConditionVariantRegistry.model_validate(payload)


def initial_condition_variant_hash(
    variant_id: str,
    *,
    path: str | Path | None = None,
) -> str:
    """Return a stable hash for one variant definition."""

    registry = load_initial_condition_variants(path)
    variant = _variant_by_id(registry, variant_id)
    canonical = json.dumps(
        variant.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_initial_condition_variant(
    scenario: BenchmarkScenario,
    world: WorldState,
    variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    *,
    path: str | Path | None = None,
) -> WorldState:
    """Apply a named nuisance-state overlay to a configured scenario world."""

    registry = load_initial_condition_variants(path)
    variant = _variant_by_id(registry, variant_id)
    overlay = variant.lab_overlays.get(scenario.lab_config_name)
    if overlay is None:
        return world

    for raw_material in overlay.add_materials:
        world.add_material(Material.model_validate(raw_material))
    for raw_sample in overlay.add_samples:
        world.add_sample(Sample.model_validate(raw_sample))
    for raw_person in overlay.add_persons:
        world.add_person(Person.model_validate(raw_person))
    return world


def _variant_by_id(
    registry: InitialConditionVariantRegistry,
    variant_id: str,
) -> InitialConditionVariant:
    try:
        return registry.variants[variant_id]
    except KeyError as exc:
        available = ", ".join(sorted(registry.variants))
        raise ValueError(
            f"Unknown initial-condition variant {variant_id!r}; available: {available}"
        ) from exc


__all__ = [
    "BASELINE_INITIAL_VARIANT_ID",
    "DEFAULT_INITIAL_VARIANTS_PATH",
    "InitialConditionVariant",
    "InitialConditionVariantRegistry",
    "LabInitialOverlay",
    "apply_initial_condition_variant",
    "initial_condition_variant_hash",
    "load_initial_condition_variants",
]
