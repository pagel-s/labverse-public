"""Tests for benchmark initial-condition overlays."""

from __future__ import annotations

import pytest

from labverse.benchmarks.initial_variants import (
    BASELINE_INITIAL_VARIANT_ID,
    apply_initial_condition_variant,
    initial_condition_variant_hash,
    load_initial_condition_variants,
)
from labverse.benchmarks.runner import BenchmarkRunner


def test_initial_condition_registry_contains_baseline_and_sensitivity_variant() -> None:
    registry = load_initial_condition_variants()

    assert BASELINE_INITIAL_VARIANT_ID in registry.variants
    assert "sensitivity_v1" in registry.variants
    assert "sensitivity_v2" in registry.variants


def test_baseline_initial_condition_variant_is_noop_for_configured_world() -> None:
    scenario = BenchmarkRunner().create_scenario("pcr_setup")
    world = scenario.configured_world()
    counts_before = world.entity_counts

    apply_initial_condition_variant(scenario, world, BASELINE_INITIAL_VARIANT_ID)

    assert world.entity_counts == counts_before


def test_sensitivity_initial_condition_variant_adds_only_distractor_entities() -> None:
    scenario = BenchmarkRunner().create_scenario("pcr_setup")
    world = apply_initial_condition_variant(
        scenario,
        scenario.configured_world(),
        "sensitivity_v1",
    )

    assert "variant_saline_1" in world.materials
    assert "variant_empty_tip_box_1" in world.materials
    assert "variant_qc_tube_1" in world.samples
    assert "variant_observer_1" in world.persons


def test_second_sensitivity_initial_condition_variant_adds_only_distractor_entities() -> None:
    scenario = BenchmarkRunner().create_scenario("pcr_setup")
    world = apply_initial_condition_variant(
        scenario,
        scenario.configured_world(),
        "sensitivity_v2",
    )

    assert "variant_water_2" in world.materials
    assert "variant_label_roll_2" in world.materials
    assert "variant_archive_sample_2" in world.samples
    assert "variant_visitor_2" in world.persons


def test_unknown_initial_condition_variant_fails_with_available_ids() -> None:
    scenario = BenchmarkRunner().create_scenario("pcr_setup")

    with pytest.raises(ValueError, match="available: baseline, sensitivity_v1, sensitivity_v2"):
        apply_initial_condition_variant(
            scenario,
            scenario.configured_world(),
            "missing_variant",
        )


def test_initial_condition_variant_hash_is_stable_sha256() -> None:
    digest = initial_condition_variant_hash("sensitivity_v1")

    assert len(digest) == 64
    assert digest == initial_condition_variant_hash("sensitivity_v1")
