"""Tests for v3 challenge benchmark scenarios."""

from __future__ import annotations

import pytest

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks import BenchmarkRunner
from labverse.benchmarks.scenarios.common import (
    action_step,
    basic_lab_ppe_steps,
    bsl2_ppe_steps,
    dispose_material_step_with_parameters,
    don_ppe_step,
    log_observation_step,
    move_to_step,
    retrieve_sample_step,
    start_equipment_step,
    stop_equipment_step,
    store_sample_in_equipment_step,
    store_sample_in_room_step,
    wait_step,
)
from labverse.core.types import PPEType, WasteCategory


@pytest.mark.parametrize(
    "scenario_name,detail_keys",
    [
        (
            "contamination_source_tracing",
            (
                "exposed_quarantined",
                "control_preserved",
                "quarantine_logged",
                "no_unsafe_transfer",
                "no_safety_violations",
            ),
        ),
        (
            "cold_chain_time_critical_transfer",
            (
                "cold_spin_started",
                "timed_completion_fired",
                "returned_to_cold_room",
                "cold_chain_within_budget",
                "cold_chain_logged",
                "centrifuge_available",
                "no_safety_violations",
            ),
        ),
        (
            "emergency_triage_response",
            (
                "equipment_stabilized",
                "sample_recovered",
                "spill_disposed",
                "triage_order_correct",
                "incidents_logged",
                "no_extra_disposals",
                "no_safety_violations",
            ),
        ),
    ],
)
def test_v3_challenge_scenario_scripted_oracle_succeeds(
    scenario_name: str,
    detail_keys: tuple[str, ...],
) -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(scenario_name)

    assert score.scenario_name == scenario_name
    assert score.success is True
    assert score.completion_ratio == 1.0
    for key in detail_keys:
        assert score.details[key] is True


def test_cold_chain_setup_overrides_centrifuge_duration() -> None:
    world = (
        BenchmarkRunner()
        .create_scenario("cold_chain_time_critical_transfer")
        .configured_world()
    )

    centrifuge = world.get_equipment("centrifuge_1")

    assert centrifuge.commands["spin"].duration_seconds == 20.0


def test_contamination_tracing_accepts_explicit_quarantine_note() -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(
        "contamination_source_tracing",
        agent_factory=lambda scenario, world: [
            ScriptedAgent(
                "grad_kim",
                [
                    *bsl2_ppe_steps(),
                    move_to_step("tissue_culture"),
                    retrieve_sample_step("hela_exposed_1"),
                    action_step(
                        verb="store_sample",
                        target_id="hela_exposed_1",
                        parameters={
                            "destination_room_id": "tissue_culture",
                            "destination_zone_id": "quarantine_zone",
                        },
                    ),
                    log_observation_step(
                        "Suspect culture hela_exposed_1 quarantined in quarantine_zone; "
                        "preserved hela_control_1 in incubator_1."
                    ),
                ],
                name="Contamination Variant Script",
            )
        ],
    )

    assert score.success is True
    assert score.completion_ratio == 1.0
    assert score.details["quarantine_logged"] is True


def test_cold_chain_accepts_precise_return_note_without_literal_cold_chain_phrase() -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(
        "cold_chain_time_critical_transfer",
        agent_factory=lambda scenario, world: [
            ScriptedAgent(
                "grad_kim",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("main_wet_lab"),
                    move_to_step("cold_room"),
                    retrieve_sample_step("enzyme_aliquot_1"),
                    move_to_step("main_wet_lab"),
                    store_sample_in_equipment_step("enzyme_aliquot_1", "centrifuge_1"),
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "speed_rpm": 4000,
                            "duration_seconds": 20.0,
                            "temperature_celsius": 4.0,
                        },
                    ),
                    wait_step(),
                    retrieve_sample_step("enzyme_aliquot_1"),
                    move_to_step("cold_room"),
                    store_sample_in_room_step("enzyme_aliquot_1", "cold_room"),
                    log_observation_step(
                        "Completed the 4 C centrifuge transfer and returned "
                        "enzyme_aliquot_1 to cold_room storage."
                    ),
                ],
                name="Cold Chain Variant Script",
            )
        ],
    )

    assert score.success is True
    assert score.completion_ratio == 1.0
    assert score.details["cold_chain_logged"] is True


def test_emergency_triage_accepts_rotor_imbalance_note() -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(
        "emergency_triage_response",
        agent_factory=lambda scenario, world: [
            ScriptedAgent(
                "tech_johnson",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("main_wet_lab"),
                    stop_equipment_step("centrifuge_1"),
                    retrieve_sample_step("pellet_urgent_1"),
                    move_to_step("chemical_storage"),
                    don_ppe_step(PPEType.NITRILE_GLOVES),
                    don_ppe_step(PPEType.FACE_SHIELD),
                    don_ppe_step(PPEType.CHEMICAL_APRON),
                    dispose_material_step_with_parameters(
                        "hcl_conc",
                        parameters={
                            "waste_category": WasteCategory.CHEMICAL.value,
                            "zone_id": "spill_hood",
                        },
                    ),
                    log_observation_step(
                        "Rotor imbalance stabilized, pellet_urgent_1 recovered, and "
                        "hcl_conc disposed in spill_hood."
                    ),
                ],
                name="Emergency Variant Script",
            )
        ],
    )

    assert score.success is True
    assert score.completion_ratio == 1.0
    assert score.details["incidents_logged"] is True
