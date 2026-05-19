"""Tests for the branch-selection benchmark scenario."""

from __future__ import annotations

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks import BenchmarkRunner
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    log_observation_step,
    move_to_step,
    retrieve_sample_step,
    start_equipment_step,
    store_sample_in_equipment_step,
    wait_step,
)


def test_branch_selection_setup_overrides_centrifuge_duration() -> None:
    world = (
        BenchmarkRunner()
        .create_scenario("branch_selection_under_uncertainty")
        .configured_world()
    )

    centrifuge = world.get_equipment("centrifuge_1")

    assert centrifuge.commands["spin"].duration_seconds == 20.0


def test_branch_selection_under_uncertainty_is_registered_and_scripted_passes() -> None:
    runner = BenchmarkRunner()

    assert "branch_selection_under_uncertainty" in runner.list_scenarios()

    score = runner.run_scenario("branch_selection_under_uncertainty")

    assert score.scenario_name == "branch_selection_under_uncertainty"
    assert score.success is True
    assert score.completion_ratio == 1.0
    assert score.details["scarce_instrument_used_once"] is True
    assert score.details["triage_sample_loaded"] is True
    assert score.details["sample_loaded_before_spin"] is True
    assert score.details["centrifuge_completed"] is True
    assert score.details["avoided_pcr_repeat"] is True
    assert score.details["rationale_logged"] is True


def test_branch_selection_accepts_valid_rationale_without_final_sample_lock_in() -> None:
    runner = BenchmarkRunner()

    score = runner.run_scenario(
        "branch_selection_under_uncertainty",
        agent_factory=lambda scenario, world: [
            ScriptedAgent(
                "tech_johnson",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("main_wet_lab"),
                    retrieve_sample_step("triage_sample_1"),
                    store_sample_in_equipment_step("triage_sample_1", "centrifuge_1"),
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "speed_rpm": 1800,
                            "duration_seconds": 20.0,
                            "temperature_celsius": 4.0,
                        },
                    ),
                    wait_step(),
                    retrieve_sample_step("triage_sample_1"),
                    log_observation_step(
                        "Choosing the scarce-instrument triage branch for triage_sample_1. "
                        "This avoids repeating the weak PCR rerun, thereby preserving the "
                        "legacy_pcr_mix reagent budget and avoiding further PCR failure."
                    ),
                ],
                name="Branch Triage Retrieval Script",
            )
        ],
    )

    assert score.success is True
    assert score.completion_ratio == 1.0
    assert score.details["triage_sample_loaded"] is True
    assert score.details["sample_loaded_before_spin"] is True
    assert score.details["centrifuge_completed"] is True
    assert score.details["avoided_pcr_repeat"] is True
    assert score.details["rationale_logged"] is True
