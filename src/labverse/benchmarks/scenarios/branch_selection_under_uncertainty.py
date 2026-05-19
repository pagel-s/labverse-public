"""Research-style benchmark for branch selection under uncertainty."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    log_observation_step,
    logged_notes,
    move_to_step,
    retrieve_sample_step,
    start_equipment_step,
    store_sample_in_equipment_step,
    successful_action_indices,
    successful_actions,
    wait_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import BSLLevel, EquipmentStatus
from labverse.models.material import Material
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class BranchSelectionUnderUncertaintyScenario(BenchmarkScenario):
    """Choose a feasible, informative branch under reagent and instrument constraints."""

    name = "branch_selection_under_uncertainty"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 22
    optimal_steps = 9
    task = ScenarioTaskSpec(
        objective=(
            "Choose the scarce-instrument triage branch for triage_sample_1 instead of "
            "repeating the weak PCR rerun."
        ),
        known_requirements=(
            "Don basic wet-lab PPE and move to main_wet_lab.",
            "Retrieve triage_sample_1, load it into centrifuge_1, and use centrifuge_1 "
            "exactly once with a 4 C spin.",
            "Preserve the legacy_pcr_mix reagent budget by avoiding a repeated PCR run.",
        ),
        required_outputs=(
            "Log the rationale, naming the chosen triage branch, the avoided PCR rerun, "
            "and the scarce reagent preservation decision.",
        ),
        safety_constraints=("Only interact with equipment and materials in the current room.",),
        referenced_ids=(
            "main_wet_lab",
            "centrifuge_1",
            "pcr_1",
            "triage_sample_1",
            "legacy_pcr_mix",
        ),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_johnson")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        world.add_material(
            Material(
                id="legacy_pcr_mix",
                name="Legacy PCR Master Mix",
                room_id="main_wet_lab",
                zone_id="bench_1",
                quantity=12.0,
                quantity_unit="uL",
            )
        )
        centrifuge = world.get_equipment("centrifuge_1")
        if "spin" in centrifuge.commands:
            centrifuge.commands["spin"].duration_seconds = 20.0
        centrifuge.status = EquipmentStatus.IDLE
        centrifuge.running_command = None
        centrifuge.running_command_started_at = None
        centrifuge.running_command_duration = None
        centrifuge.current_users.clear()

        pcr = world.get_equipment("pcr_1")
        pcr.status = EquipmentStatus.IDLE
        pcr.running_command = None
        pcr.running_command_started_at = None
        pcr.running_command_duration = None
        pcr.current_users.clear()

        world.add_sample(
            Sample(
                id="triage_sample_1",
                name="Triage Sample 1",
                sample_type="clarified_cell_lysate",
                room_id="main_wet_lab",
                zone_id="bench_1",
                status="active",
                bsl_level=BSLLevel.BSL1,
                created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
                created_by="tech_johnson",
            )
        )
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
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
                    log_observation_step(
                        "Selected centrifuge triage branch; deferred PCR rerun "
                        "because the prior PCR branch failed and reagent budget "
                        "is scarce."
                    ),
                ],
                name="Branch Triage Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        centrifuge_runs = successful_actions(
            event_log,
            "start_equipment",
            source_id="tech_johnson",
            target_id="centrifuge_1",
        )
        pcr_runs = successful_actions(
            event_log,
            "start_equipment",
            source_id="tech_johnson",
            target_id="pcr_1",
        )
        store_indices = successful_action_indices(
            event_log,
            "store_sample",
            source_id="tech_johnson",
            target_id="triage_sample_1",
        )
        store_actions = successful_actions(
            event_log,
            "store_sample",
            source_id="tech_johnson",
            target_id="triage_sample_1",
        )
        spin_indices = successful_action_indices(
            event_log,
            "start_equipment",
            source_id="tech_johnson",
            target_id="centrifuge_1",
        )
        notes = logged_notes(event_log, source_id="tech_johnson")
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)
        timed_completion_fired = any(
            event.target_id == "centrifuge_1"
            and event.event_type == EventType.TIMED_EVENT
            and "completed" in event.description
            for event in event_log.events
        )
        rationale_logged = any(_is_valid_branch_rationale(note) for note in notes)
        scarce_instrument_used_once = len(centrifuge_runs) == 1
        avoided_pcr_repeat = len(pcr_runs) == 0
        triage_sample_loaded = any(
            _stored_in_equipment(action, equipment_id="centrifuge_1")
            for action in store_actions
        )
        sample_loaded_before_spin = bool(
            store_indices and spin_indices and min(store_indices) < min(spin_indices)
        )
        legacy_reagent_preserved = (
            "legacy_pcr_mix" in final_state.materials
            and final_state.get_material("legacy_pcr_mix").quantity == 12.0
        )
        completion_ratio = (
            0.2 * float(scarce_instrument_used_once)
            + 0.15 * float(triage_sample_loaded and sample_loaded_before_spin)
            + 0.15 * float(timed_completion_fired)
            + 0.2 * float(avoided_pcr_repeat)
            + 0.15 * float(rationale_logged)
            + 0.1 * float(legacy_reagent_preserved)
            + 0.05 * float(no_safety_violations)
        )
        success = (
            scarce_instrument_used_once
            and triage_sample_loaded
            and sample_loaded_before_spin
            and timed_completion_fired
            and avoided_pcr_repeat
            and rationale_logged
            and legacy_reagent_preserved
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "scarce_instrument_used_once": scarce_instrument_used_once,
                "triage_sample_loaded": triage_sample_loaded,
                "sample_loaded_before_spin": sample_loaded_before_spin,
                "centrifuge_completed": timed_completion_fired,
                "avoided_pcr_repeat": avoided_pcr_repeat,
                "rationale_logged": rationale_logged,
                "legacy_reagent_preserved": legacy_reagent_preserved,
                "no_safety_violations": no_safety_violations,
            },
        )


def _is_valid_branch_rationale(note: str) -> bool:
    normalized = note.casefold()
    chose_branch = any(
        phrase in normalized
        for phrase in (
            "triage branch",
            "centrifuge branch",
            "centrifuge triage",
            "scarce-instrument",
            "selected centrifuge",
            "choosing the scarce-instrument",
        )
    )
    references_failed_pcr = "pcr" in normalized and any(
        phrase in normalized
        for phrase in (
            "avoid",
            "avoided",
            "avoiding",
            "defer",
            "deferred",
            "rerun",
            "repeat",
            "failed",
            "failure",
        )
    )
    preserves_reagent = any(
        phrase in normalized
        for phrase in (
            "scarce",
            "reagent budget",
            "preserv",
            "legacy_pcr_mix",
        )
    )
    return chose_branch and references_failed_pcr and preserves_reagent


def _stored_in_equipment(event: object, *, equipment_id: str) -> bool:
    data = getattr(event, "data", {})
    if not isinstance(data, dict):
        return False
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        return False
    return parameters.get("destination_equipment_id") == equipment_id


__all__ = ["BranchSelectionUnderUncertaintyScenario"]
