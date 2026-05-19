"""Sample tracking and chain-of-custody benchmark scenario."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    log_observation_step,
    move_to_step,
    retrieve_sample_step,
    store_sample_in_equipment_step,
    store_sample_in_room_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class SampleTrackingScenario(BenchmarkScenario):
    name = "sample_tracking"
    lab_config_name = "minimal_test.yaml"
    max_steps = 35
    optimal_steps = 10
    task = ScenarioTaskSpec(
        objective="Maintain chain of custody while moving incoming_sample_1 through the lab.",
        known_requirements=(
            "Don basic wet-lab PPE before entering wet_lab.",
            "Move to wet_lab and retrieve incoming_sample_1.",
            "Log accession of incoming_sample_1.",
            "Store incoming_sample_1 in centrifuge_1.",
            "Retrieve incoming_sample_1 again.",
            "Move to office and store incoming_sample_1 in office.",
        ),
        required_outputs=(
            "Log exactly accessioned incoming_sample_1 when accessioning the sample.",
        ),
        safety_constraints=("Only store or retrieve samples in the current room.",),
        referenced_ids=("incoming_sample_1", "wet_lab", "centrifuge_1", "office"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_1")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        world.add_sample(
            Sample(
                id="incoming_sample_1",
                name="Incoming Sample 1",
                sample_type="incoming_specimen",
                room_id="wet_lab",
                zone_id="bench_1",
                created_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
                created_by="tech_1",
            )
        )
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
            ScriptedAgent(
                "tech_1",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("wet_lab"),
                    retrieve_sample_step("incoming_sample_1"),
                    log_observation_step("accessioned incoming_sample_1"),
                    store_sample_in_equipment_step("incoming_sample_1", "centrifuge_1"),
                    retrieve_sample_step("incoming_sample_1"),
                    move_to_step("office"),
                    store_sample_in_room_step("incoming_sample_1", "office"),
                ],
                name="Sample Tracking Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        sample = final_state.get_sample("incoming_sample_1")
        actions = [record.action for record in sample.chain_of_custody]
        accession_logged = any(
            "accessioned incoming_sample_1" in event.description.casefold()
            for event in event_log.query(event_type=EventType.STATE_CHANGE)
            if event.source_id == "tech_1"
        )
        custody_complete = actions == ["retrieved", "stored", "retrieved", "stored"]
        final_location_correct = sample.room_id == "office" and sample.equipment_id is None
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.3 * float(custody_complete)
            + 0.2 * float(accession_logged)
            + 0.3 * float(final_location_correct)
            + 0.2 * float(no_safety_violations)
        )
        success = (
            custody_complete
            and accession_logged
            and final_location_correct
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "custody_complete": custody_complete,
                "accession_logged": accession_logged,
                "final_location_correct": final_location_correct,
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["SampleTrackingScenario"]
