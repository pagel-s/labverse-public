"""Equipment failure recovery benchmark scenario."""

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
    stop_equipment_step,
    successful_actions,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import EquipmentStatus
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class EquipmentFailureRecoveryScenario(BenchmarkScenario):
    name = "equipment_failure_recovery"
    lab_config_name = "minimal_test.yaml"
    max_steps = 20
    optimal_steps = 7
    task = ScenarioTaskSpec(
        objective="Recover safely from the centrifuge failure without losing the sample.",
        known_requirements=(
            "Don basic wet-lab PPE before entering wet_lab.",
            "Move to wet_lab.",
            "Stop the failed centrifuge_1.",
            "Retrieve pellet_1 from centrifuge_1.",
        ),
        required_outputs=("Log an incident documenting the centrifuge_1 imbalance.",),
        safety_constraints=(
            "Recover the sample only after stopping the failed equipment.",
            "Only interact with equipment and samples in the current room.",
        ),
        referenced_ids=("wet_lab", "centrifuge_1", "pellet_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_1")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        centrifuge = world.get_equipment("centrifuge_1")
        centrifuge.status = EquipmentStatus.RUNNING
        centrifuge.running_command = "spin"
        centrifuge.error_message = "Rotor imbalance detected"

        world.add_sample(
            Sample(
                id="pellet_1",
                name="Recovered Pellet",
                sample_type="cell_pellet",
                room_id="wet_lab",
                zone_id="bench_1",
                equipment_id="centrifuge_1",
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
                    stop_equipment_step("centrifuge_1"),
                    retrieve_sample_step("pellet_1"),
                    log_observation_step("incident documented for centrifuge_1 imbalance"),
                ],
                name="Failure Recovery Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        centrifuge = final_state.get_equipment("centrifuge_1")
        person = final_state.get_person("tech_1")
        pellet = final_state.get_sample("pellet_1")
        sample_recovered = bool(
            successful_actions(
                event_log,
                "retrieve_sample",
                source_id="tech_1",
                target_id="pellet_1",
            )
        ) and ("pellet_1" in person.carrying or pellet.equipment_id != "centrifuge_1")
        equipment_stopped = centrifuge.status == EquipmentStatus.IDLE
        incident_logged = any(
            _is_valid_incident_note(note) for note in logged_notes(event_log, source_id="tech_1")
        )
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.35 * float(equipment_stopped)
            + 0.25 * float(sample_recovered)
            + 0.2 * float(incident_logged)
            + 0.2 * float(no_safety_violations)
        )
        success = (
            equipment_stopped and sample_recovered and incident_logged and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "equipment_stopped": equipment_stopped,
                "sample_recovered": sample_recovered,
                "incident_logged": incident_logged,
                "no_safety_violations": no_safety_violations,
            },
        )


def _is_valid_incident_note(note: str) -> bool:
    normalized = " ".join(note.casefold().replace("-", " ").split())
    has_report_marker = any(
        marker in normalized for marker in ("incident", "report", "documented", "documentation")
    )
    has_equipment = "centrifuge_1" in normalized or "centrifuge 1" in normalized
    has_failure = any(marker in normalized for marker in ("imbalance", "failure", "failed"))
    return has_report_marker and has_equipment and has_failure


__all__ = ["EquipmentFailureRecoveryScenario"]
