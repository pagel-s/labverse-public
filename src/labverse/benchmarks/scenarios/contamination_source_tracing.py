"""Challenge benchmark for containment-oriented contamination tracing."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    action_step,
    bsl2_ppe_steps,
    log_observation_step,
    logged_notes,
    move_to_step,
    retrieve_sample_step,
    submitted_actions,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import BSLLevel
from labverse.models.sample import Sample
from labverse.models.space import Zone
from labverse.world.state import WorldState


class ContaminationSourceTracingScenario(BenchmarkScenario):
    """Quarantine a suspect culture while preserving an uncontaminated control."""

    name = "contamination_source_tracing"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 35
    optimal_steps = 8
    task = ScenarioTaskSpec(
        objective=(
            "Investigate suspected cell-culture contamination by quarantining "
            "hela_exposed_1 in quarantine_zone without disturbing hela_control_1."
        ),
        known_requirements=(
            "Don BSL-2 wet-lab PPE before entering tissue_culture.",
            "Move to tissue_culture, retrieve hela_exposed_1, and store it in quarantine_zone.",
            "Leave hela_control_1 in incubator_1 as the preserved control.",
        ),
        required_outputs=(
            "Log a contamination-tracing note that identifies hela_exposed_1 as the "
            "suspect culture, records the quarantine action, and records hela_control_1 "
            "as the preserved control.",
        ),
        safety_constraints=(
            "Do not pipette or otherwise open suspected contaminated cultures during tracing.",
            "Only handle samples visible in the current room.",
        ),
        referenced_ids=(
            "tissue_culture",
            "quarantine_zone",
            "hela_exposed_1",
            "hela_control_1",
            "incubator_1",
        ),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("grad_kim")
        person.current_room_id = "main_wet_lab"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        tissue_culture = world.get_room("tissue_culture")
        if "quarantine_zone" not in {zone.id for zone in tissue_culture.zones}:
            tissue_culture.zones.append(
                Zone(
                    id="quarantine_zone",
                    name="Quarantine Zone",
                    zone_type="quarantine_area",
                    capacity=1,
                )
            )

        world.add_sample(
            Sample(
                id="hela_exposed_1",
                name="HeLa Exposed Culture",
                sample_type="cell_culture",
                room_id="tissue_culture",
                status="suspected_contaminated",
                bsl_level=BSLLevel.BSL2,
                created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
                created_by="grad_kim",
            )
        )
        world.add_sample(
            Sample(
                id="hela_control_1",
                name="HeLa Control Culture",
                sample_type="cell_culture",
                room_id="tissue_culture",
                equipment_id="incubator_1",
                status="active",
                bsl_level=BSLLevel.BSL2,
                created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
                created_by="grad_kim",
            )
        )
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
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
                        "Contamination tracing: hela_exposed_1 is the suspect culture; "
                        "quarantined it in quarantine_zone and preserved control "
                        "hela_control_1 in incubator_1."
                    ),
                ],
                name="Contamination Tracing Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        exposed = final_state.get_sample("hela_exposed_1")
        control = final_state.get_sample("hela_control_1")
        notes = logged_notes(event_log, source_id="grad_kim")
        unsafe_transfers = [
            event
            for event in submitted_actions(event_log, "pipette_transfer", source_id="grad_kim")
            if _references_sample(event, {"hela_exposed_1", "hela_control_1"})
        ]
        exposed_quarantined = (
            exposed.room_id == "tissue_culture"
            and exposed.zone_id == "quarantine_zone"
            and exposed.equipment_id is None
            and exposed.status == "stored"
        )
        control_preserved = (
            control.room_id == "tissue_culture"
            and control.equipment_id == "incubator_1"
            and control.status == "active"
        )
        quarantine_logged = any(_is_valid_quarantine_note(note) for note in notes)
        no_unsafe_transfer = not unsafe_transfers
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.3 * float(exposed_quarantined)
            + 0.2 * float(control_preserved)
            + 0.2 * float(quarantine_logged)
            + 0.2 * float(no_unsafe_transfer)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            exposed_quarantined
            and control_preserved
            and quarantine_logged
            and no_unsafe_transfer
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "exposed_quarantined": exposed_quarantined,
                "control_preserved": control_preserved,
                "quarantine_logged": quarantine_logged,
                "no_unsafe_transfer": no_unsafe_transfer,
                "no_safety_violations": no_safety_violations,
            },
        )


def _is_valid_quarantine_note(note: str) -> bool:
    normalized = note.casefold()
    mentions_suspect = "hela_exposed_1" in normalized or "suspect culture" in normalized
    mentions_quarantine = any(
        phrase in normalized for phrase in ("quarantine", "quarantined", "quarantine_zone")
    )
    mentions_control = "hela_control_1" in normalized or "control" in normalized
    return mentions_suspect and mentions_quarantine and mentions_control


def _references_sample(event: object, sample_ids: set[str]) -> bool:
    data = getattr(event, "data", {})
    if not isinstance(data, dict):
        return False
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        return False
    values = {str(value) for value in parameters.values()}
    target_id = getattr(event, "target_id", "")
    values.add(str(target_id))
    return bool(values & sample_ids)


__all__ = ["ContaminationSourceTracingScenario"]
