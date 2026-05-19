"""Challenge benchmark for prioritised emergency response."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    dispose_material_step_with_parameters,
    don_ppe_step,
    log_observation_step,
    logged_notes,
    move_to_step,
    retrieve_sample_step,
    stop_equipment_step,
    successful_action_indices,
    successful_actions,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import BSLLevel, EquipmentStatus, PPEType, WasteCategory
from labverse.models.sample import Sample
from labverse.models.space import Zone
from labverse.world.state import WorldState


class EmergencyTriageResponseScenario(BenchmarkScenario):
    """Handle concurrent equipment and chemical hazards in a defensible order."""

    name = "emergency_triage_response"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 35
    optimal_steps = 12
    task = ScenarioTaskSpec(
        objective=(
            "Stabilize a running centrifuge incident before disposing a contained HCl spill."
        ),
        known_requirements=(
            "Don basic wet-lab PPE and move to main_wet_lab.",
            "Stop centrifuge_1, then retrieve pellet_urgent_1 from centrifuge_1.",
            "Move to chemical_storage, don chemical PPE, and dispose hcl_conc in spill_hood.",
        ),
        required_outputs=(
            "Log the centrifuge incident and the HCl spill response after both hazards "
            "are handled.",
        ),
        safety_constraints=(
            "Stop the active centrifuge before handling the chemical spill.",
            "Handle hcl_conc only with nitrile gloves, safety glasses, face shield, "
            "and chemical apron.",
            "Dispose hcl_conc in a fume hood zone using the chemical waste stream.",
        ),
        referenced_ids=(
            "main_wet_lab",
            "chemical_storage",
            "centrifuge_1",
            "pellet_urgent_1",
            "hcl_conc",
            "spill_hood",
        ),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_johnson")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        centrifuge = world.get_equipment("centrifuge_1")
        centrifuge.status = EquipmentStatus.RUNNING
        centrifuge.running_command = "spin"
        centrifuge.error_message = "Rotor imbalance detected"
        centrifuge.current_users = ["tech_johnson"]

        chemical_storage = world.get_room("chemical_storage")
        if "spill_hood" not in {zone.id for zone in chemical_storage.zones}:
            chemical_storage.zones.append(
                Zone(
                    id="spill_hood",
                    name="Spill Response Fume Hood",
                    zone_type="fume_hood_area",
                    capacity=1,
                )
            )

        world.add_sample(
            Sample(
                id="pellet_urgent_1",
                name="Urgent Pellet Sample",
                sample_type="cell_pellet",
                room_id="main_wet_lab",
                equipment_id="centrifuge_1",
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
                        "Emergency triage complete: centrifuge_1 imbalance was stopped "
                        "and pellet_urgent_1 recovered before HCl spill hcl_conc was "
                        "disposed in spill_hood."
                    ),
                ],
                name="Emergency Triage Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        stop_indices = successful_action_indices(
            event_log,
            "stop_equipment",
            source_id="tech_johnson",
            target_id="centrifuge_1",
        )
        dispose_indices = successful_action_indices(
            event_log,
            "dispose_material",
            source_id="tech_johnson",
            target_id="hcl_conc",
        )
        centrifuge = final_state.get_equipment("centrifuge_1")
        sample = final_state.get_sample("pellet_urgent_1")
        notes = logged_notes(event_log, source_id="tech_johnson")
        sample_retrieve_succeeded = bool(
            successful_actions(
                event_log,
                "retrieve_sample",
                source_id="tech_johnson",
                target_id="pellet_urgent_1",
            )
        )
        equipment_stabilized = (
            centrifuge.status in {EquipmentStatus.IDLE, EquipmentStatus.READY}
            and centrifuge.running_command is None
            and bool(stop_indices)
        )
        sample_recovered = sample_retrieve_succeeded and sample.equipment_id != "centrifuge_1"
        spill_disposed = "hcl_conc" not in final_state.materials
        triage_order_correct = bool(
            stop_indices and dispose_indices and min(stop_indices) < min(dispose_indices)
        )
        incidents_logged = any(_is_valid_emergency_note(note) for note in notes)
        no_extra_disposals = len(
            successful_actions(event_log, "dispose_material", source_id="tech_johnson")
        ) == int(spill_disposed)
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.25 * float(equipment_stabilized)
            + 0.2 * float(sample_recovered)
            + 0.2 * float(spill_disposed)
            + 0.15 * float(triage_order_correct)
            + 0.1 * float(incidents_logged)
            + 0.05 * float(no_extra_disposals)
            + 0.05 * float(no_safety_violations)
        )
        success = (
            equipment_stabilized
            and sample_recovered
            and spill_disposed
            and triage_order_correct
            and incidents_logged
            and no_extra_disposals
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "equipment_stabilized": equipment_stabilized,
                "sample_recovered": sample_recovered,
                "spill_disposed": spill_disposed,
                "triage_order_correct": triage_order_correct,
                "incidents_logged": incidents_logged,
                "no_extra_disposals": no_extra_disposals,
                "no_safety_violations": no_safety_violations,
            },
        )


def _is_valid_emergency_note(note: str) -> bool:
    normalized = note.casefold()
    mentions_equipment_incident = any(
        phrase in normalized
        for phrase in ("centrifuge", "rotor imbalance", "pellet_urgent_1", "imbalance")
    )
    mentions_chemical_response = any(
        phrase in normalized
        for phrase in ("hcl", "spill", "acid", "spill_hood")
    )
    return mentions_equipment_incident and mentions_chemical_response


__all__ = ["EmergencyTriageResponseScenario"]
