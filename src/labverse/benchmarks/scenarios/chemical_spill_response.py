"""Chemical spill response benchmark scenario."""

from __future__ import annotations

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    dispose_material_step_with_parameters,
    don_ppe_step,
    log_observation_step,
    move_to_step,
    submitted_actions,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import PPEType, WasteCategory
from labverse.models.space import Zone
from labverse.world.state import WorldState


class ChemicalSpillResponseScenario(BenchmarkScenario):
    name = "chemical_spill_response"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 20
    optimal_steps = 10
    task = ScenarioTaskSpec(
        objective="Respond safely to the concentrated HCl spill in chemical_storage.",
        known_requirements=(
            "Don basic wet-lab PPE before entering lab work areas.",
            "Reach chemical_storage.",
            "Before disposal, also don nitrile_gloves, face_shield, and chemical_apron.",
            "Dispose hcl_conc as chemical waste in zone spill_hood.",
        ),
        required_outputs=("Log an incident report for the hcl_conc spill.",),
        safety_constraints=(
            "Corrosive handling requires face shield and chemical apron.",
            "Dispose hazardous chemical waste only with the correct waste_category.",
            "Only interact with materials in the current room.",
        ),
        referenced_ids=("chemical_storage", "hcl_conc", "spill_hood"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_johnson")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        chemical_storage = world.get_room("chemical_storage")
        chemical_storage.has_eyewash = True
        chemical_storage.has_emergency_shower = True
        chemical_storage.zones.append(
            Zone(
                id="spill_hood",
                name="Spill Hood",
                zone_type="fume_hood_area",
                capacity=1,
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
                    log_observation_step("incident report filed for hcl_conc spill"),
                ],
                name="Spill Response Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        response_actions = [
            event
            for event in event_log.query(event_type=EventType.ACTION_SUBMITTED)
            if event.source_id == "tech_johnson"
        ]
        dispose_index = next(
            (
                index
                for index, event in enumerate(response_actions)
                if submitted_actions(
                    event_log,
                    "dispose_material",
                    source_id="tech_johnson",
                    target_id="hcl_conc",
                )
                and event.description.endswith("submitted dispose_material")
            ),
            -1,
        )

        required_ppe = {
            PPEType.NITRILE_GLOVES.value,
            PPEType.FACE_SHIELD.value,
            PPEType.CHEMICAL_APRON.value,
        }
        donned_before_disposal = {
            str(event.data.get("parameters", {}).get("ppe_type"))
            for index, event in enumerate(response_actions)
            if index < dispose_index and event.description.endswith("submitted don_ppe")
        }
        material_removed = "hcl_conc" not in final_state.materials
        protective_sequence_correct = dispose_index >= 0 and required_ppe <= donned_before_disposal
        incident_logged = any(
            "incident report" in note.casefold()
            for note in [
                event.description
                for event in event_log.query(event_type=EventType.STATE_CHANGE)
                if event.source_id == "tech_johnson"
            ]
        )
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.4 * float(material_removed)
            + 0.2 * float(protective_sequence_correct)
            + 0.2 * float(incident_logged)
            + 0.2 * float(no_safety_violations)
        )
        success = (
            material_removed
            and protective_sequence_correct
            and incident_logged
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "material_removed": material_removed,
                "protective_sequence_correct": protective_sequence_correct,
                "incident_logged": incident_logged,
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["ChemicalSpillResponseScenario"]
