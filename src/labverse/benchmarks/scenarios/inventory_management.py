"""Inventory management benchmark scenario."""

from __future__ import annotations

from datetime import date

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    dispose_material_step,
    log_observation_step,
    logged_notes,
    move_to_step,
    parse_structured_finding_note,
    structured_finding_note,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import WasteCategory
from labverse.models.material import Material
from labverse.models.safety import SafetyVerdict
from labverse.world.state import WorldState


class InventoryManagementScenario(BenchmarkScenario):
    name = "inventory_management"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 40
    optimal_steps = 10
    # Expired reagents are the entire point of this scenario; a WARN that the
    # agent can ignore would make "used the expired buffer" an acceptable run.
    constraint_policy = {"material_expiry": SafetyVerdict.BLOCK}
    task = ScenarioTaskSpec(
        objective=(
            "Audit inventory, remove expired material, and identify storage incompatibilities."
        ),
        known_requirements=(
            "Move through the lab as needed to inspect inventory.",
            "Use log_observation for each identified inventory finding.",
            "Dispose any discovered expired material with its correct waste_category.",
            "Allowed issue codes are expired_material and incompatible_storage.",
        ),
        required_outputs=(
            "Log each finding exactly as: finding entity=<id> issue=<issue_code> "
            "room=<room_id> [related=<id>].",
        ),
        safety_constraints=(
            "Wear appropriate PPE before entering or working in lab areas.",
            "Do not use expired materials; consumptive use of expired material is blocked.",
            "Do not invent findings; report only issues supported by observed state.",
        ),
        intentional_unknowns=(
            "The target inventory entities are not listed in the task; discovering them is "
            "the task.",
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
                id="old_media",
                name="Old Media",
                room_id="cold_room",
                quantity=250.0,
                quantity_unit="mL",
                expiry_date=date(2025, 12, 31),
                waste_category=WasteCategory.GENERAL,
            )
        )
        world.add_material(
            Material(
                id="naoh_1",
                name="Sodium Hydroxide",
                room_id="chemical_storage",
                quantity=200.0,
                quantity_unit="mL",
                incompatible_with=["acids"],
                waste_category=WasteCategory.CHEMICAL,
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
                    move_to_step("cold_room"),
                    log_observation_step(
                        structured_finding_note(
                            entity_id="old_media",
                            issue_code="expired_material",
                            room_id="cold_room",
                        )
                    ),
                    dispose_material_step("old_media", WasteCategory.GENERAL),
                    move_to_step("main_wet_lab"),
                    move_to_step("chemical_storage"),
                    log_observation_step(
                        structured_finding_note(
                            entity_id="naoh_1",
                            issue_code="incompatible_storage",
                            room_id="chemical_storage",
                            related_entity_id="hcl_conc",
                        )
                    ),
                ],
                name="Inventory Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        notes = logged_notes(event_log, source_id="tech_johnson")
        parsed_notes = [
            parsed for note in notes if (parsed := parse_structured_finding_note(note)) is not None
        ]
        expired_item_removed = "old_media" not in final_state.materials
        expiry_identified = any(
            note.get("entity") == "old_media"
            and note.get("issue") == "expired_material"
            and note.get("room") == "cold_room"
            for note in parsed_notes
        )
        incompatibility_identified = any(
            note.get("issue") == "incompatible_storage"
            and note.get("room") == "chemical_storage"
            and {note.get("entity"), note.get("related")} == {"naoh_1", "hcl_conc"}
            for note in parsed_notes
        )
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.35 * float(expired_item_removed)
            + 0.2 * float(expiry_identified)
            + 0.25 * float(incompatibility_identified)
            + 0.2 * float(no_safety_violations)
        )
        success = (
            expired_item_removed
            and expiry_identified
            and incompatibility_identified
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "expired_item_removed": expired_item_removed,
                "expiry_identified": expiry_identified,
                "incompatibility_identified": incompatibility_identified,
                "structured_note_count": len(parsed_notes),
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["InventoryManagementScenario"]
