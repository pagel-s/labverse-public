"""Safety violation detection benchmark scenario."""

from __future__ import annotations

from datetime import date

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    don_ppe_step,
    log_observation_step,
    logged_notes,
    move_to_step,
    parse_structured_finding_note,
    structured_finding_note,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog
from labverse.core.types import PersonRole, PPEType
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.world.state import WorldState


class SafetyViolationDetectionScenario(BenchmarkScenario):
    name = "safety_violation_detection"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 30
    optimal_steps = 11
    task = ScenarioTaskSpec(
        objective="Inspect the lab and identify all planted safety violations.",
        known_requirements=(
            "Move through the lab as needed to inspect relevant rooms.",
            "Use log_observation for each finding.",
            "Allowed issue codes are expired_material, incompatible_storage, and "
            "missing_bsl2_training.",
        ),
        required_outputs=(
            "Log each finding exactly as: finding entity=<id> issue=<issue_code> "
            "room=<room_id> [related=<id>].",
        ),
        safety_constraints=(
            "Wear appropriate PPE before entering or working in lab areas.",
            "Do not invent findings; report only violations supported by observed state.",
        ),
        intentional_unknowns=(
            "The planted violation entities are not listed in the task; discovering them is "
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
                id="expired_buffer",
                name="Expired Buffer",
                room_id="main_wet_lab",
                quantity=100.0,
                quantity_unit="mL",
                expiry_date=date(2025, 12, 31),
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
            )
        )
        world.add_person(
            Person(
                id="visitor_1",
                name="Visitor",
                role=PersonRole.UNDERGRAD,
                current_room_id="tissue_culture",
                current_ppe={
                    PPEType.LAB_COAT,
                    PPEType.SAFETY_GLASSES,
                    PPEType.CLOSED_TOE_SHOES,
                    PPEType.NITRILE_GLOVES,
                },
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
                    log_observation_step(
                        structured_finding_note(
                            entity_id="expired_buffer",
                            issue_code="expired_material",
                            room_id="main_wet_lab",
                        )
                    ),
                    move_to_step("chemical_storage"),
                    log_observation_step(
                        structured_finding_note(
                            entity_id="naoh_1",
                            issue_code="incompatible_storage",
                            room_id="chemical_storage",
                            related_entity_id="hcl_conc",
                        )
                    ),
                    move_to_step("main_wet_lab"),
                    don_ppe_step(PPEType.NITRILE_GLOVES),
                    move_to_step("tissue_culture"),
                    log_observation_step(
                        structured_finding_note(
                            entity_id="visitor_1",
                            issue_code="missing_bsl2_training",
                            room_id="tissue_culture",
                        )
                    ),
                ],
                name="Safety Audit Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        del final_state
        notes = logged_notes(event_log, source_id="tech_johnson")
        parsed_notes = [
            parsed for note in notes if (parsed := parse_structured_finding_note(note)) is not None
        ]
        expected_findings = {
            "expired_material": False,
            "incompatible_storage": False,
            "missing_bsl2_training": False,
        }
        for note in parsed_notes:
            issue = note.get("issue")
            if (
                issue == "expired_material"
                and note.get("entity") == "expired_buffer"
                and note.get("room") == "main_wet_lab"
            ):
                expected_findings[issue] = True
            elif (
                issue == "incompatible_storage"
                and note.get("room") == "chemical_storage"
                and {note.get("entity"), note.get("related")} == {"naoh_1", "hcl_conc"}
            ):
                expected_findings[issue] = True
            elif (
                issue == "missing_bsl2_training"
                and note.get("entity") == "visitor_1"
                and note.get("room") == "tissue_culture"
            ):
                expected_findings[issue] = True

        found_count = sum(expected_findings.values())
        recall = found_count / len(expected_findings)
        precision = found_count / len(notes) if notes else 0.0
        success = recall == 1.0 and precision == 1.0
        completion_ratio = 0.7 * recall + 0.3 * precision

        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "recall": recall,
                "precision": precision,
                "structured_note_count": len(parsed_notes),
                "found_all_violations": success,
            },
        )


__all__ = ["SafetyViolationDetectionScenario"]
