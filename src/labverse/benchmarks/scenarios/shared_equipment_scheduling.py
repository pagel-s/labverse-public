"""Shared equipment scheduling benchmark scenario."""

from __future__ import annotations

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    blocked_action_count,
    move_to_step,
    start_equipment_step,
    submitted_actions,
    successful_actions,
    wait_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import PersonRole
from labverse.models.person import Person
from labverse.world.state import WorldState


class SharedEquipmentSchedulingScenario(BenchmarkScenario):
    name = "shared_equipment_scheduling"
    lab_config_name = "minimal_test.yaml"
    max_steps = 30
    optimal_steps = 22
    task = ScenarioTaskSpec(
        objective="Coordinate three technicians so each completes exactly one centrifuge run.",
        known_requirements=(
            "Control tech_1, tech_2, and tech_3.",
            "Each technician must don basic wet-lab PPE and reach wet_lab before using "
            "centrifuge_1.",
            "Each technician must start centrifuge_1 exactly once with command_name=spin.",
            "Use speed_rpm=2000 for tech_1, speed_rpm=2500 for tech_2, and "
            "speed_rpm=3000 for tech_3.",
            "Use wait or another safe non-conflicting action while timed centrifuge runs finish.",
            "Leave centrifuge_1 idle at the end.",
        ),
        safety_constraints=(
            "Only one technician can use centrifuge_1 at a time.",
            "Do not repeat a centrifuge run for a technician that has already completed one.",
            "Avoid blocked equipment-conflict actions.",
        ),
        referenced_ids=("tech_1", "tech_2", "tech_3", "wet_lab", "centrifuge_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        for person_id in ("tech_1",):
            person = world.get_person(person_id)
            person.current_room_id = "office"
            person.current_zone_id = None
            person.current_ppe = set()
            person.carrying.clear()

        centrifuge = world.get_equipment("centrifuge_1")
        if "spin" in centrifuge.commands:
            centrifuge.commands["spin"].duration_seconds = 20.0

        world.add_person(
            Person(
                id="tech_2",
                name="Jordan Lee",
                role=PersonRole.TECHNICIAN,
                current_room_id="office",
            )
        )
        world.add_person(
            Person(
                id="tech_3",
                name="Morgan Patel",
                role=PersonRole.TECHNICIAN,
                current_room_id="office",
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
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "command_name": "spin",
                            "speed_rpm": 2000,
                            "duration_seconds": 20.0,
                        },
                    ),
                    wait_step(),
                    wait_step(),
                    wait_step(),
                ],
                name="Tech 1 Schedule Script",
            ),
            ScriptedAgent(
                "tech_2",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("wet_lab"),
                    wait_step(),
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "command_name": "spin",
                            "speed_rpm": 2500,
                            "duration_seconds": 20.0,
                        },
                    ),
                    wait_step(),
                ],
                name="Tech 2 Schedule Script",
            ),
            ScriptedAgent(
                "tech_3",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("wet_lab"),
                    wait_step(),
                    wait_step(),
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "command_name": "spin",
                            "speed_rpm": 3000,
                            "duration_seconds": 20.0,
                        },
                    ),
                ],
                name="Tech 3 Schedule Script",
            ),
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        attempted_start_counts = {
            agent_id: len(
                submitted_actions(
                    event_log,
                    "start_equipment",
                    source_id=agent_id,
                    target_id="centrifuge_1",
                )
            )
            for agent_id in ("tech_1", "tech_2", "tech_3")
        }
        successful_start_counts = {
            agent_id: len(
                successful_actions(
                    event_log,
                    "start_equipment",
                    source_id=agent_id,
                    target_id="centrifuge_1",
                )
            )
            for agent_id in ("tech_1", "tech_2", "tech_3")
        }
        every_agent_processed = all(count == 1 for count in successful_start_counts.values())
        conflict_free = blocked_action_count(event_log) == 0
        timed_completions = sum(
            1
            for event in event_log.query(event_type=EventType.TIMED_EVENT)
            if event.target_id == "centrifuge_1" and "completed" in event.description
        )
        centrifuge_available_at_end = (
            final_state.get_equipment("centrifuge_1").status.value == "idle"
        )
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.35 * float(every_agent_processed)
            + 0.25 * float(conflict_free)
            + 0.2 * float(timed_completions == 3)
            + 0.1 * float(centrifuge_available_at_end)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            every_agent_processed
            and conflict_free
            and timed_completions == 3
            and centrifuge_available_at_end
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "every_agent_processed": every_agent_processed,
                "conflict_free": conflict_free,
                "timed_completions": float(timed_completions),
                "centrifuge_available_at_end": centrifuge_available_at_end,
                "no_safety_violations": no_safety_violations,
                "tech_1_start_count": float(successful_start_counts["tech_1"]),
                "tech_2_start_count": float(successful_start_counts["tech_2"]),
                "tech_3_start_count": float(successful_start_counts["tech_3"]),
                "tech_1_attempted_start_count": float(attempted_start_counts["tech_1"]),
                "tech_2_attempted_start_count": float(attempted_start_counts["tech_2"]),
                "tech_3_attempted_start_count": float(attempted_start_counts["tech_3"]),
            },
        )


__all__ = ["SharedEquipmentSchedulingScenario"]
