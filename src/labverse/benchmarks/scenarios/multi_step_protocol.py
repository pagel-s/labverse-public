"""Multi-step protocol benchmark scenario."""

from __future__ import annotations

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    bsl2_ppe_steps,
    move_to_step,
    retrieve_sample_step,
    start_equipment_step,
    store_sample_in_equipment_step,
    submitted_actions,
    wait_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import EquipmentStatus
from labverse.world.state import WorldState


class MultiStepProtocolScenario(BenchmarkScenario):
    name = "multi_step_protocol"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 60
    optimal_steps = 11
    task = ScenarioTaskSpec(
        objective="Execute the timed cell-culture hold protocol with correct sample storage.",
        known_requirements=(
            "Don BSL-2 PPE before handling hela_p12.",
            "Move to tissue_culture.",
            "Start bsc_1 with command_name=work.",
            "Retrieve hela_p12.",
            "Store hela_p12 in incubator_1 before starting the hold.",
            "Start incubator_1 with command_name=incubate.",
            "Use wait while the timed incubator hold is running.",
            "Continue until the timed completion event fires and incubator_1 is idle.",
        ),
        safety_constraints=(
            "Do not stop incubator_1 early.",
            "Keep hela_p12 stored in incubator_1 during the hold.",
            "Only interact with samples and equipment in the current room.",
        ),
        referenced_ids=("tissue_culture", "bsc_1", "hela_p12", "incubator_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("grad_kim")
        person.current_room_id = "main_wet_lab"
        person.current_zone_id = "bench_1"
        person.current_ppe = set()
        person.carrying.clear()

        bsc = world.get_equipment("bsc_1")
        bsc.status = EquipmentStatus.READY
        bsc.running_command = None
        bsc.current_users.clear()
        if "work" in bsc.commands:
            bsc.commands["work"].duration_seconds = 20.0

        incubator = world.get_equipment("incubator_1")
        incubator.status = EquipmentStatus.READY
        incubator.running_command = None
        incubator.current_users.clear()
        if "incubate" in incubator.commands:
            incubator.commands["incubate"].duration_seconds = 20.0
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
            ScriptedAgent(
                "grad_kim",
                [
                    *bsl2_ppe_steps(),
                    move_to_step("tissue_culture"),
                    start_equipment_step(
                        "bsc_1",
                        parameters={"command_name": "work"},
                    ),
                    retrieve_sample_step("hela_p12"),
                    store_sample_in_equipment_step("hela_p12", "incubator_1"),
                    start_equipment_step(
                        "incubator_1",
                        parameters={"command_name": "incubate"},
                    ),
                    wait_step(),
                    wait_step(),
                ],
                name="Protocol Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        sample = final_state.get_sample("hela_p12")
        incubator = final_state.get_equipment("incubator_1")
        hold_started = bool(
            submitted_actions(
                event_log,
                "start_equipment",
                source_id="grad_kim",
                target_id="incubator_1",
            )
        )
        timed_completion_fired = any(
            event.target_id == "incubator_1"
            and event.event_type == EventType.TIMED_EVENT
            and "completed" in event.description
            for event in event_log.events
        )
        sample_stored_between_steps = sample.equipment_id == "incubator_1"
        protocol_completed = timed_completion_fired and incubator.status == EquipmentStatus.IDLE
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.2 * float(hold_started)
            + 0.3 * float(timed_completion_fired)
            + 0.2 * float(sample_stored_between_steps)
            + 0.2 * float(protocol_completed)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            hold_started
            and timed_completion_fired
            and sample_stored_between_steps
            and protocol_completed
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "hold_started": hold_started,
                "timed_completion_fired": timed_completion_fired,
                "sample_stored_between_steps": sample_stored_between_steps,
                "protocol_completed": protocol_completed,
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["MultiStepProtocolScenario"]
