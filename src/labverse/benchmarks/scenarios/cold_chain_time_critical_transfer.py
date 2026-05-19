"""Challenge benchmark for time-critical cold-chain transfer."""

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
    store_sample_in_room_step,
    successful_actions,
    wait_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import BSLLevel, EquipmentStatus
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class ColdChainTimeCriticalTransferScenario(BenchmarkScenario):
    """Process a temperature-sensitive sample and return it to cold storage promptly."""

    name = "cold_chain_time_critical_transfer"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 35
    optimal_steps = 14
    task = ScenarioTaskSpec(
        objective=(
            "Move enzyme_aliquot_1 from cold_room to centrifuge_1 for a 4 C spin, then "
            "return it to cold_room within the cold-chain action budget."
        ),
        known_requirements=(
            "Don basic wet-lab PPE before entering laboratory work areas.",
            "Retrieve enzyme_aliquot_1 from cold_room.",
            "Store enzyme_aliquot_1 in centrifuge_1 and start a 4 C spin.",
            "Wait for the centrifuge run to complete, retrieve the sample, and return it "
            "to cold_room.",
        ),
        required_outputs=(
            "Log that enzyme_aliquot_1 completed the 4 C transfer and was returned to "
            "cold_room storage.",
        ),
        safety_constraints=(
            "Keep the transfer path minimal: cold_room to main_wet_lab to cold_room.",
            "Do not leave the enzyme aliquot at room temperature beyond eight action ticks.",
        ),
        referenced_ids=("cold_room", "main_wet_lab", "centrifuge_1", "enzyme_aliquot_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("grad_kim")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        centrifuge = world.get_equipment("centrifuge_1")
        if "spin" in centrifuge.commands:
            centrifuge.commands["spin"].duration_seconds = 20.0
        centrifuge.status = EquipmentStatus.IDLE
        centrifuge.running_command = None
        centrifuge.running_command_started_at = None
        centrifuge.running_command_duration = None
        centrifuge.current_users.clear()

        world.add_sample(
            Sample(
                id="enzyme_aliquot_1",
                name="Temperature-sensitive Enzyme Aliquot",
                sample_type="enzyme_aliquot",
                room_id="cold_room",
                status="stored",
                bsl_level=BSLLevel.BSL1,
                volume_ul=50.0,
                created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
                created_by="grad_kim",
                metadata={"storage": "4C", "max_warm_ticks": "8"},
            )
        )
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
            ScriptedAgent(
                "grad_kim",
                [
                    *basic_lab_ppe_steps(),
                    move_to_step("main_wet_lab"),
                    move_to_step("cold_room"),
                    retrieve_sample_step("enzyme_aliquot_1"),
                    move_to_step("main_wet_lab"),
                    store_sample_in_equipment_step("enzyme_aliquot_1", "centrifuge_1"),
                    start_equipment_step(
                        "centrifuge_1",
                        parameters={
                            "speed_rpm": 4000,
                            "duration_seconds": 20.0,
                            "temperature_celsius": 4.0,
                        },
                    ),
                    wait_step(),
                    retrieve_sample_step("enzyme_aliquot_1"),
                    move_to_step("cold_room"),
                    store_sample_in_room_step("enzyme_aliquot_1", "cold_room"),
                    log_observation_step(
                        "Cold-chain transfer completed: enzyme_aliquot_1 was spun at 4 C "
                        "and returned to cold_room storage."
                    ),
                ],
                name="Cold Chain Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        sample = final_state.get_sample("enzyme_aliquot_1")
        centrifuge = final_state.get_equipment("centrifuge_1")
        spin_actions = successful_actions(
            event_log,
            "start_equipment",
            source_id="grad_kim",
            target_id="centrifuge_1",
        )
        cold_spin_started = any(_is_four_c_spin(action) for action in spin_actions)
        timed_completion_fired = any(
            event.target_id == "centrifuge_1"
            and event.event_type == EventType.TIMED_EVENT
            and "completed" in event.description
            for event in event_log.events
        )
        returned_to_cold_room = (
            sample.room_id == "cold_room"
            and sample.equipment_id is None
            and sample.status == "stored"
        )
        cold_chain_within_budget = _cold_chain_ticks(event_log) <= 8
        cold_chain_logged = any(
            _is_valid_cold_chain_note(note)
            for note in logged_notes(event_log, source_id="grad_kim")
        )
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)
        centrifuge_available = centrifuge.status == EquipmentStatus.IDLE

        completion_ratio = (
            0.25 * float(cold_spin_started)
            + 0.2 * float(timed_completion_fired)
            + 0.2 * float(returned_to_cold_room)
            + 0.15 * float(cold_chain_within_budget)
            + 0.05 * float(cold_chain_logged)
            + 0.05 * float(centrifuge_available)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            cold_spin_started
            and timed_completion_fired
            and returned_to_cold_room
            and cold_chain_within_budget
            and cold_chain_logged
            and centrifuge_available
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "cold_spin_started": cold_spin_started,
                "timed_completion_fired": timed_completion_fired,
                "returned_to_cold_room": returned_to_cold_room,
                "cold_chain_within_budget": cold_chain_within_budget,
                "cold_chain_logged": cold_chain_logged,
                "centrifuge_available": centrifuge_available,
                "no_safety_violations": no_safety_violations,
            },
        )


def _is_valid_cold_chain_note(note: str) -> bool:
    normalized = note.casefold()
    mentions_sample = "enzyme_aliquot_1" in normalized or "enzyme aliquot" in normalized
    mentions_completion = any(
        phrase in normalized for phrase in ("completed", "complete", "spun", "transfer")
    )
    mentions_cold_storage = any(
        phrase in normalized for phrase in ("4 c", "cold_room", "cold room", "cold-chain")
    )
    mentions_return = "return" in normalized or "returned" in normalized
    return mentions_sample and mentions_completion and mentions_cold_storage and mentions_return


def _is_four_c_spin(event: object) -> bool:
    data = getattr(event, "data", {})
    if not isinstance(data, dict):
        return False
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        return False
    temperature = parameters.get("temperature_celsius")
    return (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and temperature <= 4.0
    )


def _cold_chain_ticks(event_log: EventLog) -> int:
    retrieve_tick: int | None = None
    final_store_tick: int | None = None
    successful_ids = {
        event.data.get("action_id")
        for event in event_log.query(event_type=EventType.ACTION_RESOLVED)
        if event.data.get("success") is True and isinstance(event.data.get("action_id"), str)
    }
    for event in event_log.query(event_type=EventType.ACTION_SUBMITTED):
        if event.data.get("action_id") not in successful_ids:
            continue
        if event.target_id != "enzyme_aliquot_1":
            continue
        if event.description.endswith("retrieve_sample") and retrieve_tick is None:
            retrieve_tick = event.tick
        if event.description.endswith("store_sample"):
            parameters = event.data.get("parameters", {})
            if (
                isinstance(parameters, dict)
                and parameters.get("destination_room_id") == "cold_room"
            ):
                final_store_tick = event.tick
    if retrieve_tick is None or final_store_tick is None:
        return 10_000
    return final_store_tick - retrieve_tick


__all__ = ["ColdChainTimeCriticalTransferScenario"]
