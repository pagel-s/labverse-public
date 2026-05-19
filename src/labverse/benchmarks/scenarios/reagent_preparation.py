"""Reagent preparation benchmark scenario."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    log_observation_step,
    move_to_step,
    pipette_transfer_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import EquipmentStatus
from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class ReagentPreparationScenario(BenchmarkScenario):
    name = "reagent_preparation"
    lab_config_name = "minimal_test.yaml"
    max_steps = 15
    optimal_steps = 6
    task = ScenarioTaskSpec(
        objective="Prepare and label a PBS 1X working solution.",
        known_requirements=(
            "Don basic wet-lab PPE before handling materials or equipment.",
            "Move to wet_lab before working with the reagent batch.",
            "Use pipette_pbs_1 to transfer 50000 uL from pbs_10x_stock into pbs_batch_1.",
        ),
        required_outputs=("Log that PBS 1X was prepared and labeled.",),
        safety_constraints=("Only interact with materials and equipment in the current room.",),
        referenced_ids=("wet_lab", "pipette_pbs_1", "pbs_10x_stock", "pbs_batch_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("tech_1")
        person.current_room_id = "office"
        person.current_zone_id = None
        person.current_ppe = set()
        person.carrying.clear()

        world.add_equipment(
            Equipment(
                id="pipette_pbs_1",
                name="PBS Prep Pipette",
                equipment_type="pipette",
                room_id="wet_lab",
                zone_id="bench_1",
                status=EquipmentStatus.READY,
                capabilities=["transfer"],
            )
        )
        world.add_material(
            Material(
                id="pbs_10x_stock",
                name="PBS 10X Stock",
                room_id="wet_lab",
                zone_id="bench_1",
                quantity=100.0,
                quantity_unit="mL",
            )
        )
        world.add_sample(
            Sample(
                id="pbs_batch_1",
                name="PBS 1X Batch",
                sample_type="buffer",
                room_id="wet_lab",
                zone_id="bench_1",
                volume_ul=450_000.0,
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
                    pipette_transfer_step(
                        "pipette_pbs_1",
                        source_id="pbs_10x_stock",
                        destination_id="pbs_batch_1",
                        volume_ul=50_000.0,
                    ),
                    log_observation_step("PBS 1X prepared and labeled 2026-01-01 TS"),
                ],
                name="Reagent Prep Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        batch = final_state.get_sample("pbs_batch_1")
        stock = final_state.get_material("pbs_10x_stock")
        labels_logged = any(
            "pbs 1x" in event.description.casefold()
            for event in event_log.query(event_type=EventType.STATE_CHANGE)
            if event.source_id == "tech_1"
        )
        correct_volume = batch.volume_ul == 500_000.0
        stock_consumed_correctly = stock.quantity == 50.0
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.4 * float(correct_volume)
            + 0.3 * float(stock_consumed_correctly)
            + 0.2 * float(labels_logged)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            correct_volume and stock_consumed_correctly and labels_logged and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "correct_volume": correct_volume,
                "stock_consumed_correctly": stock_consumed_correctly,
                "labels_logged": labels_logged,
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["ReagentPreparationScenario"]
