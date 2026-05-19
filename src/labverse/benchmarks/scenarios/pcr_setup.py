"""PCR setup benchmark scenario."""

from __future__ import annotations

from datetime import UTC, datetime

from labverse.agent.scripted import ScriptedAgent
from labverse.benchmarks.scenarios.base import BenchmarkScenario, ScenarioTaskSpec
from labverse.benchmarks.scenarios.common import (
    basic_lab_ppe_steps,
    latest_action_parameters,
    pipette_transfer_step,
    start_equipment_step,
)
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.core.events import EventLog, EventType
from labverse.core.types import EquipmentStatus
from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.sample import Sample
from labverse.world.state import WorldState


class PCRSetupScenario(BenchmarkScenario):
    name = "pcr_setup"
    lab_config_name = "research_bsl2.yaml"
    max_steps = 25
    optimal_steps = 5
    task = ScenarioTaskSpec(
        objective="Set up and start the PCR protocol for pcr_reaction_1.",
        known_requirements=(
            "Don basic wet-lab PPE before handling materials or equipment.",
            "Use pipette_pcr_1 to transfer 25 uL from pcr_master_mix into pcr_reaction_1.",
            "Start pcr_1 with command_name=run_program.",
            "Use denaturation_temp=95, annealing_temp=55, extension_temp=72, "
            "cycle_count=30, and hold_temp=4.",
        ),
        safety_constraints=(
            "Only interact with equipment and materials in the current room.",
            "Do not stop the thermocycler early.",
        ),
        referenced_ids=("pipette_pcr_1", "pcr_master_mix", "pcr_reaction_1", "pcr_1"),
    )

    def setup(self, world: WorldState) -> WorldState:
        person = world.get_person("grad_kim")
        person.current_room_id = "main_wet_lab"
        person.current_zone_id = "bench_1"
        person.current_ppe = set()
        person.carrying.clear()

        world.add_equipment(
            Equipment(
                id="pipette_pcr_1",
                name="PCR Setup Pipette",
                equipment_type="pipette",
                room_id="main_wet_lab",
                zone_id="bench_1",
                status=EquipmentStatus.READY,
                capabilities=["transfer"],
            )
        )
        world.add_material(
            Material(
                id="pcr_master_mix",
                name="PCR Master Mix",
                room_id="main_wet_lab",
                zone_id="bench_1",
                quantity=100.0,
                quantity_unit="uL",
            )
        )
        world.add_sample(
            Sample(
                id="pcr_reaction_1",
                name="PCR Reaction 1",
                sample_type="pcr_reaction",
                room_id="main_wet_lab",
                zone_id="bench_1",
                volume_ul=0.0,
                created_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
                created_by="grad_kim",
            )
        )
        return world

    def build_scripted_agents(self) -> list[ScriptedAgent]:
        return [
            ScriptedAgent(
                "grad_kim",
                [
                    *basic_lab_ppe_steps(),
                    pipette_transfer_step(
                        "pipette_pcr_1",
                        source_id="pcr_master_mix",
                        destination_id="pcr_reaction_1",
                        volume_ul=25.0,
                    ),
                    start_equipment_step(
                        "pcr_1",
                        parameters={
                            "command_name": "run_program",
                            "denaturation_temp": 95.0,
                            "annealing_temp": 55.0,
                            "extension_temp": 72.0,
                            "cycle_count": 30,
                            "hold_temp": 4.0,
                        },
                    ),
                ],
                name="PCR Script",
            )
        ]

    def evaluate(self, event_log: EventLog, final_state: WorldState) -> BenchmarkScore:
        sample = final_state.get_sample("pcr_reaction_1")
        pcr_machine = final_state.get_equipment("pcr_1")
        parameters = latest_action_parameters(
            event_log,
            "start_equipment",
            source_id="grad_kim",
            target_id="pcr_1",
        )
        expected_parameters = {
            "command_name": "run_program",
            "denaturation_temp": 95.0,
            "annealing_temp": 55.0,
            "extension_temp": 72.0,
            "cycle_count": 30,
            "hold_temp": 4.0,
        }

        master_mix_transferred = sample.volume_ul == 25.0
        program_parameters_correct = all(
            parameters.get(name) == value for name, value in expected_parameters.items()
        )
        thermocycler_started = pcr_machine.running_command == "run_program"
        no_safety_violations = not event_log.query(event_type=EventType.SAFETY_VIOLATION)

        completion_ratio = (
            0.4 * float(master_mix_transferred)
            + 0.3 * float(program_parameters_correct)
            + 0.2 * float(thermocycler_started)
            + 0.1 * float(no_safety_violations)
        )
        success = (
            master_mix_transferred
            and program_parameters_correct
            and thermocycler_started
            and no_safety_violations
        )
        return self._score(
            event_log,
            success=success,
            completion_ratio=completion_ratio,
            details={
                "master_mix_transferred": master_mix_transferred,
                "program_parameters_correct": program_parameters_correct,
                "thermocycler_started": thermocycler_started,
                "no_safety_violations": no_safety_violations,
            },
        )


__all__ = ["PCRSetupScenario"]
