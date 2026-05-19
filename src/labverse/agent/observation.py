"""Observation builders for agent-facing world views."""

from __future__ import annotations

from collections.abc import Callable

from labverse.agent.interface import (
    DoorSummary,
    EquipmentSummary,
    MaterialSummary,
    Observation,
    PersonSummary,
    SampleSummary,
    SelfState,
)
from labverse.core.clock import SimulationClock
from labverse.core.events import EventLog, EventType
from labverse.world.state import WorldState


class ObservationBuilder:
    """Build agent-facing observations from the current world state."""

    def __init__(
        self,
        world: WorldState,
        clock: SimulationClock,
        event_log: EventLog,
        *,
        working_memory_provider: Callable[[str], str] | None = None,
        last_action_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        self._world = world
        self._clock = clock
        self._event_log = event_log
        self._working_memory_provider = working_memory_provider
        self._last_action_provider = last_action_provider

    def build(self, agent_id: str) -> Observation:
        person = self._world.get_person(agent_id)
        if person.current_room_id is None:
            raise ValueError(f"Person {person.id} is not currently in a room")

        room = self._world.get_room(person.current_room_id)
        visible_persons = [
            PersonSummary(
                id=other.id,
                name=other.name,
                role=other.role,
                current_zone_id=other.current_zone_id,
                current_activity=other.current_activity,
            )
            for other in sorted(
                self._world.persons_in_room(room.id),
                key=lambda candidate: candidate.id,
            )
            if other.id != person.id
        ]
        visible_equipment = [
            EquipmentSummary(
                id=equipment.id,
                name=equipment.name,
                equipment_type=equipment.equipment_type,
                status=equipment.status,
                zone_id=equipment.zone_id,
                running_command=equipment.running_command,
            )
            for equipment in sorted(
                self._world.equipment_in_room(room.id),
                key=lambda candidate: candidate.id,
            )
        ]
        visible_materials = [
            MaterialSummary(
                id=material.id,
                name=material.name,
                quantity=material.quantity,
                quantity_unit=material.quantity_unit,
                zone_id=material.zone_id,
                is_hazardous=material.is_hazardous,
            )
            for material in sorted(
                self._world.materials_in_room(room.id),
                key=lambda candidate: candidate.id,
            )
        ]
        visible_samples = [
            SampleSummary(
                id=sample.id,
                name=sample.name,
                sample_type=sample.sample_type,
                status=sample.status,
                zone_id=sample.zone_id,
                equipment_id=sample.equipment_id,
                bsl_level=sample.bsl_level,
                volume_ul=sample.volume_ul,
            )
            for sample in sorted(
                self._world.samples_in_room(room.id),
                key=lambda candidate: candidate.id,
            )
        ]
        visible_doors = [
            DoorSummary(
                id=door.id,
                connects=door.connects,
                is_open=door.is_open,
                locked=door.locked,
                requires_keycard=door.requires_keycard,
                requires_bsl_clearance=door.requires_bsl_clearance,
            )
            for door in sorted(
                [door for door in self._world.doors if room.id in door.connects],
                key=lambda candidate: candidate.id,
            )
        ]

        active_alerts = self._active_alerts(agent_id, room.id)
        recent_events = self._recent_events(agent_id, room.id)

        return Observation(
            agent_id=agent_id,
            timestamp=self._clock.current_time,
            tick=self._clock.tick,
            current_room=room.model_copy(deep=True),
            self_state=SelfState(
                id=person.id,
                name=person.name,
                role=person.role,
                current_room_id=person.current_room_id,
                current_zone_id=person.current_zone_id,
                current_ppe=set(person.current_ppe),
                carrying=list(person.carrying),
                current_activity=person.current_activity,
                certifications=[cert.cert_type for cert in person.certifications],
            ),
            visible_persons=visible_persons,
            visible_equipment=visible_equipment,
            visible_materials=visible_materials,
            visible_samples=visible_samples,
            visible_doors=visible_doors,
            carrying=list(person.carrying),
            current_ppe=set(person.current_ppe),
            active_alerts=active_alerts,
            recent_events=recent_events,
            working_memory=(
                self._working_memory_provider(agent_id)
                if self._working_memory_provider is not None
                else ""
            ),
            trace_parent_action_id=(
                self._last_action_provider(agent_id)
                if self._last_action_provider is not None
                else None
            ),
            narrative=self._narrative(
                room_name=room.name,
                visible_persons=visible_persons,
                visible_equipment=visible_equipment,
                visible_materials=visible_materials,
                visible_samples=visible_samples,
                visible_doors=visible_doors,
                carrying=list(person.carrying),
                current_ppe=sorted(item.value for item in person.current_ppe),
            ),
        )

    def _active_alerts(self, agent_id: str, room_id: str) -> list[str]:
        alerts: list[str] = []
        for event in reversed(self._event_log.events):
            if event.event_type not in {EventType.SAFETY_WARNING, EventType.SAFETY_VIOLATION}:
                continue
            if event.source_id == agent_id or event.target_id in {agent_id, room_id, ""}:
                alerts.append(event.description)
            if len(alerts) == 5:
                break
        return list(reversed(alerts))

    def _recent_events(self, agent_id: str, room_id: str) -> list[str]:
        descriptions: list[str] = []
        relevant_ids = {
            agent_id,
            room_id,
            *self._world.get_room(room_id).equipment_ids,
            *self._world.get_room(room_id).material_ids,
            *[sample.id for sample in self._world.samples_in_room(room_id)],
            *[door.id for door in self._world.doors if room_id in door.connects],
        }
        for event in reversed(self._event_log.events):
            if (
                event.source_id in relevant_ids
                or event.target_id in relevant_ids
                or event.source_id == agent_id
                or event.target_id == agent_id
            ):
                descriptions.append(event.description)
            if len(descriptions) == 5:
                break
        return list(reversed(descriptions))

    def _narrative(
        self,
        *,
        room_name: str,
        visible_persons: list[PersonSummary],
        visible_equipment: list[EquipmentSummary],
        visible_materials: list[MaterialSummary],
        visible_samples: list[SampleSummary],
        visible_doors: list[DoorSummary],
        carrying: list[str],
        current_ppe: list[str],
    ) -> str:
        person_phrase = (
            "no other people"
            if not visible_persons
            else ", ".join(person.name for person in visible_persons)
        )
        equipment_phrase = (
            "no equipment"
            if not visible_equipment
            else ", ".join(equipment.name for equipment in visible_equipment)
        )
        material_phrase = (
            "no materials"
            if not visible_materials
            else ", ".join(material.name for material in visible_materials)
        )
        sample_phrase = (
            "no samples"
            if not visible_samples
            else ", ".join(sample.name for sample in visible_samples)
        )
        door_phrase = (
            "no visible doors"
            if not visible_doors
            else ", ".join(
                f"{door.id} ({'open' if door.is_open else 'closed'}"
                f"{', locked' if door.locked else ''})"
                for door in visible_doors
            )
        )
        carrying_phrase = "nothing" if not carrying else ", ".join(carrying)
        ppe_phrase = "no PPE" if not current_ppe else ", ".join(current_ppe)
        return (
            f"You are in {room_name}. Visible people: {person_phrase}. "
            f"Visible equipment: {equipment_phrase}. Visible materials: {material_phrase}. "
            f"Visible samples: {sample_phrase}. Visible doors: {door_phrase}. "
            f"Carrying: {carrying_phrase}. Current PPE: {ppe_phrase}."
        )


__all__ = ["ObservationBuilder"]
