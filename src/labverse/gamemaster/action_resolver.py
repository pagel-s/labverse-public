"""Verb handlers for the GameMaster action pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from labverse.core.clock import SimulationClock
from labverse.core.events import Event, EventType
from labverse.core.types import BSLLevel, CertificationType, EquipmentStatus, PPEType
from labverse.equipment.backends.simulated import SimulatedBackend
from labverse.models.action import Action
from labverse.models.equipment import Equipment
from labverse.models.material import Material
from labverse.models.person import Person
from labverse.models.sample import CustodyRecord, Sample
from labverse.models.space import Door
from labverse.world.spatial import SpatialIndex
from labverse.world.state import WorldState


@dataclass
class ScheduledEffect:
    """A timed callback request emitted by an action handler."""

    event_type: str
    target_id: str
    delay_seconds: float
    callback_data: dict[str, object] = field(default_factory=dict)


@dataclass
class CancellationRequest:
    """A request to cancel previously scheduled timed events."""

    event_type: str | None = None
    target_id: str | None = None


@dataclass
class ResolvedAction:
    """Fully resolved side effects for a validated action."""

    outcome_description: str
    state_changes: list[dict[str, object]] = field(default_factory=list)
    time_elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    scheduled_effects: list[ScheduledEffect] = field(default_factory=list)
    cancellations: list[CancellationRequest] = field(default_factory=list)
    emitted_events: list[Event] = field(default_factory=list)


@dataclass
class ResolverContext:
    """Runtime services handlers can mutate and inspect."""

    world: WorldState
    clock: SimulationClock
    spatial: SpatialIndex
    equipment_backend: SimulatedBackend


class ActionHandler(ABC):
    """Interface and metadata for a verb-specific action handler."""

    verb: str
    description: str = ""
    requires_target: bool = True
    target_types: tuple[str, ...] = ()

    def validate(self, action: Action, world: WorldState) -> list[str]:
        """Validate action shape and parameter types before preconditions run."""
        del action, world
        return []

    @abstractmethod
    def preconditions(self, action: Action, context: ResolverContext) -> list[str]: ...

    @abstractmethod
    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction: ...


class MoveToHandler(ActionHandler):
    verb = "move_to"
    description = "Move to an adjacent room"
    target_types = ("room",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        target_room_id = _require_target(action)
        if person.current_room_id is None:
            return [f"Person {person.id} is not currently in a room"]
        can_move, reason = context.spatial.can_traverse(
            person.current_room_id,
            target_room_id,
            _person_bsl_clearance(person),
        )
        return [] if can_move else [reason]

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        source_room_id = person.current_room_id
        target_room_id = _require_target(action)
        person.current_room_id = target_room_id
        person.current_zone_id = None
        return ResolvedAction(
            outcome_description=f"{person.name} moved to {target_room_id}",
            state_changes=[
                {
                    "entity_type": "person",
                    "entity_id": person.id,
                    "field": "current_room_id",
                    "old": source_room_id,
                    "new": target_room_id,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.PERSON_MOVED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=person.id,
                    target_id=target_room_id,
                    description=f"{person.name} moved from {source_room_id} to {target_room_id}",
                )
            ],
        )


class OpenDoorHandler(ActionHandler):
    verb = "open_door"
    description = "Open a nearby door"
    target_types = ("door",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        failures = _door_interaction_failures(person, door)
        if door.locked:
            failures.append(f"Door {door.id} is locked")
        if door.is_open:
            failures.append(f"Door {door.id} is already open")
        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        door.is_open = True
        description = f"{person.name} opened {_door_description(door, person.current_room_id)}"
        return ResolvedAction(
            outcome_description=description,
            state_changes=[
                {
                    "entity_type": "door",
                    "entity_id": door.id,
                    "field": "is_open",
                    "old": False,
                    "new": True,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    target_id=door.id,
                    description=description,
                    data={"door_id": door.id, "state": "open", "connects": list(door.connects)},
                )
            ],
        )


class CloseDoorHandler(ActionHandler):
    verb = "close_door"
    description = "Close a nearby door"
    target_types = ("door",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        failures = _door_interaction_failures(person, door)
        if not door.is_open:
            failures.append(f"Door {door.id} is already closed")
        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        door.is_open = False
        description = f"{person.name} closed {_door_description(door, person.current_room_id)}"
        return ResolvedAction(
            outcome_description=description,
            state_changes=[
                {
                    "entity_type": "door",
                    "entity_id": door.id,
                    "field": "is_open",
                    "old": True,
                    "new": False,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    target_id=door.id,
                    description=description,
                    data={"door_id": door.id, "state": "closed", "connects": list(door.connects)},
                )
            ],
        )


class UnlockDoorHandler(ActionHandler):
    verb = "unlock_door"
    description = "Unlock a nearby door"
    target_types = ("door",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        failures = _door_interaction_failures(person, door)
        if not door.locked:
            failures.append(f"Door {door.id} is already unlocked")
        if door.requires_keycard and not _person_has_keycard(person, door):
            failures.append(f"Door {door.id} requires a keycard")
        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        door = context.world.get_door(_require_target(action))
        door.locked = False
        description = f"{person.name} unlocked {_door_description(door, person.current_room_id)}"
        return ResolvedAction(
            outcome_description=description,
            state_changes=[
                {
                    "entity_type": "door",
                    "entity_id": door.id,
                    "field": "locked",
                    "old": True,
                    "new": False,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    target_id=door.id,
                    description=description,
                    data={"door_id": door.id, "state": "unlocked", "connects": list(door.connects)},
                )
            ],
        )


class StartEquipmentHandler(ActionHandler):
    verb = "start_equipment"
    description = "Start an equipment command"
    target_types = ("equipment",)

    def validate(self, action: Action, world: WorldState) -> list[str]:
        if action.target_id is None or action.target_id not in world.equipment:
            return []

        equipment = world.get_equipment(action.target_id)
        duration = action.parameters.get("duration_seconds")
        if duration is not None and not isinstance(duration, (int, float)):
            return ["duration_seconds must be numeric"]
        if isinstance(duration, (int, float)) and float(duration) < 0:
            return ["duration_seconds cannot be negative"]

        command_name = _command_name(action, equipment)
        if equipment.commands and command_name not in equipment.commands:
            return [f"Equipment {equipment.id} does not support command {command_name!r}"]
        return []

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        equipment = context.world.get_equipment(_require_target(action))
        failures: list[str] = []

        if person.current_room_id != equipment.room_id:
            failures.append(
                f"Person {person.id} must be in room {equipment.room_id} to start {equipment.id}"
            )

        if (
            len(equipment.current_users) >= equipment.max_concurrent_users
            and action.agent_id not in equipment.current_users
        ):
            failures.append(f"Equipment {equipment.id} is at max concurrent users")

        if equipment.status in {
            EquipmentStatus.RUNNING,
            EquipmentStatus.COOLING_DOWN,
            EquipmentStatus.MAINTENANCE,
            EquipmentStatus.ERROR,
        }:
            failures.append(
                f"Equipment {equipment.id} is not startable from {equipment.status.value}"
            )

        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        equipment = context.world.get_equipment(_require_target(action))
        command_name = _command_name(action, equipment)
        duration_seconds = _command_duration(action, equipment)
        old_status = equipment.status.value

        context.equipment_backend.start_command(
            equipment.id,
            action.agent_id,
            command_name,
            started_at=context.clock.current_time,
            duration_seconds=duration_seconds,
        )

        scheduled_effects: list[ScheduledEffect] = []
        if duration_seconds is not None and duration_seconds > 0:
            scheduled_effects.append(
                ScheduledEffect(
                    event_type="equipment_complete",
                    target_id=equipment.id,
                    delay_seconds=duration_seconds,
                    callback_data={"command_name": command_name},
                )
            )

        return ResolvedAction(
            outcome_description=f"{equipment.name} started command {command_name}",
            state_changes=[
                {
                    "entity_type": "equipment",
                    "entity_id": equipment.id,
                    "field": "status",
                    "old": old_status,
                    "new": equipment.status.value,
                },
                {
                    "entity_type": "equipment",
                    "entity_id": equipment.id,
                    "field": "running_command",
                    "old": None,
                    "new": command_name,
                },
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            scheduled_effects=scheduled_effects,
            emitted_events=[
                Event(
                    event_type=EventType.EQUIPMENT_STATE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    target_id=equipment.id,
                    description=f"{equipment.name} entered {equipment.status.value}",
                    data={"status": equipment.status.value, "command_name": command_name},
                )
            ],
        )


class StopEquipmentHandler(ActionHandler):
    verb = "stop_equipment"
    description = "Stop a running piece of equipment"
    target_types = ("equipment",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        equipment = context.world.get_equipment(_require_target(action))
        failures: list[str] = []

        if person.current_room_id != equipment.room_id:
            failures.append(
                f"Person {person.id} must be in room {equipment.room_id} to stop {equipment.id}"
            )
        if equipment.status != EquipmentStatus.RUNNING:
            failures.append(f"Equipment {equipment.id} is not currently running")
        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        equipment = context.world.get_equipment(_require_target(action))
        old_command = equipment.running_command

        context.equipment_backend.stop_command(equipment.id, agent_id=action.agent_id)

        return ResolvedAction(
            outcome_description=f"{equipment.name} stopped command {old_command or 'run'}",
            state_changes=[
                {
                    "entity_type": "equipment",
                    "entity_id": equipment.id,
                    "field": "status",
                    "old": EquipmentStatus.RUNNING.value,
                    "new": equipment.status.value,
                },
                {
                    "entity_type": "equipment",
                    "entity_id": equipment.id,
                    "field": "running_command",
                    "old": old_command,
                    "new": None,
                },
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            cancellations=[
                CancellationRequest(event_type="equipment_complete", target_id=equipment.id)
            ],
            emitted_events=[
                Event(
                    event_type=EventType.EQUIPMENT_STATE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    target_id=equipment.id,
                    description=f"{equipment.name} entered {equipment.status.value}",
                    data={"status": equipment.status.value, "command_name": old_command},
                )
            ],
        )


class PipetteTransferHandler(ActionHandler):
    verb = "pipette_transfer"
    description = "Transfer liquid between materials or samples with a pipette"
    target_types = ("equipment",)

    def validate(self, action: Action, world: WorldState) -> list[str]:
        errors: list[str] = []
        if action.target_id is None or action.target_id not in world.equipment:
            return errors

        equipment = world.get_equipment(action.target_id)
        if equipment.equipment_type != "pipette":
            errors.append(f"Equipment {equipment.id} is not a pipette")

        volume_ul = action.parameters.get("volume_ul")
        if not isinstance(volume_ul, (int, float)):
            errors.append("volume_ul must be numeric")
        elif float(volume_ul) <= 0:
            errors.append("volume_ul must be positive")

        source_id = action.parameters.get("source_id")
        destination_id = action.parameters.get("destination_id")
        if not isinstance(source_id, str):
            errors.append("source_id is required")
        if not isinstance(destination_id, str):
            errors.append("destination_id is required")
        if (
            isinstance(source_id, str)
            and isinstance(destination_id, str)
            and source_id == destination_id
        ):
            errors.append("source_id and destination_id must differ")

        if isinstance(source_id, str) and not _entity_exists(
            source_id,
            ("material", "sample"),
            world,
        ):
            errors.append(f"Unknown transfer source: {source_id}")
        if isinstance(destination_id, str) and not _entity_exists(
            destination_id, ("material", "sample"), world
        ):
            errors.append(f"Unknown transfer destination: {destination_id}")
        return errors

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        pipette = context.world.get_equipment(_require_target(action))
        source = _transfer_entity(context.world, _require_string_parameter(action, "source_id"))
        destination = _transfer_entity(
            context.world,
            _require_string_parameter(action, "destination_id"),
        )
        volume_ul = _require_positive_float_parameter(action, "volume_ul")

        failures: list[str] = []
        if person.current_room_id != pipette.room_id:
            failures.append(
                f"Person {person.id} must be in room {pipette.room_id} to use {pipette.id}"
            )
        if _entity_room_id(source) != person.current_room_id:
            failures.append(f"Transfer source is not in room {person.current_room_id}")
        if _entity_room_id(destination) != person.current_room_id:
            failures.append(f"Transfer destination is not in room {person.current_room_id}")

        source_failure = _check_transfer_source_capacity(source, volume_ul)
        if source_failure is not None:
            failures.append(source_failure)

        if (
            isinstance(destination, Material)
            and _quantity_delta_for_material(
                destination.quantity_unit,
                volume_ul,
            )
            is None
        ):
            failures.append(
                f"Destination material {destination.id} uses unsupported unit "
                f"{destination.quantity_unit!r}"
            )

        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        source_id = _require_string_parameter(action, "source_id")
        destination_id = _require_string_parameter(action, "destination_id")
        volume_ul = _require_positive_float_parameter(action, "volume_ul")
        pipette = context.world.get_equipment(_require_target(action))
        source = _transfer_entity(context.world, source_id)
        destination = _transfer_entity(context.world, destination_id)

        state_changes: list[dict[str, object]] = []
        emitted_events: list[Event] = []

        if isinstance(source, Material):
            delta = _quantity_delta_for_material(source.quantity_unit, volume_ul)
            if delta is None:
                raise ValueError(f"Unsupported material unit: {source.quantity_unit}")
            old_quantity = source.quantity
            source.quantity -= delta
            state_changes.append(
                {
                    "entity_type": "material",
                    "entity_id": source.id,
                    "field": "quantity",
                    "old": old_quantity,
                    "new": source.quantity,
                }
            )
            emitted_events.append(
                Event(
                    event_type=EventType.MATERIAL_CONSUMED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=source.id,
                    target_id=destination.id,
                    description=f"{delta:g} {source.quantity_unit} transferred from {source.name}",
                    data={
                        "source_id": source.id,
                        "destination_id": destination.id,
                        "volume_ul": volume_ul,
                    },
                )
            )
        else:
            # The precondition refuses None-volume sources via
            # _check_transfer_source_capacity; this branch enforces the same
            # invariant defensively so the depletion arithmetic never silently
            # fabricates a starting volume.
            if source.volume_ul is None:
                raise ValueError(
                    f"Sample {source.id} has unknown volume and cannot be used as a transfer source"
                )
            old_volume = source.volume_ul
            source.volume_ul = source.volume_ul - volume_ul
            state_changes.append(
                {
                    "entity_type": "sample",
                    "entity_id": source.id,
                    "field": "volume_ul",
                    "old": old_volume,
                    "new": source.volume_ul,
                }
            )

        if isinstance(destination, Material):
            delta = _quantity_delta_for_material(destination.quantity_unit, volume_ul)
            if delta is None:
                raise ValueError(f"Unsupported material unit: {destination.quantity_unit}")
            old_quantity = destination.quantity
            destination.quantity += delta
            state_changes.append(
                {
                    "entity_type": "material",
                    "entity_id": destination.id,
                    "field": "quantity",
                    "old": old_quantity,
                    "new": destination.quantity,
                }
            )
        else:
            # Destination policy: a None starting volume means "the sample is
            # empty / freshly created" and is filled to exactly `volume_ul`.
            # This is asymmetric with the source side on purpose — an unknown
            # source volume is a contract violation (we cannot deplete what we
            # cannot measure), but a fresh destination is a common real case.
            destination_old_volume: float | None = destination.volume_ul
            destination.volume_ul = (destination.volume_ul or 0.0) + volume_ul
            state_changes.append(
                {
                    "entity_type": "sample",
                    "entity_id": destination.id,
                    "field": "volume_ul",
                    "old": destination_old_volume,
                    "new": destination.volume_ul,
                }
            )
            emitted_events.append(
                Event(
                    event_type=EventType.SAMPLE_UPDATED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=source.id,
                    target_id=destination.id,
                    description=(
                        f"{volume_ul:g} uL transferred into sample {destination.name} "
                        f"using {pipette.name}"
                    ),
                    data={
                        "source_id": source.id,
                        "destination_id": destination.id,
                        "volume_ul": volume_ul,
                    },
                )
            )

        return ResolvedAction(
            outcome_description=(
                f"Transferred {volume_ul:g} uL from {source.name} to {destination.name} "
                f"using {pipette.name}"
            ),
            state_changes=state_changes,
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=emitted_events,
        )


class DonPPEHandler(ActionHandler):
    verb = "don_ppe"
    description = "Put on a piece of PPE"
    requires_target = False

    def validate(self, action: Action, world: WorldState) -> list[str]:
        del world
        if _resolve_ppe_type(action) is None:
            return ["Action don_ppe requires a valid PPE type"]
        return []

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        ppe_type = _resolve_ppe_type(action)
        if ppe_type is None:
            return []
        if ppe_type in person.current_ppe:
            return [f"Person {person.id} is already wearing {ppe_type.value}"]
        return []

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        ppe_type = _resolve_ppe_type(action)
        if ppe_type is None:
            raise ValueError("don_ppe requires a PPE type")
        old_ppe = sorted(item.value for item in person.current_ppe)
        person.current_ppe.add(ppe_type)
        new_ppe = sorted(item.value for item in person.current_ppe)
        return ResolvedAction(
            outcome_description=f"{person.name} donned {ppe_type.value}",
            state_changes=[
                {
                    "entity_type": "person",
                    "entity_id": person.id,
                    "field": "current_ppe",
                    "old": old_ppe,
                    "new": new_ppe,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=person.id,
                    description=f"{person.name} donned {ppe_type.value}",
                    data={"ppe_type": ppe_type.value},
                )
            ],
        )


class DoffPPEHandler(ActionHandler):
    verb = "doff_ppe"
    description = "Remove a piece of PPE"
    requires_target = False

    def validate(self, action: Action, world: WorldState) -> list[str]:
        del world
        if _resolve_ppe_type(action) is None:
            return ["Action doff_ppe requires a valid PPE type"]
        return []

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        ppe_type = _resolve_ppe_type(action)
        if ppe_type is None:
            return []
        if ppe_type not in person.current_ppe:
            return [f"Person {person.id} is not wearing {ppe_type.value}"]
        return []

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        ppe_type = _resolve_ppe_type(action)
        if ppe_type is None:
            raise ValueError("doff_ppe requires a PPE type")
        old_ppe = sorted(item.value for item in person.current_ppe)
        person.current_ppe.remove(ppe_type)
        new_ppe = sorted(item.value for item in person.current_ppe)
        return ResolvedAction(
            outcome_description=f"{person.name} removed {ppe_type.value}",
            state_changes=[
                {
                    "entity_type": "person",
                    "entity_id": person.id,
                    "field": "current_ppe",
                    "old": old_ppe,
                    "new": new_ppe,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=person.id,
                    description=f"{person.name} removed {ppe_type.value}",
                    data={"ppe_type": ppe_type.value},
                )
            ],
        )


class StoreSampleHandler(ActionHandler):
    verb = "store_sample"
    description = "Store a sample in a room zone or equipment"
    target_types = ("sample",)

    def validate(self, action: Action, world: WorldState) -> list[str]:
        errors: list[str] = []
        destination_room_id = action.parameters.get("destination_room_id")
        destination_equipment_id = action.parameters.get("destination_equipment_id")
        if destination_room_id is None and destination_equipment_id is None:
            errors.append("store_sample requires destination_room_id or destination_equipment_id")
        if destination_room_id is not None and destination_equipment_id is not None:
            errors.append(
                "store_sample cannot specify both destination_room_id and destination_equipment_id"
            )
        if isinstance(destination_room_id, str) and destination_room_id not in world.rooms:
            errors.append(f"Unknown destination room: {destination_room_id}")
        if (
            isinstance(destination_equipment_id, str)
            and destination_equipment_id not in world.equipment
        ):
            errors.append(f"Unknown destination equipment: {destination_equipment_id}")
        zone_id = action.parameters.get("destination_zone_id")
        if zone_id is not None and not isinstance(zone_id, str):
            errors.append("destination_zone_id must be a string when provided")
        return errors

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        sample = context.world.get_sample(_require_target(action))
        failures: list[str] = []

        if person.current_room_id is None:
            failures.append(f"Person {person.id} is not currently in a room")
            return failures

        if sample.id not in person.carrying and sample.room_id != person.current_room_id:
            failures.append(
                f"Sample {sample.id} is not accessible from room {person.current_room_id}"
            )

        destination_room_id = action.parameters.get("destination_room_id")
        if isinstance(destination_room_id, str) and destination_room_id != person.current_room_id:
            failures.append(
                f"Person {person.id} must already be in destination room {destination_room_id}"
            )

        destination_equipment_id = action.parameters.get("destination_equipment_id")
        if isinstance(destination_equipment_id, str):
            equipment = context.world.get_equipment(destination_equipment_id)
            if equipment.room_id != person.current_room_id:
                failures.append(
                    f"Destination equipment {equipment.id} is not in room {person.current_room_id}"
                )

        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        sample = context.world.get_sample(_require_target(action))
        old_carrying = list(person.carrying)
        old_room_id = sample.room_id
        old_zone_id = sample.zone_id
        old_equipment_id = sample.equipment_id
        old_status = sample.status

        destination_equipment_id = action.parameters.get("destination_equipment_id")
        destination_zone_id = action.parameters.get("destination_zone_id")
        if sample.id in person.carrying:
            person.carrying.remove(sample.id)

        if isinstance(destination_equipment_id, str):
            equipment = context.world.get_equipment(destination_equipment_id)
            sample.room_id = equipment.room_id
            sample.zone_id = equipment.zone_id
            sample.equipment_id = equipment.id
        else:
            sample.room_id = _require_string_parameter(action, "destination_room_id")
            sample.zone_id = destination_zone_id if isinstance(destination_zone_id, str) else None
            sample.equipment_id = None
        sample.status = "stored"
        sample.chain_of_custody.append(
            _custody_record(
                sample=sample,
                person_id=person.id,
                action_name="stored",
                timestamp=context.clock.current_time,
            )
        )

        return ResolvedAction(
            outcome_description=f"{sample.name} stored successfully",
            state_changes=[
                {
                    "entity_type": "person",
                    "entity_id": person.id,
                    "field": "carrying",
                    "old": old_carrying,
                    "new": list(person.carrying),
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "room_id",
                    "old": old_room_id,
                    "new": sample.room_id,
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "zone_id",
                    "old": old_zone_id,
                    "new": sample.zone_id,
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "equipment_id",
                    "old": old_equipment_id,
                    "new": sample.equipment_id,
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "status",
                    "old": old_status,
                    "new": sample.status,
                },
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.SAMPLE_UPDATED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=person.id,
                    target_id=sample.id,
                    description=f"{sample.name} stored in {sample.equipment_id or sample.room_id}",
                )
            ],
        )


class RetrieveSampleHandler(ActionHandler):
    verb = "retrieve_sample"
    description = "Retrieve a sample into the operator's possession"
    target_types = ("sample",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        sample = context.world.get_sample(_require_target(action))
        failures: list[str] = []

        if person.current_room_id is None:
            failures.append(f"Person {person.id} is not currently in a room")
            return failures
        if sample.room_id != person.current_room_id:
            failures.append(
                f"Sample {sample.id} is in room {sample.room_id}, not {person.current_room_id}"
            )
        if sample.id in person.carrying:
            failures.append(f"Person {person.id} is already carrying sample {sample.id}")
        return failures

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        person = context.world.get_person(action.agent_id)
        sample = context.world.get_sample(_require_target(action))
        old_carrying = list(person.carrying)
        old_zone_id = sample.zone_id
        old_equipment_id = sample.equipment_id
        old_status = sample.status

        person.carrying.append(sample.id)
        sample.equipment_id = None
        sample.zone_id = person.current_zone_id
        sample.status = "active"
        sample.chain_of_custody.append(
            _custody_record(
                sample=sample,
                person_id=person.id,
                action_name="retrieved",
                timestamp=context.clock.current_time,
            )
        )

        return ResolvedAction(
            outcome_description=f"{person.name} retrieved {sample.name}",
            state_changes=[
                {
                    "entity_type": "person",
                    "entity_id": person.id,
                    "field": "carrying",
                    "old": old_carrying,
                    "new": list(person.carrying),
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "equipment_id",
                    "old": old_equipment_id,
                    "new": None,
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "zone_id",
                    "old": old_zone_id,
                    "new": sample.zone_id,
                },
                {
                    "entity_type": "sample",
                    "entity_id": sample.id,
                    "field": "status",
                    "old": old_status,
                    "new": sample.status,
                },
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.SAMPLE_UPDATED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=person.id,
                    target_id=sample.id,
                    description=f"{person.name} retrieved {sample.name}",
                )
            ],
        )


class DisposeMaterialHandler(ActionHandler):
    verb = "dispose_material"
    description = "Dispose of a material into a waste stream"
    target_types = ("material",)

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        person = context.world.get_person(action.agent_id)
        material = context.world.get_material(_require_target(action))
        if person.current_room_id != material.room_id:
            return [
                f"Person {person.id} must be in room {material.room_id} to dispose {material.id}"
            ]
        return []

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        material = context.world.remove_material(_require_target(action))
        waste_stream = action.parameters.get(
            "waste_category",
            action.parameters.get("waste_stream"),
        )
        return ResolvedAction(
            outcome_description=(
                f"{material.name} disposed into {waste_stream or material.waste_category.value}"
            ),
            state_changes=[
                {
                    "entity_type": "material",
                    "entity_id": material.id,
                    "field": "removed",
                    "old": False,
                    "new": True,
                }
            ],
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.MATERIAL_CONSUMED,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=material.id,
                    description=f"{material.name} disposed",
                    data={"waste_stream": waste_stream or material.waste_category.value},
                )
            ],
        )


class LogObservationHandler(ActionHandler):
    verb = "log_observation"
    description = "Record an observation without changing the world"
    requires_target = False

    def validate(self, action: Action, world: WorldState) -> list[str]:
        del world
        notes = action.parameters.get("notes")
        if notes is not None and not isinstance(notes, str):
            return ["notes must be a string when provided"]
        return []

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        del action, context
        return []

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        notes = action.parameters.get("notes")
        description = notes if isinstance(notes, str) and notes else "Observation logged"
        return ResolvedAction(
            outcome_description=description,
            time_elapsed_seconds=context.clock.seconds_per_tick,
            emitted_events=[
                Event(
                    event_type=EventType.STATE_CHANGE,
                    tick=context.clock.tick,
                    timestamp=context.clock.current_time,
                    source_id=action.agent_id,
                    description=description,
                )
            ],
        )


class WaitHandler(ActionHandler):
    verb = "wait"
    description = "Wait one turn without changing the world"
    requires_target = False

    def validate(self, action: Action, world: WorldState) -> list[str]:
        del action, world
        return []

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        del action, context
        return []

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        del action
        return ResolvedAction(
            outcome_description="Waited one turn",
            time_elapsed_seconds=context.clock.seconds_per_tick,
        )


class ActionResolver:
    """Registry of the built-in action handlers."""

    def __init__(self) -> None:
        handlers = (
            MoveToHandler(),
            OpenDoorHandler(),
            CloseDoorHandler(),
            UnlockDoorHandler(),
            StartEquipmentHandler(),
            StopEquipmentHandler(),
            PipetteTransferHandler(),
            DonPPEHandler(),
            DoffPPEHandler(),
            StoreSampleHandler(),
            RetrieveSampleHandler(),
            DisposeMaterialHandler(),
            WaitHandler(),
            LogObservationHandler(),
        )
        self._handlers: dict[str, ActionHandler] = {handler.verb: handler for handler in handlers}

    @property
    def handlers(self) -> dict[str, ActionHandler]:
        return dict(self._handlers)

    @property
    def supported_verbs(self) -> set[str]:
        return set(self._handlers)

    def has_handler(self, verb: str) -> bool:
        return verb in self._handlers

    def handler_for(self, verb: str) -> ActionHandler:
        return self._handlers[verb]

    def preconditions(self, action: Action, context: ResolverContext) -> list[str]:
        return self.handler_for(action.verb).preconditions(action, context)

    def resolve(self, action: Action, context: ResolverContext) -> ResolvedAction:
        return self.handler_for(action.verb).resolve(action, context)


def _require_target(action: Action) -> str:
    if action.target_id is None:
        raise ValueError(f"Action {action.verb} requires a target")
    return action.target_id


def _command_name(action: Action, equipment: Equipment) -> str:
    raw_name = action.parameters.get("command_name")
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    return next(iter(equipment.commands), "run")


def _command_duration(action: Action, equipment: Equipment) -> float | None:
    explicit_duration = action.parameters.get("duration_seconds")
    if isinstance(explicit_duration, (int, float)):
        return float(explicit_duration)
    command_name = _command_name(action, equipment)
    if command_name in equipment.commands:
        return equipment.commands[command_name].duration_seconds
    return None


def _person_bsl_clearance(person: Person) -> BSLLevel:
    if person.has_certification(CertificationType.BSL3_TRAINING):
        return BSLLevel.BSL3
    if person.has_certification(CertificationType.BSL2_TRAINING):
        return BSLLevel.BSL2
    return BSLLevel.BSL1


def _door_interaction_failures(person: Person, door: Door) -> list[str]:
    if person.current_room_id is None:
        return [f"Person {person.id} is not currently in a room"]
    if person.current_room_id not in door.connects:
        return [f"Person {person.id} must be next to door {door.id} to use it"]
    return []


def _door_description(door: Door, current_room_id: str | None) -> str:
    other_rooms = [room_id for room_id in door.connects if room_id != current_room_id]
    if other_rooms:
        return f"door to {other_rooms[0]}"
    return f"door {door.id}"


def _person_has_keycard(person: Person, door: Door) -> bool:
    return "master_keycard" in person.carrying or f"keycard:{door.id}" in person.carrying


def _resolve_ppe_type(action: Action) -> PPEType | None:
    raw_value = action.parameters.get("ppe_type", action.target_id)
    if isinstance(raw_value, str):
        try:
            return PPEType(raw_value)
        except ValueError:
            return None
    return None


def _entity_exists(target_id: str, target_types: tuple[str, ...], world: WorldState) -> bool:
    for target_type in target_types:
        if target_type == "room" and target_id in world.rooms:
            return True
        if target_type == "equipment" and target_id in world.equipment:
            return True
        if target_type == "material" and target_id in world.materials:
            return True
        if target_type == "sample" and target_id in world.samples:
            return True
    return False


def _require_string_parameter(action: Action, name: str) -> str:
    raw_value = action.parameters.get(name)
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"Action {action.verb} requires string parameter {name!r}")
    return raw_value


def _require_positive_float_parameter(action: Action, name: str) -> float:
    raw_value = action.parameters.get(name)
    if not isinstance(raw_value, (int, float)):
        raise ValueError(f"Action {action.verb} requires numeric parameter {name!r}")
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"Action {action.verb} requires positive parameter {name!r}")
    return value


def _transfer_entity(world: WorldState, entity_id: str) -> Material | Sample:
    if entity_id in world.materials:
        return world.get_material(entity_id)
    if entity_id in world.samples:
        return world.get_sample(entity_id)
    raise KeyError(f"Unknown transfer entity: {entity_id}")


def _entity_room_id(entity: Material | Sample) -> str:
    return entity.room_id


def _check_transfer_source_capacity(entity: Material | Sample, volume_ul: float) -> str | None:
    if isinstance(entity, Material):
        delta = _quantity_delta_for_material(entity.quantity_unit, volume_ul)
        if delta is None:
            return f"Material {entity.id} uses unsupported unit {entity.quantity_unit!r}"
        if entity.quantity < delta:
            return f"Material {entity.id} does not have enough quantity for the transfer"
        return None

    if entity.volume_ul is None:
        return f"Sample {entity.id} has unknown volume and cannot be used as a transfer source"
    if entity.volume_ul < volume_ul:
        return f"Sample {entity.id} does not contain {volume_ul:g} uL"
    return None


def _quantity_delta_for_material(quantity_unit: str, volume_ul: float) -> float | None:
    normalized_unit = quantity_unit.casefold()
    if normalized_unit in {"ul", "µl"}:
        return volume_ul
    if normalized_unit == "ml":
        return volume_ul / 1000.0
    if normalized_unit == "l":
        return volume_ul / 1_000_000.0
    return None


def _custody_record(
    *,
    sample: Sample,
    person_id: str,
    action_name: str,
    timestamp: datetime,
) -> CustodyRecord:
    return CustodyRecord(
        timestamp=timestamp,
        person_id=person_id,
        action=action_name,
        location_room_id=sample.room_id,
        location_zone_id=sample.zone_id,
        location_equipment_id=sample.equipment_id,
    )


__all__ = [
    "ActionHandler",
    "ActionResolver",
    "CancellationRequest",
    "ResolvedAction",
    "ResolverContext",
    "ScheduledEffect",
]
