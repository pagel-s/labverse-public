"""Action space calculation for agent-facing current options.

The surface produced here is **affordance-only**: every action it returns is
*structurally* applicable to entities visible in the agent's current room (a
door exists, an instrument is here, the agent is carrying a sample, ...). It
does **not** pre-evaluate safety preconditions, validators, or device-state
preconditions — those are discovered at attempt time, when the GameMaster
returns a BLOCKED result with a reason. This keeps the agent surface honest:
what the agent learns about the world's rules has to come from interaction,
not from a hint-leaking listing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from labverse.core.clock import SimulationClock
from labverse.core.types import ActionCategory
from labverse.models.action import Action, ActionGroup, ActionSpec, AvailableAction
from labverse.models.equipment import Equipment, EquipmentCommand, EquipmentParameter
from labverse.models.material import Material
from labverse.models.sample import Sample
from labverse.safety import minimum_ppe_for_bsl, room_required_ppe
from labverse.world.spatial import SpatialIndex
from labverse.world.state import WorldState

# Stable cap on the pipette transfer cross-product. With N entities the
# branching is N*(N-1); a busy room can blow past LLM tool-schema oneOf limits
# without this cap. The selection rule is deterministic: lexicographic order on
# (source.id, destination.id), take the first MAX_PIPETTE_TRANSFER_BRANCHES.
MAX_PIPETTE_TRANSFER_BRANCHES = 30

_GROUP_ORDER = (
    "movement",
    "doors",
    "ppe",
    "equipment",
    "materials",
    "samples",
    "wait",
    "observe",
)
_GROUP_LABELS = {
    "movement": "Movement",
    "doors": "Doors",
    "ppe": "PPE",
    "equipment": "Equipment",
    "materials": "Materials",
    "samples": "Samples",
    "wait": "Wait",
    "observe": "Observe",
}
_GROUP_PREFIXES = {
    "movement": "move",
    "doors": "door",
    "ppe": "ppe",
    "equipment": "equip",
    "materials": "mat",
    "samples": "sample",
    "wait": "wait",
    "observe": "observe",
}


class ActionSpaceCalculator:
    """Compute the set of currently available actions for an agent.

    Affordance-only: returns every action whose verb structurally applies to
    entities the agent can perceive. Precondition discovery happens via
    attempt-and-block in :class:`labverse.gamemaster.adjudicator.GameMaster`.
    """

    def __init__(
        self,
        world: WorldState,
        clock: SimulationClock,
        spatial: SpatialIndex,
    ) -> None:
        self._world = world
        self._clock = clock
        self._spatial = spatial

    def calculate(self, agent_id: str) -> ActionSpec:
        person = self._world.get_person(agent_id)
        if person.current_room_id is None:
            return ActionSpec(
                agent_id=agent_id,
                tick=self._clock.tick,
                context=f"{person.name} is not currently in a room",
            )

        available_actions: list[AvailableAction] = []
        current_room_id = person.current_room_id
        current_room = self._world.get_room(current_room_id)
        neighbors = sorted(self._spatial.neighbors(current_room_id))

        for neighbor in neighbors:
            door = self._spatial.get_door(current_room_id, neighbor)
            neighbor_room = self._world.get_room(neighbor)
            move_action = Action(
                id=f"preview_move_{neighbor}",
                agent_id=agent_id,
                category=ActionCategory.MOVE,
                verb="move_to",
                target_id=neighbor,
            )
            self._append_action(
                available_actions,
                action=move_action,
                description=f"Move to {neighbor_room.name}",
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

            if door is None:
                continue

            if door.locked:
                unlock_action = Action(
                    id=f"preview_unlock_{door.id}",
                    agent_id=agent_id,
                    category=ActionCategory.MOVE,
                    verb="unlock_door",
                    target_id=door.id,
                )
                self._append_action(
                    available_actions,
                    action=unlock_action,
                    description=f"Unlock door to {neighbor_room.name}",
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )

            if not door.is_open:
                open_action = Action(
                    id=f"preview_open_{door.id}",
                    agent_id=agent_id,
                    category=ActionCategory.MOVE,
                    verb="open_door",
                    target_id=door.id,
                )
                self._append_action(
                    available_actions,
                    action=open_action,
                    description=f"Open door to {neighbor_room.name}",
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )
            else:
                close_action = Action(
                    id=f"preview_close_{door.id}",
                    agent_id=agent_id,
                    category=ActionCategory.MOVE,
                    verb="close_door",
                    target_id=door.id,
                )
                self._append_action(
                    available_actions,
                    action=close_action,
                    description=f"Close door to {neighbor_room.name}",
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )

        room_materials = self._world.materials_in_room(current_room_id)
        room_samples = self._world.samples_in_room(current_room_id)
        room_equipment = self._world.equipment_in_room(current_room_id)

        for equipment in sorted(room_equipment, key=lambda candidate: candidate.id):
            default_command_name = next(iter(equipment.commands), "run")
            command = equipment.commands.get(default_command_name)
            fixed_parameters: dict[str, object] = {"command_name": default_command_name}
            required_parameters: list[str] = []
            optional_parameters: list[str] = []
            parameter_details: dict[str, str] = {}
            if command is not None:
                for parameter in command.parameters:
                    # Generic rule: if the command itself fixes a duration, the
                    # `duration_seconds` parameter is not free — it comes from
                    # the YAML command definition. Applies to any device.
                    if (
                        parameter.name == "duration_seconds"
                        and command.duration_seconds is not None
                    ):
                        fixed_parameters["duration_seconds"] = command.duration_seconds
                        continue
                    # Generic rule: a parameter is optional iff it has a
                    # default available (either at the parameter or at the
                    # equipment property level). Required otherwise. This
                    # replaces the previous device-specific switch.
                    if parameter.default_value is not None:
                        optional_parameters.append(parameter.name)
                    else:
                        required_parameters.append(parameter.name)
                    parameter_details[parameter.name] = self._parameter_detail(
                        equipment,
                        parameter,
                    )
            start_description = _equipment_start_description(
                equipment,
                command_name=default_command_name,
                command=command,
            )
            start_action = Action(
                id=f"preview_start_{equipment.id}",
                agent_id=agent_id,
                category=ActionCategory.USE_EQUIPMENT,
                verb="start_equipment",
                target_id=equipment.id,
                parameters=dict(fixed_parameters),
            )
            self._append_action(
                available_actions,
                action=start_action,
                description=start_description,
                fixed_parameters=fixed_parameters,
                required_parameters=required_parameters,
                optional_parameters=optional_parameters,
                parameter_details=parameter_details,
                parameter_schema=(
                    _parameter_object_schema(
                        properties={
                            parameter.name: _equipment_parameter_schema(parameter)
                            for parameter in (command.parameters if command is not None else [])
                            if parameter.name in {*required_parameters, *optional_parameters}
                        },
                        required=required_parameters,
                        description=(
                            f"Free parameters for starting {equipment.name}. "
                            "Fixed command parameters are already selected."
                        ),
                    )
                    if required_parameters or optional_parameters
                    else None
                ),
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

            stop_action = Action(
                id=f"preview_stop_{equipment.id}",
                agent_id=agent_id,
                category=ActionCategory.USE_EQUIPMENT,
                verb="stop_equipment",
                target_id=equipment.id,
            )
            self._append_action(
                available_actions,
                action=stop_action,
                description=f"Stop {equipment.name} early and interrupt the current run",
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

            if "transfer" in equipment.capabilities:
                self._enumerate_transfers(
                    available_actions,
                    agent_id=agent_id,
                    equipment=equipment,
                    room_materials=room_materials,
                    room_samples=room_samples,
                )

        required_ppe = set(room_required_ppe(current_room))
        for neighbor in neighbors:
            required_ppe.update(room_required_ppe(self._world.get_room(neighbor)))
        for material in room_materials:
            required_ppe.update(material.required_ppe)
        for sample in room_samples:
            required_ppe.update(minimum_ppe_for_bsl(sample.bsl_level))

        for ppe_type in sorted(required_ppe - person.current_ppe, key=lambda item: item.value):
            action = Action(
                id=f"preview_don_{ppe_type.value}",
                agent_id=agent_id,
                category=ActionCategory.SAFETY,
                verb="don_ppe",
                parameters={"ppe_type": ppe_type.value},
            )
            self._append_action(
                available_actions,
                action=action,
                description=f"Don {_humanize_token(ppe_type.value)}",
                fixed_parameters={"ppe_type": ppe_type.value},
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

        for ppe_type in sorted(person.current_ppe, key=lambda item: item.value):
            action = Action(
                id=f"preview_doff_{ppe_type.value}",
                agent_id=agent_id,
                category=ActionCategory.SAFETY,
                verb="doff_ppe",
                parameters={"ppe_type": ppe_type.value},
            )
            self._append_action(
                available_actions,
                action=action,
                description=f"Remove {_humanize_token(ppe_type.value)}",
                fixed_parameters={"ppe_type": ppe_type.value},
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

        for material in sorted(room_materials, key=lambda candidate: candidate.id):
            zone_candidates = [(None, None), *[(zone.id, zone.name) for zone in current_room.zones]]
            for zone_id, zone_name in zone_candidates:
                parameters = {"waste_category": material.waste_category.value}
                if zone_id is not None:
                    parameters["zone_id"] = zone_id
                action = Action(
                    id=f"preview_dispose_{material.id}_{zone_id or 'room'}",
                    agent_id=agent_id,
                    category=ActionCategory.HANDLE_MATERIAL,
                    verb="dispose_material",
                    target_id=material.id,
                    parameters=parameters,
                )
                description = f"Dispose of {material.name}"
                if zone_name is not None:
                    description += f" in {zone_name}"
                self._append_action(
                    available_actions,
                    action=action,
                    description=description,
                    fixed_parameters=parameters,
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )

        for sample in sorted(room_samples, key=lambda candidate: candidate.id):
            retrieve_action = Action(
                id=f"preview_retrieve_{sample.id}",
                agent_id=agent_id,
                category=ActionCategory.HANDLE_SAMPLE,
                verb="retrieve_sample",
                target_id=sample.id,
            )
            self._append_action(
                available_actions,
                action=retrieve_action,
                description=f"Retrieve sample {sample.name}",
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

        for sample_id in sorted(person.carrying):
            if sample_id not in self._world.samples:
                continue
            sample = self._world.get_sample(sample_id)
            room_store_action = Action(
                id=f"preview_store_room_{sample_id}",
                agent_id=agent_id,
                category=ActionCategory.HANDLE_SAMPLE,
                verb="store_sample",
                target_id=sample_id,
                parameters={"destination_room_id": current_room_id},
            )
            self._append_action(
                available_actions,
                action=room_store_action,
                description=f"Store sample {sample.name} in {current_room.name}",
                fixed_parameters={"destination_room_id": current_room_id},
                optional_parameters=["destination_zone_id"],
                parameter_details={
                    "destination_zone_id": "Optional zone within the current room for storage."
                },
                parameter_schema=_parameter_object_schema(
                    properties={
                        "destination_zone_id": {
                            "type": "string",
                            "enum": [zone.id for zone in current_room.zones],
                            "description": "Optional zone within the current room for storage.",
                        }
                    }
                    if current_room.zones
                    else {},
                    required=[],
                    description=(
                        "Optional storage placement inside the current room."
                        if current_room.zones
                        else "No additional parameter overrides are expected for this action."
                    ),
                )
                if current_room.zones
                else None,
                estimated_duration_seconds=self._clock.seconds_per_tick,
            )

            for equipment in sorted(room_equipment, key=lambda candidate: candidate.id):
                equipment_store_action = Action(
                    id=f"preview_store_{sample_id}_{equipment.id}",
                    agent_id=agent_id,
                    category=ActionCategory.HANDLE_SAMPLE,
                    verb="store_sample",
                    target_id=sample_id,
                    parameters={"destination_equipment_id": equipment.id},
                )
                self._append_action(
                    available_actions,
                    action=equipment_store_action,
                    description=f"Store sample {sample.name} in {equipment.name}",
                    fixed_parameters={"destination_equipment_id": equipment.id},
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )

        wait_action = Action(
            id="preview_wait",
            agent_id=agent_id,
            category=ActionCategory.OBSERVE,
            verb="wait",
        )
        self._append_action(
            available_actions,
            action=wait_action,
            description="Wait one turn without changing the world",
            estimated_duration_seconds=self._clock.seconds_per_tick,
        )

        log_action = Action(
            id="preview_log_observation",
            agent_id=agent_id,
            category=ActionCategory.OBSERVE,
            verb="log_observation",
        )
        self._append_action(
            available_actions,
            action=log_action,
            description="Record an observation note",
            optional_parameters=["notes"],
            parameter_details={
                "notes": (
                    "Observation or report text. If the task contract specifies an exact note "
                    "or finding format, copy that format exactly."
                )
            },
            parameter_schema=_parameter_object_schema(
                properties={
                    "notes": {
                        "type": "string",
                        "description": (
                            "Observation or report text. If the task contract specifies an exact "
                            "note or finding format, copy that format exactly."
                        ),
                    }
                },
                required=[],
                description="Optional note attached to the observation report action.",
            ),
            estimated_duration_seconds=self._clock.seconds_per_tick,
        )

        return self._build_action_spec(
            agent_id=agent_id,
            available_actions=available_actions,
            context=(
                f"{person.name} is in {current_room.name} with "
                f"{len(room_equipment)} equipment items, {len(room_materials)} materials, "
                f"and {len(room_samples)} samples visible"
            ),
        )

    def _enumerate_transfers(
        self,
        available_actions: list[AvailableAction],
        *,
        agent_id: str,
        equipment: Equipment,
        room_materials: list[Material],
        room_samples: list[Sample],
    ) -> None:
        transfer_entities: list[Material | Sample] = [
            *sorted(room_materials, key=lambda candidate: candidate.id),
            *sorted(room_samples, key=lambda candidate: candidate.id),
        ]
        emitted = 0
        for source in transfer_entities:
            if emitted >= MAX_PIPETTE_TRANSFER_BRANCHES:
                break
            for destination in transfer_entities:
                if emitted >= MAX_PIPETTE_TRANSFER_BRANCHES:
                    break
                if source.id == destination.id:
                    continue
                transfer_action = Action(
                    id=(f"preview_pipette_transfer_{equipment.id}_{source.id}_{destination.id}"),
                    agent_id=agent_id,
                    category=ActionCategory.HANDLE_SAMPLE,
                    verb="pipette_transfer",
                    target_id=equipment.id,
                    parameters={
                        "source_id": source.id,
                        "destination_id": destination.id,
                        "volume_ul": 10.0,
                    },
                )
                self._append_action(
                    available_actions,
                    action=transfer_action,
                    description=(
                        f"Transfer from {source.name} to {destination.name} using {equipment.name}"
                    ),
                    fixed_parameters={
                        "source_id": source.id,
                        "destination_id": destination.id,
                    },
                    required_parameters=["volume_ul"],
                    parameter_details={
                        "volume_ul": "Transfer volume in uL between the fixed source "
                        "and destination."
                    },
                    parameter_schema=_parameter_object_schema(
                        properties={
                            "volume_ul": {
                                "type": "number",
                                "minimum": 0.000001,
                                "description": (
                                    "Transfer volume in uL between the fixed source and "
                                    "destination."
                                ),
                            }
                        },
                        required=["volume_ul"],
                        description="Free parameters for this pipette transfer action.",
                    ),
                    estimated_duration_seconds=self._clock.seconds_per_tick,
                )
                emitted += 1

    def _append_action(
        self,
        available_actions: list[AvailableAction],
        *,
        action: Action,
        description: str,
        fixed_parameters: Mapping[str, object] | None = None,
        required_parameters: list[str] | None = None,
        optional_parameters: list[str] | None = None,
        parameter_details: Mapping[str, str] | None = None,
        parameter_schema: Mapping[str, Any] | None = None,
        estimated_duration_seconds: float | None = None,
    ) -> None:
        normalized_parameter_schema = dict(parameter_schema or {})
        available_actions.append(
            AvailableAction(
                action_id=action.id,
                verb=action.verb,
                category=action.category,
                label=description,
                description=description,
                target_id=action.target_id,
                target_type=self._target_type_for(action),
                fixed_parameters=dict(fixed_parameters or {}),
                required_parameters=required_parameters or [],
                optional_parameters=optional_parameters or [],
                parameter_details=dict(parameter_details or {}),
                parameter_schema=normalized_parameter_schema,
                free_parameter_schema=normalized_parameter_schema,
                estimated_duration_seconds=estimated_duration_seconds,
            )
        )

    def _build_action_spec(
        self,
        *,
        agent_id: str,
        available_actions: list[AvailableAction],
        context: str,
    ) -> ActionSpec:
        grouped_candidates: dict[str, list[AvailableAction]] = {key: [] for key in _GROUP_ORDER}
        group_counts = {key: 0 for key in _GROUP_ORDER}
        for action in available_actions:
            group_key = self._group_key_for(action)
            group_counts[group_key] += 1
            action.group_key = group_key
            action.group_label = _GROUP_LABELS[group_key]
            action.candidate_ref = f"{_GROUP_PREFIXES[group_key]}_{group_counts[group_key]}"
            action.target_label = self._target_label_for(action)
            if not action.label:
                action.label = action.description or _humanize_token(action.verb)
            grouped_candidates[group_key].append(action)

        groups = [
            ActionGroup(
                group_key=group_key,
                label=_GROUP_LABELS[group_key],
                candidates=grouped_candidates[group_key],
            )
            for group_key in _GROUP_ORDER
            if grouped_candidates[group_key]
        ]
        return ActionSpec(
            agent_id=agent_id,
            tick=self._clock.tick,
            groups=groups,
            available_actions=available_actions,
            context=context,
        )

    def _group_key_for(self, action: AvailableAction) -> str:
        if action.verb == "move_to":
            return "movement"
        if action.verb in {"open_door", "close_door", "unlock_door"}:
            return "doors"
        if action.verb in {"don_ppe", "doff_ppe"}:
            return "ppe"
        if action.verb in {"start_equipment", "stop_equipment"}:
            return "equipment"
        if action.verb == "dispose_material":
            return "materials"
        if action.verb in {"pipette_transfer", "retrieve_sample", "store_sample"}:
            return "samples"
        if action.verb == "log_observation":
            return "observe"
        return "observe"

    def _target_label_for(self, action: AvailableAction) -> str | None:
        if action.target_id is None:
            return None
        if action.target_id in self._world.rooms:
            return self._world.get_room(action.target_id).name
        if action.target_id in self._world.equipment:
            return self._world.get_equipment(action.target_id).name
        if action.target_id in self._world.materials:
            return self._world.get_material(action.target_id).name
        if action.target_id in self._world.samples:
            return self._world.get_sample(action.target_id).name
        for door in self._world.doors:
            if door.id != action.target_id:
                continue
            connected_names = [self._world.get_room(room_id).name for room_id in door.connects]
            if connected_names:
                return "Door: " + " / ".join(connected_names)
            return door.id
        return action.target_id

    def _target_type_for(self, action: Action) -> str | None:
        if action.target_id is None:
            return None
        if action.target_id in self._world.rooms:
            return "room"
        if any(door.id == action.target_id for door in self._world.doors):
            return "door"
        if action.target_id in self._world.equipment:
            return "equipment"
        if action.target_id in self._world.materials:
            return "material"
        if action.target_id in self._world.samples:
            return "sample"
        return None

    def _parameter_detail(
        self,
        equipment: Equipment,
        parameter: EquipmentParameter,
    ) -> str:
        detail_parts = [parameter.description or parameter.name.replace("_", " ")]
        if parameter.unit:
            detail_parts.append(f"unit={parameter.unit}")
        range_parts: list[str] = []
        if parameter.min_value is not None:
            range_parts.append(f"min={parameter.min_value:g}")
        if parameter.max_value is not None:
            range_parts.append(f"max={parameter.max_value:g}")
        if range_parts:
            detail_parts.append(", ".join(range_parts))
        property_state = equipment.properties.get(parameter.name)
        if property_state is not None:
            current_value = property_state.current_value
            default_value = property_state.default_value
            if current_value is not None:
                detail_parts.append(f"current={current_value}")
            elif default_value is not None:
                detail_parts.append(f"default={default_value}")
        elif parameter.default_value is not None:
            detail_parts.append(f"default={parameter.default_value}")
        return "; ".join(detail_parts)


def _humanize_token(value: str) -> str:
    return value.replace("_", " ")


def _equipment_start_description(
    equipment: Equipment,
    *,
    command_name: str,
    command: EquipmentCommand | None,
) -> str:
    if command is not None and command.description:
        description = f"Start {equipment.name}: {command.description}"
    else:
        description = f"Start {equipment.name} ({_humanize_token(command_name)})"
    if command is not None and command.duration_seconds is not None:
        description += (
            f"; if left running it completes automatically after {command.duration_seconds:.0f}s"
        )
    return description


def _equipment_parameter_schema(parameter: EquipmentParameter) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "description": parameter.description or _humanize_token(parameter.name),
    }
    schema["type"] = _json_schema_type(parameter.param_type)
    if parameter.allowed_values:
        schema["enum"] = list(parameter.allowed_values)
    if parameter.min_value is not None:
        schema["minimum"] = parameter.min_value
    if parameter.max_value is not None:
        schema["maximum"] = parameter.max_value
    return schema


def _json_schema_type(param_type: str) -> str:
    normalized = param_type.casefold()
    if normalized == "int":
        return "integer"
    if normalized in {"float", "number"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    return "string"


def _parameter_object_schema(
    *,
    properties: Mapping[str, Any],
    required: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(required),
        "description": description,
    }


__all__ = ["ActionSpaceCalculator", "MAX_PIPETTE_TRANSFER_BRANCHES"]
