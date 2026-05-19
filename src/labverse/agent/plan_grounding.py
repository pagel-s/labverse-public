"""Shared grounding helpers for one-shot planners and replay binding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from labverse.agent.action_binding import (
    build_action_from_available,
)
from labverse.core.types import PPEType, WasteCategory
from labverse.models.action import Action, AvailableAction
from labverse.models.equipment import Equipment
from labverse.safety.bsl_rules import minimum_ppe_for_bsl
from labverse.safety.ppe import room_required_ppe
from labverse.world.state import WorldState


def build_valid_id_inventory(
    world_state: WorldState,
    *,
    agent_ids: Sequence[str] | None = None,
    allowed_verbs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a compact planner-facing inventory of valid IDs and verb parameters."""

    equipment_entries: list[dict[str, Any]] = []
    for equipment in sorted(world_state.equipment.values(), key=lambda item: item.id):
        commands: dict[str, Any] = {}
        for command_name, command in sorted(equipment.commands.items()):
            command_entry: dict[str, Any] = {
                "parameter_names": [parameter.name for parameter in command.parameters],
            }
            if command.duration_seconds is not None:
                command_entry["fixed_parameters"] = {
                    "duration_seconds": command.duration_seconds,
                }
            commands[command_name] = command_entry
        equipment_entries.append(
            {
                "id": equipment.id,
                "name": equipment.name,
                "room_id": equipment.room_id,
                "commands": commands,
            }
        )

    verb_parameter_rules: dict[str, dict[str, Any]] = {
        "move_to": {"target_kind": "room", "parameter_names": []},
        "open_door": {"target_kind": "door", "parameter_names": []},
        "close_door": {"target_kind": "door", "parameter_names": []},
        "unlock_door": {"target_kind": "door", "parameter_names": []},
        "start_equipment": {
            "target_kind": "equipment",
            "parameter_names": ["command_name", "<equipment-command-parameters>"],
        },
        "stop_equipment": {"target_kind": "equipment", "parameter_names": []},
        "pipette_transfer": {
            "target_kind": "equipment",
            "parameter_names": ["source_id", "destination_id", "volume_ul"],
        },
        "retrieve_sample": {"target_kind": "sample", "parameter_names": []},
        "store_sample": {
            "target_kind": "sample",
            "parameter_names": [
                "destination_room_id",
                "destination_equipment_id",
                "destination_zone_id",
            ],
        },
        "dispose_material": {
            "target_kind": "material",
            "parameter_names": ["waste_category", "zone_id"],
        },
        "don_ppe": {"target_kind": None, "parameter_names": ["ppe_type"]},
        "doff_ppe": {"target_kind": None, "parameter_names": ["ppe_type"]},
        "wait": {"target_kind": None, "parameter_names": []},
        "log_observation": {"target_kind": None, "parameter_names": ["notes"]},
    }
    if allowed_verbs is not None:
        allowed = set(allowed_verbs)
        verb_parameter_rules = {
            verb: rules for verb, rules in verb_parameter_rules.items() if verb in allowed
        }

    inventory: dict[str, Any] = {
        "rooms": [
            {"id": room.id, "name": room.name, "zone_ids": [zone.id for zone in room.zones]}
            for room in sorted(world_state.rooms.values(), key=lambda item: item.id)
        ],
        "doors": [
            {"id": door.id, "connects": list(door.connects)}
            for door in sorted(world_state.doors, key=lambda item: item.id)
        ],
        "equipment": equipment_entries,
        "materials": [
            {"id": material.id, "name": material.name, "room_id": material.room_id}
            for material in sorted(world_state.materials.values(), key=lambda item: item.id)
        ],
        "samples": [
            {
                "id": sample.id,
                "name": sample.name,
                "room_id": sample.room_id,
                "equipment_id": sample.equipment_id,
            }
            for sample in sorted(world_state.samples.values(), key=lambda item: item.id)
        ],
        "verb_parameter_rules": verb_parameter_rules,
    }
    if agent_ids is not None:
        inventory["agents"] = list(agent_ids)
    return inventory


def format_valid_id_inventory_json(
    world_state: WorldState,
    *,
    agent_ids: Sequence[str] | None = None,
    allowed_verbs: Sequence[str] | None = None,
) -> str:
    """Format the compact valid-ID inventory as stable JSON."""

    return json.dumps(
        build_valid_id_inventory(
            world_state,
            agent_ids=agent_ids,
            allowed_verbs=allowed_verbs,
        ),
        indent=2,
        sort_keys=True,
    )


def validate_planned_step_grounding(
    *,
    verb: str,
    target_id: str | None,
    parameters: Mapping[str, Any],
    world_state: WorldState,
    agent_id: str | None = None,
    valid_agent_ids: Sequence[str] | None = None,
) -> None:
    """Validate that a planned step references real world IDs and legal parameters."""

    if valid_agent_ids is not None:
        if agent_id is None or agent_id not in valid_agent_ids:
            raise ValueError(f"Planner grounding failure: invalid agent_id {agent_id!r}")

    match verb:
        case "move_to":
            _require_target_kind(verb, target_id, set(world_state.rooms), "room")
            _require_no_extra_parameters(verb, parameters)
        case "open_door" | "close_door" | "unlock_door":
            door_ids = {door.id for door in world_state.doors}
            _require_target_kind(verb, target_id, door_ids, "door")
            _require_no_extra_parameters(verb, parameters)
        case "start_equipment":
            equipment = _require_equipment_target(verb, target_id, world_state)
            _validate_start_equipment_parameters(equipment, parameters)
        case "stop_equipment":
            _require_equipment_target(verb, target_id, world_state)
            _require_no_extra_parameters(verb, parameters)
        case "pipette_transfer":
            _require_equipment_target(verb, target_id, world_state)
            _validate_pipette_transfer_parameters(parameters, world_state)
        case "retrieve_sample":
            _require_target_kind(verb, target_id, set(world_state.samples), "sample")
            _require_no_extra_parameters(verb, parameters)
        case "store_sample":
            _require_target_kind(verb, target_id, set(world_state.samples), "sample")
            _validate_store_sample_parameters(parameters, world_state)
        case "dispose_material":
            material = _require_material_target(verb, target_id, world_state)
            _validate_dispose_material_parameters(material.room_id, parameters, world_state)
        case "don_ppe" | "doff_ppe":
            _require_no_target(verb, target_id)
            _validate_ppe_parameters(verb, parameters)
        case "wait":
            _require_no_target(verb, target_id)
            _require_no_extra_parameters(verb, parameters)
        case "log_observation":
            _require_no_target(verb, target_id)
            _validate_log_parameters(parameters)
        case _:
            raise ValueError(f"Planner grounding failure: unsupported canonical verb {verb!r}")


def lint_planned_sequence(
    steps: Sequence[Mapping[str, Any]],
    *,
    world_state: WorldState,
    default_agent_id: str | None = None,
    valid_agent_ids: Sequence[str] | None = None,
    max_steps: int | None = None,
) -> None:
    """Reject plans that are impossible before live replay.

    This is deliberately a validator, not a repair layer. It catches obvious
    sequence defects so the planner can retry, while preserving all failures as
    invalid model outputs in the paper artifacts.
    """

    if max_steps is not None and len(steps) > max_steps:
        raise ValueError(f"Plan has {len(steps)} steps but the scenario allows at most {max_steps}")

    allowed_agents = (
        set(valid_agent_ids) if valid_agent_ids is not None else set(world_state.persons)
    )
    agent_rooms: dict[str, str | None] = {}
    agent_ppe: dict[str, set[PPEType]] = {}
    agent_carrying: dict[str, set[str]] = {}
    sample_rooms = {
        sample_id: _sample_current_room(sample_id, world_state) for sample_id in world_state.samples
    }
    material_rooms = {
        material_id: material.room_id for material_id, material in world_state.materials.items()
    }

    def ensure_agent(agent_id: str) -> None:
        if agent_id not in allowed_agents or agent_id not in world_state.persons:
            raise ValueError(f"Plan references invalid agent_id {agent_id!r}")
        if agent_id in agent_rooms:
            return
        person = world_state.get_person(agent_id)
        agent_rooms[agent_id] = person.current_room_id
        agent_ppe[agent_id] = set(person.current_ppe)
        agent_carrying[agent_id] = set(person.carrying)

    for index, step in enumerate(steps, start=1):
        raw_agent_id = step.get("agent_id", default_agent_id)
        if not isinstance(raw_agent_id, str) or not raw_agent_id:
            raise ValueError(f"Plan step {index} is missing a valid agent_id")
        agent_id = raw_agent_id
        ensure_agent(agent_id)

        verb = step.get("verb")
        if not isinstance(verb, str):
            raise ValueError(f"Plan step {index} is missing a string verb")
        target_id = step.get("target_id")
        if target_id is not None and not isinstance(target_id, str):
            raise ValueError(f"Plan step {index} target_id must be a string when present")
        raw_parameters = step.get("parameters") or {}
        if not isinstance(raw_parameters, Mapping):
            raise ValueError(f"Plan step {index} parameters must be an object")
        parameters = dict(raw_parameters)

        validate_planned_step_grounding(
            verb=verb,
            target_id=target_id,
            parameters=parameters,
            world_state=world_state,
            agent_id=agent_id,
            valid_agent_ids=valid_agent_ids,
        )

        current_room = agent_rooms[agent_id]
        match verb:
            case "don_ppe":
                agent_ppe[agent_id].add(PPEType(parameters["ppe_type"]))
            case "doff_ppe":
                agent_ppe[agent_id].discard(PPEType(parameters["ppe_type"]))
            case "move_to":
                assert target_id is not None
                _require_adjacent_room(
                    index=index,
                    agent_id=agent_id,
                    current_room=current_room,
                    target_room=target_id,
                    world_state=world_state,
                )
                _require_planned_ppe(
                    index=index,
                    agent_id=agent_id,
                    verb=verb,
                    required=room_required_ppe(world_state.get_room(target_id)),
                    worn=agent_ppe[agent_id],
                    target_label=target_id,
                )
                agent_rooms[agent_id] = target_id
            case "start_equipment" | "stop_equipment" | "pipette_transfer":
                assert target_id is not None
                equipment = world_state.get_equipment(target_id)
                _require_same_room(
                    index,
                    agent_id,
                    verb,
                    target_id,
                    current_room,
                    equipment.room_id,
                )
                _require_planned_ppe(
                    index=index,
                    agent_id=agent_id,
                    verb=verb,
                    required=room_required_ppe(world_state.get_room(equipment.room_id)),
                    worn=agent_ppe[agent_id],
                    target_label=target_id,
                )
                if verb == "pipette_transfer":
                    for parameter_name in ("source_id", "destination_id"):
                        _require_entity_in_room(
                            index=index,
                            agent_id=agent_id,
                            verb=verb,
                            entity_id=str(parameters[parameter_name]),
                            current_room=current_room,
                            material_rooms=material_rooms,
                            sample_rooms=sample_rooms,
                            carrying=agent_carrying[agent_id],
                        )
            case "retrieve_sample":
                assert target_id is not None
                sample_room = sample_rooms[target_id]
                _require_same_room(index, agent_id, verb, target_id, current_room, sample_room)
                sample = world_state.get_sample(target_id)
                _require_planned_ppe(
                    index=index,
                    agent_id=agent_id,
                    verb=verb,
                    required=room_required_ppe(world_state.get_room(sample_room))
                    | set(minimum_ppe_for_bsl(sample.bsl_level)),
                    worn=agent_ppe[agent_id],
                    target_label=target_id,
                )
                agent_carrying[agent_id].add(target_id)
                sample_rooms[target_id] = sample_room
            case "store_sample":
                assert target_id is not None
                if target_id not in agent_carrying[agent_id]:
                    raise ValueError(
                        f"Plan step {index} for {agent_id} stores sample {target_id!r} "
                        "before retrieving it"
                    )
                destination_room = _planned_storage_room(parameters, world_state)
                if destination_room is not None:
                    _require_same_room(
                        index,
                        agent_id,
                        verb,
                        target_id,
                        current_room,
                        destination_room,
                    )
                    sample_rooms[target_id] = destination_room
                agent_carrying[agent_id].discard(target_id)
            case "dispose_material":
                assert target_id is not None
                if target_id not in material_rooms:
                    raise ValueError(
                        f"Plan step {index} for {agent_id} disposes material {target_id!r} "
                        "after it has already been removed"
                    )
                material_room = material_rooms[target_id]
                _require_same_room(index, agent_id, verb, target_id, current_room, material_room)
                _require_planned_ppe(
                    index=index,
                    agent_id=agent_id,
                    verb=verb,
                    required=room_required_ppe(world_state.get_room(material_room)),
                    worn=agent_ppe[agent_id],
                    target_label=target_id,
                )
                material_rooms.pop(target_id, None)
            case "open_door" | "close_door" | "unlock_door":
                assert target_id is not None
                door = next(door for door in world_state.doors if door.id == target_id)
                if current_room is not None and current_room not in door.connects:
                    raise ValueError(
                        f"Plan step {index} for {agent_id} uses door {target_id!r} "
                        f"from room {current_room!r}, but the door is not adjacent"
                    )
            case "wait" | "log_observation":
                pass
            case _:
                raise ValueError(f"Plan step {index} uses unsupported verb {verb!r}")


def _sample_current_room(sample_id: str, world_state: WorldState) -> str:
    sample = world_state.get_sample(sample_id)
    if sample.equipment_id is not None and sample.equipment_id in world_state.equipment:
        return world_state.get_equipment(sample.equipment_id).room_id
    return sample.room_id


def _planned_storage_room(parameters: Mapping[str, Any], world_state: WorldState) -> str | None:
    destination_room_id = parameters.get("destination_room_id")
    if isinstance(destination_room_id, str):
        return destination_room_id
    destination_equipment_id = parameters.get("destination_equipment_id")
    if isinstance(destination_equipment_id, str):
        return world_state.get_equipment(destination_equipment_id).room_id
    return None


def _require_adjacent_room(
    *,
    index: int,
    agent_id: str,
    current_room: str | None,
    target_room: str,
    world_state: WorldState,
) -> None:
    if current_room is None or current_room == target_room:
        return
    adjacent = any(
        current_room in door.connects and target_room in door.connects for door in world_state.doors
    )
    if not adjacent:
        raise ValueError(
            f"Plan step {index} for {agent_id} moves from {current_room!r} to "
            f"{target_room!r}, but those rooms are not adjacent"
        )


def _require_same_room(
    index: int,
    agent_id: str,
    verb: str,
    target_id: str,
    current_room: str | None,
    target_room: str,
) -> None:
    if current_room == target_room:
        return
    raise ValueError(
        f"Plan step {index} for {agent_id} uses {verb} on {target_id!r} in "
        f"{target_room!r} while the agent is in {current_room!r}"
    )


def _require_entity_in_room(
    *,
    index: int,
    agent_id: str,
    verb: str,
    entity_id: str,
    current_room: str | None,
    material_rooms: Mapping[str, str],
    sample_rooms: Mapping[str, str],
    carrying: set[str],
) -> None:
    if entity_id in carrying:
        return
    room = material_rooms.get(entity_id) or sample_rooms.get(entity_id)
    if room is None or room == current_room:
        return
    raise ValueError(
        f"Plan step {index} for {agent_id} uses {verb} with {entity_id!r} in "
        f"{room!r} while the agent is in {current_room!r}"
    )


def _require_planned_ppe(
    *,
    index: int,
    agent_id: str,
    verb: str,
    required: set[PPEType],
    worn: set[PPEType],
    target_label: str,
) -> None:
    missing = sorted(ppe.value for ppe in required - worn)
    if missing:
        raise ValueError(
            f"Plan step {index} for {agent_id} uses {verb} on {target_label!r} "
            f"before donning required PPE: {', '.join(missing)}"
        )


def bind_planned_action(
    *,
    agent_id: str,
    verb: str,
    target_id: str | None,
    parameters: Mapping[str, Any],
    available_actions: Sequence[AvailableAction],
) -> Action:
    """Bind a grounded planned step to the current live action surface."""

    matching_actions = [
        available_action
        for available_action in available_actions
        if available_action.verb == verb
        and available_action.target_id == target_id
        and _planner_parameters_match_fixed_parameters(
            available_action.fixed_parameters,
            parameters,
        )
    ]
    if not matching_actions:
        surface_summary = ", ".join(
            _available_action_summary(available_action) for available_action in available_actions
        )
        raise ValueError(
            "Planner grounding failure: planned step does not match the current action surface. "
            f"Requested verb={verb!r}, target_id={target_id!r}, parameters={dict(parameters)!r}. "
            f"Current candidates: {surface_summary}"
        )
    if len(matching_actions) > 1:
        # Multiple candidates share the same (verb, target_id) and don't
        # conflict with the planner's parameters. Rather than failing, we
        # deterministically pick the best match:
        #   1. Highest overlap between planner params and candidate
        #      fixed_parameters (the planner was most specific about this one).
        #   2. Fewest total fixed_parameters (the simplest action).
        #   3. Alphabetical candidate_ref (stable tiebreak).
        def _disambiguation_key(action: AvailableAction) -> tuple[int, int, str]:
            coverage = sum(1 for name in action.fixed_parameters if name in parameters)
            return (-coverage, len(action.fixed_parameters), action.candidate_ref)

        matching_actions.sort(key=_disambiguation_key)

    return build_action_from_available(
        agent_id=agent_id,
        available_action=matching_actions[0],
        provided_parameters=parameters,
    )


def _require_target_kind(
    verb: str,
    target_id: str | None,
    valid_ids: set[str],
    kind: str,
) -> str:
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(f"Planner grounding failure: {verb} requires a {kind} target_id")
    if target_id not in valid_ids:
        raise ValueError(
            f"Planner grounding failure: {verb} target_id {target_id!r} is not a valid {kind} ID"
        )
    return target_id


def _require_equipment_target(
    verb: str,
    target_id: str | None,
    world_state: WorldState,
) -> Equipment:
    target = _require_target_kind(verb, target_id, set(world_state.equipment), "equipment")
    return world_state.get_equipment(target)


def _require_material_target(verb: str, target_id: str | None, world_state: WorldState) -> Any:
    target = _require_target_kind(verb, target_id, set(world_state.materials), "material")
    return world_state.get_material(target)


def _require_no_target(verb: str, target_id: str | None) -> None:
    if target_id is not None:
        raise ValueError(f"Planner grounding failure: {verb} must not include target_id")


def _require_no_extra_parameters(verb: str, parameters: Mapping[str, Any]) -> None:
    if parameters:
        raise ValueError(
            f"Planner grounding failure: {verb} must not include parameters {sorted(parameters)}"
        )


def _validate_start_equipment_parameters(
    equipment: Equipment,
    parameters: Mapping[str, Any],
) -> None:
    command_name = parameters.get("command_name")
    if not isinstance(command_name, str):
        raise ValueError(
            "Planner grounding failure: start_equipment requires parameters.command_name"
        )
    command = equipment.commands.get(command_name)
    if command is None:
        raise ValueError(
            "Planner grounding failure: start_equipment command_name "
            f"{command_name!r} is not valid for equipment {equipment.id!r}"
        )
    allowed_names = {"command_name", *(parameter.name for parameter in command.parameters)}
    if command.duration_seconds is not None:
        allowed_names.add("duration_seconds")
    unexpected = sorted(set(parameters) - allowed_names)
    if unexpected:
        raise ValueError(
            "Planner grounding failure: start_equipment has invalid parameter keys "
            f"{unexpected} for equipment {equipment.id!r}"
        )


def _validate_pipette_transfer_parameters(
    parameters: Mapping[str, Any],
    world_state: WorldState,
) -> None:
    required = {"source_id", "destination_id", "volume_ul"}
    missing = sorted(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            f"Planner grounding failure: pipette_transfer is missing required parameters {missing}"
        )
    unexpected = sorted(set(parameters) - required)
    if unexpected:
        raise ValueError(
            f"Planner grounding failure: pipette_transfer has invalid parameter keys {unexpected}"
        )
    valid_transfer_ids = set(world_state.materials) | set(world_state.samples)
    for name in ("source_id", "destination_id"):
        value = parameters.get(name)
        if not isinstance(value, str) or value not in valid_transfer_ids:
            raise ValueError(
                "Planner grounding failure: pipette_transfer "
                f"{name} {value!r} is not a valid material/sample ID"
            )
    volume = parameters.get("volume_ul")
    if not isinstance(volume, (int, float)) or isinstance(volume, bool):
        raise ValueError("Planner grounding failure: pipette_transfer volume_ul must be numeric")


def _validate_store_sample_parameters(
    parameters: Mapping[str, Any],
    world_state: WorldState,
) -> None:
    allowed = {"destination_room_id", "destination_equipment_id", "destination_zone_id"}
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(
            f"Planner grounding failure: store_sample has invalid parameter keys {unexpected}"
        )
    destination_room_id = parameters.get("destination_room_id")
    destination_equipment_id = parameters.get("destination_equipment_id")
    if destination_room_id is None and destination_equipment_id is None:
        raise ValueError(
            "Planner grounding failure: store_sample requires destination_room_id or "
            "destination_equipment_id"
        )
    if destination_room_id is not None and destination_equipment_id is not None:
        raise ValueError(
            "Planner grounding failure: store_sample must not specify both destination_room_id "
            "and destination_equipment_id"
        )
    room_id: str | None = None
    if destination_room_id is not None:
        if not isinstance(destination_room_id, str) or destination_room_id not in world_state.rooms:
            raise ValueError(
                "Planner grounding failure: store_sample destination_room_id "
                f"{destination_room_id!r} is invalid"
            )
        room_id = destination_room_id
    if destination_equipment_id is not None:
        if (
            not isinstance(destination_equipment_id, str)
            or destination_equipment_id not in world_state.equipment
        ):
            raise ValueError(
                "Planner grounding failure: store_sample destination_equipment_id "
                f"{destination_equipment_id!r} is invalid"
            )
        if "destination_zone_id" in parameters:
            raise ValueError(
                "Planner grounding failure: store_sample destination_zone_id is only valid "
                "with destination_room_id"
            )
    if "destination_zone_id" in parameters:
        destination_zone_id = parameters["destination_zone_id"]
        if not isinstance(destination_zone_id, str):
            raise ValueError("Planner grounding failure: destination_zone_id must be a string")
        if room_id is None:
            raise ValueError(
                "Planner grounding failure: destination_zone_id requires destination_room_id"
            )
        zone_ids = {zone.id for zone in world_state.get_room(room_id).zones}
        if destination_zone_id not in zone_ids:
            raise ValueError(
                "Planner grounding failure: destination_zone_id "
                f"{destination_zone_id!r} is invalid for room {room_id!r}"
            )


def _validate_dispose_material_parameters(
    room_id: str,
    parameters: Mapping[str, Any],
    world_state: WorldState,
) -> None:
    allowed = {"waste_category", "zone_id"}
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(
            f"Planner grounding failure: dispose_material has invalid parameter keys {unexpected}"
        )
    waste_category = parameters.get("waste_category")
    valid_waste_categories = {category.value for category in WasteCategory}
    if not isinstance(waste_category, str) or waste_category not in valid_waste_categories:
        raise ValueError(
            "Planner grounding failure: dispose_material requires a valid waste_category"
        )
    zone_id = parameters.get("zone_id")
    if zone_id is None:
        return
    if not isinstance(zone_id, str):
        raise ValueError("Planner grounding failure: dispose_material zone_id must be a string")
    zone_ids = {zone.id for zone in world_state.get_room(room_id).zones}
    if zone_id not in zone_ids:
        raise ValueError(
            f"Planner grounding failure: dispose_material zone_id {zone_id!r} is invalid"
        )


def _validate_ppe_parameters(verb: str, parameters: Mapping[str, Any]) -> None:
    allowed = {"ppe_type"}
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(
            f"Planner grounding failure: {verb} has invalid parameter keys {unexpected}"
        )
    ppe_type = parameters.get("ppe_type")
    valid_ppe_types = {ppe.value for ppe in PPEType}
    if not isinstance(ppe_type, str) or ppe_type not in valid_ppe_types:
        raise ValueError(f"Planner grounding failure: {verb} requires a valid ppe_type")


def _validate_log_parameters(parameters: Mapping[str, Any]) -> None:
    allowed = {"notes"}
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(
            f"Planner grounding failure: log_observation has invalid parameter keys {unexpected}"
        )
    notes = parameters.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Planner grounding failure: log_observation notes must be a string")


def _available_action_summary(available_action: AvailableAction) -> str:
    summary = f"{available_action.candidate_ref}:{available_action.verb}"
    if available_action.target_id is not None:
        summary += f"->{available_action.target_id}"
    if available_action.fixed_parameters:
        summary += f" fixed={dict(available_action.fixed_parameters)}"
    return summary


def _planner_parameters_match_fixed_parameters(
    fixed_parameters: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> bool:
    return all(
        name not in parameters or parameters[name] == value
        for name, value in fixed_parameters.items()
    )
