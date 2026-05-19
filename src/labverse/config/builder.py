"""Build a runtime `WorldState` from validated config models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard, TypeVar

from pydantic import BaseModel

from labverse.config.schema import ConfigError, EquipmentConfig, LabConfig
from labverse.equipment.catalog import EquipmentCatalog
from labverse.models.equipment import Equipment, EquipmentParameter
from labverse.models.space import LabLayout
from labverse.world.state import WorldState

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


def build_world(config: LabConfig, equipment_catalog: EquipmentCatalog | None = None) -> WorldState:
    """Construct a `WorldState` from a validated `LabConfig`."""
    layout = LabLayout(
        id=config.lab.id,
        name=config.lab.name,
        description=config.lab.description,
        default_entry_room=config.lab.default_entry_room,
    )
    world = WorldState(lab=layout)

    for room in config.rooms:
        cloned_room = _clone_model(room)
        cloned_room.equipment_ids = []
        cloned_room.material_ids = []
        world.add_room(cloned_room)
    for door in config.doors:
        world.add_door(_clone_model(door))
    for equipment in config.equipment:
        world.add_equipment(_build_equipment(equipment, equipment_catalog))
    for material in config.materials:
        world.add_material(_clone_model(material))
    for person in config.persons:
        world.add_person(_clone_model(person))
    for sample in config.samples:
        world.add_sample(_clone_model(sample))

    return world


def _build_equipment(
    config: EquipmentConfig,
    equipment_catalog: EquipmentCatalog | None,
) -> Equipment:
    definition = (
        equipment_catalog.get_definition(config.equipment_type)
        if equipment_catalog is not None
        else None
    )
    commands = {
        **{
            name: _clone_model(command)
            for name, command in (definition.commands.items() if definition is not None else [])
        },
        **{name: _clone_model(command) for name, command in config.commands.items()},
    }
    properties: dict[str, EquipmentParameter] = {}
    if definition is not None:
        properties.update(
            {name: _clone_model(parameter) for name, parameter in definition.properties.items()}
        )
    properties.update(
        {name: _normalize_property(name, value) for name, value in config.properties.items()}
    )

    return Equipment(
        id=config.id,
        name=config.name,
        equipment_type=config.equipment_type,
        manufacturer=config.manufacturer,
        model_name=config.model_name,
        status=config.status,
        room_id=config.room_id,
        zone_id=config.zone_id,
        position=_clone_model(config.position) if config.position is not None else None,
        commands=commands,
        properties=properties,
        required_certifications=_merge_unique(
            definition.required_certifications if definition is not None else [],
            config.required_certifications,
        ),
        max_concurrent_users=config.max_concurrent_users,
        current_users=list(config.current_users),
        running_command=config.running_command,
        running_command_started_at=config.running_command_started_at,
        running_command_duration=config.running_command_duration,
        error_message=config.error_message,
        last_maintenance=config.last_maintenance,
        maintenance_interval_days=config.maintenance_interval_days,
        metadata={
            **(definition.metadata if definition is not None else {}),
            **dict(config.metadata),
        },
        capabilities=_merge_unique(
            definition.capabilities if definition is not None else [],
            config.capabilities,
        ),
    )


def _normalize_property(name: str, value: object) -> EquipmentParameter:
    if isinstance(value, EquipmentParameter):
        parameter = _clone_model(value)
        parameter.name = name
        return parameter

    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("name", name)
        return EquipmentParameter.model_validate(payload)

    if isinstance(value, list):
        if len(value) == 2 and all(_is_number(item) for item in value):
            min_value = float(value[0])
            max_value = float(value[1])
            return EquipmentParameter(
                name=name,
                param_type="range",
                min_value=min_value,
                max_value=max_value,
            )
        raise ConfigError(
            f"equipment property {name!r} must be a scalar, a two-number range, "
            "or a parameter mapping"
        )

    if isinstance(value, bool):
        return EquipmentParameter(
            name=name,
            param_type="bool",
            default_value=value,
            current_value=value,
        )

    if _is_number(value):
        numeric_value = float(value)
        return EquipmentParameter(
            name=name,
            param_type="float",
            default_value=numeric_value,
            current_value=numeric_value,
        )

    if isinstance(value, str):
        return EquipmentParameter(
            name=name,
            param_type="str",
            default_value=value,
            current_value=value,
        )

    raise ConfigError(f"equipment property {name!r} has unsupported value {value!r}")


def _clone_model(model: ModelT) -> ModelT:  # noqa: UP047
    return type(model).model_validate(
        model.model_dump(mode="python", round_trip=True),
    )


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_unique(base: list[T], override: list[T]) -> list[T]:  # noqa: UP047
    merged: list[T] = []
    for item in [*base, *override]:
        if item not in merged:
            merged.append(item)
    return merged
