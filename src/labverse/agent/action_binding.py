"""Shared helpers for binding interactive JSON action objects to available actions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from labverse.agent.json_io import coerce_json_object_field
from labverse.core.ids import generate_id
from labverse.models.action import Action, AvailableAction

RESERVED_ARGUMENT_KEYS = frozenset(
    {
        "action_id",
        "candidate_ref",
        "actor_id",
        "verb",
        "target_id",
        "parameters",
        "tool",
        "tool_name",
        "name",
    }
)


@dataclass(frozen=True)
class ActionChoiceResolution:
    """Resolved available action plus any compatibility fallback used."""

    available_action: AvailableAction
    normalization_kinds: tuple[str, ...] = ()

    @property
    def fallback_kind(self) -> str | None:
        if not self.normalization_kinds:
            return None
        return self.normalization_kinds[0]


@dataclass(frozen=True)
class ParameterExtraction:
    """Normalized action parameters plus transport-level normalizations used."""

    parameters: dict[str, Any]
    normalization_kinds: tuple[str, ...] = ()


_NUMERIC_STRING_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def build_action_input_schema(
    available_actions: Sequence[AvailableAction],
) -> dict[str, Any]:
    """Build a candidate-local JSON action schema for the current action surface."""

    if not available_actions:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_ref": {
                    "type": "string",
                    "enum": [],
                    "description": "No actions are currently available.",
                }
            },
            "required": ["candidate_ref"],
        }

    branches = [_action_choice_branch(action) for action in available_actions]
    if len(branches) == 1:
        return branches[0]
    return {
        "description": (
            "The interactive action output must be a JSON object. Choose exactly one "
            "currently available candidate_ref. The selected branch defines the exact "
            "verb, target, actor, and free parameters allowed on this turn."
        ),
        "oneOf": branches,
    }


def _action_choice_branch(available_action: AvailableAction) -> dict[str, Any]:
    parameter_schema = _normalized_parameter_schema(available_action)
    properties: dict[str, Any] = {
        "candidate_ref": {
            "type": "string",
            "const": available_action.candidate_ref,
            "description": (
                "The exact currently available action candidate to execute: "
                f"{available_action.candidate_ref}"
            ),
        },
        "verb": {
            "type": "string",
            "const": available_action.verb,
            "description": f"The action verb for {available_action.candidate_ref}.",
        },
    }
    required = ["candidate_ref"]
    if available_action.actor_id is not None:
        properties["actor_id"] = {
            "type": "string",
            "const": available_action.actor_id,
            "description": f"The actor for {available_action.candidate_ref}.",
        }
    if available_action.target_id is not None:
        properties["target_id"] = {
            "type": "string",
            "const": available_action.target_id,
            "description": f"The target entity for {available_action.candidate_ref}.",
        }
    if parameter_schema is not None:
        properties["parameters"] = parameter_schema
        if parameter_schema.get("required"):
            required.append("parameters")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
        "description": available_action.description or available_action.candidate_ref,
    }


def _normalized_parameter_schema(available_action: AvailableAction) -> dict[str, Any] | None:
    if available_action.free_parameter_schema:
        schema = dict(available_action.free_parameter_schema)
        schema.setdefault("type", "object")
        schema.setdefault("additionalProperties", False)
        schema.setdefault("properties", {})
        schema.setdefault("required", list(available_action.required_parameters))
        return schema

    parameter_names = [*available_action.required_parameters, *available_action.optional_parameters]
    if not parameter_names:
        return None

    properties: dict[str, Any] = {}
    for name in parameter_names:
        detail = available_action.parameter_details.get(
            name, "Parameter override for the selected action."
        )
        properties[name] = {"description": detail}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(available_action.required_parameters),
        "description": (
            "Parameter overrides for the selected candidate. Only fields listed for this "
            "candidate_ref are valid."
        ),
    }


def extract_provided_parameters(arguments: Mapping[str, Any]) -> ParameterExtraction:
    """Extract action parameters from nested or flattened JSON action fields."""

    nested, normalization_kinds = _normalize_parameters(arguments.get("parameters"))
    flattened = {
        key: value for key, value in arguments.items() if key not in RESERVED_ARGUMENT_KEYS
    }
    conflicts = sorted(
        key for key, value in flattened.items() if key in nested and nested[key] != value
    )
    if conflicts:
        raise ValueError(
            "Conflicting parameter values provided in action output for: "
            + ", ".join(conflicts)
        )
    combined_normalization_kinds = list(normalization_kinds)
    if flattened:
        combined_normalization_kinds.append("flattened_parameters")
    return ParameterExtraction(
        parameters={**flattened, **nested},
        normalization_kinds=tuple(combined_normalization_kinds),
    )


def resolve_available_action(
    action_id: str,
    available_actions: Sequence[AvailableAction],
) -> AvailableAction:
    """Resolve an internal executable action_id against the current action surface."""

    for action in available_actions:
        if action.action_id == action_id:
            return action
    valid = ", ".join(action.action_id for action in available_actions)
    raise ValueError(f"Unknown action_id '{action_id}'. Available action_ids: {valid}")


def resolve_available_candidate(
    candidate_ref: str,
    available_actions: Sequence[AvailableAction],
) -> AvailableAction:
    """Resolve a public candidate_ref against the current action surface."""

    for action in available_actions:
        if action.candidate_ref == candidate_ref:
            return action
    valid = ", ".join(action.candidate_ref for action in available_actions)
    raise ValueError(f"Unknown candidate_ref '{candidate_ref}'. Available candidate_refs: {valid}")


def resolve_action_choice(
    arguments: Mapping[str, Any],
    available_actions: Sequence[AvailableAction],
) -> ActionChoiceResolution:
    """Resolve a JSON action choice against the current surface."""

    candidate_ref = arguments.get("candidate_ref")
    if not isinstance(candidate_ref, str):
        raise ValueError("Missing or invalid 'candidate_ref' in action output")

    available_action = resolve_available_candidate(candidate_ref, available_actions)

    normalization_kinds: list[str] = []

    def add_normalization(kind: str) -> None:
        if kind not in normalization_kinds:
            normalization_kinds.append(kind)

    verb = arguments.get("verb")
    if verb is None:
        add_normalization("omitted_const_fields")
    elif verb != available_action.verb:
        raise ValueError(
            "Action output verb mismatch for "
            f"'{candidate_ref}': expected "
            f"'{available_action.verb}', got '{verb}'"
        )

    if available_action.actor_id is not None:
        actor_id = arguments.get("actor_id")
        if actor_id is None:
            add_normalization("omitted_const_fields")
        elif actor_id != available_action.actor_id:
            raise ValueError(
                "Action output actor_id mismatch for "
                f"'{candidate_ref}': expected "
                f"'{available_action.actor_id}', got '{actor_id}'"
            )
    elif "actor_id" in arguments:
        raise ValueError(f"Action output for '{candidate_ref}' must not include actor_id")

    if available_action.target_id is not None:
        target_id = arguments.get("target_id")
        if target_id is None:
            add_normalization("omitted_const_fields")
        elif target_id != available_action.target_id:
            raise ValueError(
                "Action output target_id mismatch for "
                f"'{candidate_ref}': expected "
                f"'{available_action.target_id}', got '{target_id}'"
            )
    elif "target_id" in arguments:
        add_normalization("dropped_superfluous_target_id")

    return ActionChoiceResolution(
        available_action=available_action,
        normalization_kinds=tuple(normalization_kinds),
    )


def normalize_parameters_for_available_action(
    available_action: AvailableAction,
    parameters: Mapping[str, Any],
) -> ParameterExtraction:
    """Apply narrow schema-aware coercions that do not change action semantics."""

    schema = available_action.free_parameter_schema or available_action.parameter_schema
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping):
        return ParameterExtraction(parameters=dict(parameters))

    normalized_parameters = dict(parameters)
    normalization_kinds: list[str] = []
    for name, value in list(normalized_parameters.items()):
        property_schema = properties.get(name)
        if not isinstance(property_schema, Mapping):
            continue
        coerced_value, value_kinds = _coerce_parameter_value(property_schema, value)
        if value_kinds:
            normalized_parameters[name] = coerced_value
            normalization_kinds.extend(value_kinds)
    return ParameterExtraction(
        parameters=normalized_parameters,
        normalization_kinds=tuple(normalization_kinds),
    )


def _coerce_parameter_value(
    schema: Mapping[str, Any],
    value: Any,
) -> tuple[Any, tuple[str, ...]]:
    expected_type = schema.get("type")
    if expected_type == "integer":
        return _coerce_integer_value(value)
    if expected_type == "number":
        return _coerce_number_value(value)
    return value, ()


def _coerce_integer_value(value: Any) -> tuple[Any, tuple[str, ...]]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value, ()
    if isinstance(value, float) and not isinstance(value, bool):
        if value.is_integer():
            return int(value), ("integral_float_to_integer",)
        return value, ()
    if isinstance(value, str):
        parsed = _parse_numeric_string(value)
        if parsed is None:
            return value, ()
        normalization_kinds = ["stringified_number"]
        if isinstance(parsed, float):
            if not parsed.is_integer():
                return value, ()
            parsed = int(parsed)
            normalization_kinds.append("integral_float_to_integer")
        return parsed, tuple(normalization_kinds)
    return value, ()


def _coerce_number_value(value: Any) -> tuple[Any, tuple[str, ...]]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, ()
    if isinstance(value, str):
        parsed = _parse_numeric_string(value)
        if parsed is None:
            return value, ()
        return parsed, ("stringified_number",)
    return value, ()


def _parse_numeric_string(value: str) -> int | float | None:
    stripped = value.strip()
    if not stripped or not _NUMERIC_STRING_RE.fullmatch(stripped):
        return None
    if "." not in stripped and "e" not in stripped.casefold():
        try:
            return int(stripped)
        except ValueError:
            return None
    try:
        return float(stripped)
    except ValueError:
        return None


def fixed_parameters_match(
    fixed_parameters: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> bool:
    """Check whether a parameter set is consistent with fixed defaults."""

    return all(parameters.get(name) == value for name, value in fixed_parameters.items())


def build_action_from_available(
    *,
    agent_id: str,
    available_action: AvailableAction,
    provided_parameters: Mapping[str, Any],
) -> Action:
    """Bind parameter overrides to a currently available action."""

    normalized_parameters = normalize_parameters_for_available_action(
        available_action,
        provided_parameters,
    ).parameters

    mismatched_fixed = [
        name
        for name, value in available_action.fixed_parameters.items()
        if name in normalized_parameters and normalized_parameters[name] != value
    ]
    if mismatched_fixed:
        raise ValueError(
            f"Fixed parameters for action '{available_action.candidate_ref}' cannot be changed: "
            f"{', '.join(mismatched_fixed)}"
        )

    allowed_names = set(available_action.required_parameters) | set(
        available_action.optional_parameters
    )
    unknown_names = sorted(
        set(normalized_parameters) - allowed_names - set(available_action.fixed_parameters)
    )
    if unknown_names:
        raise ValueError(
            f"Unexpected parameters for candidate '{available_action.candidate_ref}': "
            f"{', '.join(unknown_names)}"
        )

    missing_required = sorted(
        name for name in available_action.required_parameters if name not in normalized_parameters
    )
    if missing_required:
        raise ValueError(
            f"Missing required parameters for candidate '{available_action.candidate_ref}': "
            f"{', '.join(missing_required)}"
        )

    _validate_parameter_schema(
        available_action.candidate_ref,
        available_action.free_parameter_schema or available_action.parameter_schema,
        normalized_parameters,
    )

    parameters = {**available_action.fixed_parameters, **normalized_parameters}
    return Action(
        id=generate_id("action"),
        agent_id=agent_id,
        category=available_action.category,
        verb=available_action.verb,
        target_id=available_action.target_id,
        parameters=parameters,
    )


def _validate_parameter_schema(
    candidate_ref: str,
    schema: Mapping[str, Any],
    provided_parameters: Mapping[str, Any],
) -> None:
    if not schema:
        return

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return

    required = schema.get("required", ())
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes, bytearray)):
        missing_required = [
            str(name)
            for name in required
            if isinstance(name, str) and name not in provided_parameters
        ]
        if missing_required:
            raise ValueError(
                f"Missing required parameters for candidate '{candidate_ref}': "
                f"{', '.join(missing_required)}"
            )

    for name, value in provided_parameters.items():
        raw_property_schema = properties.get(name)
        if isinstance(raw_property_schema, Mapping):
            _validate_parameter_value(candidate_ref, name, raw_property_schema, value)


def _validate_parameter_value(
    candidate_ref: str,
    name: str,
    schema: Mapping[str, Any],
    value: Any,
) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if expected_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be a number"
                )
        elif expected_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be an integer"
                )
        elif expected_type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be a string"
                )
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be a boolean"
                )
        elif expected_type == "array":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be an array"
                )
        elif expected_type == "object":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Parameter '{name}' for candidate '{candidate_ref}' must be an object"
                )

    enum_values = schema.get("enum")
    if isinstance(enum_values, Sequence) and not isinstance(enum_values, (str, bytes, bytearray)):
        if value not in enum_values:
            allowed = ", ".join(repr(item) for item in enum_values)
            raise ValueError(
                f"Parameter '{name}' for candidate '{candidate_ref}' must be one of: {allowed}"
            )

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Parameter '{name}' for candidate '{candidate_ref}' must be a number >= {minimum}"
            )
        if value < minimum:
            raise ValueError(
                f"Parameter '{name}' for candidate '{candidate_ref}' must be >= {minimum}"
            )

    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Parameter '{name}' for candidate '{candidate_ref}' must be a number <= {maximum}"
            )
        if value > maximum:
            raise ValueError(
                f"Parameter '{name}' for candidate '{candidate_ref}' must be <= {maximum}"
            )


def _normalize_parameters(raw: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    return coerce_json_object_field(raw, field_name="parameters")
