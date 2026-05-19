"""Tests for action-binding helpers used by the interactive paper agents."""

from __future__ import annotations

import pytest

from labverse.agent.action_binding import (
    build_action_from_available,
    extract_provided_parameters,
    normalize_parameters_for_available_action,
    resolve_action_choice,
)
from labverse.core.types import ActionCategory
from labverse.models.action import AvailableAction


def _available_action(
    *,
    action_id: str,
    candidate_ref: str,
    verb: str,
    target_id: str | None = None,
) -> AvailableAction:
    return AvailableAction(
        action_id=action_id,
        candidate_ref=candidate_ref,
        verb=verb,
        category=ActionCategory.OBSERVE if verb == "log_observation" else ActionCategory.MOVE,
        target_id=target_id,
        description=candidate_ref,
    )


def test_resolve_action_choice_accepts_exact_candidate_ref() -> None:
    action = _available_action(
        action_id="preview_log_observation",
        candidate_ref="observe_1",
        verb="log_observation",
    )

    resolved = resolve_action_choice(
        {"candidate_ref": "observe_1"},
        [action],
    )

    assert resolved.available_action.candidate_ref == "observe_1"
    assert resolved.available_action.action_id == "preview_log_observation"
    assert resolved.fallback_kind == "omitted_const_fields"


def test_resolve_action_choice_rejects_bare_verb() -> None:
    action = _available_action(
        action_id="preview_log_observation",
        candidate_ref="observe_1",
        verb="log_observation",
    )

    with pytest.raises(ValueError, match="Unknown candidate_ref 'log_observation'"):
        resolve_action_choice(
            {"candidate_ref": "log_observation", "verb": "log_observation"},
            [action],
        )


def test_resolve_action_choice_rejects_unknown_candidate_ref() -> None:
    actions = [
        _available_action(
            action_id="preview_move_office",
            candidate_ref="move_1",
            verb="move_to",
            target_id="office",
        ),
        _available_action(
            action_id="preview_move_cold_room",
            candidate_ref="move_2",
            verb="move_to",
            target_id="cold_room",
        ),
    ]

    with pytest.raises(ValueError, match="Unknown candidate_ref 'move_to'"):
        resolve_action_choice(
            {"candidate_ref": "move_to", "verb": "move_to"},
            actions,
        )


def test_extract_provided_parameters_records_flattened_transport_normalization() -> None:
    extraction = extract_provided_parameters(
        {
            "candidate_ref": "observe_1",
            "verb": "log_observation",
            "notes": "wait one turn",
        }
    )

    assert extraction.parameters == {"notes": "wait one turn"}
    assert extraction.normalization_kinds == ("flattened_parameters",)


def test_extract_provided_parameters_repairs_stringified_parameters_object() -> None:
    extraction = extract_provided_parameters(
        {
            "candidate_ref": "equip_1",
            "verb": "start_equipment",
            "parameters": '{"speed_rpm": 1000, "temperature_celsius": 4}',
        }
    )

    assert extraction.parameters == {"speed_rpm": 1000, "temperature_celsius": 4}
    assert extraction.normalization_kinds == ("stringified_parameters",)


def test_extract_provided_parameters_repairs_malformed_stringified_parameters_object() -> None:
    extraction = extract_provided_parameters(
        {
            "candidate_ref": "equip_1",
            "verb": "start_equipment",
            "parameters": "{speed_rpm: 1000, temperature_celsius: 4}",
        }
    )

    assert extraction.parameters == {"speed_rpm": 1000, "temperature_celsius": 4}
    assert extraction.normalization_kinds == ("json_repair", "stringified_parameters")


def test_build_action_from_available_validates_parameter_schema() -> None:
    action = AvailableAction(
        action_id="preview_pipette_transfer",
        candidate_ref="sample_1",
        verb="pipette_transfer",
        category=ActionCategory.HANDLE_SAMPLE,
        description="Transfer sample",
        fixed_parameters={"source_id": "source_1", "destination_id": "dest_1"},
        required_parameters=["volume_ul"],
        parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "volume_ul": {
                    "type": "number",
                    "minimum": 0.1,
                }
            },
            "required": ["volume_ul"],
        },
    )

    with pytest.raises(ValueError, match="must be >= 0.1"):
        build_action_from_available(
            agent_id="tech_1",
            available_action=action,
            provided_parameters={"volume_ul": 0.05},
        )


def test_resolve_action_choice_allows_missing_redundant_const_fields() -> None:
    action = _available_action(
        action_id="preview_move_office",
        candidate_ref="move_1",
        verb="move_to",
        target_id="office",
    )

    resolved = resolve_action_choice({"candidate_ref": "move_1"}, [action])

    assert resolved.available_action.candidate_ref == "move_1"
    assert resolved.fallback_kind == "omitted_const_fields"


def test_resolve_action_choice_allows_missing_team_actor_and_target_echoes() -> None:
    action = AvailableAction(
        action_id="tech_1::preview_move_wet_lab",
        candidate_ref="t1_move_1",
        actor_id="tech_1",
        verb="move_to",
        category=ActionCategory.MOVE,
        target_id="wet_lab",
        description="Move to wet lab",
    )

    resolved = resolve_action_choice({"candidate_ref": "t1_move_1"}, [action])

    assert resolved.available_action.candidate_ref == "t1_move_1"
    assert resolved.fallback_kind == "omitted_const_fields"


def test_resolve_action_choice_drops_superfluous_target_id_for_no_target_candidate() -> None:
    action = _available_action(
        action_id="preview_doff_gloves",
        candidate_ref="ppe_1",
        verb="doff_ppe",
    )

    resolved = resolve_action_choice(
        {"candidate_ref": "ppe_1", "verb": "doff_ppe", "target_id": "safety_glasses"},
        [action],
    )

    assert resolved.available_action.candidate_ref == "ppe_1"
    assert resolved.normalization_kinds == ("dropped_superfluous_target_id",)


def test_normalize_parameters_for_available_action_coerces_integral_float_to_integer() -> None:
    action = AvailableAction(
        action_id="preview_start_centrifuge",
        candidate_ref="equip_1",
        verb="start_equipment",
        category=ActionCategory.USE_EQUIPMENT,
        description="Start centrifuge",
        target_id="centrifuge_1",
        required_parameters=["speed_rpm"],
        parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"speed_rpm": {"type": "integer"}},
            "required": ["speed_rpm"],
        },
        free_parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"speed_rpm": {"type": "integer"}},
            "required": ["speed_rpm"],
        },
    )

    extraction = normalize_parameters_for_available_action(action, {"speed_rpm": 1000.0})

    assert extraction.parameters == {"speed_rpm": 1000}
    assert extraction.normalization_kinds == ("integral_float_to_integer",)


def test_normalize_parameters_for_available_action_coerces_stringified_numbers() -> None:
    action = AvailableAction(
        action_id="preview_start_centrifuge",
        candidate_ref="equip_1",
        verb="start_equipment",
        category=ActionCategory.USE_EQUIPMENT,
        description="Start centrifuge",
        target_id="centrifuge_1",
        required_parameters=["speed_rpm"],
        optional_parameters=["temperature_celsius"],
        parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "speed_rpm": {"type": "integer"},
                "temperature_celsius": {"type": "number"},
            },
            "required": ["speed_rpm"],
        },
        free_parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "speed_rpm": {"type": "integer"},
                "temperature_celsius": {"type": "number"},
            },
            "required": ["speed_rpm"],
        },
    )

    extraction = normalize_parameters_for_available_action(
        action,
        {"speed_rpm": "1000.0", "temperature_celsius": "4.5"},
    )

    assert extraction.parameters == {"speed_rpm": 1000, "temperature_celsius": 4.5}
    assert extraction.normalization_kinds == (
        "stringified_number",
        "integral_float_to_integer",
        "stringified_number",
    )


def test_build_action_from_available_accepts_coercible_numeric_parameters() -> None:
    action = AvailableAction(
        action_id="preview_start_centrifuge",
        candidate_ref="equip_1",
        verb="start_equipment",
        category=ActionCategory.USE_EQUIPMENT,
        description="Start centrifuge",
        target_id="centrifuge_1",
        required_parameters=["speed_rpm"],
        parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"speed_rpm": {"type": "integer"}},
            "required": ["speed_rpm"],
        },
        free_parameter_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"speed_rpm": {"type": "integer"}},
            "required": ["speed_rpm"],
        },
    )

    built = build_action_from_available(
        agent_id="tech_1",
        available_action=action,
        provided_parameters={"speed_rpm": "1000.0"},
    )

    assert built.parameters["speed_rpm"] == 1000
