"""Shared parsing helpers for interactive JSON action outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from labverse.agent.action_binding import (
    ActionChoiceResolution,
    ParameterExtraction,
    extract_provided_parameters,
    normalize_parameters_for_available_action,
    resolve_action_choice,
)
from labverse.agent.json_io import (
    load_json_text_with_repair,
)
from labverse.agent.json_io import (
    normalize_json_text_response as _normalize_json_text_response,
)
from labverse.agent.json_io import (
    normalize_json_text_response_with_kinds as _normalize_json_text_response_with_kinds,
)
from labverse.agent.providers.base import (
    LLMResponse,
)
from labverse.models.action import AvailableAction

_INVALID_ACTION_OUTPUT_MESSAGE = (
    "You must return exactly one valid JSON object. Do not respond with prose, markdown, "
    "XML, tags, or nested wrapper objects."
)
_DISALLOWED_TEXT_WRAPPER_KEYS = frozenset(
    {"action_call", "tool", "arguments", "name", "tool_name", "action_id"}
)


@dataclass(frozen=True)
class InteractiveActionParseResult:
    """Parsed interactive action output plus grounding metadata."""

    arguments: dict[str, Any]
    resolution: ActionChoiceResolution
    parameter_extraction: ParameterExtraction
    normalization_kinds: tuple[str, ...]
    used_provider_tool_call_fallback: bool


def normalize_json_text_response_with_kinds(text: str) -> tuple[str, tuple[str, ...]]:
    """Compatibility export for callers importing via interactive_output."""

    return _normalize_json_text_response_with_kinds(text)


def normalize_json_text_response(text: str) -> str:
    """Compatibility export for callers importing via interactive_output."""

    return _normalize_json_text_response(text)


def parse_interactive_action_response(
    response: LLMResponse,
    available_actions: Sequence[AvailableAction],
) -> InteractiveActionParseResult:
    """Parse one interactive model response using plain JSON text only."""

    normalized_arguments: dict[str, Any]
    normalization_kinds: list[str]

    if response.text and response.text.strip():
        normalized_arguments, normalization_kinds = _parse_interactive_action_text(response.text)
    elif response.tool_call is not None:
        raise ValueError(
            "Interactive output must be plain JSON text. Provider tool calls are not accepted."
        )
    else:
        raise ValueError(_INVALID_ACTION_OUTPUT_MESSAGE)

    resolution = resolve_action_choice(normalized_arguments, available_actions)
    parameter_extraction = extract_provided_parameters(normalized_arguments)
    coerced_parameter_extraction = normalize_parameters_for_available_action(
        resolution.available_action,
        parameter_extraction.parameters,
    )
    combined_normalization_kinds = list(normalization_kinds)
    combined_normalization_kinds.extend(resolution.normalization_kinds)
    combined_normalization_kinds.extend(parameter_extraction.normalization_kinds)
    combined_normalization_kinds.extend(coerced_parameter_extraction.normalization_kinds)
    return InteractiveActionParseResult(
        arguments=normalized_arguments,
        resolution=resolution,
        parameter_extraction=coerced_parameter_extraction,
        normalization_kinds=tuple(combined_normalization_kinds),
        used_provider_tool_call_fallback=False,
    )


def _parse_interactive_action_text(text: str) -> tuple[dict[str, Any], list[str]]:
    try:
        parsed, normalization_kinds = load_json_text_with_repair(text)
    except ValueError as exc:
        raise ValueError(_INVALID_ACTION_OUTPUT_MESSAGE) from exc
    if not isinstance(parsed, dict):
        raise ValueError(_INVALID_ACTION_OUTPUT_MESSAGE)

    disallowed_keys = sorted(_DISALLOWED_TEXT_WRAPPER_KEYS & set(parsed))
    if disallowed_keys:
        raise ValueError(
            "Interactive JSON output must not include nested wrapper keys: "
            + ", ".join(disallowed_keys)
        )
    return parsed, list(normalization_kinds)
