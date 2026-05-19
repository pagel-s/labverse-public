"""Shared helpers for parsing and repairing paper-facing JSON outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from json_repair import loads as json_repair_loads


def normalize_json_text_response_with_kinds(text: str) -> tuple[str, tuple[str, ...]]:
    """Strip benign wrapper text around a JSON body without changing its semantics."""

    normalized = text.strip()
    normalization_kinds: list[str] = []
    trimmed_leading_dashes = False
    while normalized.startswith("---"):
        normalized = normalized[3:].lstrip()
        trimmed_leading_dashes = True
    if trimmed_leading_dashes:
        normalization_kinds.append("leading_dashes_removed")

    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
        normalization_kinds.append("code_fence_removed")

    return normalized, tuple(normalization_kinds)


def normalize_json_text_response(text: str) -> str:
    """Strip benign wrapper text around a JSON body without changing its semantics."""

    normalized, _ = normalize_json_text_response_with_kinds(text)
    return normalized


def load_json_text_with_repair(text: str) -> tuple[Any, tuple[str, ...]]:
    """Parse JSON text with strict stdlib JSON first and json-repair as fallback."""

    normalized, normalization_kinds = normalize_json_text_response_with_kinds(text)
    try:
        return json.loads(normalized), normalization_kinds
    except json.JSONDecodeError:
        repaired = json_repair_loads(normalized, skip_json_loads=True)
        return repaired, (*normalization_kinds, "json_repair")


def coerce_json_object_field(
    value: Any,
    *,
    field_name: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Coerce a mapping field, including stringified JSON objects when recoverable."""

    if value is None:
        return {}, ()
    if isinstance(value, Mapping):
        return dict(value), ()
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be an object when provided")

    parsed, normalization_kinds = load_json_text_with_repair(value)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"'{field_name}' must be an object when provided")
    return dict(parsed), (*normalization_kinds, f"stringified_{field_name}")
