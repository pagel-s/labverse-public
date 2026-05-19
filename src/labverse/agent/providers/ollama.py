"""Ollama provider adapter using direct raw HTTP to the native /api/chat endpoint.

We deliberately target Ollama's native chat API (``POST /api/chat``) rather than
its OpenAI-compatibility layer (``POST /v1/chat/completions``). The OpenAI-compat
layer silently drops per-request option overrides — in particular ``num_ctx`` —
which means 31B+ open-weights models run with Ollama's default 4096-token
context window regardless of what we send. Prompts that exceed that ceiling
are truncated without warning, producing the hallucinated-entity and
tool-schema-fragment failures that derailed earlier paper-benchmark pilots. The
native endpoint honors every option we care about (``num_ctx``, ``num_predict``,
``temperature``, ``keep_alive``) and exposes the same tool-calling surface we
rely on (``message.tool_calls[i].function.{name,arguments}``), so the cost of
using it is only the small adapter layer in this module.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from labverse.agent.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderExecutionError,
    TokenUsage,
    ToolCall,
)

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/"
_DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
_TOOL_CALL_TAG_PATTERN = re.compile(
    r"^<(?:tool_call|function_call|tool)>\s*(?P<body>.+?)\s*</(?:tool_call|function_call|tool)>$",
    re.DOTALL,
)


def _normalize_ollama_model(model: str) -> str:
    """Strip the provider prefix used in LabVerse CLI/paper commands."""

    if model.startswith("ollama/"):
        return model.split("/", 1)[1]
    return model


def _default_ollama_think_setting(model: str) -> bool | str | None:
    """Return a conservative default thinking mode for tool-heavy local runs."""

    normalized = _normalize_ollama_model(model).lower()
    if normalized.startswith(("qwen", "gemma", "deepseek")):
        return False
    return None


def _resolve_ollama_think_setting(model: str) -> bool | str | None:
    """Resolve the think setting from env or per-model defaults."""

    env_value = os.environ.get("OLLAMA_THINK")
    if env_value is None:
        return _default_ollama_think_setting(model)

    normalized = env_value.strip().lower()
    if normalized in {"false", "0", "off", "no"}:
        return False
    if normalized in {"true", "1", "on", "yes"}:
        return True
    if normalized == "none":
        return None
    return env_value.strip()


def _resolve_ollama_keep_alive(default: str | None) -> str | None:
    """Resolve request-level keep_alive, allowing local runs to reduce VRAM residency."""

    env_value = os.environ.get("LABVERSE_OLLAMA_KEEP_ALIVE")
    if env_value is None:
        return default
    normalized = env_value.strip()
    if normalized.lower() in {"none", "off", "false", "0"}:
        return None
    return normalized


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return dict(model_extra)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raw_dict = getattr(value, "__dict__", None)
    if isinstance(raw_dict, Mapping):
        return dict(raw_dict)
    return {}


def _parse_json_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _tool_call_from_payload(payload: Any) -> ToolCall | None:
    if payload is None:
        return None
    if hasattr(payload, "function"):
        function = getattr(payload, "function")
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(name, str):
            return ToolCall(name=name, arguments=_parse_json_arguments(arguments))

    payload_map = _as_mapping(payload)
    if not payload_map:
        return None
    function = payload_map.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str):
            return ToolCall(name=name, arguments=_parse_json_arguments(function.get("arguments")))
    name = payload_map.get("name") or payload_map.get("tool") or payload_map.get("tool_name")
    if isinstance(name, str):
        raw_arguments = payload_map.get("arguments", payload_map.get("parameters", {}))
        return ToolCall(name=name, arguments=_parse_json_arguments(raw_arguments))
    tool_call = payload_map.get("tool_call")
    if tool_call is not None:
        return _tool_call_from_payload(tool_call)
    return None


def _tool_call_from_content(content: Any) -> tuple[ToolCall | None, str | None, list[str]]:
    normalizations: list[str] = []

    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            tagged = _TOOL_CALL_TAG_PATTERN.match(stripped)
            if tagged is not None:
                payload = _parse_json_arguments(tagged.group("body"))
                tool_call = _tool_call_from_payload(payload)
                if tool_call is not None:
                    return tool_call, None, ["content_tool_call"]
            payload = _parse_json_arguments(stripped)
            tool_call = _tool_call_from_payload(payload)
            if tool_call is not None:
                return tool_call, None, ["content_tool_call"]
        return None, stripped or None, normalizations

    if isinstance(content, Mapping):
        tool_call = _tool_call_from_payload(content)
        if tool_call is not None:
            return tool_call, None, ["content_tool_call"]
        for key in ("text", "content", "value"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return None, value.strip(), normalizations
        return None, None, normalizations

    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        text_parts: list[str] = []
        for item in content:
            tool_call = _tool_call_from_payload(item)
            if tool_call is not None:
                return tool_call, None, ["content_tool_call"]
            if isinstance(item, Mapping):
                item_type = item.get("type")
                if item_type in {"text", "output_text", "input_text"}:
                    value = item.get("text") or item.get("content") or item.get("value")
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
                        continue
                for key in ("text", "content", "value"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
                        break
            elif isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
        if text_parts:
            return None, "\n".join(text_parts), normalizations

    return None, None, normalizations


def _extract_tool_call_and_text(
    message: Any,
    *,
    choice: Any | None = None,
    response: Any | None = None,
) -> tuple[ToolCall | None, str | None, list[str], dict[str, Any]]:
    metadata: dict[str, Any] = {}

    direct_tool_calls = getattr(message, "tool_calls", None)
    if isinstance(direct_tool_calls, Sequence) and not isinstance(
        direct_tool_calls, (str, bytes, bytearray)
    ):
        for candidate in direct_tool_calls:
            tool_call = _tool_call_from_payload(candidate)
            if tool_call is not None:
                return tool_call, None, [], metadata

    function_call = getattr(message, "function_call", None)
    if function_call is not None:
        tool_call = _tool_call_from_payload(function_call)
        if tool_call is not None:
            return tool_call, None, ["legacy_function_call"], metadata

    extras = _as_mapping(message)
    if extras:
        metadata["message_extra_keys"] = sorted(extras.keys())
        extra_tool_calls = extras.get("tool_calls")
        if isinstance(extra_tool_calls, Sequence) and not isinstance(
            extra_tool_calls, (str, bytes, bytearray)
        ):
            for candidate in extra_tool_calls:
                tool_call = _tool_call_from_payload(candidate)
                if tool_call is not None:
                    return tool_call, None, [], metadata
        extra_function_call = extras.get("function_call")
        if extra_function_call is not None:
            tool_call = _tool_call_from_payload(extra_function_call)
            if tool_call is not None:
                return tool_call, None, ["legacy_function_call"], metadata
        if "content" in extras:
            tool_call, text, normalizations = _tool_call_from_content(extras.get("content"))
            if tool_call is not None or text is not None:
                return tool_call, text, normalizations, metadata

    for label, container in (("choice", choice), ("response", response)):
        extra = _as_mapping(container)
        if not extra:
            continue
        metadata.setdefault(f"{label}_extra_keys", sorted(extra.keys()))
        extra_tool_calls = extra.get("tool_calls")
        if isinstance(extra_tool_calls, Sequence) and not isinstance(
            extra_tool_calls, (str, bytes, bytearray)
        ):
            for candidate in extra_tool_calls:
                tool_call = _tool_call_from_payload(candidate)
                if tool_call is not None:
                    normalization = (
                        "legacy_function_call" if label == "response" else "content_tool_call"
                    )
                    return tool_call, None, [normalization], metadata
        extra_function_call = extra.get("function_call")
        if extra_function_call is not None:
            tool_call = _tool_call_from_payload(extra_function_call)
            if tool_call is not None:
                return tool_call, None, ["legacy_function_call"], metadata
        if "content" in extra:
            tool_call, text, normalizations = _tool_call_from_content(extra.get("content"))
            if tool_call is not None or text is not None:
                return tool_call, text, normalizations, metadata

    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")
    tool_call, text, normalizations = _tool_call_from_content(content)
    return tool_call, text, normalizations, metadata


def _normalize_base_url(base_url: str) -> str:
    """Return a root base URL suitable for resolving ``api/chat``.

    Users often configure ``OLLAMA_BASE_URL`` with a trailing ``/v1/`` suffix
    because that's what OpenAI-compatible clients expect. The native Ollama
    API lives under ``/api/*`` at the server root, so we strip any ``/v1``
    segment before handing the URL to httpx.
    """

    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[: -len("/v1")]
    return stripped + "/"


def _usage_from_native_response(response_json: Mapping[str, Any]) -> TokenUsage:
    """Map Ollama's native token counters to our ``TokenUsage`` shape."""

    prompt = response_json.get("prompt_eval_count")
    completion = response_json.get("eval_count")
    return TokenUsage(
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
    )


def _protocol_error_metadata(
    *,
    reason: str,
    http_status: int | None,
    raw_request_json: Mapping[str, Any],
    raw_response_json: Any,
    parsed_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider_protocol_error": reason,
        "provider_protocol_error_kind": "opaque_empty_completion",
        "opaque_response": True,
        "opaque_response_dump": {
            "reason": reason,
            "http_status": http_status,
            "raw_request_json": dict(raw_request_json),
            "raw_response_json": raw_response_json,
        },
    }
    if parsed_summary is not None:
        metadata["opaque_response_dump"]["parsed_summary"] = dict(parsed_summary)
    return metadata


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
        max_retries: int = 0,
        num_ctx: int | None = 24576,
        keep_alive: str | None = _DEFAULT_OLLAMA_KEEP_ALIVE,
    ) -> None:
        del max_retries
        resolved_base_url = _normalize_base_url(
            base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        )
        resolved_api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        self._client = httpx.Client(
            base_url=resolved_base_url,
            headers={
                "Authorization": f"Bearer {resolved_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        self._base_url = resolved_base_url
        self._model = _normalize_ollama_model(model)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._keep_alive = _resolve_ollama_keep_alive(keep_alive)
        self._think = _resolve_ollama_think_setting(model)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        # The native /api/chat endpoint does not accept a tool_choice field.
        # The agent-level retry loop enforces "must use tool" semantics via
        # feedback messages on subsequent attempts, which is sufficient in
        # practice and portable across endpoints.
        del tool_choice

        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})

        options: dict[str, Any] = {
            "temperature": self._temperature,
            "num_predict": self._max_tokens,
        }
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": all_messages,
            "stream": False,
            "options": options,
        }
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
        if self._think is not None:
            payload["think"] = self._think

        try:
            response = self._client.post("api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(
                f"Ollama request failed: {exc}",
                metadata={
                    "http_status": None,
                    "raw_request_json": payload,
                },
            ) from exc

        raw_text = response.text
        try:
            response_json = response.json()
        except ValueError as exc:
            raise ProviderExecutionError(
                "Ollama returned a non-JSON response.",
                metadata={
                    "http_status": response.status_code,
                    "raw_request_json": payload,
                    "raw_response_text": raw_text,
                },
            ) from exc

        if response.status_code >= 400:
            error_message = None
            if isinstance(response_json, Mapping):
                error_payload = response_json.get("error")
                if isinstance(error_payload, Mapping):
                    raw_message = error_payload.get("message")
                    if isinstance(raw_message, str):
                        error_message = raw_message
                elif isinstance(error_payload, str):
                    error_message = error_payload
            message = (
                f"Ollama returned HTTP {response.status_code}: {error_message}"
                if error_message
                else f"Ollama returned HTTP {response.status_code}"
            )
            raise ProviderExecutionError(
                message,
                metadata={
                    "http_status": response.status_code,
                    "raw_request_json": payload,
                    "raw_response_json": response_json,
                },
            )

        if not isinstance(response_json, Mapping):
            raise ProviderExecutionError(
                "Ollama returned a malformed JSON response.",
                metadata={
                    "http_status": response.status_code,
                    "raw_request_json": payload,
                    "raw_response_json": response_json,
                },
            )

        message_payload = response_json.get("message")
        if not isinstance(message_payload, Mapping):
            raise ProviderExecutionError(
                "Ollama response shape error: top-level 'message' is missing or not an object.",
                metadata={
                    "http_status": response.status_code,
                    "raw_request_json": payload,
                    "raw_response_json": response_json,
                },
            )

        usage = _usage_from_native_response(response_json)
        tool_call, text, transport_normalization_kinds, extraction_metadata = (
            _extract_tool_call_and_text(message_payload, choice=None, response=response_json)
        )
        done_reason = response_json.get("done_reason")
        metadata: dict[str, Any] = {
            "provider": "ollama",
            "response_id": response_json.get("id"),
            "model": response_json.get("model", self._model),
            # Keep the OpenAI-style key name so downstream telemetry and the
            # reviewer-facing bundle inspectors don't need to learn a second
            # vocabulary. The value is Ollama's native done_reason.
            "finish_reason": done_reason,
            "base_url": self._base_url,
            "http_status": response.status_code,
            "transport_normalization_kinds": transport_normalization_kinds,
            **extraction_metadata,
        }
        if usage.completion_tokens > 0 and tool_call is None and text is None:
            metadata.update(
                _protocol_error_metadata(
                    reason=(
                        "Ollama completion returned nonzero completion tokens but "
                        "no text and no tool call."
                    ),
                    http_status=response.status_code,
                    raw_request_json=payload,
                    raw_response_json=response_json,
                    parsed_summary={
                        "done_reason": done_reason,
                        "completion_tokens": usage.completion_tokens,
                        "content": message_payload.get("content"),
                        "tool_calls": message_payload.get("tool_calls"),
                        "function_call": message_payload.get("function_call"),
                    },
                )
            )

        return LLMResponse(
            tool_call=tool_call,
            text=text,
            usage=usage,
            metadata=metadata,
        )

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }


__all__ = [
    "OllamaProvider",
    "_default_ollama_think_setting",
    "_normalize_ollama_model",
    "_resolve_ollama_think_setting",
]
