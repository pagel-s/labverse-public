"""Base LLM provider interface and response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class ToolCall:
    """A tool/function call extracted from an LLM response."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    tool_call: ToolCall | None = None
    text: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderExecutionError(RuntimeError):
    """Provider-level execution failure with structured metadata."""

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class LLMProvider(ABC):
    """Abstract base for LLM provider adapters."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request and return the response.

        Args:
            messages: Conversation messages with 'role' and 'content' keys.
            tools: JSON Schema tool definitions for structured output.
            tool_choice: "required" to force tool use, "auto" for default.
            system: System prompt (passed separately for Anthropic compatibility).
        """
        ...


def tool_name_matches(actual_name: str, expected_name: str) -> bool:
    """Return whether a provider-emitted tool name is exactly or suffix-equivalent.

    Some provider adapters can namespace tool names while preserving the exact
    tool suffix, for example ``default_api:take_action``. Accept only exact
    suffix-equivalent aliases and nothing broader.
    """

    return actual_name == expected_name or actual_name.endswith(f":{expected_name}")


def response_transport_normalization_kinds(response: LLMResponse) -> list[str]:
    """Return provider-side transport normalizations recorded on a response."""

    raw = response.metadata.get("transport_normalization_kinds", [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]
    return []


def response_provider_protocol_error(response: LLMResponse) -> str | None:
    """Return a provider-protocol anomaly message recorded on a response."""

    raw = response.metadata.get("provider_protocol_error")
    if isinstance(raw, str) and raw:
        return raw
    return None
