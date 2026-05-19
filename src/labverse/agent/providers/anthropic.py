"""Anthropic Claude provider adapter."""

from __future__ import annotations

from typing import Any

from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_retries: int = 0,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
            if tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}

        response = self._client.messages.create(**kwargs)

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        # Extract tool use or text from response
        tool_call = None
        text = None
        for block in response.content:
            if block.type == "tool_use":
                tool_call = ToolCall(
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
                break
            elif block.type == "text":
                text = block.text

        return LLMResponse(
            tool_call=tool_call,
            text=text,
            usage=usage,
            metadata={
                "provider": "anthropic",
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", self._model),
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert generic tool definition to Anthropic format."""
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {}),
        }
