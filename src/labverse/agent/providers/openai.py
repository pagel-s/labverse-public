"""OpenAI provider adapter."""

from __future__ import annotations

import json
from typing import Any

from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall


class OpenAIProvider(LLMProvider):
    """LLM provider using the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_retries: int = 0,
    ) -> None:
        import openai

        self._client = openai.OpenAI(
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
        # OpenAI takes system as a message, not a separate parameter
        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": all_messages,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"

        response = self._client.chat.completions.create(**kwargs)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens or 0,
            completion_tokens=response.usage.completion_tokens or 0,
        )

        choice = response.choices[0]
        tool_call = None
        text = None

        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_call = ToolCall(name=tc.function.name, arguments=arguments)
        elif choice.message.content:
            text = choice.message.content

        return LLMResponse(
            tool_call=tool_call,
            text=text,
            usage=usage,
            metadata={
                "provider": "openai",
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", self._model),
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert generic tool definition to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
