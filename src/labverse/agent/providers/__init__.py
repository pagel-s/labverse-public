"""LLM provider adapters for agent implementations."""

from __future__ import annotations

from typing import Any

from labverse.agent.providers.base import LLMProvider, LLMResponse, TokenUsage, ToolCall

__all__ = ["LLMProvider", "LLMResponse", "TokenUsage", "ToolCall", "create_provider"]


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        name: Provider name — "anthropic", "openai", or "ollama".
        **kwargs: Passed to the provider constructor (model, api_key, etc.).
    """
    if name == "anthropic":
        try:
            from labverse.agent.providers.anthropic import AnthropicProvider

            return AnthropicProvider(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install labverse[anthropic]"
            ) from exc

    if name == "openai":
        try:
            from labverse.agent.providers.openai import OpenAIProvider

            return OpenAIProvider(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install labverse[openai]"
            ) from exc

    if name == "ollama":
        try:
            from labverse.agent.providers.ollama import OllamaProvider

            return OllamaProvider(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "Ollama HTTP client dependency not installed. Run: pip install httpx"
            ) from exc

    raise ValueError(f"Unknown LLM provider: {name!r}. Supported: 'anthropic', 'openai', 'ollama'")
