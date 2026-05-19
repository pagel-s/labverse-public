"""Tests for LLM provider adapters."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from labverse.agent.providers import create_provider
from labverse.agent.providers.base import LLMResponse, TokenUsage, ToolCall


class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_add(self):
        a = TokenUsage(prompt_tokens=100, completion_tokens=50)
        b = TokenUsage(prompt_tokens=200, completion_tokens=30)
        result = a + b
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 80


class TestLLMResponse:
    def test_defaults(self):
        resp = LLMResponse()
        assert resp.tool_call is None
        assert resp.text is None
        assert resp.usage.total_tokens == 0

    def test_with_tool_call(self):
        resp = LLMResponse(
            tool_call=ToolCall(name="take_action", arguments={"verb": "don_ppe"}),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        assert resp.tool_call is not None
        assert resp.tool_call.name == "take_action"


class TestCreateProviderFactory:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider("unknown_provider")

    @patch.dict("sys.modules", {"anthropic": None})
    def test_anthropic_missing_sdk_raises_import_error(self):
        with pytest.raises(ImportError, match="pip install labverse"):
            create_provider("anthropic")

    @patch.dict("sys.modules", {"openai": None})
    def test_openai_missing_sdk_raises_import_error(self):
        with pytest.raises(ImportError, match="pip install labverse"):
            create_provider("openai")


def _mock_anthropic_module() -> tuple[MagicMock, MagicMock]:
    """Create a mock anthropic module and return (mock_module, mock_client)."""
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module, mock_client


def _mock_openai_module() -> tuple[MagicMock, MagicMock]:
    """Create a mock openai module and return (mock_module, mock_client)."""
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    return mock_module, mock_client


class TestAnthropicProvider:
    def _make_provider(self):
        """Create an AnthropicProvider with a mocked SDK."""
        mock_module, mock_client = _mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            # Force re-import to pick up the mock
            if "labverse.agent.providers.anthropic" in sys.modules:
                del sys.modules["labverse.agent.providers.anthropic"]
            from labverse.agent.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(model="claude-test", api_key="test-key")
        return provider, mock_client

    def test_sends_system_separately(self):
        provider, mock_client = self._make_provider()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="hello")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        provider.complete(
            [{"role": "user", "content": "test"}],
            system="You are helpful.",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_converts_tool_format(self):
        provider, mock_client = self._make_provider()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        generic_tool = {
            "name": "take_action",
            "description": "Do something",
            "input_schema": {"type": "object", "properties": {}},
        }
        provider.complete(
            [{"role": "user", "content": "test"}],
            tools=[generic_tool],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        sent_tool = call_kwargs["tools"][0]
        assert sent_tool["name"] == "take_action"
        assert "input_schema" in sent_tool

    def test_extracts_tool_use_response(self):
        provider, mock_client = self._make_provider()

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "take_action"
        tool_block.input = {"verb": "don_ppe"}

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "test"}])

        assert result.tool_call is not None
        assert result.tool_call.name == "take_action"
        assert result.tool_call.arguments == {"verb": "don_ppe"}
        assert result.usage.prompt_tokens == 50

    def test_extracts_text_response(self):
        provider, mock_client = self._make_provider()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The sample is ready."

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 10
        mock_client.messages.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "test"}])

        assert result.text == "The sample is ready."
        assert result.tool_call is None


class TestOpenAIProvider:
    def _make_provider(self):
        """Create an OpenAIProvider with a mocked SDK."""
        mock_module, mock_client = _mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            if "labverse.agent.providers.openai" in sys.modules:
                del sys.modules["labverse.agent.providers.openai"]
            from labverse.agent.providers.openai import OpenAIProvider

            provider = OpenAIProvider(model="gpt-test", api_key="test-key")
        return provider, mock_client

    def test_prepends_system_as_message(self):
        provider, mock_client = self._make_provider()

        mock_choice = MagicMock()
        mock_choice.message.tool_calls = None
        mock_choice.message.content = "hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        provider.complete(
            [{"role": "user", "content": "test"}],
            system="You are helpful.",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_converts_tool_format(self):
        provider, mock_client = self._make_provider()

        mock_choice = MagicMock()
        mock_choice.message.tool_calls = None
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        generic_tool = {
            "name": "take_action",
            "description": "Do something",
            "input_schema": {"type": "object", "properties": {}},
        }
        provider.complete(
            [{"role": "user", "content": "test"}],
            tools=[generic_tool],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        sent_tool = call_kwargs["tools"][0]
        assert sent_tool["type"] == "function"
        assert sent_tool["function"]["name"] == "take_action"
        assert "parameters" in sent_tool["function"]

    def test_extracts_function_call_response(self):
        provider, mock_client = self._make_provider()

        mock_tc = MagicMock()
        mock_tc.function.name = "take_action"
        mock_tc.function.arguments = '{"verb": "don_ppe"}'

        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "test"}])

        assert result.tool_call is not None
        assert result.tool_call.name == "take_action"
        assert result.tool_call.arguments == {"verb": "don_ppe"}

    def test_extracts_text_response(self):
        provider, mock_client = self._make_provider()

        mock_choice = MagicMock()
        mock_choice.message.tool_calls = None
        mock_choice.message.content = "The answer is 42."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "test"}])

        assert result.text == "The answer is 42."
        assert result.tool_call is None
