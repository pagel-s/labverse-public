"""Tests for the Ollama provider adapter."""

from __future__ import annotations

from typing import Any

import pytest

from labverse.agent.providers.base import ProviderExecutionError
from labverse.agent.providers.ollama import (
    OllamaProvider,
    _default_ollama_think_setting,
    _normalize_ollama_model,
    _resolve_ollama_think_setting,
)


class _FakeHTTPResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        text: str | None = None,
        json_error: ValueError | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else str(payload)
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeHTTPClient:
    def __init__(self, response: _FakeHTTPResponse, **kwargs: Any) -> None:
        self.response = response
        self.kwargs = kwargs
        self.post_calls: list[dict[str, Any]] = []

    def post(self, path: str, *, json: dict[str, Any]) -> _FakeHTTPResponse:
        self.post_calls.append({"path": path, "json": json})
        return self.response


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeHTTPResponse,
) -> dict[str, Any]:
    holder: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> _FakeHTTPClient:
        client = _FakeHTTPClient(response, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("labverse.agent.providers.ollama.httpx.Client", _factory)
    return holder


def _tool_response_payload() -> dict[str, Any]:
    """Native /api/chat tool-call response fixture.

    Ollama's native endpoint returns ``message`` at the top level (no
    ``choices`` array), uses ``done_reason`` instead of ``finish_reason``,
    and reports usage via ``prompt_eval_count`` / ``eval_count``. Tool-call
    ``arguments`` come through as an already-parsed object, not a JSON string.
    """

    return {
        "model": "gemma4:26b",
        "created_at": "2026-04-14T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "take_action",
                        "arguments": {
                            "candidate_ref": "observe_1",
                            "verb": "log_observation",
                        },
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 3,
    }


def test_normalize_ollama_model_strips_prefix() -> None:
    assert _normalize_ollama_model("ollama/qwen3:14b") == "qwen3:14b"
    assert _normalize_ollama_model("gemma4:26b") == "gemma4:26b"


def test_default_think_setting_disables_thinking_for_gemma_and_qwen() -> None:
    assert _default_ollama_think_setting("ollama/gemma4:26b") is False
    assert _default_ollama_think_setting("ollama/qwen3:14b") is False
    assert _default_ollama_think_setting("ollama/mistral-small:24b") is None


def test_resolve_think_setting_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_THINK", "low")
    assert _resolve_ollama_think_setting("ollama/gemma4:26b") == "low"
    monkeypatch.setenv("OLLAMA_THINK", "false")
    assert _resolve_ollama_think_setting("ollama/mistral-small:24b") is False
    monkeypatch.setenv("OLLAMA_THINK", "none")
    assert _resolve_ollama_think_setting("ollama/gemma4:26b") is None


def test_complete_sends_think_false_for_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))
    monkeypatch.delenv("OLLAMA_THINK", raising=False)

    provider = OllamaProvider(model="ollama/gemma4:26b")
    response = provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[
            {
                "name": "take_action",
                "input_schema": {
                    "type": "object",
                    "properties": {"candidate_ref": {"type": "string", "enum": ["a"]}},
                    "required": ["candidate_ref"],
                },
            }
        ],
        tool_choice="required",
        system="system",
    )

    assert response.tool_call is not None
    post_call = holder["client"].post_calls[0]
    assert post_call["path"] == "api/chat"
    assert post_call["json"]["think"] is False
    # tool_choice is accepted by the caller for API symmetry with other
    # providers, but the native endpoint does not support it and it must
    # never leak into the request payload.
    assert "tool_choice" not in post_call["json"]
    assert post_call["json"]["stream"] is False


def test_complete_omits_think_for_non_thinking_default(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))
    monkeypatch.delenv("OLLAMA_THINK", raising=False)

    provider = OllamaProvider(model="ollama/mistral-small:24b")
    provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[{"name": "take_action", "input_schema": {}}],
        tool_choice="required",
    )

    post_call = holder["client"].post_calls[0]
    assert "think" not in post_call["json"]


def test_complete_extracts_tool_call_from_message_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _tool_response_payload()
    payload["done_reason"] = "stop"
    payload["message"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "take_action",
                    "arguments": {"candidate_ref": "move_1", "target_id": "office"},
                }
            }
        ],
    }
    _install_fake_httpx(monkeypatch, _FakeHTTPResponse(payload))

    provider = OllamaProvider(model="ollama/gemma4:26b")
    response = provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[{"name": "take_action", "input_schema": {}}],
        tool_choice="required",
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "take_action"
    assert response.tool_call.arguments["candidate_ref"] == "move_1"
    assert response.metadata["transport_normalization_kinds"] == []


def test_complete_extracts_serialized_tool_call_from_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _tool_response_payload()
    payload["done_reason"] = "stop"
    payload["message"] = {
        "role": "assistant",
        "content": (
            '<tool_call>{"name":"take_action","arguments":{"candidate_ref":"observe_1"}}'
            "</tool_call>"
        ),
    }
    _install_fake_httpx(monkeypatch, _FakeHTTPResponse(payload))

    provider = OllamaProvider(model="ollama/gemma4:26b")
    response = provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[{"name": "take_action", "input_schema": {}}],
        tool_choice="required",
    )

    assert response.tool_call is not None
    assert response.tool_call.arguments == {"candidate_ref": "observe_1"}
    assert response.metadata["transport_normalization_kinds"] == ["content_tool_call"]


def test_complete_records_opaque_response_dump_when_no_text_or_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "model": "gemma4:26b",
        "created_at": "2026-04-14T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": None,
            "function_call": None,
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 21,
        "eval_count": 43,
    }
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(payload))

    provider = OllamaProvider(model="ollama/gemma4:26b")
    response = provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[{"name": "take_action", "input_schema": {}}],
        tool_choice="required",
    )

    assert response.tool_call is None
    assert response.text is None
    assert response.metadata["opaque_response"] is True
    assert "provider_protocol_error" in response.metadata
    dump = response.metadata["opaque_response_dump"]
    assert dump["http_status"] == 200
    # The raw request payload is captured with num_ctx lifted and tool_choice
    # intentionally absent — it's not a native-endpoint concept.
    assert dump["raw_request_json"]["options"]["num_ctx"] == 24576
    assert "tool_choice" not in dump["raw_request_json"]
    assert dump["raw_response_json"]["eval_count"] == 43
    assert holder["client"].post_calls


def test_complete_raises_provider_error_with_raw_payload_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"error": {"message": "model 'gemma4:26b' not found"}}
    _install_fake_httpx(monkeypatch, _FakeHTTPResponse(payload, status_code=404))

    provider = OllamaProvider(model="ollama/gemma4:26b")
    with pytest.raises(ProviderExecutionError) as exc_info:
        provider.complete([{"role": "user", "content": "choose"}])

    assert "HTTP 404" in str(exc_info.value)
    assert exc_info.value.metadata["http_status"] == 404
    assert exc_info.value.metadata["raw_response_json"] == payload


def test_complete_sends_default_num_ctx_and_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))

    provider = OllamaProvider(model="ollama/gemma4:26b")
    provider.complete(
        [{"role": "user", "content": "choose"}],
        tools=[{"name": "take_action", "input_schema": {}}],
        tool_choice="required",
    )

    post_call = holder["client"].post_calls[0]
    # Defaults must raise the runtime context window above the 4096-token ollama
    # default — otherwise the tool schema silently truncates the user prompt on
    # large open-weights models. 24k leaves real headroom above the ~16k peaks
    # observed on local_* interactive conditions in the gemma4:31b triage sweep.
    assert post_call["json"]["options"]["num_ctx"] == 24576
    # And must pin the model in VRAM across the whole run so cold-start cost
    # doesn't blow the per-request deadline on every scenario.
    assert post_call["json"]["keep_alive"] == "30m"


def test_complete_honors_explicit_num_ctx_and_keep_alive_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))

    provider = OllamaProvider(
        model="ollama/gemma4:26b",
        num_ctx=32768,
        keep_alive="5m",
    )
    provider.complete([{"role": "user", "content": "choose"}])

    post_call = holder["client"].post_calls[0]
    assert post_call["json"]["options"]["num_ctx"] == 32768
    assert post_call["json"]["keep_alive"] == "5m"


def test_complete_honors_env_keep_alive_override(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))
    monkeypatch.setenv("LABVERSE_OLLAMA_KEEP_ALIVE", "30s")

    provider = OllamaProvider(model="ollama/gemma4:26b")
    provider.complete([{"role": "user", "content": "choose"}])

    post_call = holder["client"].post_calls[0]
    assert post_call["json"]["keep_alive"] == "30s"


def test_complete_env_keep_alive_override_can_disable_request_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))
    monkeypatch.setenv("LABVERSE_OLLAMA_KEEP_ALIVE", "none")

    provider = OllamaProvider(model="ollama/gemma4:26b")
    provider.complete([{"role": "user", "content": "choose"}])

    post_call = holder["client"].post_calls[0]
    assert "keep_alive" not in post_call["json"]


def test_complete_omits_num_ctx_and_keep_alive_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))

    provider = OllamaProvider(
        model="ollama/gemma4:26b",
        num_ctx=None,
        keep_alive=None,
    )
    provider.complete([{"role": "user", "content": "choose"}])

    post_call = holder["client"].post_calls[0]
    # Native /api/chat always carries options for temperature/num_predict, but
    # num_ctx must disappear when explicitly disabled so callers can opt in to
    # the server-side default without us clobbering it.
    assert "num_ctx" not in post_call["json"]["options"]
    assert "keep_alive" not in post_call["json"]


def test_ollama_provider_default_timeout_accommodates_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_httpx(monkeypatch, _FakeHTTPResponse(_tool_response_payload()))

    OllamaProvider(model="ollama/gemma4:26b")

    # The default request timeout must exceed the frontier 120 s floor so that
    # first-call cold-start on 31B+ open-weights models doesn't trip the
    # provider into a retry loop before the model has finished loading.
    assert holder["client"].kwargs["timeout"] >= 300.0


def test_complete_raises_provider_error_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse("not-json", text="not-json", json_error=ValueError("bad json"))
    _install_fake_httpx(monkeypatch, response)

    provider = OllamaProvider(model="ollama/gemma4:26b")
    with pytest.raises(ProviderExecutionError) as exc_info:
        provider.complete([{"role": "user", "content": "choose"}])

    assert "non-JSON" in str(exc_info.value)
    assert exc_info.value.metadata["raw_response_text"] == "not-json"
