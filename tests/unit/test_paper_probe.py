"""Tests for paper provider probes."""

from __future__ import annotations

import re

from labverse.paper.probe import build_probe_request


def _extract_block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, flags=re.S)
    assert match is not None
    return match.group(1)


def test_build_probe_request_for_shared_equipment_multi_agent_surface() -> None:
    request = build_probe_request(
        condition="local_narrative_interactive_grounded",
        scenario="shared_equipment_scheduling",
    )

    assert request.tool_schema["name"] == "take_action"
    # Affordance-only surface: structurally-applicable actions only, no precondition
    # filtering, so this count reflects the structural fan-out for the team.
    # 3 techs; +1 `wait` candidate per tech over the previous 18 after `wait` was
    # promoted to a first-class verb alongside `log_observation`.
    assert request.available_action_count == 21
    assert "<team_state>" in str(request.user_message["content"])
    assert "<available_team_actions>" in str(request.user_message["content"])
    assert "<task_contract>" in str(request.user_message["content"])
    assert "<turn_status>" in str(request.user_message["content"])
    assert "completion_ratio" not in str(request.user_message["content"])


def test_build_probe_request_for_single_agent_structured_surface() -> None:
    request = build_probe_request(
        condition="local_structured_interactive_grounded",
        scenario="chemical_spill_response",
    )

    assert request.tool_schema["name"] == "take_action"
    assert request.available_action_count > 0
    assert "<observation>" in str(request.user_message["content"])
    assert "<available_actions>" in str(request.user_message["content"])
    assert "<task_contract>" in request.system_prompt
    assert "<turn_status>" in str(request.user_message["content"])
    assert "completion_ratio" not in request.system_prompt
    assert "completion_ratio" not in str(request.user_message["content"])


def test_interactive_probe_conditions_share_same_task_contract() -> None:
    requests = [
        build_probe_request(condition=condition, scenario="chemical_spill_response")
        for condition in (
            "local_structured_interactive_grounded",
            "local_narrative_interactive_grounded",
            "global_structured_interactive_grounded",
        )
    ]

    task_contracts = [
        _extract_block(request.system_prompt, "task_contract") for request in requests
    ]

    assert task_contracts[0] == task_contracts[1] == task_contracts[2]
