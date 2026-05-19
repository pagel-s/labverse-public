"""One-shot provider probes for frozen paper prompt surfaces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from labverse.agent.global_structured import (
    build_global_structured_system_prompt,
    format_world_state_summary_json,
)
from labverse.agent.interactive_output import normalize_json_text_response
from labverse.agent.interactive_text import build_narrative_user_message
from labverse.agent.prompt import (
    TurnStatus,
    build_action_tool,
    build_global_structured_user_message,
    build_local_structured_user_message,
    build_narrative_system_prompt,
    build_system_prompt,
)
from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    PRIMARY_INTERACTIVE_CONDITIONS,
    normalize_condition_name,
)
from labverse.benchmarks.experiment import (
    ProviderOptions,
    _create_provider,
    _interactive_team_observation_text,
    _interactive_team_system_prompt,
)
from labverse.benchmarks.paper_agents import (
    _build_joint_action_tool,
    _format_team_state_message,
    collect_team_actions,
)
from labverse.benchmarks.runner import BenchmarkRunner
from labverse.gamemaster import GameMaster
from labverse.paper.runner import _infer_provider


@dataclass(slots=True)
class ProbeRequest:
    system_prompt: str
    user_message: dict[str, Any]
    tool_schema: dict[str, Any]
    prompt_surface_kind: str
    available_action_count: int


class PaperProbeReport(BaseModel):
    condition: str
    model: str
    provider: str
    scenario: str
    prompt_surface_kind: str
    tool_name: str
    available_action_count: int
    system_prompt_chars: int
    user_message_chars: int
    tool_schema_chars: int
    total_payload_chars: int
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    returned_action_output_kind: str | None = None
    returned_tool_name: str | None = None
    returned_candidate_ref: str | None = None
    returned_text: str | None = None


def _single_agent_request(
    *,
    condition: str,
    goal: str,
    game_master: GameMaster,
    agent_id: str,
    scenario_name: str,
    max_steps: int,
) -> ProbeRequest:
    observation = game_master.get_observation(agent_id)
    action_spec = game_master.get_available_actions(agent_id)
    available_actions = action_spec.available_actions
    turn_status = TurnStatus(
        scenario_name=scenario_name,
        steps_used=0,
        max_steps=max_steps,
    )

    if condition == CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED:
        system_prompt = build_system_prompt(goal=goal)
        user_message = build_local_structured_user_message(
            observation,
            available_actions,
            turn_status=turn_status,
        )
        prompt_surface_kind = condition
    elif condition == CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED:
        system_prompt = build_narrative_system_prompt(goal=goal)
        user_message = build_narrative_user_message(
            observation,
            available_actions,
            turn_status=turn_status,
        )
        prompt_surface_kind = condition
    elif condition == CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED:
        system_prompt = build_global_structured_system_prompt(goal=goal)
        user_message = build_global_structured_user_message(
            observation,
            available_actions,
            format_world_state_summary_json(game_master.world.snapshot()),
            turn_status=turn_status,
        )
        prompt_surface_kind = condition
    else:
        raise ValueError(f"Unsupported probe condition: {condition}")

    return ProbeRequest(
        system_prompt=system_prompt,
        user_message=user_message,
        tool_schema=build_action_tool(available_actions),
        prompt_surface_kind=prompt_surface_kind,
        available_action_count=len(available_actions),
    )


def _multi_agent_request(
    *,
    condition: str,
    scenario_name: str,
) -> ProbeRequest:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    game_master = GameMaster(
        scenario.configured_world(),
        safety_engine=scenario.build_safety_engine(),
    )
    agent_ids = [agent.agent_id for agent in scenario.build_scripted_agents()]
    action_specs = {agent_id: game_master.get_available_actions(agent_id) for agent_id in agent_ids}
    team_actions = collect_team_actions(
        {agent_id: action_spec.available_actions for agent_id, action_spec in action_specs.items()}
    )
    observations = {agent_id: game_master.get_observation(agent_id) for agent_id in agent_ids}
    observation_text = _interactive_team_observation_text(
        condition=condition,
        observations=observations,
    )
    world_state_json = (
        format_world_state_summary_json(game_master.world.snapshot())
        if condition == CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED
        else None
    )
    return ProbeRequest(
        system_prompt=_interactive_team_system_prompt(condition),
        user_message={
            "role": "user",
            "content": _format_team_state_message(
                goal=scenario.get_goal_description(),
                observations=observation_text,
                team_actions=team_actions,
                last_result_text=None,
                turn_status=TurnStatus(
                    scenario_name=scenario.name,
                    steps_used=0,
                    max_steps=scenario.get_max_steps(),
                ),
                world_state_json=world_state_json,
            ),
        },
        tool_schema=_build_joint_action_tool(team_actions),
        prompt_surface_kind=f"{condition}_team",
        available_action_count=len(team_actions),
    )


def build_probe_request(*, condition: str, scenario: str) -> ProbeRequest:
    canonical_condition = normalize_condition_name(condition)
    if canonical_condition not in PRIMARY_INTERACTIVE_CONDITIONS:
        raise ValueError("paper-probe only supports the primary interactive paper conditions.")

    if scenario == "shared_equipment_scheduling":
        return _multi_agent_request(condition=canonical_condition, scenario_name=scenario)

    runner = BenchmarkRunner()
    benchmark_scenario = runner.create_scenario(scenario)
    game_master = GameMaster(
        benchmark_scenario.configured_world(),
        safety_engine=benchmark_scenario.build_safety_engine(),
    )
    agent_id = benchmark_scenario.build_scripted_agents()[0].agent_id
    return _single_agent_request(
        condition=canonical_condition,
        goal=benchmark_scenario.get_goal_description(),
        game_master=game_master,
        agent_id=agent_id,
        scenario_name=benchmark_scenario.name,
        max_steps=benchmark_scenario.get_max_steps(),
    )


def probe_paper_request(
    *,
    model: str,
    condition: str,
    scenario: str,
    timeout_seconds: float = 30.0,
) -> PaperProbeReport:
    request = build_probe_request(condition=condition, scenario=scenario)
    provider = _infer_provider(model)
    provider_instance = _create_provider(
        provider,
        model,
        options=ProviderOptions(
            temperature=0.0,
            max_tokens=256,
            request_timeout_seconds=timeout_seconds,
            client_max_retries=0,
            worker_timeout_seconds=max(timeout_seconds + 5.0, 35.0),
        ),
    )

    start = time.monotonic()
    response = provider_instance.complete(
        [request.user_message],
        tools=[request.tool_schema],
        system=request.system_prompt,
    )
    elapsed_seconds = time.monotonic() - start

    payload_chars = len(
        json.dumps(
            {
                "system": request.system_prompt,
                "messages": [request.user_message],
                "tools": [request.tool_schema],
                "tool_choice": None,
            },
            sort_keys=True,
        )
    )
    returned_action_output_kind = None
    returned_candidate_ref = None
    returned_tool_name = None
    if response.tool_call is not None:
        returned_action_output_kind = "provider_tool_call_fallback"
        returned_tool_name = response.tool_call.name
        candidate_ref = response.tool_call.arguments.get("candidate_ref")
        if isinstance(candidate_ref, str):
            returned_candidate_ref = candidate_ref
    elif response.text and response.text.strip():
        returned_action_output_kind = "json_text"
        returned_candidate_ref = _extract_candidate_ref_from_json_text(response.text)

    return PaperProbeReport(
        condition=normalize_condition_name(condition),
        model=model,
        provider=provider,
        scenario=scenario,
        prompt_surface_kind=request.prompt_surface_kind,
        tool_name=request.tool_schema["name"],
        available_action_count=request.available_action_count,
        system_prompt_chars=len(request.system_prompt),
        user_message_chars=len(str(request.user_message.get("content", ""))),
        tool_schema_chars=len(json.dumps(request.tool_schema, sort_keys=True)),
        total_payload_chars=payload_chars,
        elapsed_seconds=round(elapsed_seconds, 2),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        response_metadata=dict(response.metadata),
        returned_action_output_kind=returned_action_output_kind,
        returned_tool_name=returned_tool_name,
        returned_candidate_ref=returned_candidate_ref,
        returned_text=response.text,
    )


def _extract_candidate_ref_from_json_text(text: str) -> str | None:
    try:
        parsed = json.loads(normalize_json_text_response(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    candidate_ref = parsed.get("candidate_ref")
    return candidate_ref if isinstance(candidate_ref, str) else None


__all__ = [
    "PaperProbeReport",
    "ProbeRequest",
    "build_probe_request",
    "probe_paper_request",
]
