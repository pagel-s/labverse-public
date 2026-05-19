"""Experiment runner: orchestrates condition x model x scenario x replication.

Usage:
    python -m labverse.benchmarks.experiment \
        --conditions local_structured_interactive_grounded \
                     local_narrative_interactive_grounded \
                     global_structured_interactive_grounded \
        --models claude-sonnet-4-20250514 gpt-4o \
        --providers anthropic openai \
        --scenarios pcr_setup chemical_spill_response \
        --replications 5 \
        --output results.jsonl
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from labverse.agent.audit import (
    attach_action_result,
    build_artifact_integrity_step,
    compute_step_metrics,
    fail_step_audit,
    finalize_step_audit,
    mark_post_completion_noop,
    materialize_audit_steps,
    read_actor_audit_steps,
    start_step_audit,
)
from labverse.agent.global_structured import (
    GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS,
    GlobalStructuredInteractiveAgent,
    format_world_state_summary_json,
)
from labverse.agent.interactive_text import InteractiveTextAgent
from labverse.agent.interface import AgentInterface
from labverse.agent.llm import LLMAgent
from labverse.agent.prompt import TurnStatus
from labverse.agent.providers.base import LLMProvider, TokenUsage
from labverse.agent.text_planner import TextPlannerAgent
from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    normalize_condition_name,
    normalize_condition_names,
)
from labverse.benchmarks.initial_variants import (
    BASELINE_INITIAL_VARIANT_ID,
    apply_initial_condition_variant,
)
from labverse.benchmarks.paper_agents import (
    _GLOBAL_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT,
    _INTERACTIVE_MULTI_AGENT_SYSTEM_PROMPT,
    _STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT,
    MultiAgentInteractiveCoordinator,
    MultiAgentTextPlanner,
    bind_planned_step,
    build_team_actions_text,
    build_team_observations_text,
    build_team_structured_observations_text,
    collect_team_actions,
)
from labverse.benchmarks.replay import build_replay_frames_for_steps
from labverse.benchmarks.runner import AgentFactory, BenchmarkRunner, ScenarioExecutionCapture
from labverse.benchmarks.scenarios.base import BenchmarkScenario
from labverse.benchmarks.scorer import BenchmarkScore, BenchmarkScorer
from labverse.gamemaster import GameMaster
from labverse.world.state import WorldState

logger = logging.getLogger(__name__)

CONDITION_LOCAL_STRUCTURED = CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED
CONDITION_LOCAL_NARRATIVE = CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED
CONDITION_GLOBAL_STRUCTURED = CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED
CONDITION_GLOBAL_ONE_SHOT = CONDITION_GLOBAL_ONE_SHOT_PLANNER
ALL_CONDITIONS = [
    CONDITION_LOCAL_STRUCTURED,
    CONDITION_LOCAL_NARRATIVE,
    CONDITION_GLOBAL_STRUCTURED,
    CONDITION_GLOBAL_ONE_SHOT,
]

# Backward-compatible aliases kept for older internal references and tests.
CONDITION_EMBODIED = CONDITION_LOCAL_STRUCTURED
CONDITION_TEXT_FEEDBACK = CONDITION_LOCAL_NARRATIVE
CONDITION_TEXT_ONLY = CONDITION_GLOBAL_ONE_SHOT
PRIMARY_INTERACTIVE_CONDITIONS = [
    CONDITION_LOCAL_STRUCTURED,
    CONDITION_LOCAL_NARRATIVE,
    CONDITION_GLOBAL_STRUCTURED,
]
OLLAMA_ACTION_MAX_TOKENS = 256
OLLAMA_ONE_SHOT_MAX_TOKENS = 2048
# Minimum HTTP timeout for ollama requests. Cold-start load of large open-weights
# models (31B+) routinely exceeds the 120 s default that the frontier providers
# use, so we floor ollama's per-request timeout well above that. The provider
# also pins the model in VRAM via keep_alive, so this deadline only matters on
# the very first call of a run.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 300.0


@dataclass
class RunResult:
    """Result of a single experiment run."""

    condition: str
    model: str
    provider: str
    scenario: str
    replication: int
    score: BenchmarkScore
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "model": self.model,
            "provider": self.provider,
            "scenario": self.scenario,
            "replication": self.replication,
            "success": self.score.success,
            "completion_ratio": self.score.completion_ratio,
            "efficiency_score": self.score.efficiency_score,
            "steps_taken": self.score.steps_taken,
            "safety_violations": self.score.safety_violations,
            "blocked_actions": self.score.blocked_actions,
            "time_elapsed_seconds": self.score.time_elapsed_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_time_seconds": self.wall_time_seconds,
            "error": self.error,
            "timestamp": self.timestamp,
            "details": self.score.details,
        }


@dataclass(frozen=True)
class ProviderOptions:
    """LLM provider execution settings for experiment runs."""

    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout_seconds: float = 120.0
    client_max_retries: int = 0
    worker_timeout_seconds: float = 1800.0


@dataclass
class ScenarioRunArtifacts:
    """Artifacts captured during one scenario run."""

    score: BenchmarkScore
    usage: TokenUsage = field(default_factory=TokenUsage)
    event_ids: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    trace_ids_by_agent: dict[str, list[str]] = field(default_factory=dict)
    step_audits: list[dict[str, Any]] = field(default_factory=list)
    replay_frames: list[dict[str, Any]] = field(default_factory=list)
    run_metrics: dict[str, int] = field(default_factory=dict)


def _create_provider(
    provider_name: str,
    model: str,
    *,
    options: ProviderOptions | None = None,
) -> LLMProvider:
    """Create an LLM provider instance."""
    provider_options = options or ProviderOptions()
    max_tokens = provider_options.max_tokens
    if provider_name == "anthropic":
        from labverse.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model,
            max_tokens=max_tokens,
            temperature=provider_options.temperature,
            timeout_seconds=provider_options.request_timeout_seconds,
            max_retries=provider_options.client_max_retries,
        )
    elif provider_name == "openai":
        from labverse.agent.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            max_tokens=max_tokens,
            temperature=provider_options.temperature,
            timeout_seconds=provider_options.request_timeout_seconds,
            max_retries=provider_options.client_max_retries,
        )
    elif provider_name == "ollama":
        from labverse.agent.providers.ollama import OllamaProvider

        return OllamaProvider(
            model=model,
            max_tokens=max_tokens,
            temperature=provider_options.temperature,
            timeout_seconds=provider_options.request_timeout_seconds,
            max_retries=provider_options.client_max_retries,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def _provider_options_for_condition(
    condition: str,
    provider_name: str,
    options: ProviderOptions | None = None,
) -> ProviderOptions:
    """Apply provider-specific limits without distorting condition semantics."""

    provider_options = options or ProviderOptions()
    max_tokens = provider_options.max_tokens
    request_timeout_seconds = provider_options.request_timeout_seconds
    canonical_condition = normalize_condition_name(condition)
    if provider_name == "ollama":
        if canonical_condition in PRIMARY_INTERACTIVE_CONDITIONS:
            max_tokens = min(max_tokens, OLLAMA_ACTION_MAX_TOKENS)
        elif canonical_condition == CONDITION_GLOBAL_ONE_SHOT:
            max_tokens = max(max_tokens, OLLAMA_ONE_SHOT_MAX_TOKENS)
        # Floor the per-request timeout: 120 s is enough for frontier APIs but
        # not for first-call cold-start on 31B+ open-weights models under ollama.
        request_timeout_seconds = max(request_timeout_seconds, OLLAMA_REQUEST_TIMEOUT_SECONDS)
    return ProviderOptions(
        temperature=provider_options.temperature,
        max_tokens=max_tokens,
        request_timeout_seconds=request_timeout_seconds,
        client_max_retries=provider_options.client_max_retries,
        worker_timeout_seconds=provider_options.worker_timeout_seconds,
    )


def _make_agent_factory(
    condition: str,
    provider_name: str,
    model: str,
    *,
    provider_options: ProviderOptions | None = None,
) -> AgentFactory:
    """Return an AgentFactory function for the given condition."""
    canonical_condition = normalize_condition_name(condition)
    adapted_provider_options = _provider_options_for_condition(
        canonical_condition,
        provider_name,
        provider_options,
    )

    def factory(
        scenario: BenchmarkScenario,
        world_snapshot: WorldState,
    ) -> AgentInterface | list[AgentInterface]:
        goal = scenario.get_goal_description()

        scripted_agents = scenario.build_scripted_agents()
        agent_ids = [agent.agent_id for agent in scripted_agents]
        if not agent_ids:
            raise ValueError(f"Scenario {scenario.name} does not define any agent IDs")

        if canonical_condition == CONDITION_LOCAL_STRUCTURED:
            if len(agent_ids) > 1:
                return [
                    LLMAgent(
                        agent_id=agent_id,
                        provider=_create_provider(
                            provider_name,
                            model,
                            options=adapted_provider_options,
                        ),
                        goal=_goal_for_agent(
                            goal=goal,
                            agent_id=agent_id,
                            team_agent_ids=agent_ids,
                        ),
                        name=f"Embodied-{agent_id}-{model}",
                    )
                    for agent_id in agent_ids
                ]
            agent_id = agent_ids[0]
            return LLMAgent(
                agent_id=agent_id,
                provider=_create_provider(
                    provider_name,
                    model,
                    options=adapted_provider_options,
                ),
                goal=goal,
                name=f"Embodied-{model}",
            )
        elif canonical_condition == CONDITION_GLOBAL_ONE_SHOT:
            agent_id = agent_ids[0]
            return TextPlannerAgent(
                agent_id=agent_id,
                provider=_create_provider(
                    provider_name,
                    model,
                    options=adapted_provider_options,
                ),
                goal=goal,
                world_state=world_snapshot,
                name=f"TextPlanner-{model}",
                allowed_verbs=scenario.one_shot_allowed_verbs(),
                max_steps=scenario.get_max_steps(),
            )
        elif canonical_condition == CONDITION_LOCAL_NARRATIVE:
            agent_id = agent_ids[0]
            return InteractiveTextAgent(
                agent_id=agent_id,
                provider=_create_provider(
                    provider_name,
                    model,
                    options=adapted_provider_options,
                ),
                goal=goal,
                name=f"InteractiveText-{model}",
            )
        elif canonical_condition == CONDITION_GLOBAL_STRUCTURED:
            agent_id = agent_ids[0]
            return GlobalStructuredInteractiveAgent(
                agent_id=agent_id,
                provider=_create_provider(
                    provider_name,
                    model,
                    options=adapted_provider_options,
                ),
                goal=goal,
                name=f"GlobalStructured-{model}",
                max_history_turns=GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS,
            )
        else:
            raise ValueError(f"Unknown condition: {canonical_condition}")

    return factory


def _goal_for_agent(
    *,
    goal: str,
    agent_id: str,
    team_agent_ids: list[str],
) -> str:
    if len(team_agent_ids) <= 1:
        return goal
    teammates = ", ".join(candidate for candidate in team_agent_ids if candidate != agent_id)
    return (
        f"You are controlling technician {agent_id}. {goal} "
        f"Coordinate with teammates {teammates}. "
        "Complete only this technician's required work exactly once, do not repeat shared "
        "equipment runs after your technician is done, and leave shared resources available "
        "for the others."
    )


def _extract_token_usage(agent: AgentInterface) -> TokenUsage:
    """Extract cumulative token usage from agent if available."""
    usage = getattr(agent, "cumulative_usage", None)
    if isinstance(usage, TokenUsage):
        return usage
    return TokenUsage()


def _uses_paper_multi_agent_path(
    *,
    scenario_name: str,
    condition: str,
) -> bool:
    canonical_condition = normalize_condition_name(condition)
    return scenario_name == "shared_equipment_scheduling" and canonical_condition in {
        CONDITION_LOCAL_STRUCTURED,
        CONDITION_LOCAL_NARRATIVE,
        CONDITION_GLOBAL_STRUCTURED,
        CONDITION_GLOBAL_ONE_SHOT,
    }


def _extract_error_from_score(score: BenchmarkScore) -> str | None:
    runner_error = score.details.get("runner_error")
    return runner_error if isinstance(runner_error, str) else None


def _actor_audit_count(actor: object) -> int:
    return len(read_actor_audit_steps(actor))


def _new_actor_audit_steps(actor: object, previous_count: int) -> list[dict[str, Any]]:
    try:
        audit_steps = read_actor_audit_steps(actor)
    except Exception:
        return [
            build_artifact_integrity_step(
                error="Failed while collecting new actor audit steps.",
                actor_id=str(getattr(actor, "agent_id", None) or "unknown_actor"),
                actor_name=str(getattr(actor, "name", type(actor).__name__)),
                traceback_text=traceback.format_exc(),
            )
        ]
    return materialize_audit_steps(audit_steps[previous_count:])


def _step_delta_ids(
    *,
    game_master: GameMaster,
    agent_id: str,
    event_start_index: int,
    trace_start_count: int,
) -> tuple[list[str], list[str]]:
    event_ids = [event.id for event in game_master.event_log.events[event_start_index:]]
    traces = game_master.trace_records(agent_id, limit=10_000)
    trace_ids = [trace.trace_id for trace in traces[trace_start_count:]]
    return trace_ids, event_ids


def _extend_step_audits(
    collected: list[dict[str, Any]],
    new_steps: list[dict[str, Any]],
    *,
    result: Any | None = None,
    fallback_error: str | None = None,
    trace_ids: list[str] | None = None,
    event_ids: list[str] | None = None,
) -> None:
    if not new_steps:
        return
    final_index = len(new_steps) - 1
    for index, step in enumerate(new_steps):
        if index == final_index:
            if result is not None:
                attach_action_result(
                    step,
                    result=result,
                    trace_ids=trace_ids,
                    event_ids=event_ids,
                )
            elif fallback_error is not None:
                fail_step_audit(step, error=fallback_error, status="failed")
        collected.extend(new_steps[index : index + 1])


def _execution_only_step(
    *,
    actor_id: str,
    actor_name: str,
    prompt_surface_kind: str,
    action_description: str,
    resolved_action_id: str,
    resolved_parameters: dict[str, Any],
) -> dict[str, Any]:
    step = start_step_audit(
        actor_id=actor_id,
        actor_name=actor_name,
        actor_kind="execution",
        system_prompt=None,
        user_message={"role": "system", "content": action_description},
        tool_schema=None,
        prompt_surface_kind=prompt_surface_kind,
    )
    finalize_step_audit(
        step,
        resolved_action_id=resolved_action_id,
        resolved_parameters=resolved_parameters,
        status="executed",
    )
    return step


def _maybe_finish_after_failed_turn(
    *,
    scenario: BenchmarkScenario,
    game_master: GameMaster,
    new_steps: list[dict[str, Any]],
) -> BenchmarkScore | None:
    """Return a clean completion score if the scenario already finished before a failed turn."""

    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
    if not score.success:
        return None
    for step in new_steps:
        mark_post_completion_noop(step)
    return score


def _turn_status_for_scenario(
    *,
    scenario: BenchmarkScenario,
    steps_taken: int,
) -> TurnStatus:
    """Build non-oracle per-turn status for prompt-visible interactive feedback."""

    return TurnStatus(
        scenario_name=scenario.name,
        steps_used=steps_taken,
        max_steps=scenario.get_max_steps(),
    )


def _run_multi_agent_text_only(
    *,
    scenario: BenchmarkScenario,
    provider_name: str,
    model: str,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> ScenarioRunArtifacts:
    world = apply_initial_condition_variant(
        scenario,
        scenario.configured_world(),
        initial_variant_id,
    )
    agent_ids = [agent.agent_id for agent in scenario.build_scripted_agents()]
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    observations = {agent_id: game_master.get_observation(agent_id) for agent_id in agent_ids}
    action_specs = {
        agent_id: game_master.get_available_actions(agent_id).available_actions
        for agent_id in agent_ids
    }
    planner = MultiAgentTextPlanner(
        provider=_create_provider(
            provider_name,
            model,
            options=_provider_options_for_condition(
                CONDITION_GLOBAL_ONE_SHOT,
                provider_name,
                provider_options,
            ),
        ),
        goal=scenario.get_goal_description(),
        world_state=world.snapshot(),
        agent_ids=agent_ids,
        initial_observations=build_team_observations_text(observations),
        initial_action_text=build_team_actions_text(
            action_specs,
            allowed_verbs=scenario.one_shot_allowed_verbs(),
        ),
        name=f"TextPlanner-{model}",
        allowed_verbs=scenario.one_shot_allowed_verbs(),
        max_steps=scenario.get_max_steps(),
    )
    step_audits: list[dict[str, Any]] = []
    replay_frames: list[dict[str, Any]] = []
    try:
        plan = planner.plan()
    except Exception as exc:
        logger.warning(
            "Multi-agent text planner failed for %s/%s: %s",
            scenario.name,
            model,
            exc,
        )
        return ScenarioRunArtifacts(
            score=BenchmarkScorer.failure_score(scenario.name, error=str(exc)),
            usage=planner.cumulative_usage,
            step_audits=list(planner.audit_steps),
            replay_frames=replay_frames,
            run_metrics=compute_step_metrics(list(planner.audit_steps)),
        )
    step_audits.extend(planner.audit_steps)

    failed_error: str | None = None
    for step in plan[: scenario.get_max_steps()]:
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=planner.cumulative_usage,
                agent_ids=agent_ids,
                step_audits=step_audits,
            )
        event_start_index = len(game_master.event_log.events)
        trace_start_count = len(game_master.trace_records(step.agent_id, limit=10_000))
        execution_audit = _execution_only_step(
            actor_id=step.agent_id,
            actor_name=planner.name,
            prompt_surface_kind="global_one_shot_planner",
            action_description=(
                f"Replay planned step for {step.agent_id}: {step.verb}"
                + (f" target={step.target_id}" if step.target_id else "")
            ),
            resolved_action_id=f"planned::{step.agent_id}::{step.verb}",
            resolved_parameters=dict(step.parameters),
        )
        try:
            world_before = game_master.world.snapshot()
            tick_before = game_master.clock.tick
            current_time_before = game_master.clock.current_time
            step_observation = game_master.get_observation(step.agent_id)
            step_action_spec = game_master.get_available_actions(step.agent_id)
            available_actions = step_action_spec.available_actions
            action = bind_planned_step(step=step, available_actions=available_actions)
            result = game_master.submit_action(
                action,
                observation=step_observation,
                action_spec=step_action_spec,
            )
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            trace_ids, event_ids = _step_delta_ids(
                game_master=game_master,
                agent_id=step.agent_id,
                event_start_index=event_start_index,
                trace_start_count=trace_start_count,
            )
            attach_action_result(
                execution_audit,
                result=result,
                trace_ids=trace_ids,
                event_ids=event_ids,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=[execution_audit],
                    starting_step_index=len(step_audits) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                    result=result,
                )
            )
        except Exception as exc:
            failed_error = str(exc)
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            fail_step_audit(execution_audit, error=failed_error)
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=[execution_audit],
                    starting_step_index=len(step_audits) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                )
            )
            logger.warning(
                "Multi-agent text-only execution stopped on %s/%s: %s",
                scenario.name,
                model,
                exc,
            )
            step_audits.append(execution_audit)
            completed_score = _maybe_finish_after_failed_turn(
                scenario=scenario,
                game_master=game_master,
                new_steps=[execution_audit],
            )
            if completed_score is not None:
                return _artifacts_from_game_master(
                    game_master=game_master,
                    score=completed_score,
                    usage=planner.cumulative_usage,
                    agent_ids=agent_ids,
                    step_audits=step_audits,
                    replay_frames=replay_frames,
                )
            break
        step_audits.append(execution_audit)
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=planner.cumulative_usage,
                agent_ids=agent_ids,
                step_audits=step_audits,
                replay_frames=replay_frames,
            )

    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
    if failed_error is not None:
        score = score.model_copy(
            update={"details": {**score.details, "runner_error": failed_error}},
            deep=True,
        )
    return _artifacts_from_game_master(
        game_master=game_master,
        score=score,
        usage=planner.cumulative_usage,
        agent_ids=agent_ids,
        step_audits=step_audits,
        replay_frames=replay_frames,
    )


def _interactive_team_observation_text(
    *,
    condition: str,
    observations: dict[str, Any],
) -> dict[str, str]:
    canonical_condition = normalize_condition_name(condition)
    if canonical_condition == CONDITION_LOCAL_NARRATIVE:
        return build_team_observations_text(observations)
    return build_team_structured_observations_text(observations)


def _interactive_team_system_prompt(condition: str) -> str:
    canonical_condition = normalize_condition_name(condition)
    if canonical_condition == CONDITION_LOCAL_STRUCTURED:
        return _STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT
    if canonical_condition == CONDITION_GLOBAL_STRUCTURED:
        return _GLOBAL_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT
    return _INTERACTIVE_MULTI_AGENT_SYSTEM_PROMPT


def _run_multi_agent_interactive(
    *,
    condition: str,
    scenario: BenchmarkScenario,
    provider_name: str,
    model: str,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> ScenarioRunArtifacts:
    world = apply_initial_condition_variant(
        scenario,
        scenario.configured_world(),
        initial_variant_id,
    )
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agent_ids = [agent.agent_id for agent in scenario.build_scripted_agents()]
    canonical_condition = normalize_condition_name(condition)
    coordinator = MultiAgentInteractiveCoordinator(
        provider=_create_provider(
            provider_name,
            model,
            options=_provider_options_for_condition(condition, provider_name, provider_options),
        ),
        goal=scenario.get_goal_description(),
        agent_ids=agent_ids,
        name=f"InteractiveText-{model}",
        max_history_turns=(
            GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS
            if canonical_condition == CONDITION_GLOBAL_STRUCTURED
            else 10
        ),
        system_prompt=_interactive_team_system_prompt(condition),
    )
    step_audits: list[dict[str, Any]] = []
    replay_frames: list[dict[str, Any]] = []

    steps_taken = 0
    failed_error: str | None = None
    failed_traceback: str | None = None
    while steps_taken < scenario.get_max_steps():
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=coordinator.cumulative_usage,
                agent_ids=agent_ids,
                step_audits=step_audits,
                replay_frames=replay_frames,
            )
        action_specs = {
            agent_id: game_master.get_available_actions(agent_id) for agent_id in agent_ids
        }
        team_actions = collect_team_actions(
            {
                agent_id: action_spec.available_actions
                for agent_id, action_spec in action_specs.items()
            }
        )
        if not team_actions:
            break

        observations = {agent_id: game_master.get_observation(agent_id) for agent_id in agent_ids}
        try:
            audit_start = _actor_audit_count(coordinator)
            event_start_index = len(game_master.event_log.events)
            world_before = game_master.world.snapshot()
            tick_before = game_master.clock.tick
            current_time_before = game_master.clock.current_time
            action = coordinator.next_action(
                observations=_interactive_team_observation_text(
                    condition=canonical_condition,
                    observations=observations,
                ),
                team_actions=team_actions,
                turn_status=_turn_status_for_scenario(
                    scenario=scenario,
                    steps_taken=steps_taken,
                ),
                world_state_json=(
                    format_world_state_summary_json(game_master.world.snapshot())
                    if canonical_condition == CONDITION_GLOBAL_STRUCTURED
                    else None
                ),
            )
            trace_start_count = len(game_master.trace_records(action.agent_id, limit=10_000))
            result = game_master.submit_action(
                action,
                observation=observations.get(action.agent_id),
                action_spec=action_specs.get(action.agent_id),
            )
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            trace_ids, event_ids = _step_delta_ids(
                game_master=game_master,
                agent_id=action.agent_id,
                event_start_index=event_start_index,
                trace_start_count=trace_start_count,
            )
            new_steps = _new_actor_audit_steps(coordinator, audit_start)
            _extend_step_audits(
                step_audits,
                new_steps,
                result=result,
                trace_ids=trace_ids,
                event_ids=event_ids,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=new_steps,
                    starting_step_index=len(step_audits) - len(new_steps) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                    result=result,
                )
            )
        except Exception as exc:
            failed_error = str(exc)
            failed_traceback = traceback.format_exc()
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            new_steps = _new_actor_audit_steps(coordinator, audit_start)
            _extend_step_audits(
                step_audits,
                new_steps,
                fallback_error=failed_error,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=new_steps,
                    starting_step_index=len(step_audits) - len(new_steps) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                )
            )
            completed_score = _maybe_finish_after_failed_turn(
                scenario=scenario,
                game_master=game_master,
                new_steps=new_steps,
            )
            if completed_score is not None:
                return _artifacts_from_game_master(
                    game_master=game_master,
                    score=completed_score,
                    usage=coordinator.cumulative_usage,
                    agent_ids=agent_ids,
                    step_audits=step_audits,
                    replay_frames=replay_frames,
                )
            logger.warning(
                "Multi-agent text-feedback execution stopped on %s/%s: %s",
                scenario.name,
                model,
                exc,
            )
            break
        coordinator.notify_result(
            action,
            result.success,
            result.outcome_description or result.blocked_reason or "",
        )
        steps_taken += 1
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=coordinator.cumulative_usage,
                agent_ids=agent_ids,
                step_audits=step_audits,
                replay_frames=replay_frames,
            )

    if failed_error is not None and steps_taken == 0:
        failure_score = BenchmarkScorer.failure_score(scenario.name, error=failed_error)
        if failed_traceback is not None:
            failure_score = failure_score.model_copy(
                update={
                    "details": {
                        **failure_score.details,
                        "runner_traceback": failed_traceback,
                    }
                },
                deep=True,
            )
        return ScenarioRunArtifacts(
            score=failure_score,
            usage=coordinator.cumulative_usage,
            step_audits=step_audits,
            replay_frames=replay_frames,
            run_metrics=compute_step_metrics(step_audits),
        )
    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
    if failed_error is not None:
        extra_details = {"runner_error": failed_error}
        if failed_traceback is not None:
            extra_details["runner_traceback"] = failed_traceback
        score = score.model_copy(
            update={"details": {**score.details, **extra_details}},
            deep=True,
        )
    return _artifacts_from_game_master(
        game_master=game_master,
        score=score,
        usage=coordinator.cumulative_usage,
        agent_ids=agent_ids,
        step_audits=step_audits,
        replay_frames=replay_frames,
    )


def _run_global_structured_interactive_single_agent(
    *,
    scenario: BenchmarkScenario,
    provider_name: str,
    model: str,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> ScenarioRunArtifacts:
    world = apply_initial_condition_variant(
        scenario,
        scenario.configured_world(),
        initial_variant_id,
    )
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agent_id = scenario.build_scripted_agents()[0].agent_id
    agent = GlobalStructuredInteractiveAgent(
        agent_id=agent_id,
        provider=_create_provider(
            provider_name,
            model,
            options=_provider_options_for_condition(
                CONDITION_GLOBAL_STRUCTURED,
                provider_name,
                provider_options,
            ),
        ),
        goal=scenario.get_goal_description(),
        name=f"GlobalStructured-{model}",
    )
    step_audits: list[dict[str, Any]] = []
    replay_frames: list[dict[str, Any]] = []

    steps_taken = 0
    failed_error: str | None = None
    while steps_taken < scenario.get_max_steps():
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=agent.cumulative_usage,
                agent_ids=[agent.agent_id],
                step_audits=step_audits,
                replay_frames=replay_frames,
            )
        try:
            audit_start = _actor_audit_count(agent)
            observation = game_master.get_observation(agent.agent_id)
            agent.set_turn_status(
                _turn_status_for_scenario(
                    scenario=scenario,
                    steps_taken=steps_taken,
                )
            )
            agent.observe(observation)
            agent.set_world_state(game_master.world.snapshot())
            action_spec = game_master.get_available_actions(agent.agent_id)
            event_start_index = len(game_master.event_log.events)
            world_before = game_master.world.snapshot()
            tick_before = game_master.clock.tick
            current_time_before = game_master.clock.current_time
            action = agent.act(action_spec)
            trace_start_count = len(game_master.trace_records(agent.agent_id, limit=10_000))
            result = game_master.submit_action(
                action,
                observation=observation,
                action_spec=action_spec,
            )
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            trace_ids, event_ids = _step_delta_ids(
                game_master=game_master,
                agent_id=agent.agent_id,
                event_start_index=event_start_index,
                trace_start_count=trace_start_count,
            )
            new_steps = _new_actor_audit_steps(agent, audit_start)
            _extend_step_audits(
                step_audits,
                new_steps,
                result=result,
                trace_ids=trace_ids,
                event_ids=event_ids,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=new_steps,
                    starting_step_index=len(step_audits) - len(new_steps) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                    result=result,
                )
            )
            agent.notify_result(
                result.success,
                result.outcome_description or result.blocked_reason or result.failure_reason or "",
            )
        except StopIteration:
            break
        except RuntimeError as exc:
            failed_error = str(exc)
            failed_traceback = traceback.format_exc()
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            new_steps = _new_actor_audit_steps(agent, audit_start)
            _extend_step_audits(
                step_audits,
                new_steps,
                fallback_error=failed_error,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=new_steps,
                    starting_step_index=len(step_audits) - len(new_steps) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                )
            )
            completed_score = _maybe_finish_after_failed_turn(
                scenario=scenario,
                game_master=game_master,
                new_steps=new_steps,
            )
            if completed_score is not None:
                return _artifacts_from_game_master(
                    game_master=game_master,
                    score=completed_score,
                    usage=agent.cumulative_usage,
                    agent_ids=[agent.agent_id],
                    step_audits=step_audits,
                    replay_frames=replay_frames,
                )
            logger.warning(
                "Global structured execution stopped on %s/%s: %s",
                scenario.name,
                model,
                exc,
            )
            break
        except Exception as exc:
            failed_error = str(exc)
            failed_traceback = traceback.format_exc()
            world_after = game_master.world.snapshot()
            tick_after = game_master.clock.tick
            current_time_after = game_master.clock.current_time
            new_steps = _new_actor_audit_steps(agent, audit_start)
            _extend_step_audits(
                step_audits,
                new_steps,
                fallback_error=failed_error,
            )
            replay_frames.extend(
                build_replay_frames_for_steps(
                    steps=new_steps,
                    starting_step_index=len(step_audits) - len(new_steps) + 1,
                    world_before=world_before,
                    world_after=world_after,
                    tick_before=tick_before,
                    tick_after=tick_after,
                    current_time_before=current_time_before,
                    current_time_after=current_time_after,
                )
            )
            completed_score = _maybe_finish_after_failed_turn(
                scenario=scenario,
                game_master=game_master,
                new_steps=new_steps,
            )
            if completed_score is not None:
                return _artifacts_from_game_master(
                    game_master=game_master,
                    score=completed_score,
                    usage=agent.cumulative_usage,
                    agent_ids=[agent.agent_id],
                    step_audits=step_audits,
                    replay_frames=replay_frames,
                )
            logger.warning(
                "Global structured execution stopped on %s/%s: %s",
                scenario.name,
                model,
                exc,
            )
            break

        steps_taken += 1
        score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
        if score.success:
            return _artifacts_from_game_master(
                game_master=game_master,
                score=score,
                usage=agent.cumulative_usage,
                agent_ids=[agent.agent_id],
                step_audits=step_audits,
                replay_frames=replay_frames,
            )

    if failed_error is not None and steps_taken == 0:
        failure_score = BenchmarkScorer.failure_score(scenario.name, error=failed_error)
        if failed_traceback is not None:
            failure_score = failure_score.model_copy(
                update={
                    "details": {
                        **failure_score.details,
                        "runner_traceback": failed_traceback,
                    }
                },
                deep=True,
            )
        return ScenarioRunArtifacts(
            score=failure_score,
            usage=agent.cumulative_usage,
            step_audits=step_audits,
            replay_frames=replay_frames,
            run_metrics=compute_step_metrics(step_audits),
        )

    score = scenario.evaluate(game_master.event_log, game_master.world.snapshot())
    if failed_error is not None:
        extra_details = {"runner_error": failed_error}
        if failed_traceback is not None:
            extra_details["runner_traceback"] = failed_traceback
        score = score.model_copy(update={"details": {**score.details, **extra_details}}, deep=True)
    return _artifacts_from_game_master(
        game_master=game_master,
        score=score,
        usage=agent.cumulative_usage,
        agent_ids=[agent.agent_id],
        step_audits=step_audits,
        replay_frames=replay_frames,
    )


def _artifacts_from_capture(
    capture: ScenarioExecutionCapture,
    *,
    usage: TokenUsage | None = None,
) -> ScenarioRunArtifacts:
    step_audits = materialize_audit_steps(capture.step_audits)
    score = capture.score
    if not score.success and not isinstance(score.details.get("runner_error"), str):
        last_step_error = next(
            (
                error
                for step in reversed(step_audits)
                if isinstance((error := step.get("error")), str) and error
            ),
            None,
        )
        if last_step_error is not None:
            score = score.model_copy(
                update={"details": {**score.details, "runner_error": last_step_error}},
                deep=True,
            )
    return ScenarioRunArtifacts(
        score=score,
        usage=usage or TokenUsage(),
        event_ids=list(capture.event_ids),
        trace_ids=list(capture.trace_ids),
        trace_ids_by_agent={
            agent_id: list(ids) for agent_id, ids in capture.trace_ids_by_agent.items()
        },
        step_audits=step_audits,
        replay_frames=[dict(frame) for frame in capture.replay_frames],
        run_metrics=compute_step_metrics(step_audits),
    )


def _artifacts_from_game_master(
    *,
    game_master: GameMaster,
    score: BenchmarkScore,
    usage: TokenUsage,
    agent_ids: list[str],
    step_audits: list[dict[str, Any]] | None = None,
    replay_frames: list[dict[str, Any]] | None = None,
) -> ScenarioRunArtifacts:
    trace_ids_by_agent = {
        agent_id: [trace.trace_id for trace in game_master.trace_records(agent_id, limit=10_000)]
        for agent_id in agent_ids
    }
    trace_ids: list[str] = []
    for agent_trace_ids in trace_ids_by_agent.values():
        trace_ids.extend(agent_trace_ids)
    materialized_step_audits = materialize_audit_steps(step_audits or [])
    return ScenarioRunArtifacts(
        score=score,
        usage=usage,
        event_ids=[event.id for event in game_master.event_log.events],
        trace_ids=trace_ids,
        trace_ids_by_agent=trace_ids_by_agent,
        step_audits=materialized_step_audits,
        replay_frames=[dict(frame) for frame in (replay_frames or [])],
        run_metrics=compute_step_metrics(materialized_step_audits),
    )


def _run_paper_scenario(
    *,
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> ScenarioRunArtifacts:
    canonical_condition = normalize_condition_name(condition)
    runner = BenchmarkRunner(initial_variant_id=initial_variant_id)
    scenario = runner.create_scenario(scenario_name)
    if _uses_paper_multi_agent_path(scenario_name=scenario_name, condition=canonical_condition):
        if canonical_condition == CONDITION_GLOBAL_ONE_SHOT:
            return _run_multi_agent_text_only(
                scenario=scenario,
                provider_name=provider_name,
                model=model,
                initial_variant_id=initial_variant_id,
                provider_options=provider_options,
            )
        return _run_multi_agent_interactive(
            condition=canonical_condition,
            scenario=scenario,
            provider_name=provider_name,
            model=model,
            initial_variant_id=initial_variant_id,
            provider_options=provider_options,
        )
    if canonical_condition == CONDITION_GLOBAL_STRUCTURED:
        return _run_global_structured_interactive_single_agent(
            scenario=scenario,
            provider_name=provider_name,
            model=model,
            initial_variant_id=initial_variant_id,
            provider_options=provider_options,
        )

    created_agents: list[AgentInterface] = []

    def tracking_factory(
        scenario_obj: BenchmarkScenario,
        world_snapshot: WorldState,
    ) -> AgentInterface | list[AgentInterface]:
        base_factory = _make_agent_factory(
            canonical_condition,
            provider_name,
            model,
            provider_options=provider_options,
        )
        agents = base_factory(scenario_obj, world_snapshot)
        if isinstance(agents, AgentInterface):
            created_agents.append(agents)
            return agents
        normalized_agents = list(agents)
        created_agents.extend(normalized_agents)
        return normalized_agents

    capture = runner.run_scenario_capture(scenario_name, agent_factory=tracking_factory)
    usage = TokenUsage()
    for agent in created_agents:
        usage = usage + _extract_token_usage(agent)
    return _artifacts_from_capture(capture, usage=usage)


def run_single_capture(
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    *,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> tuple[RunResult, ScenarioRunArtifacts]:
    """Run a single experiment trial and return summary plus captured artifacts."""
    condition = normalize_condition_name(condition)
    start_time = time.monotonic()
    error = None
    artifacts = ScenarioRunArtifacts(
        score=BenchmarkScorer.failure_score(scenario_name, error="unknown")
    )
    try:
        artifacts = _run_paper_scenario(
            condition=condition,
            provider_name=provider_name,
            model=model,
            scenario_name=scenario_name,
            initial_variant_id=initial_variant_id,
            provider_options=provider_options,
        )
        error = _extract_error_from_score(artifacts.score)
    except Exception as exc:
        captured_traceback = traceback.format_exc()
        logger.error(
            "Run failed: %s/%s/%s rep=%d: %s",
            condition,
            model,
            scenario_name,
            replication,
            exc,
        )
        error = str(exc)
        score = BenchmarkScorer.failure_score(scenario_name, error=error).model_copy(
            update={
                "details": {
                    "error": error,
                    "runner_traceback": captured_traceback,
                }
            },
            deep=True,
        )
        artifacts = ScenarioRunArtifacts(score=score)

    wall_time = time.monotonic() - start_time
    result = RunResult(
        condition=condition,
        model=model,
        provider=provider_name,
        scenario=scenario_name,
        replication=replication,
        score=artifacts.score,
        prompt_tokens=artifacts.usage.prompt_tokens,
        completion_tokens=artifacts.usage.completion_tokens,
        wall_time_seconds=round(wall_time, 2),
        error=error,
        timestamp=datetime.now().isoformat(),
    )
    return result, artifacts


def run_single(
    condition: str,
    provider_name: str,
    model: str,
    scenario_name: str,
    replication: int,
    *,
    initial_variant_id: str = BASELINE_INITIAL_VARIANT_ID,
    provider_options: ProviderOptions | None = None,
) -> RunResult:
    """Run a single experiment trial."""
    result, _ = run_single_capture(
        condition=normalize_condition_name(condition),
        provider_name=provider_name,
        model=model,
        scenario_name=scenario_name,
        replication=replication,
        initial_variant_id=initial_variant_id,
        provider_options=provider_options,
    )
    return result


def run_experiment(
    *,
    conditions: list[str] | None = None,
    providers: dict[str, str] | None = None,
    models: list[str] | None = None,
    scenarios: list[str] | None = None,
    replications: int = 5,
    output_path: str | Path | None = None,
) -> list[RunResult]:
    """Run the full experiment matrix.

    Args:
        conditions: List of conditions to test. Default: all canonical benchmark conditions.
        providers: Mapping of model name -> provider name.
                  E.g., {"claude-sonnet-4-20250514": "anthropic", "gpt-4o": "openai"}
        models: List of model names. Overrides providers keys if both given.
        scenarios: List of scenario names. Default: all scenarios.
        replications: Number of runs per cell. Default: 5.
        output_path: Path to write JSONL results. Default: None (return only).
    """
    conditions = normalize_condition_names(conditions) or ALL_CONDITIONS
    runner = BenchmarkRunner()
    all_scenarios = scenarios or runner.list_scenarios()

    # Build model -> provider mapping
    if providers is None:
        providers = {}
    if models:
        for m in models:
            if m not in providers:
                # Guess provider from model name
                if "claude" in m or "anthropic" in m:
                    providers[m] = "anthropic"
                elif m.startswith("ollama/"):
                    providers[m] = "ollama"
                elif "gpt" in m or "o1" in m or "o3" in m:
                    providers[m] = "openai"
                else:
                    raise ValueError(
                        f"Cannot infer provider for model '{m}'. Pass providers dict explicitly."
                    )
    model_list = list(providers.keys())

    total_runs = len(conditions) * len(model_list) * len(all_scenarios) * replications
    logger.info(
        "Starting experiment: %d conditions x %d models x %d scenarios x %d reps = %d runs",
        len(conditions),
        len(model_list),
        len(all_scenarios),
        replications,
        total_runs,
    )

    results: list[RunResult] = []
    completed = 0

    output_file = None
    if output_path:
        output_file = Path(output_path).open("a")

    try:
        for condition in conditions:
            for model_name in model_list:
                provider_name = providers[model_name]
                for scenario_name in all_scenarios:
                    for rep in range(1, replications + 1):
                        completed += 1
                        logger.info(
                            "[%d/%d] %s / %s / %s / rep %d",
                            completed,
                            total_runs,
                            condition,
                            model_name,
                            scenario_name,
                            rep,
                        )

                        result = run_single(
                            condition=condition,
                            provider_name=provider_name,
                            model=model_name,
                            scenario_name=scenario_name,
                            replication=rep,
                        )
                        results.append(result)

                        # Stream results to file
                        if output_file:
                            output_file.write(json.dumps(result.to_dict()) + "\n")
                            output_file.flush()

                        # Log progress
                        status = "PASS" if result.score.success else "FAIL"
                        logger.info(
                            "  -> %s (completion=%.2f, safety_violations=%d, wall=%.1fs)",
                            status,
                            result.score.completion_ratio,
                            result.score.safety_violations,
                            result.wall_time_seconds,
                        )
    finally:
        if output_file:
            output_file.close()

    # Summary
    successes = sum(1 for r in results if r.score.success)
    logger.info(
        "Experiment complete: %d/%d passed (%.1f%%)",
        successes,
        len(results),
        100 * successes / max(len(results), 1),
    )

    return results


def summarize_results(results: list[RunResult]) -> dict[str, Any]:
    """Compute summary statistics from experiment results."""
    from collections import defaultdict

    by_condition: dict[str, list[RunResult]] = defaultdict(list)
    by_condition_scenario: dict[str, dict[str, list[RunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_condition_model: dict[str, dict[str, list[RunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        by_condition[r.condition].append(r)
        by_condition_scenario[r.condition][r.scenario].append(r)
        by_condition_model[r.condition][r.model].append(r)

    def _stats(runs: list[RunResult]) -> dict[str, Any]:
        n = len(runs)
        if n == 0:
            return {}
        return {
            "n": n,
            "success_rate": sum(1 for r in runs if r.score.success) / n,
            "mean_completion": sum(r.score.completion_ratio for r in runs) / n,
            "mean_efficiency": sum(r.score.efficiency_score for r in runs) / n,
            "mean_safety_violations": sum(r.score.safety_violations for r in runs) / n,
            "mean_blocked_actions": sum(r.score.blocked_actions for r in runs) / n,
            "mean_steps": sum(r.score.steps_taken for r in runs) / n,
            "mean_prompt_tokens": sum(r.prompt_tokens for r in runs) / n,
            "mean_completion_tokens": sum(r.completion_tokens for r in runs) / n,
            "mean_wall_time": sum(r.wall_time_seconds for r in runs) / n,
        }

    summary: dict[str, Any] = {
        "overall": _stats(results),
        "by_condition": {c: _stats(runs) for c, runs in by_condition.items()},
        "by_condition_scenario": {
            c: {s: _stats(runs) for s, runs in scenarios.items()}
            for c, scenarios in by_condition_scenario.items()
        },
        "by_condition_model": {
            c: {m: _stats(runs) for m, runs in models.items()}
            for c, models in by_condition_model.items()
        },
    }

    return summary
