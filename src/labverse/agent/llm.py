"""LLM-powered agent for lab simulations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from labverse.agent.action_binding import build_action_from_available
from labverse.agent.action_history import (
    ActionTrajectory,
    prepend_interactive_history_blocks,
    prepend_trajectory_block,
)
from labverse.agent.audit import (
    attach_action_result,
    fail_step_audit,
    finalize_step_audit,
    materialize_audit_steps,
    record_attempt,
    start_step_audit,
)
from labverse.agent.interactive_output import parse_interactive_action_response
from labverse.agent.interface import AgentInterface, Observation
from labverse.agent.prompt import (
    INTERACTIVE_TOOL_RETRY_INSTRUCTION,
    TurnStatus,
    build_action_tool,
    build_local_structured_user_message,
    build_system_prompt,
    format_move_action_guidance,
)
from labverse.agent.providers.base import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
    response_provider_protocol_error,
)
from labverse.models.action import Action, ActionSpec

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
MAX_RETRIES = 2


@dataclass
class StepStats:
    """Token usage and retry count for a single act() call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0


class LLMAgent(AgentInterface):
    """An agent that uses an LLM to decide actions in the lab simulation."""

    def __init__(
        self,
        agent_id: str,
        provider: LLMProvider,
        *,
        goal: str,
        name: str | None = None,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        self._agent_id = agent_id
        self._provider = provider
        self._name = name or f"LLMAgent-{agent_id}"
        self._system_prompt = build_system_prompt(goal=goal)
        # ``max_history_turns`` is retained for API compatibility and controls
        # the compact decision-trace buffer rather than a raw list of prior
        # user messages. See ``labverse.agent.action_history`` for the
        # scientific rationale.
        self._max_history_turns = max_history_turns
        self._trajectory = ActionTrajectory(max_entries=max_history_turns)
        self._pending_observation: Observation | None = None
        self._pending_turn_status: TurnStatus | None = None
        self._cumulative_usage = TokenUsage()
        self._step_stats: list[StepStats] = []
        self._audit_steps: list[dict[str, Any]] = []
        self._prompt_surface_kind = "local_structured_interactive_grounded"

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._cumulative_usage

    @property
    def total_retries(self) -> int:
        return sum(s.retries for s in self._step_stats)

    @property
    def step_history(self) -> list[StepStats]:
        return list(self._step_stats)

    @property
    def audit_steps(self) -> list[dict[str, Any]]:
        return materialize_audit_steps(
            self._audit_steps,
            actor_id=self._agent_id,
            actor_name=self._name,
        )

    def observe(self, observation: Observation) -> None:
        """Store the observation for the next act() call."""
        self._pending_observation = observation

    def set_turn_status(self, turn_status: TurnStatus) -> None:
        """Store non-oracle per-turn status for the next act() call."""
        self._pending_turn_status = turn_status

    def act(self, action_spec: ActionSpec) -> Action:
        """Call the LLM to choose an action from the available set."""
        if self._pending_observation is None:
            raise RuntimeError("act() called before observe()")

        observation = self._pending_observation
        self._pending_observation = None

        # The compact decision trace already carries the prior action's
        # outcome (via ``notify_result`` → ``record_last_result``), so the
        # current user message no longer embeds a separate
        # ``<action_result>`` block. History is expressed once, at the top
        # of the turn, with no duplication.
        base_obs_msg = build_local_structured_user_message(
            observation,
            action_spec.available_actions,
            last_action_result=None,
            turn_status=self._pending_turn_status,
        )
        self._pending_turn_status = None
        obs_msg = prepend_interactive_history_blocks(base_obs_msg, self._trajectory)

        # Build tool definition
        action_tool = build_action_tool(action_spec.available_actions)

        # Assemble messages. The trajectory block lives inside the single
        # user message above; there is no legacy ``_history`` list to
        # prepend — doing so replayed stale observations and action
        # surfaces that were either identical (goal) or superseded by the
        # current turn, and could push prompts past the provider's
        # ``num_ctx`` ceiling where they would silently truncate.
        messages = [obs_msg]
        step_audit = start_step_audit(
            actor_id=self._agent_id,
            actor_name=self._name,
            actor_kind="agent",
            system_prompt=self._system_prompt,
            user_message=obs_msg,
            tool_schema=action_tool,
            prompt_surface_kind=self._prompt_surface_kind,
        )

        # Try to get a valid action from the LLM
        stats = StepStats()
        try:
            (
                action,
                selected_candidate_ref,
                resolved_action_id,
                normalization_kinds,
                used_legacy_tool_format,
            ) = self._call_with_retries(
                messages,
                action_tool,
                action_spec,
                obs_msg,
                stats,
                step_audit,
            )
        except Exception as exc:
            fail_step_audit(step_audit, error=str(exc))
            self._cumulative_usage = self._cumulative_usage + stats_to_usage(stats)
            self._step_stats.append(stats)
            self._audit_steps.append(step_audit)
            raise
        finalize_step_audit(
            step_audit,
            resolved_candidate_ref=selected_candidate_ref,
            resolved_action_id=resolved_action_id,
            resolved_verb=action.verb,
            resolved_target_id=action.target_id,
            resolved_parameters=action.parameters,
            fallback_kind=normalization_kinds[0] if normalization_kinds else None,
            normalization_kinds=normalization_kinds,
            used_legacy_tool_format=used_legacy_tool_format,
        )

        # Record this turn's decision in the compact trajectory. The
        # resolved outcome (ok / failed + description) will be filled in
        # by ``notify_result`` once the adjudicator has run the action.
        self._trajectory.record_decision(
            verb=action.verb,
            candidate_ref=selected_candidate_ref,
            actor_id=self._agent_id,
            target_id=action.target_id,
        )

        # Track stats
        self._cumulative_usage = self._cumulative_usage + stats_to_usage(stats)
        self._step_stats.append(stats)
        self._audit_steps.append(step_audit)

        return action

    def notify_result(self, success: bool, description: str) -> None:
        """Notify the agent of the last action's result (called by external code)."""
        self._trajectory.record_last_result(success=success, description=description)
        if self._audit_steps:
            attach_action_result(
                self._audit_steps[-1],
                success=success,
                description=description,
            )

    def query(self, question: str) -> str:
        """Answer a free-text question using the LLM (no tools)."""
        query_msg = prepend_trajectory_block(
            {"role": "user", "content": question},
            self._trajectory,
        )
        response = self._provider.complete([query_msg], system=self._system_prompt)
        self._cumulative_usage = self._cumulative_usage + response.usage
        return response.text or ""

    def _call_with_retries(
        self,
        messages: list[dict[str, Any]],
        action_tool: dict[str, Any],
        action_spec: ActionSpec,
        obs_msg: dict[str, Any],
        stats: StepStats,
        step_audit: dict[str, Any],
    ) -> tuple[Action, str, str, list[str], bool]:
        """Call the LLM and retry on invalid responses."""
        current_messages = list(messages)

        for attempt in range(1 + MAX_RETRIES):
            try:
                response = self._provider.complete(
                    current_messages,
                    system=self._system_prompt,
                )
            except Exception as exc:
                error = f"Provider error: {exc}"
                logger.warning("Provider error (attempt %d): %s", attempt + 1, exc)
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=action_tool,
                    tool_choice=None,
                    response=None,
                    provider_error=str(exc),
                    provider_metadata=getattr(exc, "metadata", None),
                )
                if attempt < MAX_RETRIES:
                    move_guidance = format_move_action_guidance(action_spec.available_actions)
                    stats.retries += 1
                    current_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                f"<error>The model provider errored: {exc}. "
                                + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                                + "Use an exact candidate_ref from the current action list. "
                                "Add parameters only when the selected candidate exposes free "
                                "parameters. Do not repeat fixed fields. "
                                + (move_guidance + " " if move_guidance else "")
                                + "</error>"
                            ),
                        },
                    ]
                    continue
                raise RuntimeError(error) from exc

            stats.prompt_tokens += response.usage.prompt_tokens
            stats.completion_tokens += response.usage.completion_tokens
            provider_protocol_error = response_provider_protocol_error(response)
            if provider_protocol_error is not None:
                error = provider_protocol_error
                logger.warning("Provider protocol error (attempt %d): %s", attempt + 1, error)
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=action_tool,
                    tool_choice=None,
                    response=response,
                    provider_error=error,
                )
                if attempt < MAX_RETRIES:
                    move_guidance = format_move_action_guidance(action_spec.available_actions)
                    stats.retries += 1
                    current_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                f"<error>The model provider returned an invalid protocol "
                                f"response: {error}. "
                                + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                                + "Use an exact candidate_ref from the current action list. "
                                "Add parameters only when the selected candidate exposes free "
                                "parameters. Do not repeat fixed fields. "
                                + (move_guidance + " " if move_guidance else "")
                                + "</error>"
                            ),
                        },
                    ]
                    continue
                raise RuntimeError(f"Provider error: {error}")

            try:
                (
                    action,
                    selected_candidate_ref,
                    resolved_action_id,
                    normalization_kinds,
                    used_legacy_tool_format,
                ) = self._parse_action_output(response, action_spec)
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=action_tool,
                    tool_choice=None,
                    response=response,
                )
                return (
                    action,
                    selected_candidate_ref,
                    resolved_action_id,
                    normalization_kinds,
                    used_legacy_tool_format,
                )
            except ValueError as exc:
                error = str(exc)
                logger.warning("Invalid action output (attempt %d): %s", attempt + 1, error)
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=action_tool,
                    tool_choice=None,
                    response=response,
                    invalid_reason=error,
                )

            if attempt < MAX_RETRIES:
                move_guidance = format_move_action_guidance(action_spec.available_actions)
                stats.retries += 1
                current_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            f"<error>Your action was invalid: {error}. "
                            + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                            + "Use an exact candidate_ref from the current list. "
                            "Add parameters only when the selected candidate exposes free "
                            "parameters. Do not repeat fixed fields. "
                            + (move_guidance + " " if move_guidance else "")
                            + "</error>"
                        ),
                    },
                ]

        raise RuntimeError(
            f"LLM failed to produce a valid action after {1 + MAX_RETRIES} attempts. "
            f"Last error: {error}"
        )

    def _parse_action_output(
        self,
        response: LLMResponse,
        action_spec: ActionSpec,
    ) -> tuple[Action, str, str, list[str], bool]:
        """Parse one interactive model response into a valid Action."""

        parsed = parse_interactive_action_response(response, action_spec.available_actions)
        return (
            build_action_from_available(
                agent_id=self._agent_id,
                available_action=parsed.resolution.available_action,
                provided_parameters=parsed.parameter_extraction.parameters,
            ),
            parsed.resolution.available_action.candidate_ref,
            parsed.resolution.available_action.action_id,
            list(parsed.normalization_kinds),
            False,
        )


def stats_to_usage(stats: StepStats) -> TokenUsage:
    """Convert StepStats to TokenUsage for accumulation."""
    return TokenUsage(
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
    )
