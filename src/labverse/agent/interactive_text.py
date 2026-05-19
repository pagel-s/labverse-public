"""Condition C agent: interactive text planner with narrative feedback.

Like LLMAgent but receives narrative (natural language) observations
instead of structured data. This isolates the effect of interactivity
vs. structured spatial representation.
"""

from __future__ import annotations

import logging
from typing import Any

from labverse.agent.action_binding import (
    build_action_from_available,
)
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
    build_narrative_system_prompt,
    format_available_actions,
    format_move_action_guidance,
    format_turn_status,
)
from labverse.agent.providers.base import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
    response_provider_protocol_error,
)
from labverse.models.action import Action, ActionSpec, AvailableAction

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
MAX_RETRIES = 2


def format_observation_as_narrative(observation: Observation) -> str:
    """Format an observation as natural language (no structured IDs or enums).

    This is deliberately less precise than the structured observation
    format used by Condition A (LLMAgent). The agent receives prose
    descriptions that require interpretation.
    """
    room = observation.current_room
    parts: list[str] = []

    parts.append(
        f"You are in {room.name}, a {room.room_type.value.replace('_', ' ')} "
        f"with biosafety level {room.bsl_level.value}."
    )

    if room.zones:
        zone_names = ", ".join(z.name for z in room.zones)
        parts.append(f"The room has these areas: {zone_names}.")

    if observation.visible_persons:
        person_descs = []
        for p in observation.visible_persons:
            activity = f" ({p.current_activity})" if p.current_activity else ""
            person_descs.append(f"{p.name}, a {p.role.value.replace('_', ' ')}{activity}")
        parts.append(f"You can see: {'; '.join(person_descs)}.")

    if observation.visible_equipment:
        eq_descs = []
        for eq in observation.visible_equipment:
            status = eq.status.value.lower().replace("_", " ")
            cmd = f" running '{eq.running_command}'" if eq.running_command else ""
            eq_descs.append(f"{eq.name} ({status}{cmd})")
        parts.append(f"Equipment here: {', '.join(eq_descs)}.")

    if observation.visible_materials:
        mat_descs = []
        for mat in observation.visible_materials:
            hazard = " [hazardous]" if mat.is_hazardous else ""
            mat_descs.append(f"{mat.name} ({mat.quantity:.0f} {mat.quantity_unit}){hazard}")
        parts.append(f"Materials here: {', '.join(mat_descs)}.")

    active_equipment = [
        eq
        for eq in observation.visible_equipment
        if eq.running_command is not None or eq.status.value != "idle"
    ]
    if active_equipment:
        active_descs = []
        for eq in active_equipment:
            desc = f"{eq.name} is {eq.status.value.lower().replace('_', ' ')}"
            if eq.running_command:
                desc += f" running '{eq.running_command}'"
            active_descs.append(desc)
        parts.append("Visible timed or active equipment state: " + "; ".join(active_descs) + ".")

    stored_samples = [
        sample for sample in observation.visible_samples if sample.equipment_id is not None
    ]
    if stored_samples:
        sample_descs = [
            f"{sample.name} is currently stored in {sample.equipment_id}"
            for sample in stored_samples
        ]
        parts.append("Samples currently stored in equipment: " + "; ".join(sample_descs) + ".")
        parts.append(
            "If the goal depends on a timed hold or incubation, avoid stopping the equipment "
            "early and spend turns on safe waiting actions until completion is reported."
        )

    if observation.current_ppe:
        ppe_list = ", ".join(sorted(p.value.replace("_", " ") for p in observation.current_ppe))
        parts.append(f"You are wearing: {ppe_list}.")
    else:
        parts.append("You are not wearing any PPE.")

    if observation.carrying:
        parts.append(f"You are carrying: {', '.join(observation.carrying)}.")

    if observation.active_alerts:
        parts.append(f"Active alerts: {'; '.join(observation.active_alerts)}.")

    if observation.recent_events:
        parts.append(f"Recent events: {'; '.join(observation.recent_events)}.")

    return " ".join(parts)


def format_available_actions_as_narrative(
    available_actions: list[AvailableAction],
) -> str:
    """Return the canonical single-agent action list for narrative mode."""

    return format_available_actions(available_actions)


def build_narrative_user_message(
    observation: Observation,
    available_actions: list[AvailableAction],
    *,
    last_result_text: str | None = None,
    turn_status: TurnStatus | None = None,
) -> dict[str, Any]:
    """Build the canonical local narrative interactive user message."""

    content = (
        "<observation>\n"
        + format_observation_as_narrative(observation)
        + "\n</observation>\n\n"
        + format_available_actions_as_narrative(available_actions)
    )
    turn_status_text = format_turn_status(turn_status)
    if turn_status_text:
        content = turn_status_text + "\n\n" + content
    if last_result_text:
        content = f"<action_result>\n{last_result_text}\n</action_result>\n\n" + content
    return {"role": "user", "content": content}


class InteractiveTextAgent(AgentInterface):
    """Condition C: step-by-step LLM agent with narrative (non-structured) observations."""

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
        self._name = name or f"InteractiveText-{agent_id}"
        self._system_prompt = build_narrative_system_prompt(goal=goal)
        # ``max_history_turns`` is retained for API compatibility and controls
        # the compact decision-trace buffer rather than a raw list of prior
        # user messages. See ``labverse.agent.action_history`` for the
        # scientific rationale.
        self._max_history_turns = max_history_turns
        self._trajectory = ActionTrajectory(max_entries=max_history_turns)

        self._pending_observation: Observation | None = None
        self._pending_turn_status: TurnStatus | None = None
        self._cumulative_usage = TokenUsage()
        self._audit_steps: list[dict[str, Any]] = []

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
    def audit_steps(self) -> list[dict[str, Any]]:
        return materialize_audit_steps(
            self._audit_steps,
            actor_id=self._agent_id,
            actor_name=self._name,
        )

    def observe(self, observation: Observation) -> None:
        self._pending_observation = observation

    def set_turn_status(self, turn_status: TurnStatus) -> None:
        self._pending_turn_status = turn_status

    def act(self, action_spec: ActionSpec) -> Action:
        if self._pending_observation is None:
            raise RuntimeError("act() called before observe()")

        observation = self._pending_observation
        self._pending_observation = None

        # Historical outcomes live in the compact trajectory block rather
        # than an inline ``<action_result>`` tail, so the current user
        # message is built with ``last_result_text=None``. See the
        # rationale in ``labverse.agent.action_history``.
        base_user_msg = build_narrative_user_message(
            observation,
            action_spec.available_actions,
            last_result_text=None,
            turn_status=self._pending_turn_status,
        )
        self._pending_turn_status = None
        user_msg = prepend_interactive_history_blocks(base_user_msg, self._trajectory)
        tool = build_action_tool(action_spec.available_actions)
        step_audit = start_step_audit(
            actor_id=self._agent_id,
            actor_name=self._name,
            actor_kind="agent",
            system_prompt=self._system_prompt,
            user_message=user_msg,
            tool_schema=tool,
            prompt_surface_kind="local_narrative_interactive_grounded",
        )

        # The trajectory block lives inside ``user_msg``; the legacy
        # ``_history`` list of prior full-state dumps has been removed.
        messages = [user_msg]

        try:
            (
                action,
                selected_candidate_ref,
                resolved_action_id,
                normalization_kinds,
                used_legacy_tool_format,
            ) = self._call_with_retries(messages, tool, action_spec, user_msg, step_audit)
        except Exception as exc:
            fail_step_audit(step_audit, error=str(exc))
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
        # resolved outcome is attached in ``notify_result`` once the
        # adjudicator has run the action. The assistant-side "Action:
        # ..." text is intentionally never added to the conversation —
        # weaker models will imitate plain-text responses from such
        # history instead of emitting valid JSON action objects on
        # subsequent turns.
        self._trajectory.record_decision(
            verb=action.verb,
            candidate_ref=selected_candidate_ref,
            actor_id=self._agent_id,
            target_id=action.target_id,
        )
        self._audit_steps.append(step_audit)

        return action

    def notify_result(self, success: bool, description: str) -> None:
        """Attach the resolved outcome of the most recent decision."""
        self._trajectory.record_last_result(success=success, description=description)
        if self._audit_steps:
            attach_action_result(
                self._audit_steps[-1],
                success=success,
                description=description,
            )

    def query(self, question: str) -> str:
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
        tool: dict[str, Any],
        action_spec: ActionSpec,
        user_msg: dict[str, Any],
        step_audit: dict[str, Any],
    ) -> tuple[Action, str, str, list[str], bool]:
        current_messages = list(messages)

        for attempt in range(1 + MAX_RETRIES):
            try:
                response = self._provider.complete(
                    current_messages,
                    system=self._system_prompt,
                )
            except Exception as exc:
                error = f"Provider error: {exc}"
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=tool,
                    tool_choice=None,
                    response=None,
                    provider_error=str(exc),
                    provider_metadata=getattr(exc, "metadata", None),
                )
                if attempt < MAX_RETRIES:
                    move_guidance = format_move_action_guidance(action_spec.available_actions)
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

            self._cumulative_usage = self._cumulative_usage + response.usage
            provider_protocol_error = response_provider_protocol_error(response)
            if provider_protocol_error is not None:
                error = provider_protocol_error
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=tool,
                    tool_choice=None,
                    response=response,
                    provider_error=error,
                )
                if attempt < MAX_RETRIES:
                    move_guidance = format_move_action_guidance(action_spec.available_actions)
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
                    tool_schema=tool,
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
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=current_messages,
                    tool_schema=tool,
                    tool_choice=None,
                    response=response,
                    invalid_reason=error,
                )

            if attempt < MAX_RETRIES:
                move_guidance = format_move_action_guidance(action_spec.available_actions)
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

        raise RuntimeError(f"InteractiveTextAgent failed after {1 + MAX_RETRIES} attempts")

    def _parse_action_output(
        self,
        response: LLMResponse,
        action_spec: ActionSpec,
    ) -> tuple[Action, str, str, list[str], bool]:
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


__all__ = [
    "InteractiveTextAgent",
    "format_available_actions_as_narrative",
    "format_observation_as_narrative",
]
