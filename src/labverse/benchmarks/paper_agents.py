"""Paper-specific experiment agents and multi-agent controllers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from labverse.agent.action_binding import (
    build_action_from_available,
    build_action_input_schema,
)
from labverse.agent.action_history import ActionTrajectory, prepend_interactive_history_blocks
from labverse.agent.audit import (
    attach_action_result,
    fail_step_audit,
    finalize_step_audit,
    materialize_audit_steps,
    record_attempt,
    start_step_audit,
)
from labverse.agent.global_structured import format_world_state_json
from labverse.agent.interactive_output import parse_interactive_action_response
from labverse.agent.interactive_text import (
    format_available_actions_as_narrative,
    format_observation_as_narrative,
)
from labverse.agent.interface import Observation
from labverse.agent.json_io import coerce_json_object_field, load_json_text_with_repair
from labverse.agent.plan_grounding import (
    bind_planned_action,
    format_valid_id_inventory_json,
    lint_planned_sequence,
    validate_planned_step_grounding,
)
from labverse.agent.prompt import (
    INTERACTIVE_TOOL_RESPONSE_FORMAT,
    INTERACTIVE_TOOL_RETRY_INSTRUCTION,
    TurnStatus,
    format_observation_message,
    format_parameter_hints,
    format_turn_status,
    summarize_action_ids_for_history,
)
from labverse.agent.providers.base import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
    response_provider_protocol_error,
)
from labverse.models.action import VERB_CATEGORIES, Action, AvailableAction
from labverse.world.state import WorldState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

_MULTI_AGENT_PLANNER_VERB_ORDER = (
    "move_to",
    "open_door",
    "close_door",
    "unlock_door",
    "start_equipment",
    "stop_equipment",
    "pipette_transfer",
    "retrieve_sample",
    "store_sample",
    "dispose_material",
    "don_ppe",
    "doff_ppe",
    "wait",
    "log_observation",
)

_MULTI_AGENT_ACTION_VOCABULARY = {
    "move_to": "- move_to(target_id=room_id)",
    "open_door": "- open_door(target_id=door_id)",
    "close_door": "- close_door(target_id=door_id)",
    "unlock_door": "- unlock_door(target_id=door_id)",
    "start_equipment": "- start_equipment(target_id=equipment_id, parameters={command_name, ...})",
    "stop_equipment": "- stop_equipment(target_id=equipment_id)",
    "pipette_transfer": (
        "- pipette_transfer(target_id=pipette_id, "
        "parameters={source_id, destination_id, volume_ul})"
    ),
    "retrieve_sample": "- retrieve_sample(target_id=sample_id)",
    "store_sample": (
        "- store_sample(target_id=sample_id, "
        "parameters={destination_room_id OR destination_equipment_id})"
    ),
    "dispose_material": (
        "- dispose_material(target_id=material_id, parameters={waste_category, zone_id?})"
    ),
    "don_ppe": "- don_ppe(parameters={ppe_type})",
    "doff_ppe": "- doff_ppe(parameters={ppe_type})",
    "wait": "- wait()",
    "log_observation": "- log_observation(parameters={notes})",
}

_TEXT_ONLY_MULTI_AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are coordinating multiple laboratory technicians in a simulated lab.

You will be given:
1. The full lab state as structured JSON.
2. The task contract for the scenario.
3. The technician IDs you are allowed to control.
4. The lab action vocabulary.
5. A compact inventory of valid IDs and parameter names.

Output a COMPLETE global plan as a valid JSON array. Each step must be an object:
- "agent_id": the technician performing the action
- "verb": the action verb
- "target_id": target entity ID if needed
- "parameters": action-specific parameters if needed

Rules:
- Use only the listed technician IDs.
- Respect room movement, PPE, safety, and equipment contention rules.
- Each PPE item is one separate don_ppe step. Do not combine lab_coat,
  safety_glasses, and closed_toe_shoes into one action.
- Each technician must don lab_coat, safety_glasses, and closed_toe_shoes before
  moving into wet_lab.
- Each technician must move to wet_lab before starting centrifuge_1.
- Ensure each technician completes exactly one centrifuge run before the team stops.
- If a run duration is fixed in the available action, rely on that default.
- If parameter hints show exact values, copy them exactly.
- Coordinate access to shared equipment without conflicts.
- Use wait() when time must advance for running equipment.
- Use log_observation only when the goal explicitly requires documentation.
- If the task contract specifies an exact note, report, or finding format, copy that
  format exactly inside parameters.notes.
- Every agent_id, target_id, command_name, source_id, destination_id, and
  storage/disposal parameter must come from the provided valid-ID inventory.
  Do not invent IDs.
- Plans that are empty or contain only passive filler are invalid.
- Available canonical verbs:
{action_vocabulary}
- Output ONLY the valid JSON array. No explanation, no markdown, no code fences.
"""


def _multi_agent_action_vocabulary(allowed_verbs: set[str]) -> str:
    return "\n".join(
        _MULTI_AGENT_ACTION_VOCABULARY[verb]
        for verb in _MULTI_AGENT_PLANNER_VERB_ORDER
        if verb in allowed_verbs
    )


def _build_text_only_multi_agent_system_prompt(
    *,
    allowed_verbs: set[str],
) -> str:
    return _TEXT_ONLY_MULTI_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        action_vocabulary=_multi_agent_action_vocabulary(allowed_verbs),
    )


def _format_run_constraints(max_steps: int | None) -> str:
    if max_steps is None:
        return ""
    return "\n".join(
        [
            "<run_constraints>",
            f"Max steps: {max_steps}",
            "Each planned team action consumes one turn.",
            "</run_constraints>",
        ]
    )


_TEAM_INTERACTIVE_ENVIRONMENT_TEMPLATE = """\
<environment>
Each turn you receive:
1. The state representation for this coordinator condition.
2. The currently available actions for each technician.
3. Feedback from the previous team action.
4. Non-oracle turn status for the shared task.
</environment>"""

_TEAM_INTERACTIVE_ACTION_CONTRACT = """\
<action_contract>
- Choose exactly ONE currently available action by returning one valid JSON object.
- The available team action list is the source of truth for what can happen right now.
- Coordinate technicians to make progress safely and avoid equipment conflicts.
- Ensure each technician completes exactly one centrifuge run before repeating work.
- If parameter hints show exact values, copy them exactly.
- Put any parameter overrides inside the JSON parameters object.
- Fixed parameters are already selected for you. Do not repeat or change them.
- If no parameter overrides are needed, candidate_ref alone is enough.
- The candidate_ref must exactly match one of the listed team candidate_refs.
- If you include actor_id, verb, or target_id, they must exactly match the selected candidate.
- Never place candidate_ref, verb, actor_id, or target_id inside the parameters object.
- The parameters object is only for the free parameter fields shown by the selected candidate.
- Each action consumes one turn. Timed equipment keeps running across later turns
  until its completion event fires.
- Use non-interrupting actions by choosing an exact wait_* team
  candidate when time needs to pass while timed equipment completes. Do not stop equipment
  early unless the goal requires it.
</action_contract>"""

_TEAM_SYSTEM_TEMPLATE = """\
You are coordinating multiple laboratory technicians in a simulated lab.

{environment}

<state_representation>
{state_representation}
</state_representation>

{response_format}

{action_contract}
"""

_TEAM_NARRATIVE_STATE_REPRESENTATION = """\
- You receive a narrative summary of each technician's current local state.
- You do not receive a full lab-wide state dump; rely on local technician summaries plus the
  current team action list."""

_TEAM_STRUCTURED_STATE_REPRESENTATION = """\
- You receive structured local observation blocks for each technician.
- You do not receive a full lab-wide state dump; rely on local technician observations plus the
  current team action list."""

_TEAM_GLOBAL_STRUCTURED_STATE_REPRESENTATION = """\
- You receive the same structured local observation blocks plus a compact complete lab-state
  summary as structured JSON.
- Use the global lab-state summary for situational awareness, but only the listed team actions are
  executable on this turn."""


def _build_team_system_prompt(*, state_representation: str) -> str:
    return _TEAM_SYSTEM_TEMPLATE.format(
        environment=_TEAM_INTERACTIVE_ENVIRONMENT_TEMPLATE,
        state_representation=state_representation,
        response_format=INTERACTIVE_TOOL_RESPONSE_FORMAT,
        action_contract=_TEAM_INTERACTIVE_ACTION_CONTRACT,
    )


_INTERACTIVE_MULTI_AGENT_SYSTEM_PROMPT = _build_team_system_prompt(
    state_representation=_TEAM_NARRATIVE_STATE_REPRESENTATION,
)
_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT = _build_team_system_prompt(
    state_representation=_TEAM_STRUCTURED_STATE_REPRESENTATION,
)
_GLOBAL_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT = _build_team_system_prompt(
    state_representation=_TEAM_GLOBAL_STRUCTURED_STATE_REPRESENTATION,
)


@dataclass(slots=True)
class MultiAgentPlanStep:
    """One planned step in a multi-agent global schedule."""

    agent_id: str
    verb: str
    target_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TeamAvailableAction:
    """An available action candidate assigned to a specific agent."""

    candidate_ref: str
    agent_id: str
    action: AvailableAction


class MultiAgentTextPlanner:
    """Single-shot planner for paper Condition B on multi-agent scenarios."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        goal: str,
        world_state: WorldState,
        agent_ids: list[str],
        initial_observations: dict[str, str] | None = None,
        initial_action_text: str | None = None,
        name: str | None = None,
        allowed_verbs: set[str] | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._provider = provider
        self._goal = goal
        self._world_state = world_state.model_copy(deep=True)
        self._agent_ids = list(agent_ids)
        self._initial_observations = dict(initial_observations or {})
        self._initial_action_text = initial_action_text or ""
        self._name = name or "MultiAgentTextPlanner"
        self._cumulative_usage = TokenUsage()
        self._audit_steps: list[dict[str, Any]] = []
        self._allowed_verbs = set(allowed_verbs or VERB_CATEGORIES)
        self._max_steps = max_steps

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._cumulative_usage

    @property
    def name(self) -> str:
        return self._name

    @property
    def audit_steps(self) -> list[dict[str, Any]]:
        return materialize_audit_steps(
            self._audit_steps,
            actor_id="team_coordinator",
            actor_name=self._name,
        )

    def plan(self) -> list[MultiAgentPlanStep]:
        system_prompt = _build_text_only_multi_agent_system_prompt(
            allowed_verbs=self._allowed_verbs,
        )
        inventory_json = format_valid_id_inventory_json(
            self._world_state,
            agent_ids=self._agent_ids,
            allowed_verbs=sorted(self._allowed_verbs),
        )
        initial_team_state = _format_initial_team_state(
            self._initial_observations,
            self._initial_action_text,
        )
        run_constraints = _format_run_constraints(self._max_steps)
        context_sections = [
            f"<agent_ids>\n{json.dumps(self._agent_ids)}\n</agent_ids>",
            f"<task_contract>\n{self._goal}\n</task_contract>",
        ]
        if run_constraints:
            context_sections.append(run_constraints)
        context_sections.extend(
            [
                f"<world_state_json>\n{format_world_state_json(self._world_state)}\n"
                "</world_state_json>",
                f"<valid_id_inventory_json>\n{inventory_json}\n</valid_id_inventory_json>",
                initial_team_state,
                "Use only the canonical LabVerse action vocabulary listed in the system prompt.",
            ]
        )
        user_message = "\n\n".join(context_sections)
        messages = [{"role": "user", "content": user_message}]
        step_audit = start_step_audit(
            actor_id="team_coordinator",
            actor_name=self._name,
            actor_kind="planner",
            system_prompt=system_prompt,
            user_message=messages[0],
            tool_schema=None,
            prompt_surface_kind="global_one_shot_planner",
        )
        last_invalid_reason: str | None = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                response = self._provider.complete(
                    messages,
                    system=system_prompt,
                )
            except Exception as exc:
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=messages,
                    tool_schema=None,
                    tool_choice=None,
                    response=None,
                    provider_error=str(exc),
                    provider_metadata=getattr(exc, "metadata", None),
                )
                if attempt < MAX_RETRIES:
                    messages.extend(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Provider error: "
                                    f"{exc}. Retry and output only a valid JSON array."
                                ),
                            }
                        ]
                    )
                    continue
                fail_step_audit(step_audit, error=f"Provider error: {exc}")
                self._audit_steps.append(step_audit)
                raise RuntimeError(f"Multi-agent text planner provider failed: {exc}") from exc
            self._cumulative_usage = self._cumulative_usage + response.usage
            provider_protocol_error = response_provider_protocol_error(response)
            if provider_protocol_error is not None:
                error = provider_protocol_error
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=messages,
                    tool_schema=None,
                    tool_choice=None,
                    response=response,
                    provider_error=error,
                )
                if attempt < MAX_RETRIES:
                    messages.extend(
                        [
                            {
                                "role": "user",
                                "content": (
                                    f"Provider protocol error: {error}. Retry and output only a "
                                    "valid JSON array."
                                ),
                            }
                        ]
                    )
                    continue
                fail_step_audit(step_audit, error=error)
                self._audit_steps.append(step_audit)
                raise RuntimeError(error)

            if response.text is None:
                error = "Planner did not return text output."
                last_invalid_reason = error
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=messages,
                    tool_schema=None,
                    tool_choice=None,
                    response=response,
                    invalid_reason=error,
                )
            else:
                try:
                    plan, normalization_kinds = self._parse_plan(response.text)
                    record_attempt(
                        step_audit,
                        attempt_index=attempt + 1,
                        messages=messages,
                        tool_schema=None,
                        tool_choice=None,
                        response=response,
                    )
                    finalize_step_audit(
                        step_audit,
                        resolved_action_id="global_one_shot_team_plan",
                        resolved_parameters={"steps": len(plan)},
                        fallback_kind=normalization_kinds[0] if normalization_kinds else None,
                        normalization_kinds=list(normalization_kinds),
                    )
                    step_audit["planned_steps"] = [
                        {
                            "agent_id": step.agent_id,
                            "verb": step.verb,
                            "target_id": step.target_id,
                            "parameters": dict(step.parameters),
                        }
                        for step in plan
                    ]
                    self._audit_steps.append(step_audit)
                    return plan
                except ValueError as exc:
                    error = str(exc)
                    last_invalid_reason = error
                    logger.warning(
                        "Multi-agent text planner parse failed (attempt %d): %s",
                        attempt + 1,
                        error,
                    )
                    record_attempt(
                        step_audit,
                        attempt_index=attempt + 1,
                        messages=messages,
                        tool_schema=None,
                        tool_choice=None,
                        response=response,
                        invalid_reason=error,
                    )

            if attempt < MAX_RETRIES:
                messages.extend(
                    [
                        {"role": "assistant", "content": response.text or ""},
                        {
                            "role": "user",
                            "content": (
                                f"Your response was invalid: {error}. "
                                "Output ONLY a valid JSON array."
                            ),
                        },
                    ]
                )

        final_error = (
            last_invalid_reason or "Multi-agent text planner failed to produce a valid plan"
        )
        fail_step_audit(step_audit, error=final_error)
        self._audit_steps.append(step_audit)
        raise RuntimeError(final_error)

    def _parse_plan(self, text: str) -> tuple[list[MultiAgentPlanStep], tuple[str, ...]]:
        try:
            parsed, normalization_kinds = load_json_text_with_repair(text)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("Plan must be a JSON array")
        if not parsed:
            raise ValueError("Plan must contain at least one step")

        plan: list[MultiAgentPlanStep] = []
        for index, item in enumerate(parsed, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Plan step {index} is not an object")
            agent_id = item.get("agent_id")
            verb = item.get("verb")
            target_id = item.get("target_id")
            parameters = item.get("parameters") or {}
            if not isinstance(agent_id, str) or agent_id not in self._agent_ids:
                raise ValueError(f"Plan step {index} has invalid agent_id: {agent_id!r}")
            if not isinstance(verb, str) or verb not in VERB_CATEGORIES:
                raise ValueError(f"Plan step {index} has invalid verb: {verb!r}")
            if verb not in self._allowed_verbs:
                raise ValueError(f"Plan step {index} uses disallowed verb: {verb!r}")
            if target_id is not None and not isinstance(target_id, str):
                target_id = str(target_id)
            try:
                parameters, parameter_normalization_kinds = coerce_json_object_field(
                    parameters, field_name="parameters"
                )
            except ValueError as exc:
                raise ValueError(f"Plan step {index} parameters must be an object") from exc
            if parameter_normalization_kinds:
                item["parameters"] = parameters
                normalization_kinds = (*normalization_kinds, *parameter_normalization_kinds)
            validate_planned_step_grounding(
                verb=verb,
                target_id=target_id,
                parameters=parameters,
                world_state=self._world_state,
                agent_id=agent_id,
                valid_agent_ids=self._agent_ids,
            )
            plan.append(
                MultiAgentPlanStep(
                    agent_id=agent_id,
                    verb=verb,
                    target_id=target_id,
                    parameters=parameters,
                )
            )
        lint_planned_sequence(
            [
                {
                    "agent_id": step.agent_id,
                    "verb": step.verb,
                    "target_id": step.target_id,
                    "parameters": dict(step.parameters),
                }
                for step in plan
            ],
            world_state=self._world_state,
            valid_agent_ids=self._agent_ids,
            max_steps=self._max_steps,
        )
        _validate_shared_equipment_schedule_shape(plan, agent_ids=self._agent_ids)
        return plan, normalization_kinds


def _validate_shared_equipment_schedule_shape(
    plan: list[MultiAgentPlanStep],
    *,
    agent_ids: list[str],
) -> None:
    """Validate the paper shared-equipment one-shot plan shape.

    The multi-agent one-shot path is only used for shared_equipment_scheduling.
    This validator rejects invalid plans before replay so they are counted as
    invalid model outputs instead of becoming opaque downstream grounding errors.
    """

    expected_speeds = {"tech_1": 2000, "tech_2": 2500, "tech_3": 3000}
    starts_by_agent: dict[str, list[MultiAgentPlanStep]] = {agent_id: [] for agent_id in agent_ids}
    for step in plan:
        if step.verb == "start_equipment" and step.target_id == "centrifuge_1":
            starts_by_agent.setdefault(step.agent_id, []).append(step)

    for agent_id in agent_ids:
        starts = starts_by_agent.get(agent_id, [])
        if len(starts) != 1:
            raise ValueError(
                f"Shared-equipment plan must start centrifuge_1 exactly once for "
                f"{agent_id}; found {len(starts)} starts"
            )
        start = starts[0]
        command_name = start.parameters.get("command_name")
        if command_name != "spin":
            raise ValueError(
                f"Shared-equipment plan start for {agent_id} must use command_name='spin'"
            )
        expected_speed = expected_speeds.get(agent_id)
        if expected_speed is not None and start.parameters.get("speed_rpm") != expected_speed:
            raise ValueError(
                f"Shared-equipment plan start for {agent_id} must use speed_rpm={expected_speed}"
            )


class MultiAgentInteractiveCoordinator:
    """Single-LLM interactive coordinator for paper multi-agent runs."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        goal: str,
        agent_ids: list[str],
        name: str | None = None,
        max_history_turns: int = 10,
        system_prompt: str = _INTERACTIVE_MULTI_AGENT_SYSTEM_PROMPT,
    ) -> None:
        self._provider = provider
        self._goal = goal
        self._agent_ids = list(agent_ids)
        self._name = name or "MultiAgentInteractiveCoordinator"
        self._max_history_turns = max_history_turns
        self._system_prompt = system_prompt
        self._trajectory = ActionTrajectory(max_entries=max_history_turns)
        self._cumulative_usage = TokenUsage()
        self._audit_steps: list[dict[str, Any]] = []
        self._last_result_text: str | None = None
        self._last_team_candidate_ref: str | None = None

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._cumulative_usage

    @property
    def name(self) -> str:
        return self._name

    @property
    def audit_steps(self) -> list[dict[str, Any]]:
        return materialize_audit_steps(
            self._audit_steps,
            actor_id="team_coordinator",
            actor_name=self._name,
        )

    def next_action(
        self,
        *,
        observations: dict[str, str],
        team_actions: list[TeamAvailableAction],
        turn_status: TurnStatus | None = None,
        world_state_json: str | None = None,
    ) -> Action:
        if not team_actions:
            raise StopIteration("No team actions are available")

        tool = _build_joint_action_tool(team_actions)
        # Historical outcomes live in the compact trajectory block rather
        # than inline ``<team_action_result>`` text. The current user
        # message is built with ``last_result_text=None`` to avoid
        # duplicating history already expressed in the trajectory.
        base_user_msg = {
            "role": "user",
            "content": _format_team_state_message(
                goal=self._goal,
                observations=observations,
                team_actions=team_actions,
                last_result_text=None,
                turn_status=turn_status,
                world_state_json=world_state_json,
            ),
        }
        user_msg = prepend_interactive_history_blocks(base_user_msg, self._trajectory)
        current_messages = [user_msg]
        step_audit = start_step_audit(
            actor_id="team_coordinator",
            actor_name=self._name,
            actor_kind="coordinator",
            system_prompt=self._system_prompt,
            user_message=user_msg,
            tool_schema=tool,
            prompt_surface_kind="team_interactive_grounded",
        )
        audit_committed = False

        try:
            for attempt in range(1 + MAX_RETRIES):
                try:
                    response = self._provider.complete(
                        current_messages,
                        system=self._system_prompt,
                    )
                except Exception as exc:
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
                        move_guidance = _format_team_move_guidance(team_actions)
                        current_messages = [
                            user_msg,
                            {
                                "role": "user",
                                "content": (
                                    f"<error>The model provider errored: {exc}. "
                                    + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                                    + "Use an exact team candidate_ref from the current action "
                                    "list. Add parameters only when the selected candidate "
                                    "exposes free parameters. Do not repeat fixed fields. "
                                    + (move_guidance + " " if move_guidance else "")
                                    + "</error>"
                                ),
                            },
                        ]
                        continue
                    fail_step_audit(step_audit, error=f"Provider error: {exc}")
                    self._audit_steps.append(step_audit)
                    audit_committed = True
                    raise RuntimeError(
                        f"Multi-agent interactive coordinator failed after provider error: {exc}"
                    ) from exc
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
                        move_guidance = _format_team_move_guidance(team_actions)
                        current_messages = [
                            user_msg,
                            {
                                "role": "user",
                                "content": (
                                    f"<error>The model provider returned an invalid protocol "
                                    f"response: {error}. "
                                    + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                                    + "Use an exact team candidate_ref from the current action "
                                    "list. Add parameters only when the selected candidate "
                                    "exposes free parameters. Do not repeat fixed fields. "
                                    + (move_guidance + " " if move_guidance else "")
                                    + "</error>"
                                ),
                            },
                        ]
                        continue
                    fail_step_audit(step_audit, error=f"Provider error: {error}")
                    self._audit_steps.append(step_audit)
                    audit_committed = True
                    raise RuntimeError(
                        "Multi-agent interactive coordinator failed after provider protocol error: "
                        f"{error}"
                    )

                try:
                    (
                        action,
                        selected_candidate_ref,
                        resolved_action_id,
                        normalization_kinds,
                    ) = _parse_joint_action_output(
                        response,
                        team_actions,
                    )
                    record_attempt(
                        step_audit,
                        attempt_index=attempt + 1,
                        messages=current_messages,
                        tool_schema=tool,
                        tool_choice=None,
                        response=response,
                    )
                    self._trajectory.record_decision(
                        verb=action.verb,
                        candidate_ref=selected_candidate_ref,
                        actor_id=action.agent_id,
                        target_id=action.target_id,
                    )
                    self._last_result_text = None
                    self._last_team_candidate_ref = selected_candidate_ref
                    finalize_step_audit(
                        step_audit,
                        resolved_candidate_ref=selected_candidate_ref,
                        resolved_action_id=resolved_action_id,
                        resolved_verb=action.verb,
                        resolved_target_id=action.target_id,
                        resolved_parameters=action.parameters,
                        fallback_kind=normalization_kinds[0] if normalization_kinds else None,
                        normalization_kinds=normalization_kinds,
                    )
                    self._audit_steps.append(step_audit)
                    audit_committed = True
                    return action
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
                    move_guidance = _format_team_move_guidance(team_actions)
                    current_messages = [
                        user_msg,
                        {
                            "role": "user",
                            "content": (
                                f"<error>Your team action was invalid: {error}. "
                                + INTERACTIVE_TOOL_RETRY_INSTRUCTION
                                + "Use an exact team candidate_ref from the current list. "
                                "Add parameters only when the selected candidate exposes free "
                                "parameters. Do not repeat fixed fields. "
                                + (move_guidance + " " if move_guidance else "")
                                + "</error>"
                            ),
                        },
                    ]

            fail_step_audit(
                step_audit,
                error="Multi-agent interactive coordinator failed to produce a valid action",
            )
            self._audit_steps.append(step_audit)
            audit_committed = True
            raise RuntimeError(
                "Multi-agent interactive coordinator failed to produce a valid action"
            )
        except Exception as exc:
            if not audit_committed:
                fail_step_audit(step_audit, error=str(exc))
                self._audit_steps.append(step_audit)
            raise

    def notify_result(self, action: Action, success: bool, description: str) -> None:
        self._trajectory.record_last_result(success=success, description=description)
        if self._audit_steps:
            attach_action_result(
                self._audit_steps[-1],
                success=success,
                description=description,
            )


def _team_candidate_ref_for(agent_id: str, candidate_ref: str) -> str:
    normalized_agent = agent_id
    if agent_id.startswith("tech_"):
        suffix = agent_id.split("_", 1)[1]
        if suffix.isdigit():
            normalized_agent = f"t{suffix}"
    normalized_agent = normalized_agent.replace("-", "_")
    return f"{normalized_agent}_{candidate_ref}"


def collect_team_actions(
    action_specs: dict[str, list[AvailableAction]],
) -> list[TeamAvailableAction]:
    """Flatten per-agent action specs into a unique team candidate surface."""
    result: list[TeamAvailableAction] = []
    for agent_id in sorted(action_specs):
        for action in action_specs[agent_id]:
            candidate_ref = _team_candidate_ref_for(agent_id, action.candidate_ref)
            team_action = action.model_copy(
                update={
                    "candidate_ref": candidate_ref,
                    "actor_id": agent_id,
                },
                deep=True,
            )
            result.append(
                TeamAvailableAction(
                    candidate_ref=candidate_ref,
                    agent_id=agent_id,
                    action=team_action,
                )
            )
    return result


def bind_planned_step(
    *,
    step: MultiAgentPlanStep,
    available_actions: list[AvailableAction],
) -> Action:
    """Bind a planned step to the current action space when possible."""
    return bind_planned_action(
        agent_id=step.agent_id,
        verb=step.verb,
        target_id=step.target_id,
        parameters=step.parameters,
        available_actions=available_actions,
    )


def _format_team_state_message(
    *,
    goal: str,
    observations: dict[str, str],
    team_actions: list[TeamAvailableAction],
    last_result_text: str | None,
    turn_status: TurnStatus | None,
    world_state_json: str | None = None,
) -> str:
    parts = [f"<task_contract>\n{goal}\n</task_contract>"]
    if last_result_text:
        parts.append(f"<team_action_result>\n{last_result_text}\n</team_action_result>")
    turn_status_text = format_turn_status(turn_status)
    if turn_status_text:
        parts.append(turn_status_text)
    if world_state_json:
        parts.append(f"<world_state_json>\n{world_state_json}\n</world_state_json>")

    parts.append("<team_state>")
    for agent_id in sorted(observations):
        parts.append(f"- {agent_id}: {observations[agent_id]}")
    parts.append("</team_state>")

    grouped: dict[str, dict[str, list[TeamAvailableAction]]] = {}
    for team_action in team_actions:
        grouped.setdefault(team_action.agent_id, {}).setdefault(
            team_action.action.group_label,
            [],
        ).append(team_action)

    parts.append("<available_team_actions>")
    parts.append("Choose exactly one candidate_ref from this grouped team action surface.")
    parts.append(
        "Return exactly one valid JSON object whose candidate_ref matches the selected "
        "candidate."
    )
    parts.append(
        "Output only the JSON object; do not use XML, markdown, code fences, tags, or prose."
    )
    move_guidance = _format_team_move_guidance(team_actions)
    if move_guidance:
        parts.append(move_guidance)
    parts.append("Fixed parameters are already set and must not be changed.")
    parts.append("For fixed-parameter actions, candidate_ref alone is enough.")
    parts.append(
        "Only include a parameters object when the selected candidate lists free "
        "required or optional parameters."
    )
    for agent_id in sorted(grouped):
        parts.append(f"{agent_id}:")
        for group_key in sorted(
            grouped[agent_id],
            key=lambda key: next(
                (candidate.action.group_key for candidate in grouped[agent_id][key]),
                key,
            ),
        ):
            actions = grouped[agent_id][group_key]
            parts.append(f"  {group_key}:")
            for index, team_action in enumerate(actions, start=1):
                action = team_action.action
                details = [
                    f"{index}. [{team_action.candidate_ref}] actor={team_action.agent_id}",
                    f"verb={action.verb}",
                ]
                if action.target_id:
                    details.append(f"target={action.target_label or action.target_id}")
                    details.append(f"target_id={action.target_id}")
                if action.fixed_parameters:
                    fixed = ", ".join(
                        f"{name}={value}" for name, value in sorted(action.fixed_parameters.items())
                    )
                    details.append(f"(fixed: {fixed})")
                if action.required_parameters:
                    details.append(f"(requires: {', '.join(action.required_parameters)})")
                if action.optional_parameters:
                    details.append(f"(optional: {', '.join(action.optional_parameters)})")
                if action.parameter_details:
                    details.append(f"(parameter_hints: {format_parameter_hints(action)})")
                if action.estimated_duration_seconds is not None:
                    details.append(f"(takes≈{action.estimated_duration_seconds:.0f}s)")
                details.append(f"— {action.description}")
                parts.append(" ".join(details))
    parts.append("</available_team_actions>")
    return "\n".join(parts)


def _format_team_move_guidance(team_actions: list[TeamAvailableAction]) -> str:
    move_actions = [
        team_action for team_action in team_actions if team_action.action.verb == "move_to"
    ]
    if not move_actions:
        return ""
    if len(move_actions) == 1:
        action = move_actions[0].action
        destination = action.target_label or action.target_id or "the destination"
        return (
            "Movement is destination-specific. The only valid team move candidate on this turn is "
            f"{move_actions[0].candidate_ref} for {destination}; use that exact candidate_ref "
            "if the team needs to move. Previous move candidates may no longer be valid on this "
            "turn."
        )
    return (
        "Movement is destination-specific. Current valid team move candidates: "
        + ", ".join(
            f"{team_action.candidate_ref} "
            f"({team_action.action.target_label or team_action.action.target_id})"
            for team_action in move_actions
        )
        + ". Use one exact candidate_ref from this list if the team needs to move. Previous "
        + "move candidates may no longer be valid on this turn."
    )


def build_team_observations_text(observations: dict[str, Observation]) -> dict[str, str]:
    """Format per-agent observations using the same narrative surface as Condition C."""
    return {
        agent_id: format_observation_as_narrative(observation)
        for agent_id, observation in observations.items()
    }


def build_team_structured_observations_text(
    observations: dict[str, Observation],
) -> dict[str, str]:
    """Format per-agent observations using the structured surface from the embodied path."""
    return {
        agent_id: format_observation_message(observation)["content"]
        for agent_id, observation in observations.items()
    }


def build_team_actions_text(
    action_specs: dict[str, list[AvailableAction]],
    *,
    allowed_verbs: set[str] | None = None,
) -> str:
    """Render per-agent available actions for multi-agent text planning."""
    parts = ["<initial_available_actions>"]
    for agent_id in sorted(action_specs):
        parts.append(f"{agent_id}:")
        actions = action_specs[agent_id]
        if allowed_verbs is not None:
            actions = [action for action in actions if action.verb in allowed_verbs]
        parts.append(format_available_actions_as_narrative(actions))
    parts.append("</initial_available_actions>")
    return "\n".join(parts)


def _format_initial_team_state(
    observations: dict[str, str],
    action_text: str,
) -> str:
    parts = ["<initial_team_state>"]
    if observations:
        parts.append("Initial team observations:")
        for agent_id in sorted(observations):
            parts.append(f"- {agent_id}: {observations[agent_id]}")
    if action_text:
        parts.append(action_text)
    parts.append("</initial_team_state>")
    return "\n".join(parts)


def _summarize_team_history(history: list[dict[str, Any]]) -> str:
    actions: list[str] = []
    for message in history:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if "<team_action_result>" not in content:
            continue
        prefix = "Previous team action "
        if prefix not in content:
            continue
        try:
            remainder = content.split(prefix, 1)[1]
            action_ref = remainder.split(": ", 1)[1].split(" ", 1)[0]
        except IndexError:
            continue
        actions.append(action_ref)
    summary = summarize_action_ids_for_history(actions)
    if not summary:
        return ""
    return "Previous team actions: " + summary


def _build_joint_action_tool(team_actions: list[TeamAvailableAction]) -> dict[str, Any]:
    return {
        "name": "take_action",
        "description": (
            "Perform one currently available action for one technician. The canonical "
            "interactive response is one plain JSON object matching this schema. "
            "The candidate_ref must exactly match the selected candidate. Include parameters "
            "only when the selected candidate exposes free parameters. Compatibility echo "
            "fields such as actor_id, verb, and target_id are allowed but not required; if "
            "present they must exactly match the selected candidate. Do not emit prose, "
            "markdown, XML, tags, or nested wrapper objects."
        ),
        "input_schema": build_action_input_schema(
            [team_action.action for team_action in team_actions]
        ),
    }


def _parse_joint_action_output(
    response: LLMResponse,
    team_actions: list[TeamAvailableAction],
) -> tuple[Action, str, str, list[str]]:
    parsed = parse_interactive_action_response(
        response,
        [team_action.action for team_action in team_actions],
    )

    chosen = next(
        team_action
        for team_action in team_actions
        if team_action.candidate_ref == parsed.resolution.available_action.candidate_ref
    )
    return (
        build_action_from_available(
            agent_id=chosen.agent_id,
            available_action=chosen.action,
            provided_parameters=parsed.parameter_extraction.parameters,
        ),
        chosen.candidate_ref,
        chosen.action.action_id,
        list(parsed.normalization_kinds),
    )


def _parse_joint_tool_call(
    tool_call: Any,
    team_actions: list[TeamAvailableAction],
    *,
    response: LLMResponse | None = None,
) -> tuple[Action, str, str, list[str]]:
    """Compatibility wrapper for older tests and helper code paths."""

    effective_response = response or LLMResponse(tool_call=tool_call)
    if effective_response.tool_call is None:
        effective_response = LLMResponse(
            tool_call=tool_call,
            text=effective_response.text,
            usage=effective_response.usage,
            metadata=effective_response.metadata,
        )
    (
        action,
        selected_action_id,
        resolved_action_id,
        normalization_kinds,
    ) = _parse_joint_action_output(effective_response, team_actions)
    compatibility_normalization_kinds = [
        kind for kind in normalization_kinds if kind != "provider_tool_call_fallback"
    ]
    return (
        action,
        selected_action_id,
        resolved_action_id,
        compatibility_normalization_kinds,
    )


__all__ = [
    "MAX_RETRIES",
    "MultiAgentInteractiveCoordinator",
    "MultiAgentPlanStep",
    "MultiAgentTextPlanner",
    "_GLOBAL_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT",
    "_STRUCTURED_MULTI_AGENT_SYSTEM_PROMPT",
    "TeamAvailableAction",
    "build_team_actions_text",
    "build_team_observations_text",
    "build_team_structured_observations_text",
    "bind_planned_step",
    "collect_team_actions",
    "format_world_state_json",
]
