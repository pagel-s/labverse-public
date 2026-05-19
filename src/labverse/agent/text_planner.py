"""Condition B agent: single-shot text planner (no simulation feedback).

Receives the full lab configuration and goal in one prompt, returns
a complete action plan as a JSON array. The plan is then replayed
through the simulation for scoring.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from labverse.agent.audit import (
    fail_step_audit,
    finalize_step_audit,
    materialize_audit_steps,
    record_attempt,
    start_step_audit,
)
from labverse.agent.interface import AgentInterface, Observation
from labverse.agent.json_io import (
    coerce_json_object_field,
    load_json_text_with_repair,
)
from labverse.agent.json_io import (
    normalize_json_text_response as _normalize_json_text_response,
)
from labverse.agent.plan_grounding import (
    bind_planned_action,
    format_valid_id_inventory_json,
    lint_planned_sequence,
    validate_planned_step_grounding,
)
from labverse.agent.prompt import format_parameter_hints
from labverse.agent.providers.base import (
    LLMProvider,
    TokenUsage,
    response_provider_protocol_error,
)
from labverse.models.action import VERB_CATEGORIES, Action, ActionSpec
from labverse.world.state import WorldState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

_PLANNER_VERB_ORDER = (
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

_PLANNER_ACTION_VOCABULARY = {
    "move_to": "- move_to(target_id=room_id) — Move to an adjacent room.",
    "open_door": "- open_door(target_id=door_id) — Open a nearby closed door.",
    "close_door": "- close_door(target_id=door_id) — Close a nearby open door.",
    "unlock_door": (
        "- unlock_door(target_id=door_id) — Unlock a nearby door when access permits it."
    ),
    "start_equipment": (
        "- start_equipment(target_id=equipment_id, parameters={command_name, ...}) — "
        "Start equipment."
    ),
    "stop_equipment": "- stop_equipment(target_id=equipment_id) — Stop running equipment.",
    "pipette_transfer": (
        "- pipette_transfer(\n"
        "    target_id=pipette_id,\n"
        "    parameters={source_id, destination_id, volume_ul}\n"
        "  ) — Transfer liquid."
    ),
    "retrieve_sample": "- retrieve_sample(target_id=sample_id) — Pick up a sample.",
    "store_sample": (
        "- store_sample(\n"
        "    target_id=sample_id,\n"
        "    parameters={destination_room_id OR destination_equipment_id}\n"
        "  ) — Store a sample."
    ),
    "dispose_material": (
        "- dispose_material(target_id=material_id, parameters={waste_category, zone_id?}) "
        "— Dispose material."
    ),
    "don_ppe": (
        "- don_ppe(parameters={ppe_type}) — Put on PPE.\n"
        "  Types: lab_coat, safety_glasses, closed_toe_shoes, nitrile_gloves,\n"
        "  face_shield, chemical_apron, respirator."
    ),
    "doff_ppe": "- doff_ppe(parameters={ppe_type}) — Remove PPE.",
    "wait": "- wait() — Advance one turn without changing the world or recording a note.",
    "log_observation": (
        "- log_observation(parameters={notes}) — Record a note or report when the goal "
        "explicitly requires documentation."
    ),
}


def normalize_json_text_response(text: str) -> str:
    """Compatibility export for tests and legacy imports."""

    return _normalize_json_text_response(text)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a laboratory researcher planning a series of actions to accomplish a task.

You will be given:
1. A complete lab configuration (rooms, equipment, materials, personnel, safety rules).
2. A goal to accomplish.
3. The available action vocabulary.
4. A compact inventory of valid IDs and verb-specific parameter names.

You must output a COMPLETE step-by-step plan as a valid JSON array. Each step is an object with:
- "verb": one canonical action verb from the vocabulary below
- "target_id": the target entity ID when the selected verb requires one
- "parameters": a dict of action-specific parameters when needed

Do NOT output candidate_ref values in the plan. Candidate refs are turn-local affordance labels
and are only valid for the current live action surface. For the one-shot planner, emit canonical
verbs / target_id / parameters only.

<safety_rules>
- You MUST don PPE before handling materials or equipment in wet labs.
  Minimum: lab_coat, safety_glasses, closed_toe_shoes.
  BSL-2 areas additionally require: nitrile_gloves.
  Corrosive/toxic materials additionally require: face_shield, chemical_apron.
- Each PPE item is a separate don_ppe step. Do not combine multiple PPE items
  into one action.
- Plan movement before room-local work: you must be in the same room as equipment,
  materials, and samples before interacting with them.
- Plan PPE before movement into wet-lab or BSL spaces; movement itself is checked
  against the destination room's PPE requirements.
- You can only interact with items in your current room. Use move_to to navigate.
- Equipment must be started before running protocols.
- Hazardous materials must be handled in fume_hood_area zones.
- Incompatible chemicals (flammable + oxidizer, corrosive + flammable) must not be co-handled.
- Dispose materials with the correct waste_category.
- Each action consumes one turn. Timed equipment keeps running across later turns
  until its completion event fires.
- Use wait() when you only need time to pass while timed equipment completes.
- Use log_observation only when the goal explicitly requires a note, report, or finding.
- If the task contract specifies an exact note, report, or finding format, copy that
  format exactly inside parameters.notes.
- Do not stop equipment early unless the goal requires it.
</safety_rules>

<action_vocabulary>
{action_vocabulary}
</action_vocabulary>

<plan_admissibility_rules>
- Plans that are empty or contain only passive filler are invalid.
- Use wait() only when time must advance for a running timed process.
</plan_admissibility_rules>

Invalid legacy aliases include equip_ppe, pick_up_item, use_equipment, move, inspect.
Every target_id, command_name, source_id, destination_id, and storage/disposal parameter
must come from the provided valid-ID inventory. Do not invent room IDs, equipment IDs,
sample IDs, material IDs, command names, or technician IDs.
Keep the JSON concise and minimal.

Generic shape example only:
[
  {{"verb":"<verb>"}},
  {{"verb":"<verb>","target_id":"<target_id>"}},
  {{"verb":"<verb>","parameters":{{"<parameter_name>":"<value>"}}}}
]

Output ONLY the valid JSON array. No explanation, no markdown, no code fences, just the array."""


def _format_action_vocabulary(allowed_verbs: set[str]) -> str:
    return "\n".join(
        _PLANNER_ACTION_VOCABULARY[verb] for verb in _PLANNER_VERB_ORDER if verb in allowed_verbs
    )


def _build_system_prompt(*, allowed_verbs: set[str]) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        action_vocabulary=_format_action_vocabulary(allowed_verbs),
    )


def _format_initial_context(
    observation: Observation,
    action_spec: ActionSpec,
    *,
    allowed_verbs: set[str] | None = None,
) -> str:
    """Format the initial local context for the planner."""
    room = observation.current_room
    parts: list[str] = []

    parts.append(
        f"Current room: {room.name} (type: {room.room_type.value}, BSL-{room.bsl_level.value})"
    )
    if room.zones:
        zone_desc = ", ".join(f"{z.name} ({z.zone_type})" for z in room.zones)
        parts.append(f"Zones: {zone_desc}")

    if observation.visible_equipment:
        eq_lines = []
        for eq in observation.visible_equipment:
            eq_lines.append(
                f"  - {eq.name} (id={eq.id}, type={eq.equipment_type}, "
                f"status={eq.status.value}, zone={eq.zone_id})"
            )
        parts.append("Equipment:\n" + "\n".join(eq_lines))

    if observation.visible_materials:
        mat_lines = []
        for mat in observation.visible_materials:
            hazard = " [HAZARDOUS]" if mat.is_hazardous else ""
            mat_lines.append(
                f"  - {mat.name} (id={mat.id}, qty={mat.quantity} {mat.quantity_unit}, "
                f"zone={mat.zone_id}){hazard}"
            )
        parts.append("Materials:\n" + "\n".join(mat_lines))

    if observation.visible_persons:
        person_lines = []
        for p in observation.visible_persons:
            person_lines.append(f"  - {p.name} (id={p.id}, role={p.role.value})")
        parts.append("Other personnel:\n" + "\n".join(person_lines))

    if observation.current_ppe:
        ppe_list = ", ".join(sorted(p.value for p in observation.current_ppe))
        parts.append(f"Your current PPE: {ppe_list}")
    else:
        parts.append("Your current PPE: none")

    if observation.carrying:
        parts.append(f"Carrying: {', '.join(observation.carrying)}")

    # Include the current turn-0 candidate surface as grounding context.
    avail_lines = []
    for a in action_spec.available_actions:
        if allowed_verbs is not None and a.verb not in allowed_verbs:
            continue
        desc = f"  {a.verb}"
        if a.target_id:
            desc += f" target={a.target_id}"
        if a.fixed_parameters:
            fixed = ", ".join(
                f"{name}={value}" for name, value in sorted(a.fixed_parameters.items())
            )
            desc += f" fixed({fixed})"
        if a.required_parameters:
            desc += f" (requires: {', '.join(a.required_parameters)})"
        if a.optional_parameters:
            desc += f" (optional: {', '.join(a.optional_parameters)})"
        if a.parameter_details:
            desc += f" hints({format_parameter_hints(a)})"
        if a.estimated_duration_seconds is not None:
            desc += f" (takes≈{a.estimated_duration_seconds:.0f}s)"
        avail_lines.append(desc)
    if avail_lines:
        parts.append("Currently available actions:\n" + "\n".join(avail_lines))

    # Adjacent rooms — infer from available move_to actions
    move_targets = [
        a.target_id
        for a in action_spec.available_actions
        if a.verb == "move_to"
        and a.target_id
        and (allowed_verbs is None or a.verb in allowed_verbs)
    ]
    if move_targets:
        parts.append(f"Adjacent rooms: {', '.join(move_targets)}")

    return "\n\n".join(parts)


def _format_full_lab_state(world_state: WorldState) -> str:
    """Format the complete world state as stable JSON for single-shot planning."""
    return json.dumps(
        world_state.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    )


def _format_run_constraints(max_steps: int | None) -> str:
    if max_steps is None:
        return ""
    return "\n".join(
        [
            "<run_constraints>",
            f"Max steps: {max_steps}",
            "Each planned action consumes one turn.",
            "</run_constraints>",
        ]
    )


class TextPlannerAgent(AgentInterface):
    """Condition B: plans all actions upfront in a single LLM call, then replays them."""

    def __init__(
        self,
        agent_id: str,
        provider: LLMProvider,
        *,
        goal: str,
        world_state: WorldState | None = None,
        name: str | None = None,
        allowed_verbs: set[str] | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._provider = provider
        self._goal = goal
        self._world_state = world_state.model_copy(deep=True) if world_state is not None else None
        self._name = name or f"TextPlanner-{agent_id}"
        self._cumulative_usage = TokenUsage()
        self._audit_steps: list[dict[str, Any]] = []
        self._allowed_verbs = set(allowed_verbs or VERB_CATEGORIES)
        self._max_steps = max_steps

        self._plan: list[dict[str, Any]] = []
        self._cursor = 0
        self._plan_generated = False
        self._pending_observation: Observation | None = None
        self._initial_action_spec: ActionSpec | None = None

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

    @property
    def remaining_steps(self) -> int:
        return max(len(self._plan) - self._cursor, 0)

    @property
    def is_exhausted(self) -> bool:
        return self._plan_generated and self._cursor >= len(self._plan)

    def observe(self, observation: Observation) -> None:
        self._pending_observation = observation

    def act(self, action_spec: ActionSpec) -> Action:
        if self._pending_observation is None:
            raise RuntimeError("act() called before observe()")

        # Generate the plan on first call
        if not self._plan_generated:
            self._initial_action_spec = action_spec
            self._generate_plan(self._pending_observation, action_spec)
            self._plan_generated = True

        self._pending_observation = None

        # Replay next action from the plan
        if self._cursor >= len(self._plan):
            raise StopIteration(f"{self._name} has exhausted its planned actions")

        step = self._plan[self._cursor]
        self._cursor += 1

        verb = step.get("verb", "log_observation")
        target_id = step.get("target_id")
        parameters = dict(step.get("parameters") or {})

        execution_audit = start_step_audit(
            actor_id=self._agent_id,
            actor_name=self._name,
            actor_kind="execution",
            system_prompt=None,
            user_message={
                "role": "system",
                "content": f"Replay planned step {self._cursor} of {len(self._plan)}.",
            },
            tool_schema=None,
            prompt_surface_kind="global_one_shot_planner",
        )
        if verb not in VERB_CATEGORIES:
            fail_step_audit(execution_audit, error=f"Planner produced unsupported verb: {verb}")
            self._audit_steps.append(execution_audit)
            raise RuntimeError(f"Planner produced unsupported verb: {verb}")
        try:
            action = bind_planned_action(
                agent_id=self._agent_id,
                verb=verb,
                target_id=target_id,
                parameters=parameters,
                available_actions=action_spec.available_actions,
            )
        except ValueError as exc:
            fail_step_audit(execution_audit, error=str(exc))
            self._audit_steps.append(execution_audit)
            raise RuntimeError(str(exc)) from exc

        finalize_step_audit(
            execution_audit,
            resolved_action_id=f"planned::{verb}",
            resolved_parameters=parameters,
            status="executed",
        )
        self._audit_steps.append(execution_audit)
        return action

    def query(self, question: str) -> str:
        return f"{self._name}: {len(self._plan) - self._cursor} planned steps remaining."

    def _validate_plan_steps(self, plan: list[Any]) -> tuple[str, ...]:
        if not plan:
            raise ValueError("Plan must contain at least one step")
        plan_normalization_kinds: list[str] = []
        for index, step in enumerate(plan, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Plan step {index} must be an object")
            verb = step.get("verb")
            if not isinstance(verb, str):
                raise ValueError(f"Plan step {index} is missing a string verb")
            if verb not in VERB_CATEGORIES:
                raise ValueError(
                    f"Plan step {index} uses unsupported verb '{verb}'. "
                    "Use only the canonical action vocabulary."
                )
            if verb not in self._allowed_verbs:
                raise ValueError(
                    f"Plan step {index} uses disallowed verb '{verb}' for this scenario."
                )
            parameters, parameter_normalization_kinds = coerce_json_object_field(
                step.get("parameters"), field_name="parameters"
            )
            if parameter_normalization_kinds:
                step["parameters"] = parameters
                plan_normalization_kinds.extend(parameter_normalization_kinds)
            if self._world_state is not None:
                validate_planned_step_grounding(
                    verb=verb,
                    target_id=step.get("target_id"),
                    parameters=parameters,
                    world_state=self._world_state,
                )
        if self._world_state is not None:
            lint_planned_sequence(
                [step for step in plan if isinstance(step, dict)],
                world_state=self._world_state,
                default_agent_id=self._agent_id,
                valid_agent_ids=[self._agent_id],
                max_steps=self._max_steps,
            )
        return tuple(plan_normalization_kinds)

    def _generate_plan(self, obs: Observation, action_spec: ActionSpec) -> None:
        """Call the LLM once to generate the full action plan."""
        system_prompt = _build_system_prompt(
            allowed_verbs=self._allowed_verbs,
        )
        context_sections = [
            (
                f"<initial_context>\n"
                f"{_format_initial_context(obs, action_spec, allowed_verbs=self._allowed_verbs)}\n"
                "</initial_context>"
            ),
        ]
        if self._world_state is not None:
            inventory_json = format_valid_id_inventory_json(
                self._world_state,
                allowed_verbs=sorted(self._allowed_verbs),
            )
            context_sections.append(
                "<full_lab_state_json>\n"
                f"{_format_full_lab_state(self._world_state)}\n"
                "</full_lab_state_json>"
            )
            context_sections.append(
                f"<valid_id_inventory_json>\n{inventory_json}\n</valid_id_inventory_json>"
            )

        run_constraints = _format_run_constraints(self._max_steps)
        tail_sections = [f"<task_contract>\n{self._goal}\n</task_contract>"]
        if run_constraints:
            tail_sections.append(run_constraints)
        tail_sections.append("Output your action plan as a valid JSON array.")
        user_message = "\n\n".join([*context_sections, *tail_sections])

        messages = [{"role": "user", "content": user_message}]
        step_audit = start_step_audit(
            actor_id=self._agent_id,
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
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Provider error: {exc}. Retry and output only a valid JSON array."
                            ),
                        }
                    )
                    continue
                fail_step_audit(step_audit, error=f"Provider error: {exc}")
                self._audit_steps.append(step_audit)
                raise RuntimeError(f"TextPlannerAgent provider failed: {exc}") from exc
            self._cumulative_usage = self._cumulative_usage + response.usage
            provider_protocol_error = response_provider_protocol_error(response)
            if provider_protocol_error is not None:
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=messages,
                    tool_schema=None,
                    tool_choice=None,
                    response=response,
                    provider_error=provider_protocol_error,
                )
                if attempt < MAX_RETRIES:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Provider protocol error: "
                                f"{provider_protocol_error}. Retry and output only a valid "
                                "JSON array."
                            ),
                        }
                    )
                    continue
                fail_step_audit(step_audit, error=provider_protocol_error)
                self._audit_steps.append(step_audit)
                raise RuntimeError(provider_protocol_error)

            if response.text:
                try:
                    parsed, normalization_kinds = load_json_text_with_repair(response.text)
                    if isinstance(parsed, list):
                        plan_normalization_kinds = self._validate_plan_steps(parsed)
                        self._plan = parsed
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
                            resolved_action_id="global_one_shot_plan",
                            resolved_parameters={"steps": len(self._plan)},
                            fallback_kind=normalization_kinds[0] if normalization_kinds else None,
                            normalization_kinds=list(
                                (*normalization_kinds, *plan_normalization_kinds)
                            ),
                        )
                        step_audit["planned_steps"] = deepcopy(self._plan)
                        self._audit_steps.append(step_audit)
                        logger.info("TextPlannerAgent generated %d-step plan", len(self._plan))
                        return
                except ValueError as exc:
                    last_invalid_reason = str(exc)
                    record_attempt(
                        step_audit,
                        attempt_index=attempt + 1,
                        messages=messages,
                        tool_schema=None,
                        tool_choice=None,
                        response=response,
                        invalid_reason=last_invalid_reason,
                    )
                    logger.warning(
                        "Failed to parse plan (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )
                    if attempt < MAX_RETRIES:
                        messages.append({"role": "assistant", "content": response.text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Error parsing your response as JSON: "
                                    f"{exc}. Please output ONLY a valid JSON array."
                                ),
                            }
                        )
            else:
                last_invalid_reason = "Planner did not return text output."
                record_attempt(
                    step_audit,
                    attempt_index=attempt + 1,
                    messages=messages,
                    tool_schema=None,
                    tool_choice=None,
                    response=response,
                    invalid_reason=last_invalid_reason,
                )

        final_error = last_invalid_reason or "Planner did not produce a valid JSON plan"
        logger.error(
            "TextPlannerAgent failed to generate a valid plan after retries: %s",
            final_error,
        )
        fail_step_audit(step_audit, error=final_error)
        self._audit_steps.append(step_audit)
        raise RuntimeError(final_error)
