"""Interactive benchmark agent with compact global structured context each turn."""

from __future__ import annotations

import json

from labverse.agent.action_history import prepend_interactive_history_blocks
from labverse.agent.audit import fail_step_audit, finalize_step_audit, start_step_audit
from labverse.agent.interface import Observation
from labverse.agent.llm import LLMAgent, StepStats, stats_to_usage
from labverse.agent.prompt import (
    build_action_tool,
    build_global_structured_user_message,
)
from labverse.agent.prompt import (
    build_global_structured_system_prompt as _build_global_structured_system_prompt,
)
from labverse.agent.providers.base import LLMProvider
from labverse.models.action import Action, ActionSpec
from labverse.world.state import WorldState

GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS = 2


def build_global_structured_system_prompt(*, goal: str) -> str:
    """Build the system prompt for the global structured interactive control."""
    return _build_global_structured_system_prompt(goal=goal)


def _enum_value(value: object) -> object:
    """Return an enum's value when available, otherwise the object itself."""
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _enum_text(value: object) -> str:
    """Return a stable string for enum-like values used in prompt summaries."""
    return str(_enum_value(value))


def _sorted_room_ids(world_state: WorldState) -> list[str]:
    return sorted(world_state.rooms)


def _build_compact_world_state_summary(world_state: WorldState) -> dict[str, object]:
    """Build a compact, deterministic whole-lab summary for prompting."""
    room_ids = _sorted_room_ids(world_state)
    room_summaries: list[dict[str, object]] = []
    for room_id in room_ids:
        room = world_state.get_room(room_id)
        adjacent_rooms = sorted(
            {
                other
                for door in world_state.doors
                if room_id in door.connects
                for other in door.connects
                if other != room_id
            }
        )
        room_summaries.append(
            {
                "id": room.id,
                "type": _enum_text(room.room_type),
                "bsl": _enum_text(room.bsl_level),
                "adjacent_rooms": adjacent_rooms,
            }
        )

    return {
        "lab": {
            "id": world_state.lab.id,
            "entry_room": world_state.lab.default_entry_room or None,
            "counts": world_state.entity_counts,
        },
        "rooms": room_summaries,
        "doors": [
            {
                "id": door.id,
                "connects": list(door.connects),
                "is_open": door.is_open,
                "locked": door.locked,
                "requires_keycard": door.requires_keycard,
                "requires_bsl_clearance": _enum_text(door.requires_bsl_clearance)
                if door.requires_bsl_clearance is not None
                else None,
            }
            for door in sorted(world_state.doors, key=lambda door: door.id)
        ],
        "equipment": [
            {
                "id": equipment.id,
                "type": equipment.equipment_type,
                "status": _enum_text(equipment.status),
                "room_id": equipment.room_id,
                "current_users": sorted(equipment.current_users),
                "running_command": equipment.running_command,
                "error_message": equipment.error_message,
            }
            for equipment in sorted(
                world_state.equipment.values(),
                key=lambda equipment: equipment.id,
            )
        ],
        "materials": [
            {
                "id": material.id,
                "room_id": material.room_id,
                "quantity": material.quantity,
                "unit": material.quantity_unit,
                "hazardous": material.is_hazardous,
                "required_ppe": sorted(_enum_text(ppe) for ppe in material.required_ppe),
                "waste_category": _enum_text(material.waste_category),
                "is_controlled": material.is_controlled,
                "incompatible_with": sorted(material.incompatible_with),
                "expiry_date": (
                    material.expiry_date.isoformat() if material.expiry_date is not None else None
                ),
            }
            for material in sorted(world_state.materials.values(), key=lambda material: material.id)
        ],
        "samples": [
            {
                "id": sample.id,
                "status": sample.status,
                "room_id": sample.room_id,
                "equipment_id": sample.equipment_id,
                "bsl": _enum_text(sample.bsl_level),
                "derived_from": sorted(sample.derived_from),
            }
            for sample in sorted(world_state.samples.values(), key=lambda sample: sample.id)
        ],
        "personnel": [
            {
                "id": person.id,
                "role": _enum_text(person.role),
                "current_room_id": person.current_room_id,
                "current_ppe": sorted(_enum_text(ppe) for ppe in person.current_ppe),
                "current_activity": person.current_activity,
            }
            for person in sorted(world_state.persons.values(), key=lambda person: person.id)
        ],
    }


def format_world_state_json(world_state: WorldState) -> str:
    """Serialize the full world snapshot into stable JSON for prompting."""
    return json.dumps(world_state.model_dump(mode="json"), indent=2, sort_keys=True)


def format_world_state_summary_json(world_state: WorldState) -> str:
    """Serialize a compact whole-lab summary into deterministic JSON for prompting."""
    return json.dumps(
        _build_compact_world_state_summary(world_state),
        sort_keys=True,
        separators=(",", ":"),
    )


class GlobalStructuredInteractiveAgent(LLMAgent):
    """Step-wise grounded agent with full structured world-state context each turn."""

    def __init__(
        self,
        agent_id: str,
        provider: LLMProvider,
        *,
        goal: str,
        name: str | None = None,
        max_history_turns: int = 10,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            provider=provider,
            goal=goal,
            name=name or f"GlobalStructured-{agent_id}",
            max_history_turns=min(
                max_history_turns,
                GLOBAL_STRUCTURED_VERBATIM_HISTORY_TURNS,
            ),
        )
        self._system_prompt = build_global_structured_system_prompt(goal=goal)
        self._prompt_surface_kind = "global_structured_interactive_grounded"
        self._world_state_json: str | None = None

    def set_world_state(self, world_state: WorldState) -> None:
        """Store the current world snapshot for the next act() call."""
        self._world_state_json = format_world_state_summary_json(world_state)

    def act(self, action_spec: ActionSpec) -> Action:
        if self._pending_observation is None:
            raise RuntimeError("act() called before observe()")
        if self._world_state_json is None:
            raise RuntimeError("act() called before set_world_state()")

        observation: Observation = self._pending_observation
        self._pending_observation = None

        # Prior-turn outcomes are carried by the compact trajectory block;
        # no separate ``<action_result>`` embedding needed.
        base_obs_msg = build_global_structured_user_message(
            observation,
            action_spec.available_actions,
            self._world_state_json,
            last_action_result=None,
            turn_status=self._pending_turn_status,
        )
        self._pending_turn_status = None
        obs_msg = prepend_interactive_history_blocks(base_obs_msg, self._trajectory)

        action_tool = build_action_tool(action_spec.available_actions)
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

        # Record the decision in the compact trajectory.
        self._trajectory.record_decision(
            verb=action.verb,
            candidate_ref=selected_candidate_ref,
            actor_id=self._agent_id,
            target_id=action.target_id,
        )

        self._cumulative_usage = self._cumulative_usage + stats_to_usage(stats)
        self._step_stats.append(stats)
        self._audit_steps.append(step_audit)

        return action


__all__ = [
    "GlobalStructuredInteractiveAgent",
    "_build_compact_world_state_summary",
    "build_global_structured_system_prompt",
    "format_world_state_json",
    "format_world_state_summary_json",
]
