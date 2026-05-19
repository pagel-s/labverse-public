"""Validation for incoming agent actions before safety or resolution."""

from __future__ import annotations

from labverse.gamemaster.action_resolver import ActionHandler
from labverse.models.action import Action
from labverse.world.state import WorldState


class ValidationPipeline:
    """Well-formedness and dispatch validation for submitted actions."""

    def __init__(self, handlers: dict[str, ActionHandler]) -> None:
        self._handlers = handlers

    def validate_action(self, action: Action, world: WorldState) -> str | None:
        errors: list[str] = []

        if not action.id:
            errors.append("Action ID is required")
        if action.agent_id not in world.persons:
            errors.append(f"Unknown agent: {action.agent_id}")
        if not action.verb:
            errors.append("Action verb is required")
            return "; ".join(errors)
        if action.verb not in self._handlers:
            errors.append(f"Unsupported action verb: {action.verb}")
            return "; ".join(errors)

        handler = self._handlers[action.verb]
        if handler.requires_target and action.target_id is None:
            errors.append(f"Action {action.verb} requires a target_id")
        if action.target_id is not None and handler.target_types:
            if not _target_exists(action.target_id, handler.target_types, world):
                target_types = ", ".join(handler.target_types)
                errors.append(
                    f"Action {action.verb} target {action.target_id!r} must reference one of: "
                    f"{target_types}"
                )

        errors.extend(handler.validate(action, world))
        return "; ".join(errors) if errors else None


def _target_exists(target_id: str, target_types: tuple[str, ...], world: WorldState) -> bool:
    for target_type in target_types:
        if target_type == "room" and target_id in world.rooms:
            return True
        if target_type == "door":
            try:
                world.get_door(target_id)
            except KeyError:
                pass
            else:
                return True
        if target_type == "equipment" and target_id in world.equipment:
            return True
        if target_type == "material" and target_id in world.materials:
            return True
        if target_type == "sample" and target_id in world.samples:
            return True
    return False


__all__ = ["ValidationPipeline"]
