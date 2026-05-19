"""Deterministic scripted agent for tests and benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from labverse.agent.interface import AgentInterface, Observation
from labverse.core.ids import generate_id
from labverse.models.action import Action, ActionSpec

ScriptedStep = Action | Callable[[Observation, ActionSpec], Action]


class ScriptedAgent(AgentInterface):
    """Agent that follows a predefined sequence of actions."""

    def __init__(
        self,
        agent_id: str,
        script: Sequence[ScriptedStep],
        *,
        name: str | None = None,
        query_responses: Mapping[str, str] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._name = name or agent_id
        self._script = list(script)
        self._cursor = 0
        self._query_responses = dict(query_responses or {})
        self._last_observation: Observation | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def remaining_steps(self) -> int:
        return len(self._script) - self._cursor

    def observe(self, observation: Observation) -> None:
        self._last_observation = observation

    def act(self, action_spec: ActionSpec) -> Action:
        if self._cursor >= len(self._script):
            raise RuntimeError(f"ScriptedAgent {self._agent_id} has no remaining scripted actions")

        if self._last_observation is None:
            raise RuntimeError("ScriptedAgent.act() called before observe()")

        step = self._script[self._cursor]
        self._cursor += 1
        action = (
            step(self._last_observation, action_spec)
            if callable(step)
            else step.model_copy(deep=True)
        )
        if not action.id:
            action = action.model_copy(update={"id": generate_id("action")}, deep=True)
        return action.model_copy(update={"agent_id": self._agent_id}, deep=True)

    def query(self, question: str) -> str:
        if question in self._query_responses:
            return self._query_responses[question]
        if self._last_observation is None:
            return f"{self._name} has not observed the world yet."
        return (
            f"{self._name} is in {self._last_observation.current_room.name}. "
            f"{self.remaining_steps} scripted actions remain."
        )


__all__ = ["ScriptedAgent", "ScriptedStep"]
