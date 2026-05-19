"""Trace and memory models for embodied agent integrations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from labverse.agent.interface import Observation
from labverse.core.ids import generate_id
from labverse.models.action import Action, ActionResult, ActionResultStatus, ActionSpec


class AgentTraceRecord(BaseModel):
    """A single canonical step record for an agent."""

    trace_id: str = Field(default_factory=lambda: generate_id("trace"))
    agent_id: str
    tick: int
    timestamp: datetime
    observation: Observation
    action_spec: ActionSpec
    action: Action
    result: ActionResult
    event_ids: list[str] = Field(default_factory=list)
    post_observation: Observation | None = None


class AgentMemorySnapshot(BaseModel):
    """Minimal platform-managed memory state for an agent."""

    agent_id: str
    working_memory: str = ""
    last_observation_id: str | None = None
    last_trace_id: str | None = None
    last_action_status: ActionResultStatus | None = None
    trace_count: int = 0
    recent_trace_ids: list[str] = Field(default_factory=list)


class AgentTraceStore:
    """Append-only per-agent trace storage with lightweight working memory."""

    def __init__(self) -> None:
        self._traces_by_agent: dict[str, list[AgentTraceRecord]] = defaultdict(list)
        self._working_memory: dict[str, str] = {}

    def append(self, trace: AgentTraceRecord) -> None:
        self._traces_by_agent[trace.agent_id].append(trace)

    def recent(self, agent_id: str, *, limit: int = 20) -> list[AgentTraceRecord]:
        traces = self._traces_by_agent.get(agent_id, [])
        if limit <= 0:
            return []
        return list(traces[-limit:])

    def snapshot(self, agent_id: str, *, limit: int = 10) -> AgentMemorySnapshot:
        traces = self._traces_by_agent.get(agent_id, [])
        last_trace = traces[-1] if traces else None
        return AgentMemorySnapshot(
            agent_id=agent_id,
            working_memory=self._working_memory.get(agent_id, ""),
            last_observation_id=(
                last_trace.post_observation.observation_id
                if last_trace and last_trace.post_observation is not None
                else last_trace.observation.observation_id
                if last_trace
                else None
            ),
            last_trace_id=last_trace.trace_id if last_trace else None,
            last_action_status=last_trace.result.status if last_trace else None,
            trace_count=len(traces),
            recent_trace_ids=[trace.trace_id for trace in traces[-limit:]],
        )

    def set_working_memory(self, agent_id: str, value: str) -> AgentMemorySnapshot:
        self._working_memory[agent_id] = value
        return self.snapshot(agent_id)

    def working_memory(self, agent_id: str) -> str:
        return self._working_memory.get(agent_id, "")

    def last_trace(self, agent_id: str) -> AgentTraceRecord | None:
        traces = self._traces_by_agent.get(agent_id, [])
        if not traces:
            return None
        return traces[-1]
