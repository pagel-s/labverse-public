"""Timed event queue used by the GameMaster."""

from __future__ import annotations

import heapq
from itertools import count
from typing import Any

from pydantic import BaseModel, Field

from labverse.core.clock import SimulationClock
from labverse.core.ids import generate_id


class TimedEvent(BaseModel):
    """A future event that should fire when the clock reaches a target tick."""

    event_id: str
    completes_at_tick: int
    event_type: str
    target_id: str
    callback_data: dict[str, Any] = Field(default_factory=dict)


class TimeManager:
    """Min-heap priority queue for timed simulation events."""

    def __init__(self, clock: SimulationClock) -> None:
        self._clock = clock
        self._queue: list[tuple[int, int, TimedEvent]] = []
        self._counter = count()

    def schedule(
        self,
        *,
        event_type: str,
        target_id: str,
        completes_at_tick: int,
        callback_data: dict[str, Any] | None = None,
    ) -> TimedEvent:
        event = TimedEvent(
            event_id=generate_id("evt"),
            completes_at_tick=completes_at_tick,
            event_type=event_type,
            target_id=target_id,
            callback_data=callback_data or {},
        )
        heapq.heappush(self._queue, (event.completes_at_tick, next(self._counter), event))
        return event

    def schedule_in_seconds(
        self,
        *,
        event_type: str,
        target_id: str,
        delay_seconds: float,
        callback_data: dict[str, Any] | None = None,
    ) -> TimedEvent:
        delay_ticks = self._clock.ticks_for_duration(delay_seconds)
        return self.schedule(
            event_type=event_type,
            target_id=target_id,
            completes_at_tick=self._clock.tick + delay_ticks,
            callback_data=callback_data,
        )

    def cancel(self, *, event_type: str | None = None, target_id: str | None = None) -> int:
        kept: list[tuple[int, int, TimedEvent]] = []
        removed = 0
        for item in self._queue:
            event = item[2]
            matches_type = event_type is None or event.event_type == event_type
            matches_target = target_id is None or event.target_id == target_id
            if matches_type and matches_target:
                removed += 1
                continue
            kept.append(item)
        if removed:
            self._queue = kept
            heapq.heapify(self._queue)
        return removed

    def pop_due_events(self) -> list[TimedEvent]:
        due: list[TimedEvent] = []
        while self._queue and self._queue[0][0] <= self._clock.tick:
            due.append(heapq.heappop(self._queue)[2])
        return due

    def __len__(self) -> int:
        return len(self._queue)
