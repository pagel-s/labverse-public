"""Event system: typed events, synchronous pub/sub bus, and event log."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from labverse.core.ids import generate_id


class EventType(StrEnum):
    """Categories of simulation events."""

    ACTION_SUBMITTED = "action_submitted"
    ACTION_BLOCKED = "action_blocked"
    ACTION_RESOLVED = "action_resolved"
    STATE_CHANGE = "state_change"
    EQUIPMENT_STATE = "equipment_state"
    SAFETY_VIOLATION = "safety_violation"
    SAFETY_WARNING = "safety_warning"
    TIMED_EVENT = "timed_event"
    PERSON_MOVED = "person_moved"
    MATERIAL_CONSUMED = "material_consumed"
    SAMPLE_UPDATED = "sample_updated"
    CLOCK_ADVANCED = "clock_advanced"


class Event(BaseModel):
    """A single simulation event."""

    id: str = Field(default_factory=lambda: generate_id("evt"))
    event_type: EventType
    tick: int
    timestamp: datetime
    source_id: str = ""
    target_id: str = ""
    description: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


EventHandler = Callable[[Event], None]


class EventBus:
    """Synchronous publish/subscribe event bus.

    Handlers subscribe to specific event types (or all events via "*")
    and are called synchronously when events are published.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type. Use "*" for all events."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        self._handlers[key].append(handler)

    def unsubscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """Remove a handler from an event type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        handlers = self._handlers.get(key, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""
        for handler in self._handlers.get(event.event_type.value, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)


class EventLog:
    """Append-only log of all simulation events, with optional filtering."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, event: Event) -> None:
        """Add an event to the log."""
        self._events.append(event)

    @property
    def events(self) -> list[Event]:
        """All events in chronological order."""
        return list(self._events)

    def query(
        self,
        event_type: EventType | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        since_tick: int | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Filter events by criteria."""
        result = self._events
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if source_id is not None:
            result = [e for e in result if e.source_id == source_id]
        if target_id is not None:
            result = [e for e in result if e.target_id == target_id]
        if since_tick is not None:
            result = [e for e in result if e.tick >= since_tick]
        if limit is not None:
            result = result[-limit:]
        return result

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"EventLog({len(self._events)} events)"


def create_logging_bus() -> tuple[EventBus, EventLog]:
    """Create an EventBus with an attached EventLog that records everything."""
    bus = EventBus()
    log = EventLog()
    bus.subscribe("*", log.append)
    return bus, log
