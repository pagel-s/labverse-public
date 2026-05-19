"""Tests for the event system."""

from datetime import UTC, datetime

from labverse.core.events import (
    Event,
    EventBus,
    EventLog,
    EventType,
    create_logging_bus,
)


def _make_event(
    event_type: EventType = EventType.STATE_CHANGE,
    tick: int = 0,
    source_id: str = "",
    target_id: str = "",
    description: str = "",
) -> Event:
    return Event(
        event_type=event_type,
        tick=tick,
        timestamp=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        source_id=source_id,
        target_id=target_id,
        description=description,
    )


class TestEvent:
    def test_create_event(self) -> None:
        e = _make_event(description="test")
        assert e.event_type == EventType.STATE_CHANGE
        assert e.description == "test"
        assert e.data == {}

    def test_event_with_data(self) -> None:
        e = _make_event()
        e.data["key"] = "value"
        assert e.data["key"] == "value"

    def test_event_serialization(self) -> None:
        e = _make_event(source_id="agent_1", description="moved")
        d = e.model_dump()
        assert d["event_type"] == "state_change"
        assert d["source_id"] == "agent_1"
        restored = Event.model_validate(d)
        assert restored == e


class TestEventBus:
    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.STATE_CHANGE, received.append)
        event = _make_event()
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_subscribe_filters_by_type(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.EQUIPMENT_STATE, received.append)
        bus.publish(_make_event(EventType.STATE_CHANGE))
        bus.publish(_make_event(EventType.EQUIPMENT_STATE))
        assert len(received) == 1
        assert received[0].event_type == EventType.EQUIPMENT_STATE

    def test_wildcard_subscriber(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("*", received.append)
        bus.publish(_make_event(EventType.STATE_CHANGE))
        bus.publish(_make_event(EventType.SAFETY_VIOLATION))
        assert len(received) == 2

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        a: list[Event] = []
        b: list[Event] = []
        bus.subscribe(EventType.STATE_CHANGE, a.append)
        bus.subscribe(EventType.STATE_CHANGE, b.append)
        bus.publish(_make_event())
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.STATE_CHANGE, received.append)
        bus.publish(_make_event())
        assert len(received) == 1
        bus.unsubscribe(EventType.STATE_CHANGE, received.append)
        bus.publish(_make_event())
        assert len(received) == 1  # no new events

    def test_unsubscribe_nonexistent_handler(self) -> None:
        bus = EventBus()
        bus.unsubscribe(EventType.STATE_CHANGE, lambda e: None)  # no error


class TestEventLog:
    def test_append_and_query(self) -> None:
        log = EventLog()
        log.append(_make_event(tick=0, source_id="a"))
        log.append(_make_event(tick=1, source_id="b"))
        assert len(log) == 2
        assert log.events[0].source_id == "a"

    def test_query_by_type(self) -> None:
        log = EventLog()
        log.append(_make_event(EventType.STATE_CHANGE))
        log.append(_make_event(EventType.EQUIPMENT_STATE))
        log.append(_make_event(EventType.STATE_CHANGE))
        result = log.query(event_type=EventType.STATE_CHANGE)
        assert len(result) == 2

    def test_query_by_source(self) -> None:
        log = EventLog()
        log.append(_make_event(source_id="agent_1"))
        log.append(_make_event(source_id="agent_2"))
        log.append(_make_event(source_id="agent_1"))
        result = log.query(source_id="agent_1")
        assert len(result) == 2

    def test_query_by_target(self) -> None:
        log = EventLog()
        log.append(_make_event(target_id="centrifuge_1"))
        log.append(_make_event(target_id="pcr_1"))
        result = log.query(target_id="centrifuge_1")
        assert len(result) == 1

    def test_query_since_tick(self) -> None:
        log = EventLog()
        log.append(_make_event(tick=0))
        log.append(_make_event(tick=5))
        log.append(_make_event(tick=10))
        result = log.query(since_tick=5)
        assert len(result) == 2

    def test_query_with_limit(self) -> None:
        log = EventLog()
        for i in range(10):
            log.append(_make_event(tick=i))
        result = log.query(limit=3)
        assert len(result) == 3
        assert result[0].tick == 7  # last 3

    def test_query_combined_filters(self) -> None:
        log = EventLog()
        log.append(_make_event(EventType.STATE_CHANGE, tick=0, source_id="a"))
        log.append(_make_event(EventType.EQUIPMENT_STATE, tick=1, source_id="a"))
        log.append(_make_event(EventType.STATE_CHANGE, tick=2, source_id="b"))
        log.append(_make_event(EventType.STATE_CHANGE, tick=3, source_id="a"))
        result = log.query(event_type=EventType.STATE_CHANGE, source_id="a")
        assert len(result) == 2

    def test_clear(self) -> None:
        log = EventLog()
        log.append(_make_event())
        log.clear()
        assert len(log) == 0

    def test_repr(self) -> None:
        log = EventLog()
        assert "0 events" in repr(log)


class TestCreateLoggingBus:
    def test_bus_logs_all_events(self) -> None:
        bus, log = create_logging_bus()
        bus.publish(_make_event(EventType.STATE_CHANGE))
        bus.publish(_make_event(EventType.EQUIPMENT_STATE))
        bus.publish(_make_event(EventType.SAFETY_VIOLATION))
        assert len(log) == 3
