"""Shared test fixtures for LabVerse."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from labverse.config import load_world
from labverse.core.clock import SimulationClock
from labverse.core.events import EventBus, EventLog, create_logging_bus
from labverse.gamemaster import GameMaster
from labverse.safety import SafetyEngine
from labverse.world.state import WorldState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clock() -> SimulationClock:
    """A simulation clock starting at 2026-01-01 08:00 UTC, 10s per tick."""
    return SimulationClock(
        start_time=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        seconds_per_tick=10.0,
    )


@pytest.fixture
def fast_clock() -> SimulationClock:
    """A fast clock with 1s per tick for fine-grained testing."""
    return SimulationClock(
        start_time=datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC),
        seconds_per_tick=1.0,
    )


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def logging_bus() -> tuple[EventBus, EventLog]:
    return create_logging_bus()


@pytest.fixture
def minimal_world() -> WorldState:
    return load_world(ROOT / "configs" / "labs" / "minimal_test.yaml")


@pytest.fixture
def research_world() -> WorldState:
    return load_world(ROOT / "configs" / "labs" / "research_bsl2.yaml")


@pytest.fixture
def safety_engine() -> SafetyEngine:
    return SafetyEngine()


@pytest.fixture
def game_master(
    research_world: WorldState,
    clock: SimulationClock,
    event_bus: EventBus,
) -> GameMaster:
    return GameMaster(research_world, clock=clock, event_bus=event_bus)
