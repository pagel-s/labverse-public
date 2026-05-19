"""Tests for the simulation clock."""

from datetime import UTC, datetime

import pytest

from labverse.core.clock import SimulationClock


class TestSimulationClock:
    def test_initial_state(self, clock: SimulationClock) -> None:
        assert clock.tick == 0
        assert clock.elapsed_seconds == 0.0
        assert clock.current_time == datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC)

    def test_advance_one_tick(self, clock: SimulationClock) -> None:
        clock.advance(1)
        assert clock.tick == 1
        assert clock.elapsed_seconds == 10.0
        assert clock.current_time == datetime(2026, 1, 1, 8, 0, 10, tzinfo=UTC)

    def test_advance_multiple_ticks(self, clock: SimulationClock) -> None:
        clock.advance(6)
        assert clock.tick == 6
        assert clock.elapsed_seconds == 60.0

    def test_advance_by_seconds(self, clock: SimulationClock) -> None:
        ticks = clock.advance_by_seconds(30.0)
        assert ticks == 3
        assert clock.tick == 3

    def test_advance_by_seconds_rounds(self, clock: SimulationClock) -> None:
        ticks = clock.advance_by_seconds(15.0)
        assert ticks == 2  # rounds 1.5 -> 2
        assert clock.tick == 2

    def test_advance_by_seconds_rounds_up(self, clock: SimulationClock) -> None:
        ticks = clock.advance_by_seconds(11.0)
        assert ticks == 2
        assert clock.tick == 2

    def test_advance_by_seconds_minimum_one_tick(self, clock: SimulationClock) -> None:
        ticks = clock.advance_by_seconds(0.1)
        assert ticks == 1

    def test_advance_by_zero_seconds(self, clock: SimulationClock) -> None:
        ticks = clock.advance_by_seconds(0.0)
        assert ticks == 0
        assert clock.tick == 0

    def test_ticks_for_duration(self, clock: SimulationClock) -> None:
        assert clock.ticks_for_duration(30.0) == 3
        assert clock.ticks_for_duration(0.0) == 0
        assert clock.ticks_for_duration(1.0) == 1  # minimum 1 for positive durations

    def test_negative_advance_raises(self, clock: SimulationClock) -> None:
        with pytest.raises(ValueError, match="negative ticks"):
            clock.advance(-1)

    def test_negative_seconds_raises(self, clock: SimulationClock) -> None:
        with pytest.raises(ValueError, match="negative seconds"):
            clock.advance_by_seconds(-5.0)

    def test_reset(self, clock: SimulationClock) -> None:
        clock.advance(100)
        assert clock.tick == 100
        clock.reset()
        assert clock.tick == 0
        assert clock.elapsed_seconds == 0.0

    def test_custom_seconds_per_tick(self) -> None:
        clock = SimulationClock(seconds_per_tick=60.0)
        clock.advance(5)
        assert clock.elapsed_seconds == 300.0

    def test_fine_grained_clock(self, fast_clock: SimulationClock) -> None:
        fast_clock.advance(30)
        assert fast_clock.elapsed_seconds == 30.0

    def test_time_accumulates(self, clock: SimulationClock) -> None:
        clock.advance(3)
        clock.advance(2)
        clock.advance_by_seconds(50.0)
        assert clock.tick == 10  # 3 + 2 + 5

    def test_repr(self, clock: SimulationClock) -> None:
        r = repr(clock)
        assert "tick=0" in r
        assert "spt=10.0" in r

    def test_default_start_time(self) -> None:
        clock = SimulationClock()
        assert clock.start_time.year == 2026

    def test_non_positive_seconds_per_tick_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SimulationClock(seconds_per_tick=0.0)
