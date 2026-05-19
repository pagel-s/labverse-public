"""Simulation clock: tick-based time with configurable resolution."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta


class SimulationClock:
    """Manages simulation time as discrete ticks mapped to wall-clock durations.

    The clock starts at a configurable datetime and advances in discrete ticks.
    Each tick represents a configurable number of seconds of simulated time.

    Args:
        start_time: The simulated wall-clock time at tick 0.
        seconds_per_tick: How many seconds of simulated time each tick represents.
    """

    def __init__(
        self,
        start_time: datetime | None = None,
        seconds_per_tick: float = 10.0,
    ) -> None:
        if start_time is None:
            start_time = datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC)
        if seconds_per_tick <= 0:
            raise ValueError("seconds_per_tick must be positive")
        self._start_time = start_time
        self._seconds_per_tick = seconds_per_tick
        self._tick: int = 0

    @property
    def tick(self) -> int:
        """Current tick count."""
        return self._tick

    @property
    def current_time(self) -> datetime:
        """Current simulated wall-clock time."""
        return self._start_time + timedelta(seconds=self._tick * self._seconds_per_tick)

    @property
    def elapsed_seconds(self) -> float:
        """Total simulated seconds elapsed since start."""
        return self._tick * self._seconds_per_tick

    @property
    def seconds_per_tick(self) -> float:
        return self._seconds_per_tick

    @property
    def start_time(self) -> datetime:
        return self._start_time

    def advance(self, ticks: int = 1) -> None:
        """Advance the clock by the given number of ticks."""
        if ticks < 0:
            raise ValueError(f"Cannot advance by negative ticks: {ticks}")
        self._tick += ticks

    def advance_by_seconds(self, seconds: float) -> int:
        """Advance the clock by at least the given number of simulated seconds.

        Returns the number of ticks advanced.
        """
        if seconds < 0:
            raise ValueError(f"Cannot advance by negative seconds: {seconds}")
        ticks = self.ticks_for_duration(seconds)
        self._tick += ticks
        return ticks

    def ticks_for_duration(self, seconds: float) -> int:
        """Calculate how many ticks a duration in seconds corresponds to (rounded up)."""
        if seconds <= 0:
            return 0
        return max(1, math.ceil(seconds / self._seconds_per_tick))

    def reset(self) -> None:
        """Reset the clock to tick 0."""
        self._tick = 0

    def __repr__(self) -> str:
        return (
            f"SimulationClock(tick={self._tick}, "
            f"time={self.current_time.isoformat()}, "
            f"spt={self._seconds_per_tick})"
        )
