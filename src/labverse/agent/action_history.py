"""Compact decision-trace history for interactive agents.

Historical background
---------------------
Early interactive-agent implementations kept a ``_history`` list of prior
``user``-role messages. Each entry was the full per-turn prompt: goal,
observation, action surface, progress status, and world-state dump. On
every new turn the agent re-sent the entire list so the model could "see"
the conversation so far. In practice this meant that on turn 15 of a
multi-agent scenario we replayed ~11 full state dumps at ~1.8 k tokens
each, pushing single-turn prompts to ~24 k tokens — 97 % of the 24 576
``num_ctx`` ceiling on ``gemma4:31b``. The previous ``num_ctx`` default of
16 384 was silently truncating the oldest turns off the top of the
context window, quietly corrupting any benchmark that ran on it.

The right answer is not a bigger context window. Most of what we were
replaying was either **identical** across turns (the goal), **stale**
(prior observations, prior action surfaces, prior progress counters), or
**redundant with authoritative state** that the current turn already
carries. The only history-dependent information the agent actually needs
from prior turns is the list of decisions it has already tried and what
happened when they resolved. Everything else is noise.

This module captures exactly that history-dependent information as a
bounded ``ActionTrajectory`` of ``ActionTrajectoryEntry`` records. The
rendered block is compact (typically < 100 tokens per run) and replaces
the legacy ``_history`` list of prior user messages entirely.

Scientific rationale
--------------------
The agent's next-action decision has two inputs:

1. The *current* authoritative state — observation, progress, available
   actions. This is present on every turn and has always been
   authoritative; no change here.
2. The agent's *own* trajectory so far — which actions it chose, in what
   order, and whether each resolved successfully. This is the only
   history-dependent signal that matters. The trajectory block captures
   it faithfully and compactly.

Replaying stale observations and stale action surfaces is not
additionally informative: nothing in a prior turn's observation can
override the current turn's observation. Dropping those entries
therefore **cannot** cost the agent information it would otherwise use
to decide correctly. It can only remove a source of confusion and
silent-truncation risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

_DEFAULT_MAX_ENTRIES: Final[int] = 50


@dataclass(frozen=True)
class ActionTrajectoryEntry:
    """One prior-turn decision and its resolved outcome."""

    turn_index: int
    actor_id: str | None
    candidate_ref: str | None
    verb: str
    target_id: str | None
    success: bool | None
    description: str | None

    def format_line(self) -> str:
        parts: list[str] = [f"{self.turn_index}."]
        if self.candidate_ref:
            parts.append(f"[{self.candidate_ref}]")
        if self.actor_id:
            parts.append(f"actor={self.actor_id}")
        parts.append(f"verb={self.verb}")
        if self.target_id:
            parts.append(f"target_id={self.target_id}")
        if self.success is True:
            parts.append("→ ok")
        elif self.success is False:
            parts.append("→ failed")
        else:
            parts.append("→ pending")
        if self.description:
            clean = " ".join(self.description.split())
            if len(clean) > 120:
                clean = clean[:117] + "..."
            parts.append(f"({clean})")
        return " ".join(parts)


class ActionTrajectory:
    """Bounded record of the agent's prior decisions and their outcomes.

    Use ``record_decision`` immediately after the adjudicator has bound
    an action output to a concrete ``Action``, and ``record_last_result`` once
    the world has told us whether that action succeeded. Render the
    current trace onto the top of the next turn's user prompt via
    ``format_trajectory_block``.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._entries: list[ActionTrajectoryEntry] = []
        self._max_entries = max_entries
        self._dropped_entry_count = 0

    @property
    def entries(self) -> tuple[ActionTrajectoryEntry, ...]:
        return tuple(self._entries)

    @property
    def dropped_entry_count(self) -> int:
        return self._dropped_entry_count

    @property
    def is_empty(self) -> bool:
        return not self._entries and self._dropped_entry_count == 0

    def __len__(self) -> int:
        return len(self._entries)

    def record_decision(
        self,
        *,
        verb: str,
        candidate_ref: str | None = None,
        actor_id: str | None = None,
        target_id: str | None = None,
    ) -> None:
        """Append one decision with an unresolved outcome.

        The caller must invoke ``record_last_result`` before the next
        ``record_decision`` so each entry carries its resolved success
        state. An unresolved (pending) entry is still a valid render
        target — it just shows ``→ pending``.
        """

        turn_index = self._entries[-1].turn_index + 1 if self._entries else 1
        self._entries.append(
            ActionTrajectoryEntry(
                turn_index=turn_index,
                actor_id=actor_id,
                candidate_ref=candidate_ref,
                verb=verb,
                target_id=target_id,
                success=None,
                description=None,
            )
        )
        self._trim()

    def record_last_result(
        self,
        *,
        success: bool,
        description: str | None = None,
    ) -> None:
        """Attach a resolved outcome to the most recently recorded decision."""

        if not self._entries:
            return
        last = self._entries[-1]
        self._entries[-1] = ActionTrajectoryEntry(
            turn_index=last.turn_index,
            actor_id=last.actor_id,
            candidate_ref=last.candidate_ref,
            verb=last.verb,
            target_id=last.target_id,
            success=success,
            description=description,
        )

    def _trim(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)
            self._dropped_entry_count += 1

    def format_trajectory_block(self) -> str:
        """Render the trajectory as a compact ``<trajectory>`` block.

        Returns the empty string when nothing has been recorded yet, so
        callers can simply prepend the result to the current user
        message when non-empty.
        """

        if self.is_empty:
            return ""
        parts: list[str] = ["<trajectory>"]
        if self._dropped_entry_count:
            parts.append(
                f"(... {self._dropped_entry_count} earlier step"
                f"{'' if self._dropped_entry_count == 1 else 's'} elided ...)"
            )
        for entry in self._entries:
            parts.append(entry.format_line())
        parts.append("</trajectory>")
        return "\n".join(parts)

    def format_turn_feedback_block(self) -> str:
        """Render the most recent resolved action result for the next turn."""

        if not self._entries:
            return ""
        entry = self._entries[-1]
        if entry.success is None:
            return ""

        result = "succeeded" if entry.success else "failed"
        parts = ["<turn_feedback>"]
        action_bits: list[str] = []
        if entry.candidate_ref:
            action_bits.append(f"candidate_ref={entry.candidate_ref}")
        if entry.actor_id:
            action_bits.append(f"actor_id={entry.actor_id}")
        action_bits.append(f"verb={entry.verb}")
        if entry.target_id:
            action_bits.append(f"target_id={entry.target_id}")
        parts.append("Previous action: " + " ".join(action_bits))
        parts.append(f"Result: {result}")
        if entry.description:
            clean = " ".join(entry.description.split())
            parts.append(f"Details: {clean}")
        parts.append("</turn_feedback>")
        return "\n".join(parts)


def prepend_trajectory_block(
    user_msg: dict[str, Any],
    trajectory: ActionTrajectory,
) -> dict[str, Any]:
    """Return a new user message with the trajectory block prepended.

    The helper is side-effect free and returns a fresh dict so the
    original ``user_msg`` (which may be logged into the step audit by
    reference) is not mutated. When the trajectory is empty the message
    is returned unchanged; when its ``content`` is not a plain string
    (e.g. a multimodal content list) the message is also returned
    unchanged, because the block prefix only makes sense inline with
    plain-text turns.
    """

    block = trajectory.format_trajectory_block()
    if not block:
        return user_msg
    existing_content = user_msg.get("content")
    if not isinstance(existing_content, str):
        return user_msg
    return {
        **user_msg,
        "content": f"{block}\n\n{existing_content}",
    }


def prepend_interactive_history_blocks(
    user_msg: dict[str, Any],
    trajectory: ActionTrajectory,
) -> dict[str, Any]:
    """Prepend simulator feedback and compact trajectory to an interactive prompt."""

    blocks = [
        block
        for block in (
            trajectory.format_turn_feedback_block(),
            trajectory.format_trajectory_block(),
        )
        if block
    ]
    if not blocks:
        return user_msg
    existing_content = user_msg.get("content")
    if not isinstance(existing_content, str):
        return user_msg
    return {
        **user_msg,
        "content": "\n\n".join([*blocks, existing_content]),
    }


__all__ = [
    "ActionTrajectory",
    "ActionTrajectoryEntry",
    "prepend_interactive_history_blocks",
    "prepend_trajectory_block",
]
