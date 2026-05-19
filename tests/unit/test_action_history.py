"""Tests for the compact action-trajectory history used by interactive agents."""

from __future__ import annotations

import pytest

from labverse.agent.action_history import ActionTrajectory, ActionTrajectoryEntry


def test_empty_trajectory_renders_empty_string() -> None:
    trace = ActionTrajectory()
    assert trace.is_empty
    assert len(trace) == 0
    assert trace.format_trajectory_block() == ""


def test_single_decision_without_result_renders_as_pending() -> None:
    trace = ActionTrajectory()
    trace.record_decision(
        verb="don_ppe",
        candidate_ref="ppe_1",
        actor_id="tech_1",
    )

    block = trace.format_trajectory_block()

    assert "<trajectory>" in block
    assert "</trajectory>" in block
    assert "1." in block
    assert "[ppe_1]" in block
    assert "actor=tech_1" in block
    assert "verb=don_ppe" in block
    assert "→ pending" in block


def test_single_decision_with_result_renders_outcome_and_description() -> None:
    trace = ActionTrajectory()
    trace.record_decision(
        verb="move_to",
        candidate_ref="move_1",
        actor_id="tech_1",
        target_id="wet_lab",
    )
    trace.record_last_result(success=True, description="Moved from office to wet_lab")

    block = trace.format_trajectory_block()

    assert "verb=move_to" in block
    assert "target_id=wet_lab" in block
    assert "→ ok" in block
    assert "Moved from office to wet_lab" in block


def test_failed_result_marks_entry_as_failed() -> None:
    trace = ActionTrajectory()
    trace.record_decision(verb="start_equipment", candidate_ref="equip_1")
    trace.record_last_result(success=False, description="missing PPE")

    block = trace.format_trajectory_block()

    assert "→ failed" in block
    assert "missing PPE" in block


def test_record_last_result_on_empty_trace_is_noop() -> None:
    trace = ActionTrajectory()
    trace.record_last_result(success=True, description="irrelevant")
    assert trace.is_empty
    assert trace.format_trajectory_block() == ""


def test_multiple_decisions_emit_sequential_turn_indices() -> None:
    trace = ActionTrajectory()
    trace.record_decision(verb="don_ppe", candidate_ref="ppe_1", actor_id="tech_1")
    trace.record_last_result(success=True, description="ok")
    trace.record_decision(verb="move_to", candidate_ref="move_1", actor_id="tech_1")
    trace.record_last_result(success=True, description="ok")
    trace.record_decision(verb="start_equipment", candidate_ref="equip_1", actor_id="tech_1")

    lines = trace.format_trajectory_block().splitlines()
    decision_lines = [line for line in lines if line and line[0].isdigit()]
    assert len(decision_lines) == 3
    assert decision_lines[0].startswith("1.")
    assert decision_lines[1].startswith("2.")
    assert decision_lines[2].startswith("3.")
    assert "→ pending" in decision_lines[2]


def test_trim_elides_oldest_entries_and_reports_drop_count() -> None:
    trace = ActionTrajectory(max_entries=2)
    for index in range(4):
        trace.record_decision(
            verb="wait",
            candidate_ref=f"wait_{index + 1}",
            actor_id="tech_1",
        )
        trace.record_last_result(success=True, description=None)

    assert len(trace) == 2
    assert trace.dropped_entry_count == 2
    block = trace.format_trajectory_block()
    assert "2 earlier steps elided" in block
    # Only the two most recent entries remain rendered; the oldest two
    # are represented by the elision marker only.
    assert "wait_1" not in block
    assert "wait_2" not in block
    assert "wait_3" in block
    assert "wait_4" in block
    # And their original turn indices are preserved — dropped entries
    # do not renumber the survivors.
    assert "3." in block
    assert "4." in block


def test_single_dropped_entry_uses_singular_noun() -> None:
    trace = ActionTrajectory(max_entries=1)
    trace.record_decision(verb="wait", candidate_ref="wait_1")
    trace.record_decision(verb="wait", candidate_ref="wait_2")

    assert trace.dropped_entry_count == 1
    block = trace.format_trajectory_block()
    assert "1 earlier step elided" in block


def test_description_is_collapsed_and_truncated() -> None:
    trace = ActionTrajectory()
    trace.record_decision(verb="log_observation", candidate_ref="obs_1")
    trace.record_last_result(
        success=True,
        description="line one\nline two   with   runs   of   spaces  " + ("x" * 200),
    )

    line = trace.entries[0].format_line()
    assert "\n" not in line
    # Multiple whitespace runs are collapsed to single spaces.
    assert "line one line two" in line
    assert "with runs of spaces" in line
    # And the very long tail is truncated to a sensible length.
    assert "..." in line
    assert len(line) < 300


def test_trajectory_entry_is_frozen() -> None:
    entry = ActionTrajectoryEntry(
        turn_index=1,
        actor_id="tech_1",
        candidate_ref="ppe_1",
        verb="don_ppe",
        target_id=None,
        success=True,
        description="ok",
    )
    with pytest.raises(AttributeError):
        entry.success = False  # type: ignore[misc]


def test_max_entries_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ActionTrajectory(max_entries=0)
    with pytest.raises(ValueError):
        ActionTrajectory(max_entries=-3)


def test_sequential_turn_indices_include_dropped_entries() -> None:
    trace = ActionTrajectory(max_entries=2)
    trace.record_decision(verb="wait", candidate_ref="wait_1")
    trace.record_decision(verb="wait", candidate_ref="wait_2")
    trace.record_decision(verb="wait", candidate_ref="wait_3")

    # Turn counter never reverses even as older entries are elided,
    # so rendered turn indices remain monotonic and match the agent's
    # real step count.
    entries = trace.entries
    assert [entry.turn_index for entry in entries] == [2, 3]
