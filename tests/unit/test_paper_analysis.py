"""Tests for frozen paper analysis helpers."""

from __future__ import annotations

from labverse.benchmarks.conditions import PRIMARY_CONDITIONS
from labverse.paper.analysis import (
    PAIRWISE_CONDITIONS,
    PRIMARY_METRICS,
    build_paper_analysis_report,
)
from labverse.paper.runner import PaperRunRecord


def _record(
    *,
    condition: str,
    model: str,
    scenario: str,
    replication: int,
    success: bool,
    completion_ratio: float,
) -> PaperRunRecord:
    return PaperRunRecord(
        bundle_id="bundle",
        manifest_id="manifest",
        manifest_version="1.0.0",
        manifest_hash="abc",
        git_commit_sha="deadbeef",
        git_dirty=False,
        profile_name="deterministic",
        temperature=0.0,
        max_tokens=1024,
        prompt_family_hash="prompt",
        scenario_source_hash="scenario",
        config_hash="config",
        condition=condition,
        model=model,
        provider="openai",
        scenario=scenario,
        replication=replication,
        order_index=replication,
        success=success,
        completion_ratio=completion_ratio,
        steps_taken=5,
        time_elapsed_seconds=30.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=1.0,
        prompt_tokens=100,
        completion_tokens=10,
        wall_time_seconds=1.2,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_paper_analysis_builds_pairwise_report_and_coverage() -> None:
    records = [
        _record(
            condition="local_structured_interactive_grounded",
            model="gpt-4o",
            scenario="pcr_setup",
            replication=1,
            success=True,
            completion_ratio=1.0,
        ),
        _record(
            condition="local_narrative_interactive_grounded",
            model="gpt-4o",
            scenario="pcr_setup",
            replication=1,
            success=False,
            completion_ratio=0.4,
        ),
        _record(
            condition="global_structured_interactive_grounded",
            model="gpt-4o",
            scenario="pcr_setup",
            replication=1,
            success=False,
            completion_ratio=0.2,
        ),
    ]

    report = build_paper_analysis_report(
        records,
        conditions=[
            "local_structured_interactive_grounded",
            "local_narrative_interactive_grounded",
            "global_structured_interactive_grounded",
        ],
        models=["gpt-4o"],
        scenarios=["pcr_setup"],
        replications=1,
    )

    assert report.total_runs == 3
    assert report.expected_cells == 3
    assert report.complete_cells == 3
    assert report.by_condition["local_structured_interactive_grounded"]["success_rate"] == 1.0
    assert "mean_efficiency" not in report.by_condition["local_structured_interactive_grounded"]
    assert any(
        row.comparison
        == "local_structured_interactive_grounded - global_structured_interactive_grounded"
        and row.metric == "completion_ratio"
        and row.mean_delta > 0
        for row in report.pairwise
    )
    assert any(
        row.grouping_axis == "task_family"
        and row.grouping_value == "anchors"
        and row.condition == "local_structured_interactive_grounded"
        for row in report.task_family_summaries
    )
    assert any(
        row.grouping_axis == "paper_role"
        and row.grouping_value == "anchor"
        and row.condition == "global_structured_interactive_grounded"
        for row in report.paper_role_summaries
    )


def test_paper_analysis_covers_all_four_primary_conditions() -> None:
    """Regression: analyse_paper_bundle must emit every primary condition and
    every unordered pair (4C2 = 6). Catches any regression that reintroduces a
    3-condition assumption into the analysis pipeline."""

    completion_by_condition = {
        "local_structured_interactive_grounded": 1.0,
        "local_narrative_interactive_grounded": 0.4,
        "global_structured_interactive_grounded": 0.2,
        "global_one_shot_planner": 0.1,
    }
    assert set(completion_by_condition) == set(PRIMARY_CONDITIONS)

    records = [
        _record(
            condition=condition,
            model="gpt-4o",
            scenario="pcr_setup",
            replication=1,
            success=completion_ratio >= 0.5,
            completion_ratio=completion_ratio,
        )
        for condition, completion_ratio in completion_by_condition.items()
    ]

    report = build_paper_analysis_report(
        records,
        conditions=list(PRIMARY_CONDITIONS),
        models=["gpt-4o"],
        scenarios=["pcr_setup"],
        replications=1,
    )

    # All four primary conditions must appear in the headline condition summary.
    for condition in PRIMARY_CONDITIONS:
        assert condition in report.by_condition, f"missing condition in by_condition: {condition}"

    # All 4C2 = 6 pairs must appear across every primary metric.
    assert len(PAIRWISE_CONDITIONS) == 6
    expected_pairwise_rows = len(PAIRWISE_CONDITIONS) * len(PRIMARY_METRICS)
    assert len(report.pairwise) == expected_pairwise_rows

    expected_comparisons = {f"{left} - {right}" for left, right in PAIRWISE_CONDITIONS}
    actual_comparisons = {row.comparison for row in report.pairwise}
    assert expected_comparisons == actual_comparisons

    # Every metric must be represented for each pair.
    for comparison in expected_comparisons:
        metrics_for_pair = {row.metric for row in report.pairwise if row.comparison == comparison}
        assert metrics_for_pair == set(PRIMARY_METRICS), (
            f"pair {comparison} missing metrics: {set(PRIMARY_METRICS) - metrics_for_pair}"
        )


def test_manuscript_blueprint_renders_all_four_primary_conditions() -> None:
    """Regression: the rendered writeup blueprint must name every primary
    condition and every pairwise completion delta. Catches any regression that
    reintroduces 3-condition prose into writeup.py."""

    from labverse.paper.writeup import (
        PaperWriteupPacket,
        _render_manuscript_blueprint,
    )

    headline_by_condition: dict[str, dict[str, float | int]] = {
        condition: {
            "success_rate": 0.5,
            "mean_completion_ratio": 0.5,
            "mean_safety_violations": 0.0,
            "mean_blocked_actions": 0.0,
            "mean_steps_taken": 5.0,
            "mean_time_elapsed_seconds": 30.0,
            "mean_prompt_tokens": 100,
            "mean_completion_tokens": 10,
        }
        for condition in PRIMARY_CONDITIONS
    }
    packet = PaperWriteupPacket(
        bundle_dir="/tmp/fake-bundle",
        analysis_dir="/tmp/fake-bundle/analysis",
        manifest_id="fake",
        profile_name="deterministic",
        replications=1,
        total_runs=len(PRIMARY_CONDITIONS),
        conditions=list(PRIMARY_CONDITIONS),
        models=["gpt-4o"],
        scenarios=["pcr_setup"],
        ready_for_main_results=True,
        headline_by_condition=headline_by_condition,
    )

    blueprint = _render_manuscript_blueprint(packet)

    # Every primary condition must be named in the blueprint prose.
    for condition in PRIMARY_CONDITIONS:
        assert condition in blueprint, f"blueprint must name condition {condition}"

    # Every pairwise completion-delta line must be present in label form.
    from labverse.paper.writeup import _condition_label

    for left, right in PAIRWISE_CONDITIONS:
        expected_line = f"{_condition_label(left)} vs {_condition_label(right)} completion delta"
        assert expected_line in blueprint, f"missing pairwise delta line: {expected_line}"

    # The legacy 3-condition hook text must not reappear verbatim.
    assert "three main paper conditions" not in blueprint
    # global_one_shot_planner must appear prominently (abstract + results + outline + plan).
    assert blueprint.count("global_one_shot_planner") >= 3
