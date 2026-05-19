"""Paper-grade analysis helpers for frozen benchmark bundles."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, Field

from labverse.benchmarks.conditions import (
    PRIMARY_CONDITIONS,
    normalize_condition_name,
)
from labverse.paper.judgment import load_scenario_judgment_matrix
from labverse.paper.runner import PaperRunRecord, load_paper_run_records
from labverse.paper.validate import assert_valid_paper_bundle

# All unordered pairs of primary conditions. For the current paper that is 4C2 = 6
# comparisons (three interactive-vs-interactive plus the three
# interactive-vs-one-shot contrasts that test the paper's central hypothesis).
PAIRWISE_CONDITIONS = list(combinations(PRIMARY_CONDITIONS, 2))
PRIMARY_METRICS = [
    "success",
    "completion_ratio",
    "safety_violations",
    "blocked_actions",
    "steps_taken",
    "time_elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
]


class PairwiseMetricReport(BaseModel):
    comparison: str
    metric: str
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    sign_test_p: float
    holm_adjusted_p: float = 1.0
    wins_left: int = 0
    wins_right: int = 0
    ties: int = 0


class GroupedConditionSummary(BaseModel):
    grouping_axis: str
    grouping_value: str
    condition: str
    n: int
    success_rate: float
    mean_completion_ratio: float
    mean_safety_violations: float
    mean_blocked_actions: float
    mean_prompt_tokens: float
    mean_completion_tokens: float


class PaperAnalysisReport(BaseModel):
    total_runs: int
    observed_cells: int
    expected_cells: int
    complete_cells: int
    missing_cells: list[str] = Field(default_factory=list)
    by_condition: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    task_family_summaries: list[GroupedConditionSummary] = Field(default_factory=list)
    paper_role_summaries: list[GroupedConditionSummary] = Field(default_factory=list)
    pairwise: list[PairwiseMetricReport] = Field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_ci(
    values: list[float],
) -> tuple[float, float, float]:
    """Return deterministic normal-approximation CI for a paired mean delta."""

    if not values:
        return 0.0, 0.0, 0.0
    mean = _mean(values)
    if len(values) < 2:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    margin = 1.96 * standard_error
    return mean, mean - margin, mean + margin


def _sign_test(deltas: list[float]) -> tuple[float, int, int, int]:
    wins_left = sum(1 for delta in deltas if delta > 0)
    wins_right = sum(1 for delta in deltas if delta < 0)
    ties = sum(1 for delta in deltas if abs(delta) < 1e-12)
    trials = wins_left + wins_right
    if trials == 0:
        return 1.0, wins_left, wins_right, ties
    tail = max(wins_left, wins_right)
    numerator = sum(math.comb(trials, k) for k in range(tail, trials + 1))
    p_value = min(1.0, 2.0 * numerator / (2**trials))
    return p_value, wins_left, wins_right, ties


def _holm_adjust(reports: list[PairwiseMetricReport]) -> None:
    ordered = sorted(enumerate(reports), key=lambda pair: pair[1].sign_test_p)
    m = len(ordered)
    running_max = 0.0
    for rank, (index, report) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * report.sign_test_p)
        running_max = max(running_max, adjusted)
        reports[index].holm_adjusted_p = running_max


def _coverage(
    records: list[PaperRunRecord],
    *,
    conditions: list[str],
    models: list[str],
    scenarios: list[str],
    replications: int,
) -> tuple[int, int, int, list[str]]:
    observed: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in records:
        observed[(normalize_condition_name(record.condition), record.model, record.scenario)].add(
            record.replication
        )
    expected_model_map = {
        condition: (["scripted_oracle"] if condition == "scripted_oracle" else models)
        for condition in conditions
    }
    expected_cells = sum(
        len(expected_model_map[condition]) * len(scenarios) for condition in conditions
    )
    complete_cells = 0
    missing_cells: list[str] = []
    for condition in conditions:
        for model in expected_model_map[condition]:
            for scenario in scenarios:
                reps = observed.get((condition, model, scenario), set())
                if len(reps) >= replications:
                    complete_cells += 1
                if not reps:
                    missing_cells.append(f"{condition}/{model}/{scenario}")
    return len(observed), expected_cells, complete_cells, missing_cells


def _condition_summary(records: list[PaperRunRecord]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        grouped[normalize_condition_name(record.condition)].append(record)

    summary: dict[str, dict[str, float | int]] = {}
    for condition, items in sorted(grouped.items()):
        n = len(items)
        summary[condition] = {
            "n": n,
            "success_rate": sum(1 for item in items if item.success) / n,
            "mean_completion_ratio": sum(item.completion_ratio for item in items) / n,
            "mean_safety_violations": sum(item.safety_violations for item in items) / n,
            "mean_blocked_actions": sum(item.blocked_actions for item in items) / n,
            "mean_steps_taken": sum(item.steps_taken for item in items) / n,
            "mean_time_elapsed_seconds": sum(item.time_elapsed_seconds for item in items) / n,
            "mean_prompt_tokens": sum(item.prompt_tokens for item in items) / n,
            "mean_completion_tokens": sum(item.completion_tokens for item in items) / n,
            "mean_invalid_action_outputs": (sum(item.invalid_action_outputs for item in items) / n),
            "mean_action_output_retry_count": (
                sum(item.action_output_retry_count for item in items) / n
            ),
            "mean_provider_tool_call_fallback_count": (
                sum(item.provider_tool_call_fallback_count for item in items) / n
            ),
            "mean_invalid_tool_calls": sum(item.invalid_tool_calls for item in items) / n,
            "mean_tool_retry_count": sum(item.tool_retry_count for item in items) / n,
            "mean_invalid_model_outputs": (sum(item.invalid_model_outputs for item in items) / n),
            "mean_model_output_retry_count": (
                sum(item.model_output_retry_count for item in items) / n
            ),
            "mean_model_output_format_errors": (
                sum(item.model_output_format_errors for item in items) / n
            ),
            "mean_grounding_error_count": sum(item.grounding_error_count for item in items) / n,
            "mean_action_id_fallback_count": (
                sum(item.action_id_fallback_count for item in items) / n
            ),
            "mean_bare_verb_fallback_count": (
                sum(item.bare_verb_fallback_count for item in items) / n
            ),
            "mean_transport_normalization_count": (
                sum(item.transport_normalization_count for item in items) / n
            ),
            "mean_tool_name_alias_count": (sum(item.tool_name_alias_count for item in items) / n),
            "mean_flattened_parameter_count": (
                sum(item.flattened_parameter_count for item in items) / n
            ),
            "mean_json_repair_count": sum(item.json_repair_count for item in items) / n,
            "mean_stringified_parameters_repair_count": (
                sum(item.stringified_parameters_repair_count for item in items) / n
            ),
            "mean_semantic_fallback_count": (
                sum(item.semantic_fallback_count for item in items) / n
            ),
            "mean_provider_errors": sum(item.provider_errors for item in items) / n,
            "max_prompt_tokens_single_turn": max(
                item.max_prompt_tokens_single_turn for item in items
            ),
            "max_completion_tokens_single_turn": max(
                item.max_completion_tokens_single_turn for item in items
            ),
        }
    return summary


def _pairwise(records: list[PaperRunRecord]) -> list[PairwiseMetricReport]:
    index = {
        (
            normalize_condition_name(record.condition),
            record.model,
            record.scenario,
            record.replication,
        ): record
        for record in records
    }
    reports: list[PairwiseMetricReport] = []
    for left_condition, right_condition in PAIRWISE_CONDITIONS:
        common_keys = sorted(
            {
                (record.model, record.scenario, record.replication)
                for record in records
                if normalize_condition_name(record.condition) == left_condition
            }
            & {
                (record.model, record.scenario, record.replication)
                for record in records
                if normalize_condition_name(record.condition) == right_condition
            }
        )
        comparison = f"{left_condition} - {right_condition}"
        for metric in PRIMARY_METRICS:
            deltas = [
                float(getattr(index[(left_condition, model, scenario, replication)], metric))
                - float(getattr(index[(right_condition, model, scenario, replication)], metric))
                for model, scenario, replication in common_keys
            ]
            mean_delta, ci_low, ci_high = _mean_ci(deltas)
            p_value, wins_left, wins_right, ties = _sign_test(deltas)
            reports.append(
                PairwiseMetricReport(
                    comparison=comparison,
                    metric=metric,
                    n=len(deltas),
                    mean_delta=mean_delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    sign_test_p=p_value,
                    wins_left=wins_left,
                    wins_right=wins_right,
                    ties=ties,
                )
            )
    _holm_adjust(reports)
    return reports


def _group_condition_summaries(
    records: list[PaperRunRecord],
    *,
    grouping_axis: str,
    scenario_to_group: dict[str, str],
) -> list[GroupedConditionSummary]:
    grouped: dict[tuple[str, str], list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        group_value = scenario_to_group.get(record.scenario)
        if group_value is None:
            continue
        grouped[(group_value, normalize_condition_name(record.condition))].append(record)

    summaries: list[GroupedConditionSummary] = []
    for (group_value, condition), items in sorted(grouped.items()):
        n = len(items)
        summaries.append(
            GroupedConditionSummary(
                grouping_axis=grouping_axis,
                grouping_value=group_value,
                condition=condition,
                n=n,
                success_rate=sum(1 for item in items if item.success) / n,
                mean_completion_ratio=_mean([item.completion_ratio for item in items]),
                mean_safety_violations=_mean([float(item.safety_violations) for item in items]),
                mean_blocked_actions=_mean([float(item.blocked_actions) for item in items]),
                mean_prompt_tokens=_mean([float(item.prompt_tokens) for item in items]),
                mean_completion_tokens=_mean([float(item.completion_tokens) for item in items]),
            )
        )
    return summaries


def build_paper_analysis_report(
    records: list[PaperRunRecord],
    *,
    conditions: list[str],
    models: list[str],
    scenarios: list[str],
    replications: int,
) -> PaperAnalysisReport:
    conditions = [normalize_condition_name(condition) for condition in conditions]
    judgment_matrix = load_scenario_judgment_matrix()
    observed_cells, expected_cells, complete_cells, missing_cells = _coverage(
        records,
        conditions=conditions,
        models=models,
        scenarios=scenarios,
        replications=replications,
    )
    judgments = judgment_matrix.scenario_map()
    return PaperAnalysisReport(
        total_runs=len(records),
        observed_cells=observed_cells,
        expected_cells=expected_cells,
        complete_cells=complete_cells,
        missing_cells=missing_cells,
        by_condition=_condition_summary(records),
        task_family_summaries=_group_condition_summaries(
            records,
            grouping_axis="task_family",
            scenario_to_group={
                scenario_name: judgment.task_family for scenario_name, judgment in judgments.items()
            },
        ),
        paper_role_summaries=_group_condition_summaries(
            records,
            grouping_axis="paper_role",
            scenario_to_group={
                scenario_name: judgment.paper_role for scenario_name, judgment in judgments.items()
            },
        ),
        pairwise=_pairwise(records),
    )


def analyze_paper_bundle(bundle_dir: str | Path, *, output_dir: str | Path | None = None) -> Path:
    """Analyze a paper bundle and write reviewer-facing summary files."""

    bundle_path = Path(bundle_dir)
    assert_valid_paper_bundle(bundle_path)
    metadata = json.loads((bundle_path / "bundle_metadata.json").read_text())
    records = load_paper_run_records(bundle_path / "results.jsonl")
    analysis_dir = Path(output_dir) if output_dir is not None else bundle_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report = build_paper_analysis_report(
        records,
        conditions=list(metadata["conditions"]),
        models=list(metadata["models"]),
        scenarios=list(metadata["scenarios"]),
        replications=int(metadata["replications"]),
    )
    (analysis_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    with (analysis_dir / "condition_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "n",
                "success_rate",
                "mean_completion_ratio",
                "mean_safety_violations",
                "mean_blocked_actions",
                "mean_steps_taken",
                "mean_time_elapsed_seconds",
                "mean_prompt_tokens",
                "mean_completion_tokens",
                "mean_invalid_action_outputs",
                "mean_action_output_retry_count",
                "mean_provider_tool_call_fallback_count",
                "mean_invalid_tool_calls",
                "mean_tool_retry_count",
                "mean_invalid_model_outputs",
                "mean_model_output_retry_count",
                "mean_model_output_format_errors",
                "mean_grounding_error_count",
                "mean_action_id_fallback_count",
                "mean_bare_verb_fallback_count",
                "mean_transport_normalization_count",
                "mean_tool_name_alias_count",
                "mean_flattened_parameter_count",
                "mean_json_repair_count",
                "mean_stringified_parameters_repair_count",
                "mean_semantic_fallback_count",
                "mean_provider_errors",
                "max_prompt_tokens_single_turn",
                "max_completion_tokens_single_turn",
            ]
        )
        for condition, summary in sorted(report.by_condition.items()):
            writer.writerow(
                [
                    condition,
                    summary["n"],
                    summary["success_rate"],
                    summary["mean_completion_ratio"],
                    summary["mean_safety_violations"],
                    summary["mean_blocked_actions"],
                    summary["mean_steps_taken"],
                    summary["mean_time_elapsed_seconds"],
                    summary["mean_prompt_tokens"],
                    summary["mean_completion_tokens"],
                    summary["mean_invalid_action_outputs"],
                    summary["mean_action_output_retry_count"],
                    summary["mean_provider_tool_call_fallback_count"],
                    summary["mean_invalid_tool_calls"],
                    summary["mean_tool_retry_count"],
                    summary["mean_invalid_model_outputs"],
                    summary["mean_model_output_retry_count"],
                    summary["mean_model_output_format_errors"],
                    summary["mean_grounding_error_count"],
                    summary["mean_action_id_fallback_count"],
                    summary["mean_bare_verb_fallback_count"],
                    summary["mean_transport_normalization_count"],
                    summary["mean_tool_name_alias_count"],
                    summary["mean_flattened_parameter_count"],
                    summary["mean_json_repair_count"],
                    summary["mean_stringified_parameters_repair_count"],
                    summary["mean_semantic_fallback_count"],
                    summary["mean_provider_errors"],
                    summary["max_prompt_tokens_single_turn"],
                    summary["max_completion_tokens_single_turn"],
                ]
            )

    with (analysis_dir / "pairwise_deltas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "metric",
                "n",
                "mean_delta",
                "ci_low",
                "ci_high",
                "sign_test_p",
                "holm_adjusted_p",
                "wins_left",
                "wins_right",
                "ties",
            ]
        )
        for report_row in report.pairwise:
            writer.writerow(
                [
                    report_row.comparison,
                    report_row.metric,
                    report_row.n,
                    report_row.mean_delta,
                    report_row.ci_low,
                    report_row.ci_high,
                    report_row.sign_test_p,
                    report_row.holm_adjusted_p,
                    report_row.wins_left,
                    report_row.wins_right,
                    report_row.ties,
                ]
            )

    grouped_outputs: list[tuple[str, list[GroupedConditionSummary]]] = [
        ("task_family_summary.csv", list(report.task_family_summaries)),
        ("paper_role_summary.csv", list(report.paper_role_summaries)),
    ]
    for filename, summaries in grouped_outputs:
        with (analysis_dir / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "grouping_axis",
                    "grouping_value",
                    "condition",
                    "n",
                    "success_rate",
                    "mean_completion_ratio",
                    "mean_safety_violations",
                    "mean_blocked_actions",
                    "mean_prompt_tokens",
                    "mean_completion_tokens",
                ]
            )
            for group_row in summaries:
                writer.writerow(
                    [
                        group_row.grouping_axis,
                        group_row.grouping_value,
                        group_row.condition,
                        group_row.n,
                        group_row.success_rate,
                        group_row.mean_completion_ratio,
                        group_row.mean_safety_violations,
                        group_row.mean_blocked_actions,
                        group_row.mean_prompt_tokens,
                        group_row.mean_completion_tokens,
                    ]
                )

    return analysis_dir


__all__ = [
    "PAIRWISE_CONDITIONS",
    "PRIMARY_METRICS",
    "GroupedConditionSummary",
    "PaperAnalysisReport",
    "PairwiseMetricReport",
    "analyze_paper_bundle",
    "build_paper_analysis_report",
]
