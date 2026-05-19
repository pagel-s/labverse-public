#!/usr/bin/env python3
"""Generate compact camera-ready LabVerse benchmark figures."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from plot_paper_main_results import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    FAILURE_CATEGORIES,
    GRID,
    INK,
    MISSING,
    MODEL_LABELS,
    MODEL_ORDER,
    MUTED,
    SENSITIVITY_BUNDLES,
    _failure_category,
    _load_results,
    _mean,
    _ordered_conditions,
    _ordered_models,
    _save,
    _scenario_order,
    _set_theme,
    _success_summary,
    _write_csv,
)


SUCCESS = "#4f8d86"
FAILURE = "#e9dccd"
RANK_GRAY = "#9aa0a6"
ROW_SUMMARY = "#5f6c73"

MODEL_COLORS = {
    "ollama/gemma4:31b": "#4f8d86",
    "ollama/gemma4:26b": "#6f8fb3",
    "ollama/qwen3.5:27b": "#c59b4a",
    "ollama/mistral-small3.2:24b": "#9b6b62",
}

FAILURE_COLORS = {
    "Behavioral/task failure": "#c99b55",
    "Format/argument failure": "#99a16b",
    "Grounding/interface failure": "#7896b1",
    "Provider/runner failure": "#a4746a",
}

CONDITION_COMPACT = {
    "local_structured_interactive_grounded": "Local-structured",
    "local_narrative_interactive_grounded": "Local-narrative",
    "global_structured_interactive_grounded": "Global-structured",
    "global_one_shot_planner": "One-shot",
}


def _camera_theme() -> None:
    _set_theme()
    mpl.rcParams.update(
        {
            "font.size": 6.9,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.removeprefix("ollama/"))


def _model_short(model: str) -> str:
    return {
        "ollama/gemma4:31b": "Gemma 31B",
        "ollama/gemma4:26b": "Gemma 26B",
        "ollama/qwen3.5:27b": "Qwen 27B",
        "ollama/mistral-small3.2:24b": "Mistral 24B",
    }.get(model, _model_label(model))


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _condition_compact(condition: str) -> str:
    return CONDITION_COMPACT.get(condition, _condition_label(condition))


def _scenario_labels(scenarios: list[str]) -> dict[str, str]:
    return {scenario: f"S{index}" for index, scenario in enumerate(scenarios, start=1)}


def _scenario_difficulty_order(rows: list[dict[str, Any]], scenarios: list[str]) -> list[str]:
    rates: dict[str, float] = {}
    for scenario in scenarios:
        items = [row for row in rows if row["scenario"] == scenario]
        rates[scenario] = _mean([1.0 if item.get("success") else 0.0 for item in items])
    return sorted(scenarios, key=lambda scenario: (rates[scenario], scenarios.index(scenario)))


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _cell_records(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
) -> list[dict[str, Any]]:
    summary = _success_summary(rows, models=models, conditions=conditions)
    records: list[dict[str, Any]] = []
    for model in models:
        for condition in conditions:
            item = summary[(model, condition)]
            successes = int(item["successes"])
            n = int(item["n"])
            low, high = _wilson_interval(successes, n)
            records.append(
                {
                    "model": model,
                    "planner": condition,
                    "successes": successes,
                    "n": n,
                    "success_rate": item["success_rate"],
                    "completion_ratio": item["completion_ratio"],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return records


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )
    ax.text(
        0.05,
        1.035,
        title,
        transform=ax.transAxes,
        fontsize=7.1,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )


def _draw_success_matrix(
    ax: plt.Axes,
    records: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
) -> None:
    value_by_key = {(row["model"], row["planner"]): row for row in records}
    matrix = np.full((len(models), len(conditions)), np.nan)
    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            matrix[i, j] = value_by_key[(model, condition)]["success_rate"]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "camera_success",
        ["#f8f4ed", "#b7ccc5", "#4f8d86"],
    )
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(
        [_condition_compact(condition) for condition in conditions], rotation=28, ha="right"
    )
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_model_short(model) for model in models])
    ax.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, model in enumerate(models):
        best = np.nanmax(matrix[i])
        for j, condition in enumerate(conditions):
            record = value_by_key[(model, condition)]
            rate = record["success_rate"]
            color = "white" if rate >= 0.50 else INK
            ax.text(
                j,
                i - 0.06,
                f"{rate * 100:.0f}",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color=color,
            )
            ax.text(
                j,
                i + 0.22,
                f"{record['successes']}/{record['n']}",
                ha="center",
                va="center",
                fontsize=5.4,
                color=color,
            )
            if np.isclose(rate, best):
                ax.add_patch(
                    Rectangle(
                        (j - 0.46, i - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor=INK,
                        linewidth=0.9,
                    )
                )


def _draw_leaderboard(ax: plt.Axes, records: list[dict[str, Any]]) -> None:
    ranked = sorted(
        records,
        key=lambda item: (item["success_rate"], item["completion_ratio"]),
        reverse=True,
    )
    y_positions = np.arange(len(ranked))[::-1]
    for y, record in zip(y_positions, ranked, strict=True):
        rate = record["success_rate"] * 100
        low = record["ci_low"] * 100
        high = record["ci_high"] * 100
        color = MODEL_COLORS.get(record["model"], RANK_GRAY)
        ax.plot([low, high], [y, y], color=color, linewidth=1.2, alpha=0.72)
        ax.scatter(rate, y, s=18, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(
            2,
            y,
            f"{_model_short(record['model'])} / {_condition_compact(record['planner'])}",
            va="center",
            ha="left",
            fontsize=5.8,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88},
        )
        ax.text(99, y, f"{rate:.0f}", va="center", ha="right", fontsize=5.8, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(ranked) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Success rate, Wilson 95% interval (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def _draw_scenario_difficulty(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    labels: dict[str, str],
) -> None:
    ordered = _scenario_difficulty_order(rows, scenarios)
    values = []
    for scenario in ordered:
        items = [row for row in rows if row["scenario"] == scenario]
        values.append(_mean([1.0 if item.get("success") else 0.0 for item in items]) * 100)

    colors = ["#a4746a" if value < 25 else "#c59b4a" if value < 50 else SUCCESS for value in values]
    ax.bar(np.arange(len(ordered)), values, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels([labels[scenario] for scenario in ordered])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Success\nacross systems")
    ax.set_xlabel("Scenarios ordered by aggregate success (harder to easier)", labelpad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)


def plot_camera_ready_main(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    scenarios: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    records = _cell_records(rows, models=models, conditions=conditions)
    labels = _scenario_labels(scenarios)

    _write_csv(
        output_dir / "camera_ready_main_benchmark.csv",
        [
            {
                "bundle": bundle_name,
                "model": item["model"],
                "planner": item["planner"],
                "successes": item["successes"],
                "n": item["n"],
                "success_rate": round(item["success_rate"], 6),
                "ci_low": round(item["ci_low"], 6),
                "ci_high": round(item["ci_high"], 6),
                "mean_completion_ratio": round(item["completion_ratio"], 6),
            }
            for item in records
        ],
    )

    fig = plt.figure(figsize=(7.2, 4.95))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.36],
        height_ratios=[3.2, 1.05],
        left=0.08,
        right=0.985,
        bottom=0.12,
        top=0.84,
        hspace=0.62,
        wspace=0.24,
    )
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])
    ax_scenario = fig.add_subplot(grid[1, :])

    _draw_success_matrix(ax_matrix, records, models=models, conditions=conditions)
    _draw_leaderboard(ax_rank, records)
    _draw_scenario_difficulty(ax_scenario, rows, scenarios=scenarios, labels=labels)

    _panel_label(ax_matrix, "a", "model x planner success")
    _panel_label(ax_rank, "b", "ranked model-planner cells")
    _panel_label(ax_scenario, "c", "scenario difficulty")

    fig.text(
        0.08,
        0.962,
        "LabVerse benchmark: local models under planner-interface conditions",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.928,
        f"Canonical bundle: {bundle_name}; 14 scenarios per cell.",
        fontsize=6.6,
        color=MUTED,
    )
    _save(fig, output_dir, "camera_ready_main_benchmark", formats, dpi)


def plot_camera_ready_scenario_matrix(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    scenarios: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    labels = _scenario_labels(scenarios)
    ordered_scenarios = _scenario_difficulty_order(rows, scenarios)
    by_key = {(row["model"], row["condition"], row["scenario"]): row for row in rows}
    row_keys = [(model, condition) for model in models for condition in conditions]
    matrix = np.full((len(row_keys), len(ordered_scenarios)), np.nan)
    source_rows: list[dict[str, Any]] = []

    for row_index, (model, condition) in enumerate(row_keys):
        for scenario_index, scenario in enumerate(ordered_scenarios):
            row = by_key.get((model, condition, scenario))
            if row is None:
                continue
            value = 1.0 if row.get("success") else 0.0
            matrix[row_index, scenario_index] = value
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "planner": condition,
                    "scenario_label": labels[scenario],
                    "scenario": scenario,
                    "success": int(value),
                    "error_taxonomy": row.get("error_taxonomy") or "none",
                    "completion_ratio": round(float(row.get("completion_ratio", 0.0)), 6),
                }
            )
    _write_csv(output_dir / "camera_ready_scenario_matrix.csv", source_rows)

    row_success = np.nanmean(matrix, axis=1) * 100
    col_success = np.nanmean(matrix, axis=0) * 100
    display = np.where(np.isnan(matrix), 2.0, matrix)
    cmap = mpl.colors.ListedColormap([FAILURE, SUCCESS, MISSING])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig = plt.figure(figsize=(7.2, 6.35))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.2, 0.9],
        height_ratios=[0.72, 5.25],
        left=0.17,
        right=0.97,
        bottom=0.1,
        top=0.88,
        hspace=0.05,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)

    ax.imshow(display, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(ordered_scenarios)))
    ax.set_xticklabels([labels[scenario] for scenario in ordered_scenarios], rotation=0)
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([_condition_compact(condition) for _, condition in row_keys])
    ax.set_xticks(np.arange(-0.5, len(ordered_scenarios), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for group_index, model in enumerate(models):
        start = group_index * len(conditions)
        end = start + len(conditions) - 1
        ax.axhline(end + 0.5, color="white", linewidth=2.1)
        ax.text(
            -1.6,
            start + (len(conditions) - 1) / 2,
            _model_short(model),
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=MODEL_COLORS.get(model, INK),
            clip_on=False,
        )

    ax_top.bar(np.arange(len(ordered_scenarios)), col_success, color=ROW_SUMMARY, width=0.72)
    ax_top.set_ylim(0, 100)
    ax_top.set_yticks([0, 50, 100])
    ax_top.set_ylabel("Column\nsuccess")
    ax_top.tick_params(axis="x", labelbottom=False, length=0)
    ax_top.grid(axis="y", color=GRID, linewidth=0.6)
    ax_top.set_axisbelow(True)

    ax_right.barh(np.arange(len(row_keys)), row_success, color=ROW_SUMMARY, height=0.65)
    ax_right.set_xlim(0, 100)
    ax_right.set_xticks([0, 50, 100])
    ax_right.set_xlabel("Row\nsuccess")
    ax_right.tick_params(axis="y", labelleft=False, length=0)
    ax_right.grid(axis="x", color=GRID, linewidth=0.6)
    ax_right.set_axisbelow(True)

    fig.text(0.17, 0.955, "Scenario response matrix", fontsize=9.2, fontweight="bold", color=INK)
    fig.text(
        0.17,
        0.917,
        "Rows are model-planner pairs; columns are scenarios ordered from hardest to easiest.",
        fontsize=6.6,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(facecolor=SUCCESS, edgecolor="none", label="success"),
            Patch(facecolor=FAILURE, edgecolor="none", label="failure"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.97, 0.955),
        frameon=False,
        ncol=2,
    )
    _save(fig, output_dir, "camera_ready_scenario_matrix", formats, dpi)


def plot_camera_ready_failure_taxonomy(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    failure_categories = [category for category in FAILURE_CATEGORIES if category != "Success"]
    row_keys = [(model, condition) for model in models for condition in conditions]
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: dict[tuple[str, str], int] = Counter()
    failures: dict[tuple[str, str], int] = Counter()

    for row in rows:
        key = (row["model"], row["condition"])
        totals[key] += 1
        category = _failure_category(row)
        if category != "Success":
            failures[key] += 1
            buckets[key][category] += 1

    source_rows: list[dict[str, Any]] = []
    for model, condition in row_keys:
        key = (model, condition)
        for category in failure_categories:
            count = buckets[key][category]
            denom = failures[key]
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "planner": condition,
                    "category": category,
                    "failed_runs": denom,
                    "count": count,
                    "share_of_failures": round(count / denom if denom else 0.0, 6),
                    "share_of_all_runs": round(count / totals[key] if totals[key] else 0.0, 6),
                }
            )
    _write_csv(output_dir / "camera_ready_failure_taxonomy.csv", source_rows)

    y_positions = np.arange(len(row_keys))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.31, right=0.965, bottom=0.13, top=0.83)

    for y, (model, condition) in zip(y_positions, row_keys, strict=True):
        key = (model, condition)
        fail_count = failures[key]
        left = 0.0
        if fail_count == 0:
            ax.scatter(0, y, s=16, color=SUCCESS, zorder=3)
        for category in failure_categories:
            width = buckets[key][category] / fail_count * 100 if fail_count else 0.0
            ax.barh(
                y,
                width,
                left=left,
                height=0.68,
                color=FAILURE_COLORS[category],
                edgecolor="white",
                linewidth=0.55,
            )
            if width >= 18:
                text_color = "white" if category != "Behavioral/task failure" else INK
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    fontweight="bold",
                    color=text_color,
                )
            left += width
        ax.text(
            103,
            y,
            f"n={fail_count}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=MUTED,
            clip_on=False,
        )

    ax.set_xlim(0, 100)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_condition_compact(condition) for _, condition in row_keys])
    ax.set_xlabel("Composition of failed runs (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for group_index, model in enumerate(models):
        start = y_positions[group_index * len(conditions)]
        end = y_positions[group_index * len(conditions) + len(conditions) - 1]
        mid = (start + end) / 2
        ax.text(
            -28,
            mid,
            _model_short(model),
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=MODEL_COLORS.get(model, INK),
            clip_on=False,
        )
        ax.axhline(end - 0.5, color="#efede5", linewidth=0.9, zorder=-1)

    handles = [
        Patch(facecolor=FAILURE_COLORS[category], edgecolor="none", label=category)
        for category in failure_categories
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.31, 0.91), frameon=False, ncol=2
    )
    fig.text(
        0.31,
        0.965,
        "Failure taxonomy among failed runs",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.31,
        0.928,
        "Success rate is not re-plotted here; the denominator labels show failed-run counts.",
        fontsize=6.6,
        color=MUTED,
    )
    _save(fig, output_dir, "camera_ready_failure_taxonomy", formats, dpi)


def plot_camera_ready_sensitivity(
    *,
    bundle_rows: dict[str, list[dict[str, Any]]],
    bundle_names: dict[str, str],
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    bundle_order = [key for key, _ in SENSITIVITY_BUNDLES if key in bundle_rows]
    summaries = {
        key: _success_summary(bundle_rows[key], models=models, conditions=conditions)
        for key in bundle_order
    }
    source_rows: list[dict[str, Any]] = []
    for key in bundle_order:
        for model in models:
            for condition in conditions:
                item = summaries[key][(model, condition)]
                source_rows.append(
                    {
                        "bundle_key": key,
                        "bundle": bundle_names[key],
                        "model": model,
                        "planner": condition,
                        "success_rate": round(item["success_rate"], 6),
                        "successes": int(item["successes"]),
                        "n": int(item["n"]),
                    }
                )
    _write_csv(output_dir / "camera_ready_sensitivity.csv", source_rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.75),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.08,
            "right": 0.985,
            "bottom": 0.15,
            "top": 0.82,
            "hspace": 0.28,
            "wspace": 0.14,
        },
    )
    x = np.arange(len(bundle_order))
    x_labels = [dict(SENSITIVITY_BUNDLES)[key] for key in bundle_order]

    for ax, condition in zip(axes.flat, conditions, strict=True):
        for model in models:
            values = np.array(
                [summaries[key][(model, condition)]["success_rate"] * 100 for key in bundle_order]
            )
            ax.plot(
                x,
                values,
                color=MODEL_COLORS.get(model, RANK_GRAY),
                linewidth=1.35,
                marker="o",
                markersize=3.6,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
        ax.set_title(_condition_label(condition), loc="left", fontsize=7.4, fontweight="bold")
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)

    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success (%)")

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS.get(model, RANK_GRAY),
            marker="o",
            linewidth=1.35,
            label=_model_short(model),
        )
        for model in models
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.91), ncol=4, frameon=False
    )
    fig.text(0.08, 0.965, "Initial-state sensitivity", fontsize=9.2, fontweight="bold", color=INK)
    fig.text(
        0.08,
        0.928,
        "Each point aggregates 14 scenarios; perturbation 2 is the canonical headline bundle.",
        fontsize=6.6,
        color=MUTED,
    )
    _save(fig, output_dir, "camera_ready_initial_sensitivity", formats, dpi)


def build_camera_ready_figures(
    *,
    canonical_bundle: Path,
    reference_bundle: Path,
    perturbation_1_bundle: Path,
    perturbation_2_bundle: Path,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    _camera_theme()
    rows = _load_results(canonical_bundle)
    scenarios = _scenario_order(canonical_bundle)
    models = [model for model in MODEL_ORDER if model in {row["model"] for row in rows}]
    models.extend(model for model in _ordered_models(rows) if model not in models)
    conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in {row["condition"] for row in rows}
    ]
    conditions.extend(
        condition for condition in _ordered_conditions(rows) if condition not in conditions
    )

    plot_camera_ready_main(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        scenarios=scenarios,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_camera_ready_scenario_matrix(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        scenarios=scenarios,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_camera_ready_failure_taxonomy(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )

    bundle_rows = {
        "reference": _load_results(reference_bundle),
        "perturbation_1": _load_results(perturbation_1_bundle),
        "perturbation_2": _load_results(perturbation_2_bundle),
    }
    bundle_names = {
        "reference": reference_bundle.name,
        "perturbation_1": perturbation_1_bundle.name,
        "perturbation_2": perturbation_2_bundle.name,
    }
    plot_camera_ready_sensitivity(
        bundle_rows=bundle_rows,
        bundle_names=bundle_names,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact camera-ready LabVerse benchmark figures."
    )
    parser.add_argument(
        "--canonical-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-2"),
    )
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-reference"),
    )
    parser.add_argument(
        "--perturbation-1-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-1"),
    )
    parser.add_argument(
        "--perturbation-2-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-2"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"]
    )
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    build_camera_ready_figures(
        canonical_bundle=args.canonical_bundle,
        reference_bundle=args.reference_bundle,
        perturbation_1_bundle=args.perturbation_1_bundle,
        perturbation_2_bundle=args.perturbation_2_bundle,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
