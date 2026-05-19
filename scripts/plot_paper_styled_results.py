#!/usr/bin/env python3
"""Generate DeepSeek-inspired styled LabVerse benchmark and SI figures."""

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
from matplotlib.patches import Patch

from plot_paper_main_results import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    FAILURE_CATEGORIES,
    MODEL_LABELS,
    MODEL_ORDER,
    SENSITIVITY_BUNDLES,
    _failure_category,
    _load_results,
    _mean,
    _ordered_conditions,
    _ordered_models,
    _save,
    _scenario_order,
    _success_summary,
    _write_csv,
)


PAPER = "#fcfbf7"
INK = "#20242a"
MUTED = "#747b84"
GRID = "#e7e0d4"
HAIRLINE = "#c9c1b4"

MODEL_COLORS = {
    "ollama/gemma4:31b": "#4f8d86",
    "ollama/gemma4:26b": "#6f8fb3",
    "ollama/qwen3.5:27b": "#c59b4a",
    "ollama/mistral-small3.2:24b": "#9b6b62",
}

PLANNER_COLORS = {
    "local_structured_interactive_grounded": "#7f8790",
    "local_narrative_interactive_grounded": "#92b7ae",
    "global_structured_interactive_grounded": "#91a7bd",
    "global_one_shot_planner": "#c8a761",
}

FAILURE_COLORS = {
    "Behavioral/task failure": "#c99b55",
    "Format/argument failure": "#99a16b",
    "Grounding/interface failure": "#7896b1",
    "Provider/runner failure": "#a4746a",
}

DIFFICULTY_COLORS = {
    "low": "#a4746a",
    "mid": "#c59b4a",
    "high": "#4f8d86",
}

CONDITION_COMPACT = {
    "local_structured_interactive_grounded": "Local\nstruct.",
    "local_narrative_interactive_grounded": "Local\nnarr.",
    "global_structured_interactive_grounded": "Global\nstruct.",
    "global_one_shot_planner": "One-shot",
}

CONDITION_INLINE = {
    "local_structured_interactive_grounded": "Local structured",
    "local_narrative_interactive_grounded": "Local narrative",
    "global_structured_interactive_grounded": "Global structured",
    "global_one_shot_planner": "One-shot",
}


def _set_styled_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "axes.edgecolor": HAIRLINE,
            "xtick.color": "#343941",
            "ytick.color": "#343941",
            "axes.labelcolor": "#343941",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def _model_short(model: str) -> str:
    return {
        "ollama/gemma4:31b": "Gemma 31B",
        "ollama/gemma4:26b": "Gemma 26B",
        "ollama/qwen3.5:27b": "Qwen 27B",
        "ollama/mistral-small3.2:24b": "Mistral 24B",
    }.get(model, MODEL_LABELS.get(model, model.removeprefix("ollama/")))


def _condition_label(condition: str) -> str:
    return CONDITION_INLINE.get(
        condition, CONDITION_LABELS.get(condition, condition.replace("_", " "))
    )


def _condition_compact(condition: str) -> str:
    return CONDITION_COMPACT.get(condition, _condition_label(condition))


def _scenario_labels(scenarios: list[str]) -> dict[str, str]:
    return {scenario: f"S{index}" for index, scenario in enumerate(scenarios, start=1)}


def _metric_label(metric: str) -> str:
    return {
        "invalid_action_outputs": "Invalid\nactions",
        "action_output_retry_count": "Action\nretries",
        "model_output_format_errors": "Format\nerrors",
        "grounding_error_count": "Grounding\nerrors",
        "provider_errors": "Provider\nerrors",
        "json_repair_count": "JSON\nrepair",
        "stringified_parameters_repair_count": "Stringified\nparams",
        "semantic_fallback_count": "Semantic\nfallback",
        "transport_normalization_count": "Transport\nnorm.",
    }.get(metric, metric.replace("_", "\n"))


def _style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HAIRLINE)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(length=2.2, width=0.65)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.55)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def _figure_header(fig: plt.Figure, title: str, subtitle: str, *, x: float = 0.07) -> None:
    fig.text(x, 0.962, title, fontsize=9.6, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(x, 0.92, subtitle, fontsize=6.7, color=MUTED, ha="left", va="top")


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.052,
        1.045,
        title,
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def _model_handles(models: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS.get(model, "#777777"),
            marker="s",
            markersize=5,
            linestyle="none",
            label=_model_short(model),
        )
        for model in models
    ]


def _planner_handles(conditions: list[str]) -> list[Patch]:
    return [
        Patch(
            facecolor=PLANNER_COLORS.get(condition, "#777777"),
            edgecolor="none",
            label=_condition_label(condition),
        )
        for condition in conditions
    ]


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _success_records(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
) -> list[dict[str, Any]]:
    summary = _success_summary(rows, models=models, conditions=conditions)
    records: list[dict[str, Any]] = []
    for condition in conditions:
        for model in models:
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


def _scenario_difficulty_order(rows: list[dict[str, Any]], scenarios: list[str]) -> list[str]:
    rates: dict[str, float] = {}
    for scenario in scenarios:
        items = [row for row in rows if row["scenario"] == scenario]
        rates[scenario] = _mean([1.0 if item.get("success") else 0.0 for item in items])
    return sorted(scenarios, key=lambda scenario: (rates[scenario], scenarios.index(scenario)))


def _value_by_model_condition(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
    field: str,
    scale: float = 1.0,
) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["condition"])].append(float(row.get(field) or 0.0) * scale)
    return {
        (model, condition): _mean(buckets[(model, condition)])
        for model in models
        for condition in conditions
    }


def _counts_by_model(
    rows: list[dict[str, Any]], *, models: list[str], fields: list[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        for field in fields:
            out.append(
                {
                    "model": model,
                    "metric": field,
                    "total": sum(float(row.get(field) or 0.0) for row in model_rows),
                    "mean_per_run": _mean([float(row.get(field) or 0.0) for row in model_rows]),
                    "n_runs": len(model_rows),
                }
            )
    return out


def _draw_grouped_bars(
    ax: plt.Axes,
    values: dict[tuple[str, str], float],
    *,
    models: list[str],
    conditions: list[str],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    annotate: bool = False,
    yerr: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> None:
    x = np.arange(len(conditions))
    width = min(0.17, 0.72 / max(len(models), 1))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for model_index, model in enumerate(models):
        heights = [values[(model, condition)] for condition in conditions]
        errors = None
        if yerr is not None:
            lows = [
                values[(model, condition)] - yerr[(model, condition)][0] for condition in conditions
            ]
            highs = [
                yerr[(model, condition)][1] - values[(model, condition)] for condition in conditions
            ]
            errors = np.array([lows, highs])
        bars = ax.bar(
            x + offsets[model_index],
            heights,
            width=width * 0.9,
            color=MODEL_COLORS.get(model, "#777777"),
            edgecolor=PAPER,
            linewidth=0.6,
            yerr=errors,
            error_kw={"elinewidth": 0.65, "ecolor": "#4a5158", "capsize": 1.7, "capthick": 0.65},
        )
        if annotate:
            for bar, height in zip(bars, heights, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color=MUTED,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_condition_compact(condition) for condition in conditions])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style_axis(ax)


def plot_styled_main_success(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    records = _success_records(rows, models=models, conditions=conditions)
    _write_csv(
        output_dir / "styled_main_success_bars.csv",
        [
            {
                "bundle": bundle_name,
                "model": record["model"],
                "planner": record["planner"],
                "successes": record["successes"],
                "n": record["n"],
                "success_rate": round(record["success_rate"], 6),
                "ci_low": round(record["ci_low"], 6),
                "ci_high": round(record["ci_high"], 6),
                "mean_completion_ratio": round(record["completion_ratio"], 6),
            }
            for record in records
        ],
    )

    value = {(item["model"], item["planner"]): item["success_rate"] * 100 for item in records}
    yerr = {
        (item["model"], item["planner"]): (item["ci_low"] * 100, item["ci_high"] * 100)
        for item in records
    }
    model_means = {
        model: _mean([value[(model, condition)] for condition in conditions]) for model in models
    }
    planner_means = {
        condition: _mean([value[(model, condition)] for model in models])
        for condition in conditions
    }

    fig = plt.figure(figsize=(7.2, 4.55))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.1, 1.45],
        width_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.82,
        hspace=0.58,
        wspace=0.34,
    )
    ax_main = fig.add_subplot(grid[0, :])
    ax_model = fig.add_subplot(grid[1, 0])
    ax_planner = fig.add_subplot(grid[1, 1])

    _draw_grouped_bars(
        ax_main,
        value,
        models=models,
        conditions=conditions,
        ylabel="Success (%)",
        ylim=(0, 105),
        annotate=True,
        yerr=yerr,
    )
    ax_main.set_yticks([0, 25, 50, 75, 100])
    _panel_label(ax_main, "a", "success by planner interface and model")

    ranked_models = sorted(models, key=lambda model: model_means[model], reverse=True)
    ax_model.barh(
        np.arange(len(ranked_models)),
        [model_means[model] for model in ranked_models],
        color=[MODEL_COLORS.get(model, "#777777") for model in ranked_models],
        height=0.58,
    )
    ax_model.set_yticks(np.arange(len(ranked_models)))
    ax_model.set_yticklabels([_model_short(model) for model in ranked_models])
    ax_model.invert_yaxis()
    ax_model.set_xlim(0, 75)
    ax_model.set_xlabel("Mean success (%)")
    _style_axis(ax_model, xgrid=True, ygrid=False)
    _panel_label(ax_model, "b", "model average")

    ranked_conditions = sorted(
        conditions, key=lambda condition: planner_means[condition], reverse=True
    )
    ax_planner.barh(
        np.arange(len(ranked_conditions)),
        [planner_means[condition] for condition in ranked_conditions],
        color=[PLANNER_COLORS.get(condition, "#777777") for condition in ranked_conditions],
        height=0.58,
    )
    ax_planner.set_yticks(np.arange(len(ranked_conditions)))
    ax_planner.set_yticklabels([_condition_label(condition) for condition in ranked_conditions])
    ax_planner.invert_yaxis()
    ax_planner.set_xlim(0, 75)
    ax_planner.set_xlabel("Mean success (%)")
    _style_axis(ax_planner, xgrid=True, ygrid=False)
    _panel_label(ax_planner, "c", "planner average")

    fig.legend(
        handles=_model_handles(models),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.963),
        ncol=4,
        frameon=False,
    )
    _figure_header(
        fig,
        "LabVerse headline benchmark",
        (
            f"Canonical bundle: {bundle_name}; bars show success over 14 scenarios, "
            "with Wilson intervals."
        ),
    )
    _save(fig, output_dir, "styled_main_success_bars", formats, dpi)


def plot_styled_scenario_difficulty(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    scenarios: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    labels = _scenario_labels(scenarios)
    ordered = _scenario_difficulty_order(rows, scenarios)
    source_rows: list[dict[str, Any]] = []
    values: list[float] = []
    for scenario in ordered:
        items = [row for row in rows if row["scenario"] == scenario]
        successes = sum(1 for item in items if item.get("success"))
        rate = successes / len(items) if items else 0.0
        values.append(rate * 100)
        source_rows.append(
            {
                "bundle": bundle_name,
                "scenario_label": labels[scenario],
                "scenario": scenario,
                "successes": successes,
                "n": len(items),
                "success_rate": round(rate, 6),
            }
        )
    _write_csv(output_dir / "styled_scenario_difficulty_bars.csv", source_rows)

    colors = [
        DIFFICULTY_COLORS["low"]
        if value < 25
        else DIFFICULTY_COLORS["mid"]
        if value < 50
        else DIFFICULTY_COLORS["high"]
        for value in values
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.85))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.76)
    x = np.arange(len(ordered))
    bars = ax.bar(x, values, color=colors, width=0.68, edgecolor=PAPER, linewidth=0.6)
    for bar, value in zip(bars, values, strict=True):
        if value >= 45:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.5,
                f"{value:.0f}",
                ha="center",
                fontsize=5.4,
                color=MUTED,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[scenario] for scenario in ordered])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success across\nall systems (%)")
    ax.set_xlabel("Scenarios ordered by aggregate success")
    _style_axis(ax)
    _figure_header(
        fig,
        "Scenario difficulty profile",
        "Each bar aggregates all model-planner cells in the canonical bundle.",
    )
    _save(fig, output_dir, "styled_scenario_difficulty_bars", formats, dpi)


def plot_styled_failure_taxonomy(
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
    failures: dict[tuple[str, str], int] = Counter()
    totals: dict[tuple[str, str], int] = Counter()

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
    _write_csv(output_dir / "styled_failure_taxonomy_bars.csv", source_rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.25))
    fig.subplots_adjust(left=0.31, right=0.965, bottom=0.11, top=0.80)
    y_positions = np.arange(len(row_keys))[::-1]
    for y, (model, condition) in zip(y_positions, row_keys, strict=True):
        key = (model, condition)
        left = 0.0
        denom = failures[key]
        for category in failure_categories:
            width = buckets[key][category] / denom * 100 if denom else 0.0
            ax.barh(
                y,
                width,
                left=left,
                height=0.66,
                color=FAILURE_COLORS[category],
                edgecolor=PAPER,
                linewidth=0.55,
            )
            if width >= 18:
                text_color = PAPER if category != "Behavioral/task failure" else INK
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color=text_color,
                    fontweight="bold",
                )
            left += width
        ax.text(
            102.5, y, f"n={denom}", ha="left", va="center", fontsize=5.7, color=MUTED, clip_on=False
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_condition_label(condition) for _, condition in row_keys])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Composition of failed runs (%)")
    _style_axis(ax, xgrid=True, ygrid=False)
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
            color=MODEL_COLORS.get(model, INK),
            fontweight="bold",
            clip_on=False,
        )
        if group_index < len(models) - 1:
            ax.axhline(end - 0.5, color=GRID, linewidth=0.7, zorder=-1)

    handles = [
        Patch(facecolor=FAILURE_COLORS[category], edgecolor="none", label=category)
        for category in failure_categories
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.31, 0.895), frameon=False, ncol=2
    )
    _figure_header(
        fig,
        "Failure taxonomy",
        "Bars condition on failed runs; denominator labels report failed-run counts.",
        x=0.31,
    )
    _save(fig, output_dir, "styled_failure_taxonomy_bars", formats, dpi)


def plot_styled_sensitivity(
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
    bundle_label = dict(SENSITIVITY_BUNDLES)
    summaries = {
        key: _success_summary(bundle_rows[key], models=models, conditions=conditions)
        for key in bundle_order
    }

    source_rows: list[dict[str, Any]] = []
    for key in bundle_order:
        for condition in conditions:
            for model in models:
                item = summaries[key][(model, condition)]
                source_rows.append(
                    {
                        "bundle_key": key,
                        "bundle": bundle_names[key],
                        "planner": condition,
                        "model": model,
                        "successes": int(item["successes"]),
                        "n": int(item["n"]),
                        "success_rate": round(item["success_rate"], 6),
                    }
                )
    _write_csv(output_dir / "styled_sensitivity_bars.csv", source_rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.8),
        sharey=True,
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "bottom": 0.13,
            "top": 0.80,
            "hspace": 0.42,
            "wspace": 0.18,
        },
    )
    x = np.arange(len(bundle_order))
    width = min(0.17, 0.72 / max(len(models), 1))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for ax, condition in zip(axes.flat, conditions, strict=True):
        for model_index, model in enumerate(models):
            values = [
                summaries[key][(model, condition)]["success_rate"] * 100 for key in bundle_order
            ]
            ax.bar(
                x + offsets[model_index],
                values,
                width=width * 0.9,
                color=MODEL_COLORS.get(model, "#777777"),
                edgecolor=PAPER,
                linewidth=0.55,
            )
        ax.set_title(
            _condition_label(condition), loc="left", fontsize=7.5, fontweight="bold", color=INK
        )
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [bundle_label[key].replace("Perturbation ", "Pert. ") for key in bundle_order]
        )
        _style_axis(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("Success (%)")

    fig.legend(
        handles=_model_handles(models),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.912),
        ncol=4,
        frameon=False,
    )
    _figure_header(
        fig,
        "Initial-state sensitivity",
        "Same model/planner cells across the reference state and two perturbations.",
    )
    _save(fig, output_dir, "styled_sensitivity_bars", formats, dpi)


def plot_si_completion_efficiency(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    metrics = [
        ("completion_ratio", "Completion (%)", 100.0, (0, 105)),
        ("steps_taken", "Steps taken", 1.0, None),
        ("efficiency_score", "Efficiency score", 100.0, (0, 105)),
    ]
    source_rows: list[dict[str, Any]] = []
    values_by_metric: dict[str, dict[tuple[str, str], float]] = {}
    for field, _, scale, _ in metrics:
        values = _value_by_model_condition(
            rows, models=models, conditions=conditions, field=field, scale=scale
        )
        values_by_metric[field] = values
        for (model, condition), value in values.items():
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "metric": field,
                    "model": model,
                    "planner": condition,
                    "mean": round(value, 6),
                }
            )
    _write_csv(output_dir / "styled_si_completion_efficiency.csv", source_rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.05),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.21, "top": 0.74, "wspace": 0.34},
    )
    for ax, (field, ylabel, _, ylim) in zip(axes, metrics, strict=True):
        _draw_grouped_bars(
            ax,
            values_by_metric[field],
            models=models,
            conditions=conditions,
            ylabel=ylabel,
            ylim=ylim,
        )
        ax.set_title(ylabel, loc="left", fontsize=7.4, color=INK, fontweight="bold")

    fig.legend(
        handles=_model_handles(models),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.91),
        ncol=4,
        frameon=False,
    )
    _figure_header(
        fig,
        "SI: completion and efficiency",
        f"Canonical bundle: {bundle_name}; bars show cell means over scenarios.",
    )
    _save(fig, output_dir, "styled_si_completion_efficiency", formats, dpi)


def plot_si_reliability_counts(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    hard_fields = [
        "invalid_action_outputs",
        "action_output_retry_count",
        "model_output_format_errors",
        "grounding_error_count",
        "provider_errors",
    ]
    mediation_fields = [
        "json_repair_count",
        "stringified_parameters_repair_count",
        "semantic_fallback_count",
        "transport_normalization_count",
    ]
    source_rows = _counts_by_model(rows, models=models, fields=hard_fields + mediation_fields)
    for row in source_rows:
        row["bundle"] = bundle_name
    _write_csv(output_dir / "styled_si_reliability_counts.csv", source_rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.45),
        gridspec_kw={"left": 0.10, "right": 0.985, "bottom": 0.22, "top": 0.74, "wspace": 0.32},
    )

    def draw_metric_panel(
        ax: plt.Axes, fields: list[str], title: str, *, log_scale: bool = False
    ) -> None:
        x = np.arange(len(fields))
        width = min(0.17, 0.72 / max(len(models), 1))
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
        by_key = {(row["model"], row["metric"]): row["total"] for row in source_rows}
        for model_index, model in enumerate(models):
            values = [by_key[(model, field)] for field in fields]
            ax.bar(
                x + offsets[model_index],
                values,
                width=width * 0.9,
                color=MODEL_COLORS.get(model, "#777777"),
                edgecolor=PAPER,
                linewidth=0.55,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_metric_label(field) for field in fields], rotation=0)
        ax.set_ylabel("Total events")
        ax.set_title(title, loc="left", fontsize=7.4, color=INK, fontweight="bold")
        if log_scale:
            ax.set_yscale("symlog", linthresh=1)
        _style_axis(ax)

    draw_metric_panel(axes[0], hard_fields, "hard parser/grounding/provider events")
    draw_metric_panel(axes[1], mediation_fields, "repair and mediation events", log_scale=True)

    fig.legend(
        handles=_model_handles(models),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.91),
        ncol=4,
        frameon=False,
    )
    _figure_header(
        fig,
        "SI: output reliability and mediation",
        (
            "Counts are totals over all planner/scenario cells for each model; "
            "mediation is not necessarily failure."
        ),
    )
    _save(fig, output_dir, "styled_si_reliability_counts", formats, dpi)


def plot_si_runtime_cost(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    token_rows: list[dict[str, Any]] = []
    token_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        total_tokens = float(row.get("prompt_tokens") or 0.0) + float(
            row.get("completion_tokens") or 0.0
        )
        token_values[(row["model"], row["condition"])].append(total_tokens / 1000.0)
    mean_tokens = {
        (model, condition): _mean(token_values[(model, condition)])
        for model in models
        for condition in conditions
    }

    wall_time = _value_by_model_condition(
        rows, models=models, conditions=conditions, field="wall_time_seconds"
    )
    max_prompt = _value_by_model_condition(
        rows,
        models=models,
        conditions=conditions,
        field="max_prompt_tokens_single_turn",
        scale=1 / 1000,
    )
    for model in models:
        for condition in conditions:
            token_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "planner": condition,
                    "mean_total_tokens_k": round(mean_tokens[(model, condition)], 6),
                    "mean_wall_time_seconds": round(wall_time[(model, condition)], 6),
                    "mean_max_prompt_tokens_k": round(max_prompt[(model, condition)], 6),
                }
            )
    _write_csv(output_dir / "styled_si_runtime_cost.csv", token_rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.05),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.21, "top": 0.74, "wspace": 0.34},
    )
    panels = [
        (mean_tokens, "Total tokens (k)"),
        (wall_time, "Wall time (s)"),
        (max_prompt, "Max prompt / turn (k)"),
    ]
    for ax, (values, ylabel) in zip(axes, panels, strict=True):
        _draw_grouped_bars(ax, values, models=models, conditions=conditions, ylabel=ylabel)
        ax.set_title(ylabel, loc="left", fontsize=7.4, color=INK, fontweight="bold")

    fig.legend(
        handles=_model_handles(models),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.91),
        ncol=4,
        frameon=False,
    )
    _figure_header(
        fig,
        "SI: runtime and context cost",
        f"Canonical bundle: {bundle_name}; bars show cell means over scenarios.",
    )
    _save(fig, output_dir, "styled_si_runtime_cost", formats, dpi)


def plot_si_scenario_planner_bars(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    conditions: list[str],
    scenarios: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    labels = _scenario_labels(scenarios)
    ordered = _scenario_difficulty_order(rows, scenarios)
    source_rows: list[dict[str, Any]] = []
    values: dict[tuple[str, str], float] = {}
    for condition in conditions:
        for scenario in ordered:
            items = [
                row for row in rows if row["condition"] == condition and row["scenario"] == scenario
            ]
            successes = sum(1 for item in items if item.get("success"))
            rate = successes / len(items) if items else 0.0
            values[(condition, scenario)] = rate * 100
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "planner": condition,
                    "scenario_label": labels[scenario],
                    "scenario": scenario,
                    "successes": successes,
                    "n": len(items),
                    "success_rate": round(rate, 6),
                }
            )
    _write_csv(output_dir / "styled_si_scenario_planner_bars.csv", source_rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.15),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "bottom": 0.14,
            "top": 0.78,
            "hspace": 0.34,
            "wspace": 0.14,
        },
    )
    x = np.arange(len(ordered))
    for ax, condition in zip(axes.flat, conditions, strict=True):
        ax.bar(
            x,
            [values[(condition, scenario)] for scenario in ordered],
            color=PLANNER_COLORS.get(condition, "#777777"),
            width=0.7,
            edgecolor=PAPER,
            linewidth=0.55,
        )
        ax.set_title(
            _condition_label(condition), loc="left", fontsize=7.4, color=INK, fontweight="bold"
        )
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(x)
        ax.set_xticklabels([labels[scenario] for scenario in ordered])
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Scenarios ordered by aggregate success")

    fig.legend(
        handles=_planner_handles(conditions),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.91),
        ncol=2,
        frameon=False,
    )
    _figure_header(
        fig,
        "SI: scenario-level planner profile",
        "Each subplot aggregates across the four local models for a planner/interface condition.",
    )
    _save(fig, output_dir, "styled_si_scenario_planner_bars", formats, dpi)


def build_styled_figures(
    *,
    canonical_bundle: Path,
    reference_bundle: Path,
    perturbation_1_bundle: Path,
    perturbation_2_bundle: Path,
    output_dir: Path,
    si_output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    _set_styled_theme()
    rows = _load_results(canonical_bundle)
    scenarios = _scenario_order(canonical_bundle)
    observed_models = {row["model"] for row in rows}
    observed_conditions = {row["condition"] for row in rows}
    models = [model for model in MODEL_ORDER if model in observed_models]
    models.extend(model for model in _ordered_models(rows) if model not in models)
    conditions = [condition for condition in CONDITION_ORDER if condition in observed_conditions]
    conditions.extend(
        condition for condition in _ordered_conditions(rows) if condition not in conditions
    )

    plot_styled_main_success(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_styled_scenario_difficulty(
        rows=rows,
        bundle_name=canonical_bundle.name,
        scenarios=scenarios,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_styled_failure_taxonomy(
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
    plot_styled_sensitivity(
        bundle_rows=bundle_rows,
        bundle_names=bundle_names,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )

    plot_si_completion_efficiency(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=si_output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_si_reliability_counts(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        output_dir=si_output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_si_runtime_cost(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=si_output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_si_scenario_planner_bars(
        rows=rows,
        bundle_name=canonical_bundle.name,
        conditions=conditions,
        scenarios=scenarios,
        output_dir=si_output_dir,
        formats=formats,
        dpi=dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek-inspired styled LabVerse benchmark and SI figures."
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
    parser.add_argument("--si-output-dir", type=Path, default=Path("paper/figures/si"))
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"]
    )
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    build_styled_figures(
        canonical_bundle=args.canonical_bundle,
        reference_bundle=args.reference_bundle,
        perturbation_1_bundle=args.perturbation_1_bundle,
        perturbation_2_bundle=args.perturbation_2_bundle,
        output_dir=args.output_dir,
        si_output_dir=args.si_output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
