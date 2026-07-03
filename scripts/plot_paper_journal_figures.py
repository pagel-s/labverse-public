#!/usr/bin/env python3
"""Generate minimal journal-style LabVerse benchmark figures."""

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


INK = "#22272b"
MID = "#626970"
MUTED = "#87909a"
GRID = "#ded9cf"
PAPER = "#fbfaf7"

MODEL_COLORS = {
    "ollama/gemma4:31b": "#e0ebeb",
    "ollama/gemma4:26b": "#c0d7d7",
    "ollama/qwen3.5:27b": "#a3c1c3",
    "ollama/mistral-small3.2:24b": "#8ea4ae",
}


FAILURE_COLORS = {
    "Behavioral/task failure": "#686d72",
    "Format/argument failure": "#b4b79f",
    "Grounding/interface failure": "#849aaa",
    "Provider/runner failure": "#aa8d84",
}

HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "labverse_heatmap",
    ["#f6f3ed", "#d9dfdd", "#98adb4", "#566a72"],
)

COMPLETION_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "labverse_completion",
    ["#f5f7f7", "#e0ebeb", "#c0d7d7", "#a3c1c3", "#708b98"],
)

SENSITIVITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "labverse_sensitivity",
    ["#f7f3f2", "#ecd9d6", "#d9aaa5", "#b97673", "#925354"],
)

SUCCESS_CMAP = mpl.colors.ListedColormap(["#ffffff", "#74c476"])

CONDITION_COMPACT = {
    "local_structured_interactive_grounded": "Local structured",
    "local_narrative_interactive_grounded": "Local narrative",
    "global_structured_interactive_grounded": "Global structured",
    "global_one_shot_planner": "Global one-shot",
}

CONDITION_CODES = {
    "local_structured_interactive_grounded": "LS",
    "local_narrative_interactive_grounded": "LN",
    "global_structured_interactive_grounded": "GS",
    "global_one_shot_planner": "OS",
}


def _journal_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 6.4,
            "axes.titlesize": 6.8,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.55,
            "axes.edgecolor": "#b9b3a9",
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
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
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _condition_compact(condition: str) -> str:
    return CONDITION_COMPACT.get(condition, _condition_label(condition))


def _condition_code(condition: str) -> str:
    return CONDITION_CODES.get(condition, _condition_label(condition))


def _condition_code_key(conditions: list[str]) -> str:
    return "; ".join(
        f"{_condition_code(condition)}={_condition_compact(condition)}" for condition in conditions
    )


def _style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b9b3a9")
    ax.spines["bottom"].set_color("#b9b3a9")
    ax.tick_params(length=2.0, width=0.55)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.45)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)


def _selected_models_conditions(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    observed_models = {row["model"] for row in rows}
    observed_conditions = {row["condition"] for row in rows}

    models = [model for model in MODEL_ORDER if model in observed_models]
    models.extend(model for model in _ordered_models(rows) if model not in models)

    conditions = [condition for condition in CONDITION_ORDER if condition in observed_conditions]
    conditions.extend(
        condition for condition in _ordered_conditions(rows) if condition not in conditions
    )
    return models, conditions


def _success_records(
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
            records.append(
                {
                    "model": model,
                    "planner": condition,
                    "successes": int(item["successes"]),
                    "n": int(item["n"]),
                    "success_rate": float(item["success_rate"]),
                    "completion_ratio": float(item["completion_ratio"]),
                }
            )
    return records


def _success_matrix(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
) -> tuple[np.ndarray, dict[tuple[str, str], dict[str, float]]]:
    summary = _success_summary(rows, models=models, conditions=conditions)
    matrix = np.zeros((len(models), len(conditions)))
    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            matrix[i, j] = summary[(model, condition)]["success_rate"] * 100.0
    return matrix, summary


def _model_handles(models: list[str]) -> list[Patch]:
    return [
        Patch(facecolor=MODEL_COLORS.get(model, MID), edgecolor="none", label=_model_short(model))
        for model in models
    ]


def _scenario_labels(scenarios: list[str]) -> dict[str, str]:
    return {scenario: f"S{index}" for index, scenario in enumerate(scenarios, start=1)}


def _scenario_success_grid(
    rows: list[dict[str, Any]],
    *,
    model: str,
    condition: str,
    scenarios: list[str],
) -> tuple[list[float], list[int], list[int]]:
    values: list[float] = []
    successes: list[int] = []
    totals: list[int] = []
    for scenario in scenarios:
        items = [
            row
            for row in rows
            if row["model"] == model
            and row["condition"] == condition
            and row["scenario"] == scenario
        ]
        total = len(items)
        success_count = sum(1 for item in items if item.get("success"))
        values.append(success_count / total * 100.0 if total else np.nan)
        successes.append(success_count)
        totals.append(total)
    return values, successes, totals


def _scenario_cell_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    condition: str,
    scenario: str,
) -> tuple[float, float, int, int]:
    items = [
        row
        for row in rows
        if row["model"] == model
        and row["condition"] == condition
        and row["scenario"] == scenario
    ]
    total = len(items)
    successes = sum(1 for item in items if item.get("success"))
    success_rate = successes / total if total else float("nan")
    completion = _mean([float(item.get("completion_ratio") or 0.0) for item in items])
    return success_rate, completion, successes, total


def _scenario_sensitivity(
    bundle_rows: dict[str, list[dict[str, Any]]],
    *,
    model: str,
    condition: str,
    scenario: str,
) -> float:
    values: list[float] = []
    for rows in bundle_rows.values():
        _, completion, _, total = _scenario_cell_metrics(
            rows,
            model=model,
            condition=condition,
            scenario=scenario,
        )
        if total:
            values.append(completion)
    return max(values) - min(values) if values else float("nan")


def _scenario_metric_matrices(
    rows: list[dict[str, Any]],
    *,
    sensitivity_rows: dict[str, list[dict[str, Any]]],
    model: str,
    conditions: list[str],
    scenarios: list[str],
    labels: dict[str, str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    matrices = {
        "success": np.full((len(conditions), len(scenarios)), np.nan),
        "completion": np.full((len(conditions), len(scenarios)), np.nan),
        "sensitivity": np.full((len(conditions), len(scenarios)), np.nan),
    }
    source_rows: list[dict[str, Any]] = []
    for row_index, condition in enumerate(conditions):
        for column_index, scenario in enumerate(scenarios):
            success_rate, completion_ratio, successes, total = _scenario_cell_metrics(
                rows,
                model=model,
                condition=condition,
                scenario=scenario,
            )
            sensitivity = _scenario_sensitivity(
                sensitivity_rows,
                model=model,
                condition=condition,
                scenario=scenario,
            )
            matrices["success"][row_index, column_index] = success_rate * 100.0
            matrices["completion"][row_index, column_index] = completion_ratio * 100.0
            matrices["sensitivity"][row_index, column_index] = sensitivity * 100.0
            source_rows.append(
                {
                    "model": model,
                    "planner": condition,
                    "scenario": scenario,
                    "scenario_label": labels[scenario],
                    "successes": successes,
                    "n": total,
                    "success_rate": round(success_rate if total else 0.0, 6),
                    "completion_ratio": round(completion_ratio if total else 0.0, 6),
                    "perturbation_completion_range": round(
                        sensitivity if not np.isnan(sensitivity) else 0.0,
                        6,
                    ),
                }
            )
    return matrices, source_rows


def _draw_scenario_metric_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    model: str,
    conditions: list[str],
    scenarios: list[str],
    labels: dict[str, str],
    show_ylabels: bool,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    summary_label: str,
) -> None:
    ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="none",
    )
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([labels[scenario] for scenario in scenarios])
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(
        [_condition_compact(condition) for condition in conditions] if show_ylabels else []
    )
    ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(conditions), 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=1.2)
    ax.tick_params(which="both", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    mean_value = float(np.nanmean(matrix)) if matrix.size else float("nan")
    ax.set_title(
        f"{_model_short(model)}  |  {summary_label} {mean_value:.0f}%",
        loc="left",
        fontsize=6.7,
        fontweight="bold",
        color=INK,
        pad=3,
    )


def _plot_scenario_metric_figure(
    *,
    matrices: dict[str, np.ndarray],
    models: list[str],
    conditions: list[str],
    scenarios: list[str],
    labels: dict[str, str],
    title: str,
    subtitle: str,
    summary_label: str,
    output_stem: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    colorbar_label: str | None,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 3.05), squeeze=False)
    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        bottom=0.15,
        top=0.79,
        hspace=0.34,
        wspace=0.10,
    )

    for index, (ax, model) in enumerate(zip(axes.flat, models, strict=False)):
        _draw_scenario_metric_panel(
            ax,
            matrices[model],
            model=model,
            conditions=conditions,
            scenarios=scenarios,
            labels=labels,
            show_ylabels=index % ncols == 0,
            cmap=cmap,
            norm=norm,
            summary_label=summary_label,
        )
    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)

    fig.text(0.17, 0.975, title, fontsize=7.4, fontweight="bold", color=INK, va="top")
    fig.text(
        0.17,
        0.910,
        subtitle,
        fontsize=5.9,
        color=MUTED,
        va="top",
    )

    if colorbar_label is None:
        fig.legend(
            handles=[
                Patch(facecolor=SUCCESS_CMAP(0), edgecolor="none", label="Failure"),
                Patch(facecolor=SUCCESS_CMAP(1), edgecolor="none", label="Success"),
            ],
            loc="upper right",
            bbox_to_anchor=(0.985, 0.935),
            frameon=False,
            ncol=2,
            columnspacing=1.0,
            handlelength=1.0,
        )
    else:
        colorbar_axis = fig.add_axes((0.37, 0.035, 0.42, 0.018))
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=[0, 50, 100],
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=5.4)
        colorbar.set_label(colorbar_label, fontsize=5.6, labelpad=2)

    _save(fig, output_dir, output_stem, formats, dpi)


def plot_journal_main(
    *,
    rows: list[dict[str, Any]],
    sensitivity_rows: dict[str, list[dict[str, Any]]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    scenarios: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    labels = _scenario_labels(scenarios)
    source_rows: list[dict[str, Any]] = []
    metric_matrices: dict[str, dict[str, np.ndarray]] = {
        "success": {},
        "completion": {},
        "sensitivity": {},
    }
    for model in models:
        model_matrices, model_records = _scenario_metric_matrices(
            rows,
            sensitivity_rows=sensitivity_rows,
            model=model,
            conditions=conditions,
            scenarios=scenarios,
            labels=labels,
        )
        source_rows.extend(model_records)
        for metric, matrix in model_matrices.items():
            metric_matrices[metric][model] = matrix

    _write_csv(
        output_dir / "journal_main_benchmark.csv",
        [
            {
                "bundle": bundle_name,
                **record,
            }
            for record in source_rows
        ],
    )

    common = {
        "models": models,
        "conditions": conditions,
        "scenarios": scenarios,
        "labels": labels,
        "output_dir": output_dir,
        "formats": formats,
        "dpi": dpi,
    }
    _plot_scenario_metric_figure(
        matrices=metric_matrices["success"],
        title="Scenario-level success",
        subtitle=f"Canonical bundle: {bundle_name}",
        summary_label="mean success",
        output_stem="journal_main_benchmark",
        cmap=SUCCESS_CMAP,
        norm=mpl.colors.BoundaryNorm([-0.5, 50, 100.5], SUCCESS_CMAP.N),
        colorbar_label=None,
        **common,
    )
    _plot_scenario_metric_figure(
        matrices=metric_matrices["completion"],
        title="Scenario-level completion",
        subtitle=f"Canonical bundle: {bundle_name}",
        summary_label="mean completion",
        output_stem="journal_main_completion",
        cmap=COMPLETION_CMAP,
        norm=mpl.colors.Normalize(vmin=0, vmax=100),
        colorbar_label="Completion ratio (%)",
        **common,
    )
    _plot_scenario_metric_figure(
        matrices=metric_matrices["sensitivity"],
        title="Sensitivity to initial-state perturbations",
        subtitle="Range across reference, perturbation 1, and perturbation 2",
        summary_label="mean range",
        output_stem="journal_main_perturbation_sensitivity",
        cmap=SENSITIVITY_CMAP,
        norm=mpl.colors.Normalize(vmin=0, vmax=100),
        colorbar_label="Completion-ratio range (percentage points)",
        **common,
    )


def plot_journal_failure_taxonomy(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    failure_categories = [category for category in FAILURE_CATEGORIES if category != "Success"]
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    failed_totals: dict[str, int] = Counter()
    all_totals: dict[str, int] = Counter()

    for row in rows:
        model = row["model"]
        all_totals[model] += 1
        category = _failure_category(row)
        if category != "Success":
            failed_totals[model] += 1
            buckets[model][category] += 1

    source_rows: list[dict[str, Any]] = []
    for model in models:
        failed = failed_totals[model]
        total = all_totals[model]
        for category in failure_categories:
            count = buckets[model][category]
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "category": category,
                    "failed_runs": failed,
                    "all_runs": total,
                    "count": count,
                    "share_of_failed_runs": round(count / failed if failed else 0.0, 6),
                    "share_of_all_runs": round(count / total if total else 0.0, 6),
                }
            )
    _write_csv(output_dir / "journal_failure_taxonomy.csv", source_rows)

    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.17, right=0.93, bottom=0.22, top=0.68)

    y_positions = np.arange(len(models))
    for y, model in zip(y_positions, models, strict=True):
        left = 0.0
        failed = failed_totals[model]
        for category in failure_categories:
            share = buckets[model][category] / failed * 100.0 if failed else 0.0
            ax.barh(
                y,
                share,
                left=left,
                height=0.56,
                color=FAILURE_COLORS[category],
                edgecolor=PAPER,
                linewidth=0.5,
            )
            left += share
        ax.text(
            102,
            y,
            f"n={failed}",
            ha="left",
            va="center",
            fontsize=5.7,
            color=MUTED,
            clip_on=False,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_model_short(model) for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of failed runs (%)")
    _style_axis(ax, xgrid=True, ygrid=False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Patch(facecolor=FAILURE_COLORS[category], edgecolor="none", label=category)
        for category in failure_categories
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.17, 0.91),
        frameon=False,
        ncol=2,
        columnspacing=1.35,
        handlelength=1.25,
    )
    fig.text(
        0.17,
        0.985,
        "Failure composition",
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.17,
        0.945,
        "Failed runs only, aggregated over planner/interface conditions.",
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED,
    )
    _save(fig, output_dir, "journal_failure_taxonomy", formats, dpi)


def _draw_matrix_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    models: list[str],
    conditions: list[str],
    show_ylabels: bool,
    title: str,
    vmin: float,
    vmax: float,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([_condition_code(condition) for condition in conditions])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_model_short(model) for model in models] if show_ylabels else [])
    ax.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=1.0)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=6.6, fontweight="bold", color=INK)
    return image


def plot_journal_sensitivity(
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

    source_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for key in bundle_order:
        matrix, summary = _success_matrix(
            bundle_rows[key],
            models=models,
            conditions=conditions,
        )
        matrices[key] = matrix
        summaries[key] = summary
        for model in models:
            for condition in conditions:
                item = summary[(model, condition)]
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
    _write_csv(output_dir / "journal_sensitivity.csv", source_rows)

    fig, axes = plt.subplots(1, len(bundle_order), figsize=(7.2, 2.85))
    if len(bundle_order) == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.20, top=0.73, wspace=0.22)

    image = None
    for ax, key in zip(axes, bundle_order, strict=True):
        image = _draw_matrix_heatmap(
            ax,
            matrices[key],
            models=models,
            conditions=conditions,
            show_ylabels=ax is axes[0],
            title=bundle_label[key],
            vmin=0,
            vmax=100,
        )

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes,
            fraction=0.024,
            pad=0.035,
            ticks=[0, 50, 100],
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=5.6)
        colorbar.set_label("Success (%)", fontsize=5.8)

    fig.text(
        0.13,
        0.975,
        "Initial-state sensitivity",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.13,
        0.930,
        f"Same model x planner grid under each start state. {_condition_code_key(conditions)}.",
        fontsize=5.9,
        color=MUTED,
        va="top",
    )
    _save(fig, output_dir, "journal_sensitivity", formats, dpi)


def _metric_matrix(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
    field: str,
    scale: float = 1.0,
) -> np.ndarray:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row.get(field) or 0.0) * scale
        buckets[(row["model"], row["condition"])].append(value)
    matrix = np.zeros((len(models), len(conditions)))
    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            matrix[i, j] = _mean(buckets[(model, condition)])
    return matrix


def _draw_metric_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    models: list[str],
    conditions: list[str],
    title: str,
    show_ylabels: bool,
) -> mpl.image.AxesImage:
    vmax = float(np.nanmax(matrix)) if matrix.size else 1.0
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=max(vmax, 1.0), aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([_condition_code(condition) for condition in conditions])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_model_short(model) for model in models] if show_ylabels else [])
    ax.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=1.0)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=6.6, fontweight="bold", color=INK)
    return image


def plot_journal_si_diagnostics(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    completion = _metric_matrix(
        rows,
        models=models,
        conditions=conditions,
        field="completion_ratio",
        scale=100.0,
    )
    steps = _metric_matrix(
        rows,
        models=models,
        conditions=conditions,
        field="steps_taken",
    )
    wall = _metric_matrix(
        rows,
        models=models,
        conditions=conditions,
        field="wall_time_seconds",
    )
    error_fields = [
        "invalid_action_outputs",
        "action_output_retry_count",
        "model_output_format_errors",
        "grounding_error_count",
        "provider_errors",
    ]
    errors_by_model = {
        model: sum(
            float(row.get(field) or 0.0)
            for row in rows
            if row["model"] == model
            for field in error_fields
        )
        for model in models
    }

    source_rows: list[dict[str, Any]] = []
    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            source_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "planner": condition,
                    "mean_completion_percent": round(completion[i, j], 6),
                    "mean_steps": round(steps[i, j], 6),
                    "mean_wall_time_seconds": round(wall[i, j], 6),
                    "model_error_events_total": round(errors_by_model[model], 6),
                }
            )
    _write_csv(output_dir / "journal_si_diagnostics.csv", source_rows)

    fig = plt.figure(figsize=(7.2, 3.8))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.965,
        bottom=0.16,
        top=0.76,
        hspace=0.55,
        wspace=0.34,
    )
    ax_completion = fig.add_subplot(grid[0, 0])
    ax_steps = fig.add_subplot(grid[0, 1])
    ax_wall = fig.add_subplot(grid[1, 0])
    ax_errors = fig.add_subplot(grid[1, 1])

    _draw_metric_heatmap(
        ax_completion,
        completion,
        models=models,
        conditions=conditions,
        title="a  completion",
        show_ylabels=True,
    )
    _draw_metric_heatmap(
        ax_steps,
        steps,
        models=models,
        conditions=conditions,
        title="b  steps",
        show_ylabels=False,
    )
    _draw_metric_heatmap(
        ax_wall,
        wall,
        models=models,
        conditions=conditions,
        title="c  wall time",
        show_ylabels=True,
    )

    ranked = sorted(models, key=lambda model: errors_by_model[model], reverse=True)
    y = np.arange(len(ranked))
    ax_errors.barh(
        y,
        [errors_by_model[model] for model in ranked],
        height=0.55,
        color=[MODEL_COLORS.get(model, MID) for model in ranked],
        edgecolor=PAPER,
        linewidth=0.5,
    )
    ax_errors.set_yticks(y)
    ax_errors.set_yticklabels([_model_short(model) for model in ranked])
    ax_errors.invert_yaxis()
    ax_errors.set_xlabel("Total events")
    _style_axis(ax_errors, xgrid=True, ygrid=False)
    ax_errors.tick_params(axis="y", length=0)
    ax_errors.set_title(
        "d  format/grounding/provider events",
        loc="left",
        fontsize=6.6,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.12,
        0.975,
        "Supplementary diagnostics",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.12,
        0.930,
        f"Canonical bundle: {bundle_name}. Exact values are in the source CSV.",
        fontsize=5.9,
        color=MUTED,
        va="top",
    )
    _save(fig, output_dir, "journal_si_diagnostics", formats, dpi)


def build_journal_figures(
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
    _journal_theme()
    rows = _load_results(canonical_bundle)
    scenarios = _scenario_order(canonical_bundle)
    models, conditions = _selected_models_conditions(rows)
    bundle_rows = {
        "reference": _load_results(reference_bundle),
        "perturbation_1": _load_results(perturbation_1_bundle),
        "perturbation_2": _load_results(perturbation_2_bundle),
    }

    plot_journal_main(
        rows=rows,
        sensitivity_rows=bundle_rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        scenarios=scenarios,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_journal_failure_taxonomy(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )

    bundle_names = {
        "reference": reference_bundle.name,
        "perturbation_1": perturbation_1_bundle.name,
        "perturbation_2": perturbation_2_bundle.name,
    }
    plot_journal_sensitivity(
        bundle_rows=bundle_rows,
        bundle_names=bundle_names,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_journal_si_diagnostics(
        rows=rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=si_output_dir,
        formats=formats,
        dpi=dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal journal-style LabVerse figures.")
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
        "--si-output-dir",
        type=Path,
        default=Path("paper/figures/si"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        choices=["pdf", "svg", "png"],
    )
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    build_journal_figures(
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
