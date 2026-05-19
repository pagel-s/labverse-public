#!/usr/bin/env python3
"""Generate manuscript-grade LabVerse paper figures from frozen run bundles."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


CONDITION_ORDER = [
    "local_structured_interactive_grounded",
    "local_narrative_interactive_grounded",
    "global_structured_interactive_grounded",
    "global_one_shot_planner",
]

CONDITION_LABELS = {
    "local_structured_interactive_grounded": "Local structured",
    "local_narrative_interactive_grounded": "Local narrative",
    "global_structured_interactive_grounded": "Global structured",
    "global_one_shot_planner": "Global one-shot",
}

CONDITION_SHORT = {
    "local_structured_interactive_grounded": "Local\nstructured",
    "local_narrative_interactive_grounded": "Local\nnarrative",
    "global_structured_interactive_grounded": "Global\nstructured",
    "global_one_shot_planner": "Global\none-shot",
}

MODEL_ORDER = [
    "ollama/gemma4:31b",
    "ollama/gemma4:26b",
    "ollama/qwen3.5:27b",
    "ollama/mistral-small3.2:24b",
]

MODEL_LABELS = {
    "ollama/gemma4:31b": "Gemma 4 31B",
    "ollama/gemma4:26b": "Gemma 4 26B",
    "ollama/qwen3.5:27b": "Qwen 3.5 27B",
    "ollama/mistral-small3.2:24b": "Mistral Small 3.2 24B",
}

SENSITIVITY_BUNDLES = [
    ("reference", "Reference"),
    ("perturbation_1", "Perturbation 1"),
    ("perturbation_2", "Perturbation 2"),
]

FAILURE_CATEGORIES = [
    "Success",
    "Behavioral/task failure",
    "Format/argument failure",
    "Grounding/interface failure",
    "Provider/runner failure",
]

FAILURE_COLORS = {
    "Success": "#0b6e69",
    "Behavioral/task failure": "#d98c21",
    "Format/argument failure": "#7a8f2a",
    "Grounding/interface failure": "#3b6ea8",
    "Provider/runner failure": "#8d3b2f",
}

MODEL_COLORS = {
    "ollama/gemma4:31b": "#0b6e69",
    "ollama/gemma4:26b": "#2f7ebc",
    "ollama/qwen3.5:27b": "#d98c21",
    "ollama/mistral-small3.2:24b": "#8d3b2f",
}

BACKGROUND = "#fbfaf6"
INK = "#15171a"
MUTED = "#5e6470"
GRID = "#dad6ca"
PALE = "#f2eee4"
FAIL = "#ead1c4"
MISSING = "#d9d9d9"


def _set_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#9d978c",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_results(bundle_dir: Path) -> list[dict[str, Any]]:
    path = bundle_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return _load_jsonl(path)


def _scenario_order(bundle_dir: Path) -> list[str]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    return [item["name"] for item in manifest["scenarios"]]


def _ordered_models(rows: list[dict[str, Any]]) -> list[str]:
    observed = {row["model"] for row in rows}
    ordered = [model for model in MODEL_ORDER if model in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def _ordered_conditions(rows: list[dict[str, Any]]) -> list[str]:
    observed = {row["condition"] for row in rows}
    ordered = [condition for condition in CONDITION_ORDER if condition in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.removeprefix("ollama/"))


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _condition_short(condition: str) -> str:
    return CONDITION_SHORT.get(condition, condition.replace("_", "\n"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            _strip_trailing_whitespace(path)
        print(path)
    plt.close(fig)


def _strip_trailing_whitespace(path: Path) -> None:
    cleaned = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    path.write_text(cleaned)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def _success_summary(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
) -> dict[tuple[str, str], dict[str, float]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["condition"])].append(row)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for model in models:
        for condition in conditions:
            items = buckets[(model, condition)]
            successes = sum(1 for item in items if bool(item.get("success")))
            n = len(items)
            summary[(model, condition)] = {
                "n": float(n),
                "successes": float(successes),
                "success_rate": successes / n if n else float("nan"),
                "completion_ratio": _mean(
                    [float(item.get("completion_ratio", 0.0)) for item in items]
                ),
            }
    return summary


def _success_matrix(
    summary: dict[tuple[str, str], dict[str, float]],
    *,
    models: list[str],
    conditions: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.full((len(models), len(conditions)), np.nan)
    counts = np.zeros_like(rates)
    totals = np.zeros_like(rates)
    for model_index, model in enumerate(models):
        for condition_index, condition in enumerate(conditions):
            item = summary[(model, condition)]
            rates[model_index, condition_index] = item["success_rate"]
            counts[model_index, condition_index] = item["successes"]
            totals[model_index, condition_index] = item["n"]
    return rates, counts, totals


def _failure_category(row: dict[str, Any]) -> str:
    if bool(row.get("success")):
        return "Success"

    taxonomy = str(row.get("error_taxonomy") or "none")
    if taxonomy in {"provider_failure", "runner_failure", "artifact_integrity_failure"}:
        return "Provider/runner failure"
    if taxonomy in {"grounding_failure"}:
        return "Grounding/interface failure"
    if taxonomy in {"parameter_failure"}:
        return "Format/argument failure"
    if taxonomy in {"safety_failure", "scenario_behavior_failure", "sequencing_failure", "none"}:
        return "Behavioral/task failure"
    if int(row.get("provider_errors") or 0) or int(row.get("artifact_integrity_errors") or 0):
        return "Provider/runner failure"
    if int(row.get("grounding_error_count") or 0):
        return "Grounding/interface failure"
    if int(row.get("model_output_format_errors") or 0) or int(
        row.get("invalid_action_outputs") or 0
    ):
        return "Format/argument failure"
    return "Behavioral/task failure"


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    label: str,
    sublabel: str = "",
    facecolor: str = PALE,
    edgecolor: str = "#b7ad9b",
    label_color: str = INK,
    lw: float = 0.9,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + width * 0.07,
        y + height * 0.68,
        label,
        color=label_color,
        fontsize=8.2,
        fontweight="bold",
        va="center",
    )
    if sublabel:
        ax.text(
            x + width * 0.07,
            y + height * 0.36,
            sublabel,
            color=MUTED,
            fontsize=6.2,
            va="center",
            linespacing=1.15,
        )
    return box


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=0.9,
            color="#6e756f",
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_benchmark_schematic(
    *,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    models: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor("white")

    fig.text(
        0.02, 0.96, "A. LabVerse benchmark design", fontsize=11.5, fontweight="bold", color=INK
    )
    fig.text(
        0.02,
        0.905,
        "Frozen lab worlds, planner interfaces, local models, and simulator-grounded outcomes.",
        fontsize=7.4,
        color=MUTED,
    )

    def panel(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        title: str,
        lines: list[str],
        facecolor: str,
        edgecolor: str,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.95,
            )
        )
        ax.text(
            x + 0.02,
            y + height - 0.055,
            title,
            fontsize=8.4,
            fontweight="bold",
            color=INK,
            va="top",
        )
        for line_index, line in enumerate(lines):
            ax.text(
                x + 0.02,
                y + height - 0.105 - line_index * 0.04,
                line,
                fontsize=6.2,
                color=MUTED,
                va="top",
            )

    panel(
        0.035,
        0.52,
        0.19,
        0.28,
        title="LabVerse world",
        lines=["typed rooms", "equipment state", "samples, staff"],
        facecolor="#edf3ef",
        edgecolor="#8ba494",
    )

    panel(
        0.275,
        0.52,
        0.17,
        0.28,
        title="14 scenarios",
        lines=["frozen task contracts", "hidden scoring oracles"],
        facecolor="#f7f0df",
        edgecolor="#d1b66d",
    )
    for idx in range(len(scenarios)):
        col = idx % 7
        row = idx // 7
        color = "#0b6e69" if idx % 3 != 1 else "#d98c21"
        ax.add_patch(
            Rectangle(
                (0.305 + col * 0.017, 0.57 + row * 0.04),
                0.011,
                0.026,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
            )
        )

    panel(
        0.49,
        0.52,
        0.21,
        0.28,
        title="Interface",
        lines=["local/global context", "structured/narrative", "one-shot planning"],
        facecolor="#eef2f7",
        edgecolor="#92a4bf",
    )

    panel(
        0.755,
        0.52,
        0.205,
        0.28,
        title="Grounded run",
        lines=["candidate action", "simulator transition", "replayable trace"],
        facecolor="#f2ece8",
        edgecolor="#c49a86",
    )
    for x, color in [(0.84, "#0b6e69"), (0.885, "#f4d3c1"), (0.93, "#3b6ea8")]:
        ax.add_patch(
            Rectangle((x, 0.575), 0.032, 0.055, facecolor=color, edgecolor="#ba8874", linewidth=0.5)
        )

    _arrow(ax, (0.226, 0.66), (0.275, 0.66))
    _arrow(ax, (0.446, 0.66), (0.49, 0.66))
    _arrow(ax, (0.702, 0.66), (0.755, 0.66))

    panel(
        0.035,
        0.19,
        0.29,
        0.20,
        title="Local model panel",
        lines=[],
        facecolor="#eef6f4",
        edgecolor="#9db9b6",
    )
    chip_labels = ["Gemma 31B", "Gemma 26B", "Qwen 27B", "Mistral 24B"]
    for i, (model, label) in enumerate(zip(models[:4], chip_labels, strict=False)):
        x = 0.06 + (i % 2) * 0.13
        y = 0.225 + (1 - i // 2) * 0.041
        ax.add_patch(
            Rectangle(
                (x, y), 0.025, 0.026, facecolor=MODEL_COLORS.get(model, "#777"), edgecolor="none"
            )
        )
        ax.text(x + 0.033, y + 0.013, label, va="center", fontsize=5.4, color=MUTED)

    panel(
        0.375,
        0.19,
        0.25,
        0.20,
        title="Outcomes",
        lines=[],
        facecolor="#f7f4ee",
        edgecolor="#c9c0ad",
    )
    for i, (label, color) in enumerate(
        [
            ("success", "#0b6e69"),
            ("completion", "#a78c4f"),
            ("safety", "#d98c21"),
            ("format", "#7a8f2a"),
            ("grounding", "#3b6ea8"),
            ("provider", "#8d3b2f"),
        ]
    ):
        x = 0.405 + (i % 2) * 0.105
        y = 0.218 + (2 - i // 2) * 0.034
        ax.add_patch(Rectangle((x, y), 0.012, 0.019, facecolor=color, edgecolor="none"))
        ax.text(x + 0.017, y + 0.0095, label, va="center", fontsize=4.9, color=MUTED)

    panel(
        0.69,
        0.19,
        0.27,
        0.20,
        title="Manuscript figures",
        lines=["ranking matrix + heatmap", "failure taxonomy", "sensitivity analysis"],
        facecolor="#f4f2f8",
        edgecolor="#b3adc4",
    )
    _arrow(ax, (0.33, 0.29), (0.375, 0.29))
    _arrow(ax, (0.63, 0.29), (0.69, 0.29))

    n_runs = len(rows)
    fig.text(
        0.02,
        0.035,
        (
            f"Canonical bundle contains {n_runs} runs: {len(models)} models x "
            f"{len(CONDITION_ORDER)} planner/interface conditions x {len(scenarios)} scenarios."
        ),
        fontsize=6.8,
        color=MUTED,
    )
    _save(fig, output_dir, "benchmark_schematic", formats, dpi)


def plot_main_success(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    summary = _success_summary(rows, models=models, conditions=conditions)
    rates, counts, totals = _success_matrix(summary, models=models, conditions=conditions)
    model_avg = np.nanmean(rates, axis=1)
    planner_avg = np.nanmean(rates, axis=0)

    csv_rows: list[dict[str, Any]] = []
    for model in models:
        for condition in conditions:
            item = summary[(model, condition)]
            csv_rows.append(
                {
                    "bundle": bundle_name,
                    "model": model,
                    "planner": condition,
                    "successes": int(item["successes"]),
                    "n": int(item["n"]),
                    "success_rate": round(item["success_rate"], 6),
                    "mean_completion_ratio": round(item["completion_ratio"], 6),
                }
            )
    _write_csv(output_dir / "main_success_by_model_planner.csv", csv_rows)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "labverse_success", ["#f7f1e4", "#8fb9ad", "#064f57"]
    )
    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.1, 1.45],
        height_ratios=[4.0, 1.0],
        left=0.16,
        right=0.965,
        bottom=0.18,
        top=0.75,
        hspace=0.18,
        wspace=0.09,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1], sharey=ax)
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax)

    im = ax.imshow(rates, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_model_label(model) for model in models])
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(models)):
        row_best = np.nanmax(rates[i])
        for j in range(len(conditions)):
            if np.isnan(rates[i, j]):
                continue
            color = "white" if rates[i, j] >= 0.55 else "#2c2c2c"
            ax.text(
                j,
                i - 0.08,
                f"{rates[i, j] * 100:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9.2,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.22,
                f"{int(counts[i, j])}/{int(totals[i, j])}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.2,
            )
            if np.isclose(rates[i, j], row_best):
                ax.add_patch(
                    Rectangle(
                        (j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.05
                    )
                )

    ax_right.barh(np.arange(len(models)), model_avg * 100, color="#21343a", height=0.58)
    ax_right.set_xlim(0, 100)
    ax_right.set_xticks([0, 50, 100])
    ax_right.set_xlabel("Mean")
    ax_right.grid(axis="x", color=GRID, linewidth=0.7)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.tick_params(axis="x", length=2.5)
    ax_right.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax_right.spines[spine].set_visible(False)
    for y, value in enumerate(model_avg):
        ax_right.text(
            min(value * 100 + 3, 98),
            y,
            f"{value * 100:.0f}",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )

    ax_bottom.bar(np.arange(len(conditions)), planner_avg * 100, color="#a78c4f", width=0.55)
    ax_bottom.set_xticks(np.arange(len(conditions)))
    ax_bottom.set_xticklabels([_condition_short(condition) for condition in conditions])
    ax_bottom.set_ylim(0, 100)
    ax_bottom.set_yticks([0, 50, 100])
    ax_bottom.set_ylabel("Planner\nmean")
    ax_bottom.grid(axis="y", color=GRID, linewidth=0.7)
    ax_bottom.tick_params(axis="x", labelbottom=True, length=0, pad=6)
    ax_bottom.tick_params(axis="y", length=2.5)
    ax_bottom.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_bottom.spines[spine].set_visible(False)
    for x, value in enumerate(planner_avg):
        ax_bottom.text(
            x, value * 100 + 4, f"{value * 100:.0f}", ha="center", fontsize=7.0, fontweight="bold"
        )

    cax = fig.add_axes([0.835, 0.782, 0.13, 0.018])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1.0])
    colorbar.ax.set_xticklabels(["0", "50", "100"])
    colorbar.ax.tick_params(labelsize=6.2, length=2)
    colorbar.outline.set_visible(False)
    fig.text(0.835, 0.81, "Success rate (%)", fontsize=6.5, color=MUTED)

    fig.text(
        0.02, 0.95, "B. Headline benchmark result", fontsize=11.5, fontweight="bold", color=INK
    )
    fig.text(
        0.02,
        0.895,
        "Success rate by local model and planner/interface condition",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.848,
        (
            f"Canonical bundle: {bundle_name}. Cells aggregate 14 scenarios; "
            "outline marks the best planner for each model."
        ),
        fontsize=7.1,
        color=MUTED,
    )
    fig.text(
        0.16,
        0.06,
        "Cell text: success rate and successful runs / total runs.",
        fontsize=6.6,
        color=MUTED,
    )

    _save(fig, output_dir, "main_success_by_model_planner", formats, dpi)


def plot_scenario_heatmap(
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
    by_key = {(row["model"], row["condition"], row["scenario"]): row for row in rows}
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"left": 0.17, "right": 0.985, "bottom": 0.105, "top": 0.775, "hspace": 0.36},
    )
    if len(models) == 1:
        axes = np.array([axes])

    cmap = mpl.colors.ListedColormap([FAIL, "#0b6e69", MISSING])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    csv_rows: list[dict[str, Any]] = []

    for model_index, model in enumerate(models):
        ax = axes[model_index]
        matrix = np.full((len(conditions), len(scenarios)), 2.0)
        for condition_index, condition in enumerate(conditions):
            for scenario_index, scenario in enumerate(scenarios):
                row = by_key.get((model, condition, scenario))
                if row is None:
                    continue
                matrix[condition_index, scenario_index] = 1.0 if row.get("success") else 0.0
                csv_rows.append(
                    {
                        "bundle": bundle_name,
                        "model": model,
                        "planner": condition,
                        "scenario_index": scenario_index + 1,
                        "scenario": scenario,
                        "success": int(bool(row.get("success"))),
                        "completion_ratio": round(float(row.get("completion_ratio", 0.0)), 6),
                        "error_taxonomy": row.get("error_taxonomy") or "none",
                    }
                )

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels(
            [_condition_short(condition).replace("\n", " ") for condition in conditions]
        )
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels([str(i) for i in range(1, len(scenarios) + 1)])
        ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(conditions), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        success_count = int(np.nansum(matrix == 1.0))
        total = int(np.sum(matrix != 2.0))
        ax.set_title(
            _model_label(model),
            loc="left",
            pad=3,
            fontsize=8.2,
            fontweight="bold",
            color=MODEL_COLORS.get(model, INK),
        )
        ax.text(
            1.002,
            1.035,
            f"{success_count}/{total} successes",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color=MUTED,
        )

    _write_csv(output_dir / "scenario_heatmap_success.csv", csv_rows)

    axes[-1].set_xlabel("Scenario index (Scenario 1-14, frozen manifest order)")
    fig.text(
        0.02, 0.965, "C. Scenario-level success map", fontsize=11.5, fontweight="bold", color=INK
    )
    fig.text(
        0.02,
        0.913,
        "Binary success across the 14 frozen scenarios for each planner and local model",
        fontsize=8.9,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.872,
        (
            f"Canonical bundle: {bundle_name}. Scenario names are intentionally "
            "anonymized as Scenario 1-14."
        ),
        fontsize=7.1,
        color=MUTED,
    )
    legend_handles = [
        Patch(facecolor="#0b6e69", edgecolor="none", label="Success"),
        Patch(facecolor=FAIL, edgecolor="none", label="Failure"),
        Patch(facecolor=MISSING, edgecolor="none", label="Missing"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.84),
        frameon=False,
        ncol=3,
    )

    _save(fig, output_dir, "scenario_heatmap_model_planner", formats, dpi)


def plot_failure_taxonomy(
    *,
    rows: list[dict[str, Any]],
    bundle_name: str,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: dict[tuple[str, str], int] = Counter()
    for row in rows:
        key = (row["model"], row["condition"])
        buckets[key][_failure_category(row)] += 1
        totals[key] += 1

    csv_rows: list[dict[str, Any]] = []
    for model in models:
        for condition in conditions:
            key = (model, condition)
            total = totals[key]
            for category in FAILURE_CATEGORIES:
                count = buckets[key][category]
                csv_rows.append(
                    {
                        "bundle": bundle_name,
                        "model": model,
                        "planner": condition,
                        "category": category,
                        "count": count,
                        "n": total,
                        "share": round(count / total if total else 0.0, 6),
                    }
                )
    _write_csv(output_dir / "failure_taxonomy_by_model_planner.csv", csv_rows)

    labels: list[str] = []
    keys: list[tuple[str, str]] = []
    y_positions: list[float] = []
    y = 0.0
    for model in models:
        for condition in conditions:
            keys.append((model, condition))
            labels.append(_condition_label(condition))
            y_positions.append(y)
            y += 1.0
        y += 0.55

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.11, top=0.78)

    for model_index in range(len(models)):
        start = model_index * (len(conditions) + 0.55) - 0.45
        height = len(conditions) - 0.1
        if model_index % 2 == 0:
            ax.add_patch(
                Rectangle((0, start), 100, height, facecolor="#f8f6f1", edgecolor="none", zorder=-2)
            )

    for y_pos, key in zip(y_positions, keys, strict=True):
        left = 0.0
        total = totals[key]
        for category in FAILURE_CATEGORIES:
            width = (buckets[key][category] / total * 100.0) if total else 0.0
            ax.barh(
                y_pos,
                width,
                left=left,
                height=0.68,
                color=FAILURE_COLORS[category],
                edgecolor="white",
                linewidth=0.6,
            )
            if width >= 12.0:
                text_color = (
                    "white"
                    if category
                    in {"Success", "Grounding/interface failure", "Provider/runner failure"}
                    else INK
                )
                ax.text(
                    left + width / 2,
                    y_pos,
                    f"{width:.0f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6.5,
                    fontweight="bold",
                )
            left += width

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of runs per model x planner (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for model_index, model in enumerate(models):
        first_y = model_index * (len(conditions) + 0.55)
        mid_y = first_y + (len(conditions) - 1) / 2
        ax.text(
            -31,
            mid_y,
            _model_label(model),
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=MODEL_COLORS.get(model, INK),
            clip_on=False,
        )

    handles = [
        Patch(facecolor=FAILURE_COLORS[category], edgecolor="none", label=category)
        for category in FAILURE_CATEGORIES
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.285, 0.86),
        frameon=False,
        ncol=3,
        columnspacing=1.0,
    )

    fig.text(
        0.02,
        0.965,
        "D. Outcome and failure decomposition",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.913,
        "Failures are separated from infrastructure and interface failures",
        fontsize=8.9,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.872,
        (
            f"Canonical bundle: {bundle_name}. Bars sum to 100% over 14 scenarios "
            "for each model/planner pair."
        ),
        fontsize=7.1,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.035,
        (
            "Primary category is derived from each run's final error_taxonomy; "
            "successful runs remain success even if minor repairs occurred."
        ),
        fontsize=6.5,
        color=MUTED,
    )

    _save(fig, output_dir, "failure_taxonomy_by_model_planner", formats, dpi)


def plot_sensitivity(
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
    summary_by_bundle = {
        bundle_key: _success_summary(bundle_rows[bundle_key], models=models, conditions=conditions)
        for bundle_key in bundle_order
    }
    csv_rows: list[dict[str, Any]] = []
    for bundle_key in bundle_order:
        for model in models:
            for condition in conditions:
                item = summary_by_bundle[bundle_key][(model, condition)]
                csv_rows.append(
                    {
                        "bundle_key": bundle_key,
                        "bundle": bundle_names[bundle_key],
                        "model": model,
                        "planner": condition,
                        "successes": int(item["successes"]),
                        "n": int(item["n"]),
                        "success_rate": round(item["success_rate"], 6),
                        "mean_completion_ratio": round(item["completion_ratio"], 6),
                    }
                )
    _write_csv(output_dir / "sensitivity_initial_conditions.csv", csv_rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.0),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.08,
            "right": 0.985,
            "bottom": 0.14,
            "top": 0.78,
            "hspace": 0.31,
            "wspace": 0.16,
        },
    )
    axes_flat = list(axes.flat)
    x = np.arange(len(bundle_order))
    x_labels = [dict(SENSITIVITY_BUNDLES)[key] for key in bundle_order]

    for ax, condition in zip(axes_flat, conditions, strict=False):
        for model in models:
            y_values = np.array(
                [
                    summary_by_bundle[bundle_key][(model, condition)]["success_rate"] * 100.0
                    for bundle_key in bundle_order
                ]
            )
            ax.plot(
                x,
                y_values,
                color=MODEL_COLORS.get(model, "#777777"),
                linewidth=1.55,
                marker="o",
                markersize=4.2,
                markeredgecolor="white",
                markeredgewidth=0.55,
            )
            ax.text(
                x[-1] + 0.04,
                y_values[-1],
                f"{y_values[-1]:.0f}",
                ha="left",
                va="center",
                fontsize=6.2,
                color=MODEL_COLORS.get(model, "#777777"),
            )
        ax.set_title(_condition_label(condition), loc="left", fontsize=8.2, fontweight="bold")
        ax.set_xlim(-0.15, len(bundle_order) - 0.72)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=2.5, color="#8c887f")

    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success (%)")

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS.get(model, "#777777"),
            marker="o",
            linewidth=1.6,
            label=_model_label(model),
        )
        for model in models
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.855),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
    )

    fig.text(
        0.02, 0.965, "E. Initial-condition sensitivity", fontsize=11.5, fontweight="bold", color=INK
    )
    fig.text(
        0.02,
        0.913,
        "Model/planner rankings across three nuisance-state perturbations",
        fontsize=8.9,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.872,
        (
            "Each point aggregates 14 scenarios; lines preserve the same model "
            "within a planner condition."
        ),
        fontsize=7.1,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.035,
        "Perturbation 2 is the canonical bundle used for the headline result.",
        fontsize=6.5,
        color=MUTED,
    )

    _save(fig, output_dir, "sensitivity_initial_conditions", formats, dpi)


def build_figures(
    *,
    canonical_bundle: Path,
    reference_bundle: Path,
    perturbation_1_bundle: Path,
    perturbation_2_bundle: Path,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    _set_theme()

    canonical_rows = _load_results(canonical_bundle)
    scenarios = _scenario_order(canonical_bundle)
    models = _ordered_models(canonical_rows)
    conditions = _ordered_conditions(canonical_rows)

    if len(scenarios) != 14:
        print(f"warning: expected 14 scenarios, found {len(scenarios)} in {canonical_bundle}")
    if len(models) != 4:
        print(f"warning: expected 4 models, found {len(models)} in {canonical_bundle}")
    if len(conditions) != 4:
        print(
            f"warning: expected 4 planner conditions, found {len(conditions)} in {canonical_bundle}"
        )

    plot_benchmark_schematic(
        rows=canonical_rows,
        scenarios=scenarios,
        models=models,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_main_success(
        rows=canonical_rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_scenario_heatmap(
        rows=canonical_rows,
        bundle_name=canonical_bundle.name,
        models=models,
        conditions=conditions,
        scenarios=scenarios,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_failure_taxonomy(
        rows=canonical_rows,
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
    plot_sensitivity(
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
        description="Generate the core LabVerse paper figures from frozen benchmark bundles."
    )
    parser.add_argument(
        "--canonical-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-2"),
        help="Bundle used for headline success, heatmap, and failure taxonomy figures.",
    )
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-reference"),
        help="Reference initial-state bundle for sensitivity figure.",
    )
    parser.add_argument(
        "--perturbation-1-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-1"),
        help="First initial-state perturbation bundle for sensitivity figure.",
    )
    parser.add_argument(
        "--perturbation-2-bundle",
        type=Path,
        default=Path("paper_runs/initial-state-perturbation-2"),
        help="Second initial-state perturbation bundle for sensitivity figure.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        choices=["pdf", "svg", "png"],
        help="Figure formats to write.",
    )
    parser.add_argument("--dpi", type=int, default=450, help="PNG export resolution.")
    args = parser.parse_args()

    build_figures(
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
