#!/usr/bin/env python3
"""Plot planner success across frozen paper scenarios for one model."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import wrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = {
    "local_structured_interactive_grounded": "Local\nstructured",
    "local_narrative_interactive_grounded": "Local\nnarrative",
    "global_structured_interactive_grounded": "Global\nstructured",
    "global_one_shot_planner": "One-shot\nplanner",
}

CONDITION_ORDER = [
    "local_structured_interactive_grounded",
    "local_narrative_interactive_grounded",
    "global_structured_interactive_grounded",
    "global_one_shot_planner",
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenario_order(bundle_dir: Path) -> list[str]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [item["name"] for item in manifest["scenarios"]]


def _model_display_name(model: str) -> str:
    return model.removeprefix("ollama/")


def _wrap_title(text: str, width: int = 84) -> str:
    return "\n".join(wrap(text, width=width))


def build_figure(bundle_dir: Path, model: str) -> plt.Figure:
    rows = [row for row in _load_jsonl(bundle_dir / "results.jsonl") if row["model"] == model]
    if not rows:
        raise ValueError(f"No rows found for model {model!r} in {bundle_dir}")

    scenario_order = _scenario_order(bundle_dir)
    by_key = {(row["condition"], row["scenario"]): row for row in rows}
    matrix = np.full((len(CONDITION_ORDER), len(scenario_order)), np.nan)
    completion = np.full_like(matrix, np.nan, dtype=float)

    for condition_index, condition in enumerate(CONDITION_ORDER):
        for scenario_index, scenario in enumerate(scenario_order):
            row = by_key.get((condition, scenario))
            if row is None:
                continue
            matrix[condition_index, scenario_index] = 1.0 if row["success"] else 0.0
            completion[condition_index, scenario_index] = float(row["completion_ratio"])

    success_rates = np.nanmean(matrix, axis=1)
    mean_completion = np.nanmean(completion, axis=1)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(8.6, 3.55), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[14, 3.2],
        left=0.075,
        right=0.985,
        bottom=0.36,
        top=0.78,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1], sharey=ax)

    cmap = mpl.colors.ListedColormap(["#f2f0e8", "#0e7c7b"])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels(
        [f"Scenario {i}" for i in range(1, len(scenario_order) + 1)],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(CONDITION_ORDER)))
    ax.set_yticklabels([CONDITION_LABELS[item] for item in CONDITION_ORDER])
    ax.set_xlabel("Frozen scenario", labelpad=14)
    ax.set_ylabel("Planner condition")

    ax.set_xticks(np.arange(-0.5, len(scenario_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONDITION_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isnan(matrix[y, x]):
                continue
            text = "✓" if matrix[y, x] == 1 else "×"
            color = "white" if matrix[y, x] == 1 else "#6f6a60"
            ax.text(x, y, text, ha="center", va="center", color=color, fontsize=7.8)

    bar_colors = ["#2d6f93", "#2d6f93", "#2d6f93", "#0e7c7b"]
    y_positions = np.arange(len(CONDITION_ORDER))
    ax_bar.barh(y_positions, success_rates * 100, color=bar_colors, height=0.56)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Success (%)")
    ax_bar.set_xticks([0, 50, 100])
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.tick_params(axis="x", length=2.5, color="#8a867c")
    ax_bar.grid(axis="x", color="#ded9cc", linewidth=0.8)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["bottom"].set_color("#9b9588")

    for y, rate, comp in zip(y_positions, success_rates, mean_completion, strict=True):
        ax_bar.text(
            min(rate * 100 + 2.0, 96.0),
            y,
            f"{rate * 100:.0f}%",
            va="center",
            ha="left" if rate < 0.92 else "right",
            color="#222222",
            fontsize=8.2,
            fontweight="bold",
        )
        ax_bar.text(
            100,
            y + 0.32,
            f"c={comp:.2f}",
            va="center",
            ha="right",
            color="#6f6a60",
            fontsize=6.8,
        )

    fig.text(
        0.075,
        0.925,
        _wrap_title(
            "Planner-level success across 14 frozen LabVerse scenarios",
        ),
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.075,
        0.84,
        (
            f"Model: {_model_display_name(model)} · one replicate per scenario/planner · "
            "cells show binary task success; c = mean completion ratio"
        ),
        ha="left",
        va="top",
        fontsize=7.6,
        color="#5e5a52",
    )
    fig.text(
        0.075,
        0.035,
        "Scenario labels are anonymized in frozen-manifest order (Scenario 1–14).",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#6f6a60",
    )
    fig.text(
        0.985,
        0.035,
        f"Bundle: {bundle_dir.name}",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#6f6a60",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--model", default="ollama/gemma4:31b")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--stem", default="planner_success_gemma4_31b")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.bundle_dir, args.model)
    for suffix in ("pdf", "svg", "png"):
        path = args.output_dir / f"{args.stem}.{suffix}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            save_kwargs["dpi"] = 450
        fig.savefig(path, **save_kwargs)
        print(path)


if __name__ == "__main__":
    main()
