"""Scenario-level success matrix in the house style (manuscript Figure 5).

A 2x2 grid of model panels; within each, interface (rows) x scenario S1-S14
(columns) binary success in the canonical state. Styled to match the journal
figure family used elsewhere (warm paper, DejaVu Sans, panel-header typography,
green success fill, thin gridlines) rather than the stark white/flat-green grid.

Run: .venv/bin/python scripts/plot_scenario_matrix.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "paper_runs/initial-state-perturbation-2/results.jsonl"

PAPER, INK, MUTED, HAIRLINE = "#ffffff", "#20242a", "#747b84", "#c9c1b4"
FAIL_FILL, SUCCESS_FILL = "#ededed", "#31a354"  # neutral miss vs green hit

MODELS = [
    ("ollama/gemma4:31b", "Gemma 31B"),
    ("ollama/gemma4:26b", "Gemma 26B"),
    ("ollama/qwen3.5:27b", "Qwen 27B"),
    ("ollama/mistral-small3.2:24b", "Mistral 24B"),
]
CONDS = [
    ("local_structured_interactive_grounded", "Local structured"),
    ("local_narrative_interactive_grounded", "Local narrative"),
    ("global_structured_interactive_grounded", "Global structured"),
    ("global_one_shot_planner", "One-shot"),
]
SCENARIOS = [
    "pcr_setup", "reagent_preparation", "cell_culture_passage", "chemical_spill_response",
    "equipment_failure_recovery", "safety_violation_detection", "inventory_management",
    "sample_tracking", "multi_step_protocol", "shared_equipment_scheduling",
    "branch_selection_under_uncertainty", "contamination_source_tracing",
    "cold_chain_time_critical_transfer", "emergency_triage_response",
]
SLABELS = [f"S{i}" for i in range(1, len(SCENARIOS) + 1)]


def set_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans", "font.size": 7.0, "axes.linewidth": 0.0,
            "xtick.color": "#343941", "ytick.color": "#343941",
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 450,
        }
    )


def success_cell(rows, model, cond, scenario) -> float:
    items = [r for r in rows if r["model"] == model and r["condition"] == cond
             and r["scenario"] == scenario]
    if not items:
        return float("nan")
    return sum(1 for r in items if r.get("success")) / len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "LLMBIO" / "figures"))
    ap.add_argument("--stem", default="Figure5")
    args = ap.parse_args()
    rows = [json.loads(line) for line in CANON.open()]

    set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.5))
    nrow, ncol = len(CONDS), len(SCENARIOS)

    for ax, (model, mlabel) in zip(axes.flat, MODELS):
        for ci, (cond, _) in enumerate(CONDS):
            for si, scenario in enumerate(SCENARIOS):
                v = success_cell(rows, model, cond, scenario)
                hit = (not v != v) and v >= 0.5  # v==v guards NaN
                ax.add_patch(
                    plt.Rectangle(
                        (si, nrow - 1 - ci), 1, 1,
                        facecolor=SUCCESS_FILL if hit else FAIL_FILL,
                        edgecolor=PAPER, linewidth=1.1,
                    )
                )
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect("equal")
        # scenario ticks (bottom)
        ax.set_xticks([i + 0.5 for i in range(ncol)])
        ax.set_xticklabels(SLABELS)
        ax.tick_params(axis="x", length=0, labelsize=5.6, labelcolor=MUTED)
        # interface ticks (left)
        ax.set_yticks([nrow - 1 - i + 0.5 for i in range(nrow)])
        ax.set_yticklabels([lab for _, lab in CONDS])
        ax.tick_params(axis="y", length=0, labelsize=6.4, labelcolor="#343941")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(mlabel, fontsize=8.0, fontweight="bold", color=INK, loc="left", pad=4)

    # shared legend, top-right, frameless
    fig.legend(
        handles=[
            Patch(facecolor=FAIL_FILL, edgecolor=HAIRLINE, linewidth=0.4, label="Failure"),
            Patch(facecolor=SUCCESS_FILL, edgecolor="none", label="Success"),
        ],
        loc="upper right", bbox_to_anchor=(0.995, 1.02), frameon=False, ncol=2,
        handlelength=1.0, columnspacing=1.1, fontsize=7.0,
    )
    fig.text(0.012, 1.005, "Scenario-level success in the canonical state",
             fontsize=8.5, fontweight="bold", color=INK, va="top")

    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.4, w_pad=2.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.stem}.{ext}", bbox_inches="tight")
    print(f"wrote {out/args.stem}.png and .pdf")


if __name__ == "__main__":
    main()
