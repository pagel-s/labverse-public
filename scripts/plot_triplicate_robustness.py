"""Figure 4: decoding-noise robustness from the temp-0.2 triplicate bundle.

Per-interface mean success (bars) with the three replications shown as points
and the std as a capped error bar; the temperature-0 canonical estimate is
overlaid as an open diamond. Styled to match the journal figure family
(panel-header typography, green interface gradient, off-white paper).

Run: .venv/bin/python scripts/plot_triplicate_robustness.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "paper_runs/triplicate-sensitivity_v2/results.jsonl"
CANON = ROOT / "paper_runs/initial-state-perturbation-2/results.jsonl"

PAPER, INK, MUTED, HAIRLINE, GRID = "#ffffff", "#20242a", "#747b84", "#c9c1b4", "#e7e0d4"
# Green interface gradient, matching the paper's planner-condition mapping.
COND = [
    ("local_structured_interactive_grounded", "Local\nstructured", "#c7e9c0"),
    ("local_narrative_interactive_grounded", "Local\nnarrative", "#a1d99b"),
    ("global_structured_interactive_grounded", "Global\nstructured", "#74c476"),
    ("global_one_shot_planner", "One-shot", "#31a354"),
]
INFRA = [
    "Isolated subprocess crashed", "object is not subscriptable", "has no attribute",
    "unsupported operand", "PyUnicode_New", "SafeLoader", "Negative size",
    "provider failed", "Provider error", "request failed", "timeout", "HTTP 5",
    "check_token", "Connection",
]


def is_infra(e: object) -> bool:
    return any(s in str(e) for s in INFRA)


def valid(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if not (r.get("error") and is_infra(r["error"]))]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0)
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)))


def set_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 8.0,
            "axes.labelsize": 7.5, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6, "axes.linewidth": 0.65, "axes.edgecolor": HAIRLINE,
            "xtick.color": "#343941", "ytick.color": "#343941", "axes.labelcolor": "#343941",
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 450,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "LLMBIO" / "figures"))
    ap.add_argument("--stem", default="Figure4")
    args = ap.parse_args()

    trip = valid(TRIP)
    by_cond_rep: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in trip:
        by_cond_rep[r["condition"]][r["replication"]].append(int(bool(r["success"])))

    canon = valid(CANON)
    canon_rate = {}
    for cond, _, _ in COND:
        cells = [int(bool(r["success"])) for r in canon if r["condition"] == cond]
        canon_rate[cond] = 100 * (sum(cells) / len(cells)) if cells else 0.0

    set_theme()
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    xs = list(range(len(COND)))
    rep_offsets = (-0.13, 0.0, 0.13)

    for x, (cond, _, color) in zip(xs, COND):
        rep_rates = [
            100 * (sum(v) / len(v))
            for rep in (1, 2, 3)
            if (v := by_cond_rep[cond].get(rep))
        ]
        m, sd = mean_sd(rep_rates)
        # mean bar
        ax.bar(x, m, width=0.42, color=color, edgecolor="none", zorder=2)
        # std whisker
        ax.errorbar(
            x, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.9, capsize=3.0,
            capthick=0.9, zorder=4,
        )
        # individual replication points (the robustness evidence)
        for off, rr in zip(rep_offsets, rep_rates):
            ax.plot(
                x + off, rr, marker="o", markersize=2.8, markerfacecolor=PAPER,
                markeredgecolor=INK, markeredgewidth=0.7, zorder=5,
            )
        # temperature-0 canonical estimate
        ax.plot(
            x, canon_rate[cond], marker="D", markersize=4.6, markerfacecolor="none",
            markeredgecolor="#a4746a", markeredgewidth=1.0, zorder=6,
        )
        ax.text(x, m + sd + 2.6, f"{m:.0f}", ha="center", va="bottom", fontsize=6.6, color=MUTED)

    ax.set_ylim(0, 82)
    ax.set_yticks(range(0, 81, 20))
    ax.set_ylabel("Mean task success (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, _ in COND])
    ax.set_xlim(-0.6, len(COND) - 0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.text(
        0.0, 1.04, "Success is stable across replications (temperature 0.2)",
        transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=INK,
        ha="left", va="bottom",
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markeredgewidth=0.7, markersize=3.2,
               label="Replication (n=3)"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="none",
               markeredgecolor="#a4746a", markeredgewidth=1.0, markersize=4.2,
               label="Temperature 0"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)

    fig.tight_layout(pad=0.5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.stem}.{ext}", bbox_inches="tight")
    print(f"wrote {out/args.stem}.png and .pdf")


if __name__ == "__main__":
    main()
