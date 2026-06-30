"""Figure 4: decoding-noise robustness from the temp-0.2 triplicate bundle.

Per-condition mean success with error bars (std across 3 replications), with the
temperature-0 canonical point estimate overlaid. Matches the journal/styled
figure theme. Writes Figure4.{png,pdf} to the manuscript figures dir.

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

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "paper_runs/triplicate-sensitivity_v2/results.jsonl"
CANON = ROOT / "paper_runs/initial-state-perturbation-2/results.jsonl"

PAPER, INK, MUTED, HAIRLINE = "#fcfbf7", "#20242a", "#747b84", "#c9c1b4"
COND = [
    ("local_structured_interactive_grounded", "Local\nstructured", "#c7e9c0"),
    ("local_narrative_interactive_grounded", "Local\nnarrative", "#a1d99b"),
    ("global_structured_interactive_grounded", "Global\nstructured", "#74c476"),
    ("global_one_shot_planner", "Global\none-shot", "#31a354"),
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
            "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4, "axes.linewidth": 0.65, "axes.edgecolor": HAIRLINE,
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

    means, sds, labels, colors, canon_pts = [], [], [], [], []
    for cond, label, color in COND:
        rep_rates = [
            sum(v) / len(v)
            for rep in (1, 2, 3)
            if (v := by_cond_rep[cond].get(rep))
        ]
        m, sd = mean_sd(rep_rates)
        means.append(100 * m)
        sds.append(100 * sd)
        labels.append(label)
        colors.append(color)
        canon_pts.append(canon_rate[cond])

    set_theme()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = range(len(COND))
    ax.bar(
        x, means, yerr=sds, width=0.62, color=colors, edgecolor=INK, linewidth=0.6,
        error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
        zorder=2,
    )
    # temperature-0 point estimate as a reference marker
    ax.scatter(
        x, canon_pts, marker="D", s=14, facecolor=PAPER, edgecolor=INK,
        linewidth=0.8, zorder=3, label="Temp 0 (canonical)",
    )
    for xi, m, sd in zip(x, means, sds):
        ax.text(xi, m + sd + 2.2, f"{m:.0f}", ha="center", va="bottom", fontsize=6.3, color=INK)

    ax.set_ylim(0, 85)
    ax.set_ylabel("Mean task success (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, color="#e7e0d4", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4)
    ax.set_title("Robustness to decoding noise (temp 0.2, 3 reps)", color=INK)
    fig.tight_layout(pad=0.4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.stem}.{ext}", bbox_inches="tight")
    print(f"wrote {out/args.stem}.png and .pdf")
    print("bars (mean±sd):", [f"{l.replace(chr(10),' ')}: {m:.1f}±{s:.1f}" for l, m, s in zip(labels, means, sds)])


if __name__ == "__main__":
    main()
