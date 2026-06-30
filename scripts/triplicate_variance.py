"""Robustness / variance analysis of the temp-0.2 triplicate bundle (item C).

Addresses the reviewer gap "no stochastic replication / no error bars". The
canonical paper bundles are temperature 0 (deterministic). This bundle reruns
the canonical state (perturbation-2) at temperature 0.2 with 3 replications, so
we can report decoding-noise error bars on per-condition success.

Infrastructure failures (subprocess crashes / GPU-pressure corruption on the
16 GB card) are excluded as non-results; genuine model failures (invalid action,
grounding failure, safety violation) are kept as legitimate failures.

Run: python3 scripts/triplicate_variance.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "paper_runs/triplicate-sensitivity_v2/results.jsonl"
CANON = ROOT / "paper_runs/initial-state-perturbation-2/results.jsonl"

CONDS = [
    ("local_structured_interactive_grounded", "LS"),
    ("local_narrative_interactive_grounded", "LN"),
    ("global_structured_interactive_grounded", "GS"),
    ("global_one_shot_planner", "OS"),
]
SHORT = dict(CONDS)

# Infra signatures -> excluded as non-results (environmental, not model behavior).
INFRA = [
    "Isolated subprocess crashed", "object is not subscriptable", "has no attribute",
    "unsupported operand", "PyUnicode_New", "SafeLoader", "Negative size",
    "provider failed", "Provider error", "request failed", "timeout", "HTTP 5",
    "check_token", "Connection",
]


def is_infra(err: object) -> bool:
    return any(s in str(err) for s in INFRA)


def valid_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return [r for r in rows if not (r.get("error") and is_infra(r["error"]))]


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (m, math.sqrt(var))


def main() -> None:
    trip = valid_rows(TRIP)
    print(f"valid triplicate rows (temp 0.2): {len(trip)}")

    # Per condition, per replication: mean success over that rep's valid cells.
    by_cond_rep: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in trip:
        if r["condition"] in SHORT:
            by_cond_rep[r["condition"]][r["replication"]].append(int(bool(r["success"])))

    # Canonical (temp 0) point estimate for comparison.
    canon = valid_rows(CANON)
    canon_rate: dict[str, float] = {}
    for cond, _ in CONDS:
        cells = [int(bool(r["success"])) for r in canon if r["condition"] == cond]
        canon_rate[cond] = (sum(cells) / len(cells)) if cells else 0.0

    print("\n=== Per-condition success under stochastic decoding (temp 0.2) ===")
    print(f"{'cond':5s} {'rep1':>7s} {'rep2':>7s} {'rep3':>7s}   {'mean±sd (0.2)':>16s} {'temp0':>7s}")
    order = []
    for cond, sh in CONDS:
        rep_rates = []
        cells_per_rep = []
        for rep in (1, 2, 3):
            cells = by_cond_rep[cond].get(rep, [])
            cells_per_rep.append(len(cells))
            rep_rates.append((sum(cells) / len(cells)) if cells else float("nan"))
        present = [x for x in rep_rates if not math.isnan(x)]
        m, sd = mean_std(present)
        order.append((sh, m))
        rep_str = " ".join(f"{x:7.3f}" if not math.isnan(x) else "    n/a" for x in rep_rates)
        print(f"{sh:5s} {rep_str}   {f'{m:.3f}±{sd:.3f}':>16s} {canon_rate[cond]:7.3f}")

    print("\n(rep columns = mean success over that replication's valid cells;")
    print(" mean±sd is across the 3 reps = decoding-noise error bar;")
    print(" temp0 = canonical perturbation-2 point estimate for comparison.)")

    ranking = " > ".join(s for s, _ in sorted(order, key=lambda x: -x[1]))
    print(f"\nranking under temp 0.2: {ranking}")

    # Does OS stay separated from interactive once we add error bars?
    os_m, os_sd = mean_std(
        [
            (sum(by_cond_rep["global_one_shot_planner"][rep]) / len(by_cond_rep["global_one_shot_planner"][rep]))
            for rep in (1, 2, 3)
            if by_cond_rep["global_one_shot_planner"].get(rep)
        ]
    )
    best_inter = max(
        mean_std(
            [
                sum(by_cond_rep[c][rep]) / len(by_cond_rep[c][rep])
                for rep in (1, 2, 3)
                if by_cond_rep[c].get(rep)
            ]
        )[0]
        for c in [
            "local_structured_interactive_grounded",
            "local_narrative_interactive_grounded",
            "global_structured_interactive_grounded",
        ]
    )
    print(
        f"\nOS mean−2sd = {os_m - 2 * os_sd:.3f}; best interactive mean = {best_inter:.3f} "
        f"-> {'OS separated' if os_m - 2 * os_sd > best_inter else 'OVERLAP'}"
    )


if __name__ == "__main__":
    main()
