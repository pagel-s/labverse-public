"""Statistical analysis of LabVerse paper bundles (stdlib only, deterministic).

Produces, from the three frozen initial-state bundles:
  1. Per-condition success with Wilson 95% CI, and across-state mean +/- range.
  2. Paired McNemar exact tests for the key interface contrasts (canonical state).
  3. Cluster bootstrap CIs treating the 14 scenarios as the replication unit.
  4. Failure-taxonomy decomposition by condition (task vs contract vs infra).
  5. Weight-robustness: condition ranking under reported vs equal criterion weights.

Run: python3 scripts/paper_statistics.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {
    "reference": ROOT / "paper_runs/initial-state-reference",
    "perturbation-1": ROOT / "paper_runs/initial-state-perturbation-1",
    "perturbation-2": ROOT / "paper_runs/initial-state-perturbation-2",
}
CANONICAL = "perturbation-2"

CONDITIONS = [
    ("local_structured_interactive_grounded", "LS"),
    ("local_narrative_interactive_grounded", "LN"),
    ("global_structured_interactive_grounded", "GS"),
    ("global_one_shot_planner", "OS"),
]
SHORT = dict(CONDITIONS)

# Taxonomy grouping: is a failure the model's task ability, the interaction
# contract, or infrastructure outside the model's control?
TAXO_GROUP = {
    "none": "success",
    "safety_failure": "safety",
    "scenario_behavior_failure": "task_incomplete",
    "grounding_failure": "contract/grounding",
    "parameter_failure": "contract/grounding",
    "sequencing_failure": "contract/grounding",
    "runner_failure": "infrastructure",
    "provider_failure": "infrastructure",
    "artifact_integrity_failure": "infrastructure",
}


def load(bundle: Path) -> list[dict]:
    return [json.loads(line) for line in (bundle / "results.jsonl").open()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def cell_key(r: dict) -> tuple[str, str]:
    return (r["model"], r["scenario"])


def per_condition_success(rows: list[dict]) -> dict[str, list[int]]:
    """condition -> list of 0/1 success over its (model, scenario) cells."""
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        cond = r["condition"]
        if cond in SHORT:
            out[cond].append(int(bool(r["success"])))
    return out


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    data = {name: load(p) for name, p in BUNDLES.items()}

    # ---- 1. Per-condition success: Wilson CI + across-state variability ----
    section("1. Per-condition success rate (Wilson 95% CI) and across-state range")
    per_state_rate: dict[str, dict[str, float]] = {}
    for name, rows in data.items():
        succ = per_condition_success(rows)
        per_state_rate[name] = {c: sum(v) / len(v) for c, v in succ.items()}
    print(f"{'condition':10s} " + " ".join(f"{n[:11]:>13s}" for n in data) + f"{'mean±range':>16s}")
    for cond, sh in CONDITIONS:
        rates = [per_state_rate[n][cond] for n in data]
        cells = []
        for n in data:
            succ = per_condition_success(data[n])[cond]
            k, tot = sum(succ), len(succ)
            lo, hi = wilson_ci(k, tot)
            cells.append(f"{k:2d}/{tot} [{lo:.2f},{hi:.2f}]")
        mean = sum(rates) / len(rates)
        rng = max(rates) - min(rates)
        print(f"{sh:10s} " + " ".join(f"{c:>13s}" for c in cells) + f"  {mean:.3f}±{rng/2:.3f}")
    print("\n(Each cell n=56 = 4 models x 14 scenarios. CI is binomial Wilson;")
    print(" the across-state range is the no-extra-compute variability estimate.)")

    # ---- 2. Paired McNemar exact tests (canonical state) ----
    section(f"2. Paired McNemar exact tests, canonical state ({CANONICAL})")
    rows = data[CANONICAL]
    by_cond_cell: dict[str, dict[tuple, int]] = defaultdict(dict)
    for r in rows:
        if r["condition"] in SHORT:
            by_cond_cell[r["condition"]][cell_key(r)] = int(bool(r["success"]))
    pairs = [
        ("global_one_shot_planner", "global_structured_interactive_grounded"),
        ("global_one_shot_planner", "local_narrative_interactive_grounded"),
        ("global_one_shot_planner", "local_structured_interactive_grounded"),
        ("local_narrative_interactive_grounded", "global_structured_interactive_grounded"),
        ("local_narrative_interactive_grounded", "local_structured_interactive_grounded"),
        ("global_structured_interactive_grounded", "local_structured_interactive_grounded"),
    ]
    print(f"{'contrast':12s} {'A>B':>4s} {'B>A':>4s} {'disc':>5s} {'p(exact)':>10s}  sig")
    for a, b in pairs:
        ca, cb = by_cond_cell[a], by_cond_cell[b]
        keys = set(ca) & set(cb)
        b_ab = sum(1 for k in keys if ca[k] == 1 and cb[k] == 0)
        c_ab = sum(1 for k in keys if ca[k] == 0 and cb[k] == 1)
        p = mcnemar_exact(b_ab, c_ab)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{SHORT[a]+' vs '+SHORT[b]:12s} {b_ab:4d} {c_ab:4d} {b_ab+c_ab:5d} {p:10.4f}  {star}")
    print("\n(A>B = cells where A succeeds and B fails. Exact two-sided binomial on")
    print(" discordant pairs. n=56 paired model x scenario cells.)")

    # ---- 3. Cluster bootstrap CI (scenario = unit of replication) ----
    section(f"3. Cluster bootstrap 95% CI, scenarios resampled ({CANONICAL}, 10000 reps)")
    random.seed(7)
    scenarios = sorted({r["scenario"] for r in rows})
    # per scenario, per condition: mean success across the 4 models
    sc_cond: dict[str, dict[str, float]] = defaultdict(dict)
    grp: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if r["condition"] in SHORT:
            grp[(r["scenario"], r["condition"])].append(int(bool(r["success"])))
    for (sc, cond), v in grp.items():
        sc_cond[sc][cond] = sum(v) / len(v)
    B = 10000
    boot: dict[str, list[float]] = {c: [] for c, _ in CONDITIONS}
    for _ in range(B):
        sample = [random.choice(scenarios) for _ in scenarios]
        for cond, _ in CONDITIONS:
            boot[cond].append(sum(sc_cond[sc][cond] for sc in sample) / len(sample))
    print(f"{'condition':10s} {'mean':>7s} {'95% CI (cluster bootstrap)':>30s}")
    for cond, sh in CONDITIONS:
        vals = sorted(boot[cond])
        mean = sum(vals) / len(vals)
        lo, hi = vals[int(0.025 * B)], vals[int(0.975 * B)]
        print(f"{sh:10s} {mean:7.3f} {f'[{lo:.3f}, {hi:.3f}]':>30s}")
    print("\n(Resamples the 14 scenarios with replacement -- the honest replication")
    print(" unit -- so CIs reflect scenario-to-scenario variation, not n=224.)")

    # ---- 4. Failure-taxonomy decomposition by condition (canonical) ----
    section(f"4. Failure taxonomy by condition ({CANONICAL})")
    groups = ["safety", "task_incomplete", "contract/grounding", "infrastructure"]
    print(f"{'cond':5s} {'succ':>5s} " + " ".join(f"{g[:13]:>14s}" for g in groups) + f"{'fail n':>7s}")
    for cond, sh in CONDITIONS:
        crows = [r for r in rows if r["condition"] == cond]
        gc = Counter(TAXO_GROUP.get(str(r.get("error_taxonomy")), "other") for r in crows)
        succ = gc.get("success", 0)
        fails = len(crows) - succ
        cells = []
        for g in groups:
            n = gc.get(g, 0)
            pct = (100 * n / fails) if fails else 0
            cells.append(f"{n:2d} ({pct:4.0f}%)")
        print(f"{sh:5s} {succ:5d} " + " ".join(f"{c:>14s}" for c in cells) + f"{fails:7d}")
    print("\n(% is share of that condition's FAILURES. 'contract/grounding' =")
    print(" grounding+parameter+sequencing = interaction-contract errors;")
    print(" 'infrastructure' = runner/provider/artifact, outside model control.)")

    # ---- 5. Weight-robustness: ranking under reported vs equal weights ----
    section(f"5. Completion ranking robustness to criterion weights ({CANONICAL})")
    print("Reported completion_ratio (scenario weights) vs equal-weight recompute")
    print("from per-criterion booleans in `details`:\n")
    print(f"{'cond':5s} {'reported':>9s} {'equal-wt':>9s} {'delta':>7s}")
    rep_rank, eq_rank = {}, {}
    for cond, sh in CONDITIONS:
        crows = [r for r in rows if r["condition"] == cond]
        rep = sum(r.get("completion_ratio", 0) for r in crows) / len(crows)
        eqs = []
        for r in crows:
            d = r.get("details") or {}
            bools = [v for v in d.values() if isinstance(v, bool)]
            eqs.append(sum(bools) / len(bools) if bools else 0.0)
        eq = sum(eqs) / len(eqs)
        rep_rank[sh], eq_rank[sh] = rep, eq
        print(f"{sh:5s} {rep:9.3f} {eq:9.3f} {eq-rep:7.3f}")
    order_rep = [s for s, _ in sorted(rep_rank.items(), key=lambda x: -x[1])]
    order_eq = [s for s, _ in sorted(eq_rank.items(), key=lambda x: -x[1])]
    print(f"\nranking (reported): {' > '.join(order_rep)}")
    print(f"ranking (equal wt): {' > '.join(order_eq)}")
    print("stable" if order_rep == order_eq else "RANKING CHANGES under reweighting")


if __name__ == "__main__":
    main()
