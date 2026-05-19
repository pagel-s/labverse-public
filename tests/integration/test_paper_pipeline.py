"""End-to-end smoke test for the paper pipeline across all four primary conditions.

This test is gated on ``LABVERSE_OLLAMA_SMOKE=1`` because it requires a live
local ollama daemon and a real LLM call per run. It is the lightest
mechanical check that the full paper-run → status → analyze → package →
writeup pipeline works on a 4-condition bundle before spending frontier
model budget on the full 1600-run launch.

Run it explicitly before launching the real study::

    LABVERSE_OLLAMA_SMOKE=1 pytest tests/integration/test_paper_pipeline.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from labverse.benchmarks.conditions import PRIMARY_CONDITIONS
from labverse.paper import (
    analyze_paper_bundle,
    build_pipeline_health_report,
    build_writeup_packet,
    generate_paper_writeup,
    package_paper_bundle,
    run_paper_experiment,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("LABVERSE_OLLAMA_SMOKE") != "1",
    reason="set LABVERSE_OLLAMA_SMOKE=1 to run the 4-condition ollama pipeline smoke",
)


def test_paper_pipeline_runs_four_primary_conditions_end_to_end(tmp_path: Path) -> None:
    """Drive paper-run → status → analyze → package → writeup on the full
    4-condition primary matrix. Uses a small local ollama model so the test
    finishes in a few minutes without burning API budget."""

    model = os.environ.get("LABVERSE_OLLAMA_SMOKE_MODEL", "ollama/qwen3:14b")
    bundle_dir = tmp_path / "paper-4cond-smoke"

    bundle = run_paper_experiment(
        models=[model],
        conditions=list(PRIMARY_CONDITIONS),
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    # Every primary condition must have produced at least one run record.
    # PaperRunBundle exposes only file paths; records are read from results.jsonl.
    with open(bundle.results_path) as results_file:
        records = [json.loads(line) for line in results_file if line.strip()]
    observed_conditions = {record["condition"] for record in records}
    assert observed_conditions == set(PRIMARY_CONDITIONS), (
        f"missing conditions from run bundle: {set(PRIMARY_CONDITIONS) - observed_conditions}"
    )

    # Health gate: the bundle must be readable and not scale-blocked by any
    # structural issue (a provider error on a single run is tolerable in a
    # smoke; we only assert the pipeline itself did not fail).
    health = build_pipeline_health_report(bundle_dir)
    assert health.total_runs == len(PRIMARY_CONDITIONS), (
        f"expected {len(PRIMARY_CONDITIONS)} runs, got {health.total_runs}"
    )

    # Analysis must emit all 4C2 = 6 pairwise comparisons. analyze_paper_bundle
    # returns the analysis directory path; the structured report is persisted as
    # report.json inside it, which is what reviewers actually consume.
    analysis_dir = analyze_paper_bundle(bundle_dir)
    report = json.loads((analysis_dir / "report.json").read_text())
    analysis_conditions = set(report["by_condition"].keys())
    assert set(PRIMARY_CONDITIONS) <= analysis_conditions, (
        f"analysis missing conditions: {set(PRIMARY_CONDITIONS) - analysis_conditions}"
    )
    pairwise_comparisons = {row["comparison"] for row in report["pairwise"]}
    assert len(pairwise_comparisons) == 6, (
        f"expected 6 pairwise comparisons, got {len(pairwise_comparisons)}: "
        f"{sorted(pairwise_comparisons)}"
    )

    # Package and writeup must run without errors on the 4-condition bundle.
    package_paper_bundle(bundle_dir)
    writeup_packet = build_writeup_packet(bundle_dir)
    assert set(writeup_packet.conditions) >= set(PRIMARY_CONDITIONS)

    writeup_dir = generate_paper_writeup(bundle_dir)
    packet = json.loads((writeup_dir / "writeup_packet.json").read_text())

    # The writeup packet must name every primary condition (including the
    # newly-promoted global_one_shot_planner).
    for condition in PRIMARY_CONDITIONS:
        assert condition in packet["conditions"], f"writeup packet missing condition {condition}"
    assert "global_one_shot_planner" in packet["headline_by_condition"]
