"""Writeup packet generation for the frozen grounded-interaction paper."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    PRIMARY_CONDITIONS,
    normalize_condition_name,
)
from labverse.paper.analysis import (
    PAIRWISE_CONDITIONS,
    GroupedConditionSummary,
    PairwiseMetricReport,
    PaperAnalysisReport,
    analyze_paper_bundle,
)
from labverse.paper.judgment import (
    ScenarioJudgment,
    load_scenario_judgment_matrix,
)
from labverse.paper.manifest import PaperManifest, load_paper_manifest
from labverse.paper.runner import PaperRunRecord, load_paper_run_records
from labverse.paper.validate import assert_valid_paper_bundle

PRIMARY_PAPER_CONDITIONS = list(PRIMARY_CONDITIONS)
PRIMARY_WRITEUP_METRICS = [
    "success_rate",
    "mean_completion_ratio",
    "mean_safety_violations",
    "mean_blocked_actions",
    "mean_steps_taken",
    "mean_time_elapsed_seconds",
    "mean_prompt_tokens",
    "mean_completion_tokens",
]


def _condition_label(condition: str) -> str:
    labels = {
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED: "local structured interactive grounded",
        CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED: "local narrative interactive grounded",
        CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED: "global structured interactive grounded",
        CONDITION_GLOBAL_ONE_SHOT_PLANNER: "global one-shot planner",
    }
    return labels.get(condition, condition)


class ScenarioConditionSummary(BaseModel):
    scenario: str
    condition: str
    n: int
    success_rate: float
    mean_completion_ratio: float
    mean_safety_violations: float
    mean_blocked_actions: float
    mean_steps_taken: float
    mean_time_elapsed_seconds: float
    mean_prompt_tokens: float
    mean_completion_tokens: float


class ModelConditionSummary(BaseModel):
    model: str
    condition: str
    n: int
    success_rate: float
    mean_completion_ratio: float
    mean_safety_violations: float
    mean_blocked_actions: float
    mean_prompt_tokens: float
    mean_completion_tokens: float


class TaxonomySummary(BaseModel):
    taxonomy_axis: str
    taxonomy_value: str
    condition: str
    n: int
    success_rate: float
    mean_completion_ratio: float
    mean_safety_violations: float
    mean_blocked_actions: float


class PaperWriteupPacket(BaseModel):
    bundle_dir: str
    analysis_dir: str
    manifest_id: str
    profile_name: str
    replications: int
    total_runs: int
    conditions: list[str]
    models: list[str]
    scenarios: list[str]
    ready_for_main_results: bool
    readiness_notes: list[str] = Field(default_factory=list)
    headline_by_condition: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    pairwise: list[PairwiseMetricReport] = Field(default_factory=list)
    scenario_summaries: list[ScenarioConditionSummary] = Field(default_factory=list)
    model_summaries: list[ModelConditionSummary] = Field(default_factory=list)
    taxonomy_summaries: list[TaxonomySummary] = Field(default_factory=list)
    task_family_summaries: list[GroupedConditionSummary] = Field(default_factory=list)
    paper_role_summaries: list[GroupedConditionSummary] = Field(default_factory=list)
    scenario_judgments: list[ScenarioJudgment] = Field(default_factory=list)
    strongest_local_grounding_gains: list[dict[str, str | float]] = Field(default_factory=list)
    hardest_scenarios: list[dict[str, str | float]] = Field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scenario_condition_summaries(
    records: list[PaperRunRecord],
) -> list[ScenarioConditionSummary]:
    grouped: dict[tuple[str, str], list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        condition = normalize_condition_name(record.condition)
        if condition not in PRIMARY_PAPER_CONDITIONS:
            continue
        grouped[(record.scenario, condition)].append(record)

    summaries: list[ScenarioConditionSummary] = []
    for (scenario, condition), items in sorted(grouped.items()):
        n = len(items)
        summaries.append(
            ScenarioConditionSummary(
                scenario=scenario,
                condition=condition,
                n=n,
                success_rate=sum(1 for item in items if item.success) / n,
                mean_completion_ratio=_mean([item.completion_ratio for item in items]),
                mean_safety_violations=_mean([float(item.safety_violations) for item in items]),
                mean_blocked_actions=_mean([float(item.blocked_actions) for item in items]),
                mean_steps_taken=_mean([float(item.steps_taken) for item in items]),
                mean_time_elapsed_seconds=_mean([item.time_elapsed_seconds for item in items]),
                mean_prompt_tokens=_mean([float(item.prompt_tokens) for item in items]),
                mean_completion_tokens=_mean([float(item.completion_tokens) for item in items]),
            )
        )
    return summaries


def _model_condition_summaries(records: list[PaperRunRecord]) -> list[ModelConditionSummary]:
    grouped: dict[tuple[str, str], list[PaperRunRecord]] = defaultdict(list)
    for record in records:
        condition = normalize_condition_name(record.condition)
        if condition not in PRIMARY_PAPER_CONDITIONS:
            continue
        grouped[(record.model, condition)].append(record)

    summaries: list[ModelConditionSummary] = []
    for (model, condition), items in sorted(grouped.items()):
        n = len(items)
        summaries.append(
            ModelConditionSummary(
                model=model,
                condition=condition,
                n=n,
                success_rate=sum(1 for item in items if item.success) / n,
                mean_completion_ratio=_mean([item.completion_ratio for item in items]),
                mean_safety_violations=_mean([float(item.safety_violations) for item in items]),
                mean_blocked_actions=_mean([float(item.blocked_actions) for item in items]),
                mean_prompt_tokens=_mean([float(item.prompt_tokens) for item in items]),
                mean_completion_tokens=_mean([float(item.completion_tokens) for item in items]),
            )
        )
    return summaries


def _taxonomy_summaries(
    records: list[PaperRunRecord],
    manifest: PaperManifest,
) -> list[TaxonomySummary]:
    scenario_map = manifest.scenario_map()
    summaries: list[TaxonomySummary] = []
    for axis in ("spatial_complexity", "safety_complexity", "category"):
        grouped: dict[tuple[str, str], list[PaperRunRecord]] = defaultdict(list)
        for record in records:
            condition = normalize_condition_name(record.condition)
            if condition not in PRIMARY_PAPER_CONDITIONS:
                continue
            scenario = scenario_map[record.scenario]
            grouped[(getattr(scenario, axis), condition)].append(record)
        for (axis_value, condition), items in sorted(grouped.items()):
            n = len(items)
            summaries.append(
                TaxonomySummary(
                    taxonomy_axis=axis,
                    taxonomy_value=axis_value,
                    condition=condition,
                    n=n,
                    success_rate=sum(1 for item in items if item.success) / n,
                    mean_completion_ratio=_mean([item.completion_ratio for item in items]),
                    mean_safety_violations=_mean([float(item.safety_violations) for item in items]),
                    mean_blocked_actions=_mean([float(item.blocked_actions) for item in items]),
                )
            )
    return summaries


def _pairwise_lookup(
    pairwise: list[PairwiseMetricReport],
    comparison: str,
    metric: str,
) -> PairwiseMetricReport | None:
    for row in pairwise:
        if row.comparison == comparison and row.metric == metric:
            return row
    return None


def _strongest_local_grounding_gains(
    scenario_summaries: list[ScenarioConditionSummary],
) -> list[dict[str, str | float]]:
    by_scenario: dict[str, dict[str, ScenarioConditionSummary]] = defaultdict(dict)
    for summary in scenario_summaries:
        by_scenario[summary.scenario][summary.condition] = summary

    rows: list[dict[str, str | float]] = []
    for scenario, summaries in by_scenario.items():
        local_structured = summaries.get(CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED)
        global_structured = summaries.get(CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED)
        if local_structured is None or global_structured is None:
            continue
        rows.append(
            {
                "scenario": scenario,
                "completion_delta_vs_global_structured": (
                    local_structured.mean_completion_ratio - global_structured.mean_completion_ratio
                ),
                "success_delta_vs_global_structured": (
                    local_structured.success_rate - global_structured.success_rate
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: cast(float, row["completion_delta_vs_global_structured"]),
        reverse=True,
    )[:3]


def _hardest_local_structured_scenarios(
    scenario_summaries: list[ScenarioConditionSummary],
) -> list[dict[str, str | float]]:
    local_structured_rows: list[dict[str, str | float]] = [
        {
            "scenario": summary.scenario,
            "local_structured_completion_ratio": summary.mean_completion_ratio,
            "local_structured_success_rate": summary.success_rate,
        }
        for summary in scenario_summaries
        if summary.condition == CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED
    ]
    return sorted(
        local_structured_rows,
        key=lambda row: (
            cast(float, row["local_structured_completion_ratio"]),
            cast(float, row["local_structured_success_rate"]),
        ),
    )[:3]


def _judgment_rows(
    scenario_judgments: list[ScenarioJudgment],
    *,
    paper_role: str,
) -> list[str]:
    return [
        (f"- {judgment.scenario_name}: {judgment.task_family} / {judgment.expected_best_condition}")
        for judgment in scenario_judgments
        if judgment.paper_role == paper_role
    ]


def _readiness_notes(
    *,
    manifest: PaperManifest,
    analysis_report: PaperAnalysisReport,
    records: list[PaperRunRecord],
    conditions: list[str],
    models: list[str],
    scenarios: list[str],
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ready = True

    missing_conditions = [
        condition for condition in PRIMARY_PAPER_CONDITIONS if condition not in conditions
    ]
    if missing_conditions:
        ready = False
        notes.append("Missing primary paper conditions: " + ", ".join(missing_conditions))
    if not models:
        ready = False
        notes.append("No model-backed paper conditions are present in this bundle.")
    if set(scenarios) != set(manifest.scenario_names()):
        ready = False
        notes.append(f"Bundle does not cover all {len(manifest.scenarios)} frozen paper scenarios.")
    complete_cells = analysis_report.complete_cells
    expected_cells = analysis_report.expected_cells
    if complete_cells != expected_cells:
        ready = False
        notes.append(f"Bundle is incomplete: {complete_cells}/{expected_cells} complete cells.")
    if not any(
        normalize_condition_name(record.condition)
        == CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED
        for record in records
    ):
        ready = False
        notes.append("No local structured interactive rows available for the main paper claim.")
    if ready:
        notes.append("Bundle is complete enough for the main manuscript results sections.")
    return ready, notes


def build_writeup_packet(
    bundle_dir: str | Path,
    *,
    analysis_output_dir: str | Path | None = None,
) -> PaperWriteupPacket:
    """Build derived writeup tables and summaries from a frozen paper bundle."""

    bundle_path = Path(bundle_dir)
    assert_valid_paper_bundle(bundle_path)
    analysis_dir = analyze_paper_bundle(bundle_path, output_dir=analysis_output_dir)
    metadata = json.loads((bundle_path / "bundle_metadata.json").read_text())
    analysis_report = PaperAnalysisReport.model_validate_json(
        (analysis_dir / "report.json").read_text()
    )
    records = load_paper_run_records(bundle_path / "results.jsonl")
    manifest = load_paper_manifest(bundle_path / "manifest.json")
    judgment_matrix = load_scenario_judgment_matrix(manifest=manifest)

    scenario_summaries = _scenario_condition_summaries(records)
    model_summaries = _model_condition_summaries(records)
    taxonomy_summaries = _taxonomy_summaries(records, manifest)
    pairwise = list(analysis_report.pairwise)
    ready, readiness_notes = _readiness_notes(
        manifest=manifest,
        analysis_report=analysis_report,
        records=records,
        conditions=[normalize_condition_name(condition) for condition in metadata["conditions"]],
        models=list(metadata["models"]),
        scenarios=list(metadata["scenarios"]),
    )

    return PaperWriteupPacket(
        bundle_dir=str(bundle_path),
        analysis_dir=str(analysis_dir),
        manifest_id=manifest.manifest_id,
        profile_name=str(metadata["profile_name"]),
        replications=int(metadata["replications"]),
        total_runs=analysis_report.total_runs,
        conditions=[normalize_condition_name(condition) for condition in metadata["conditions"]],
        models=list(metadata["models"]),
        scenarios=list(metadata["scenarios"]),
        ready_for_main_results=ready,
        readiness_notes=readiness_notes,
        headline_by_condition={
            condition: dict(summary) for condition, summary in analysis_report.by_condition.items()
        },
        pairwise=pairwise,
        scenario_summaries=scenario_summaries,
        model_summaries=model_summaries,
        taxonomy_summaries=taxonomy_summaries,
        task_family_summaries=list(analysis_report.task_family_summaries),
        paper_role_summaries=list(analysis_report.paper_role_summaries),
        scenario_judgments=list(judgment_matrix.scenarios),
        strongest_local_grounding_gains=_strongest_local_grounding_gains(scenario_summaries),
        hardest_scenarios=_hardest_local_structured_scenarios(scenario_summaries),
    )


def _fmt_percent(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _fmt_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _write_csv(path: Path, header: list[str], rows: list[list[str | float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _render_manuscript_blueprint(packet: PaperWriteupPacket) -> str:
    headline_by_condition: dict[str, dict[str, float | int]] = {
        condition: packet.headline_by_condition.get(condition, {})
        for condition in PRIMARY_CONDITIONS
    }
    local_structured = headline_by_condition[CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED]
    local_narrative = headline_by_condition[CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED]
    global_structured = headline_by_condition[CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED]
    global_one_shot = headline_by_condition[CONDITION_GLOBAL_ONE_SHOT_PLANNER]
    completion_pairwise_deltas: list[str] = []
    for left, right in PAIRWISE_CONDITIONS:
        pair = _pairwise_lookup(
            packet.pairwise,
            f"{left} - {right}",
            "completion_ratio",
        )
        delta = _fmt_float(pair.mean_delta) if pair else "n/a"
        completion_pairwise_deltas.append(
            f"- {_condition_label(left)} vs {_condition_label(right)} completion delta: {delta}"
        )

    strongest = (
        "\n".join(
            (
                f"- {row['scenario']}: Δ completion vs global structured = "
                f"{float(row['completion_delta_vs_global_structured']):.2f}"
            )
            for row in packet.strongest_local_grounding_gains
        )
        or "- Pending full primary bundle"
    )
    hardest = (
        "\n".join(
            (
                f"- {row['scenario']}: local structured completion = "
                f"{float(row['local_structured_completion_ratio']):.2f}"
            )
            for row in packet.hardest_scenarios
        )
        or "- Pending full primary bundle"
    )
    readiness = "\n".join(f"- {note}" for note in packet.readiness_notes)
    flagship = "\n".join(_judgment_rows(packet.scenario_judgments, paper_role="flagship"))
    flagship_secondary = "\n".join(
        _judgment_rows(packet.scenario_judgments, paper_role="flagship_secondary")
    )
    boundary = "\n".join(_judgment_rows(packet.scenario_judgments, paper_role="boundary_case"))
    anchors = "\n".join(_judgment_rows(packet.scenario_judgments, paper_role="anchor"))

    return "\n".join(
        [
            "# LabVerse Embodiment Paper Blueprint",
            "",
            "## Bundle Status",
            "",
            f"- Bundle: `{packet.bundle_dir}`",
            f"- Profile: `{packet.profile_name}`",
            f"- Replications per complete cell: `{packet.replications}`",
            f"- Total runs in bundle: `{packet.total_runs}`",
            f"- Models: `{', '.join(packet.models) if packet.models else 'none'}`",
            f"- Ready for main results: `{packet.ready_for_main_results}`",
            *readiness.splitlines(),
            "",
            "## Paper Story",
            "",
            "Primary claim:",
            (
                "- Local, grounded, interactive control improves laboratory-task performance "
                "on execution-heavy tasks, while global structured access can remain competitive "
                "or superior on inspection-style tasks."
            ),
            "",
            "Main hypotheses:",
            (
                "- H1: local structured interactive control outperforms local narrative "
                "interactive control on execution-heavy tasks."
            ),
            (
                "- H2: local structured interactive control outperforms global structured "
                "interactive control on timed, safety-critical, resource-contention, and "
                "multi-room execution tasks."
            ),
            (
                "- H3: global structured interactive control can match or beat local control "
                "on global inspection and search tasks."
            ),
            "- H4: stronger models shrink but do not erase the local grounding advantage.",
            "",
            "## Title",
            "",
            (
                "**LabVerse: Does Embodied Simulation Improve LLM Agent Performance "
                "in Laboratory Tasks?**"
            ),
            "",
            "## Abstract Fill Packet",
            "",
            "Use this exact 4-move structure:",
            "1. problem",
            "2. method",
            "3. result",
            "4. implication",
            "",
            "Draft fill text:",
            "",
            (
                "We study whether large language model agents perform laboratory tasks more "
                "reliably when they operate through different grounded control interfaces. "
                f"We evaluate four primary conditions on a frozen {len(packet.scenarios)}-scenario "
                "benchmark spanning "
                "procedural, safety-critical, timed, inspection, and multi-agent tasks: three "
                "fully interactive, action-grounded conditions "
                "(`local_structured_interactive_grounded`, "
                "`local_narrative_interactive_grounded`, and "
                "`global_structured_interactive_grounded`) plus a flagship contrast condition, "
                "`global_one_shot_planner`, that emits a full plan from a single global "
                "observation. Across the current bundle, local structured interactive control "
                f"achieves {_fmt_percent(local_structured.get('success_rate'))} success and "
                f"{_fmt_float(local_structured.get('mean_completion_ratio'))} mean completion, "
                f"versus {_fmt_percent(local_narrative.get('success_rate'))} / "
                f"{_fmt_float(local_narrative.get('mean_completion_ratio'))} for local narrative "
                f"interactive, {_fmt_percent(global_structured.get('success_rate'))} / "
                f"{_fmt_float(global_structured.get('mean_completion_ratio'))} for global "
                f"structured interactive, and {_fmt_percent(global_one_shot.get('success_rate'))} "
                f"/ {_fmt_float(global_one_shot.get('mean_completion_ratio'))} for the global "
                "one-shot planner. These results support locality, information scope, and "
                "interactive grounding as meaningful design dimensions for laboratory agents."
            ),
            "",
            "## Main Results Sentences",
            "",
            (
                f"- Overall completion: local structured "
                f"{_fmt_float(local_structured.get('mean_completion_ratio'))}, "
                f"local narrative {_fmt_float(local_narrative.get('mean_completion_ratio'))}, "
                f"global structured {_fmt_float(global_structured.get('mean_completion_ratio'))}, "
                f"global one-shot {_fmt_float(global_one_shot.get('mean_completion_ratio'))}"
            ),
            *completion_pairwise_deltas,
            (
                f"- Safety violations: local structured "
                f"{_fmt_float(local_structured.get('mean_safety_violations'))}, "
                f"local narrative {_fmt_float(local_narrative.get('mean_safety_violations'))}, "
                f"global structured {_fmt_float(global_structured.get('mean_safety_violations'))}, "
                f"global one-shot {_fmt_float(global_one_shot.get('mean_safety_violations'))}"
            ),
            (
                f"- Blocked actions: local structured "
                f"{_fmt_float(local_structured.get('mean_blocked_actions'))}, "
                f"local narrative {_fmt_float(local_narrative.get('mean_blocked_actions'))}, "
                f"global structured {_fmt_float(global_structured.get('mean_blocked_actions'))}, "
                f"global one-shot {_fmt_float(global_one_shot.get('mean_blocked_actions'))}"
            ),
            "",
            "## Scenario Judgment Matrix",
            "",
            "Flagship scenarios:",
            flagship or "- Pending judgment matrix load",
            "",
            "Secondary flagship scenarios:",
            flagship_secondary or "- None",
            "",
            "Boundary-case scenarios:",
            boundary or "- None",
            "",
            "Anchor scenarios:",
            anchors or "- None",
            "",
            "## Strongest Local Grounding Gains",
            "",
            strongest,
            "",
            "## Hardest Scenarios Even Under Local Structured Control",
            "",
            hardest,
            "",
            "## Section Outline",
            "",
            "### 1. Introduction",
            (
                "- Motivate why lab work is physically grounded: PPE, room access, "
                "equipment state, timing, and safety constraints."
            ),
            (
                "- Position prior science-agent work as strong on planning but weak "
                "on grounded execution under local constraints."
            ),
            (
                "- State the question cleanly: when does local grounded interaction help, "
                "when does it not help, and why?"
            ),
            "- End with H1-H4 and a short contribution list.",
            "",
            "### 2. LabVerse Environment",
            (
                "- Describe the world model: rooms, equipment, materials, samples, "
                "people, event traces, and clocks."
            ),
            "- Describe the canonical loop: `observe -> actions -> action -> trace -> memory`.",
            "- State that observations are local and actions are grounded by "
            "exact turn-local `candidate_ref`.",
            (
                "- Mention LabVille only as visualization/debugging support, not the "
                "canonical agent interface."
            ),
            "",
            "### 3. Benchmark Design",
            (
                "- Define the four primary paper conditions: "
                "local_structured_interactive_grounded, "
                "local_narrative_interactive_grounded, "
                "global_structured_interactive_grounded, and global_one_shot_planner."
            ),
            (
                "- Call out that `global_one_shot_planner` is the flagship contrast for "
                "the central hypothesis (interactive grounded simulation vs one-shot text "
                "planning) and uses an open-verb JSON planning interface by construction."
            ),
            (
                f"- Describe the frozen {len(packet.scenarios)}-scenario benchmark and why "
                "each scenario "
                "should or should not benefit from local grounding."
            ),
            (
                "- Define the primary metrics: success, completion_ratio, "
                "safety_violations, blocked_actions, steps, simulated time, "
                "prompt tokens, and completion tokens."
            ),
            "- Keep `efficiency_score` appendix-only.",
            "",
            "### 4. Experimental Setup",
            (
                f"- The current bundle contains `{len(packet.conditions)}` conditions, "
                f"`{len(packet.models)}` models, `{len(packet.scenarios)}` scenarios, and "
                f"`{packet.replications}` replications per complete cell."
            ),
            (
                "- Explain what is held constant across conditions: scenario config, "
                "scorer, step budget, traces, and simulator semantics."
            ),
            (
                "- Explain what varies: locality, representation, and planning horizon, "
                "while keeping action grounding and simulator semantics fixed in the main paper."
            ),
            (
                "- State that the four primary conditions are the three interactive "
                "grounded conditions plus `global_one_shot_planner`, and that "
                "`scripted_oracle` is the appendix deterministic control."
            ),
            (
                "- Emphasise that `global_one_shot_planner` exists as a primary "
                "condition specifically to test the paper's central hypothesis — "
                "interactive grounded simulation versus one-shot text planning — "
                "and is NOT an appendix control in the current manifest."
            ),
            "",
            "### 5. Results",
            "- 5.1 Main effect: overall success and completion by condition.",
            "- 5.2 Safety and constraint handling: safety violations and blocked actions.",
            (
                "- 5.3 Scenario taxonomy moderation: timed, safety-critical, "
                "resource-contention, multi-room, and inspection/search effects."
            ),
            (
                "- 5.4 Model interaction: whether stronger models shrink the local "
                "grounding advantage."
            ),
            "- 5.5 Cost-quality tradeoff: prompt/completion tokens versus performance.",
            "",
            "### 6. Failure Analysis",
            (
                "- Use the fixed taxonomy: grounding failure, safety failure, "
                "sequencing failure, parameter failure, resource contention failure, "
                "and timeout/no-progress failure."
            ),
            (
                "- Include one local structured recovery example, one local narrative "
                "representation miss, one global structured search win or loss, and one "
                "shared-equipment coordination failure."
            ),
            "",
            "### 7. Discussion",
            (
                "- Explain what local grounding buys, where global structured access "
                "remains competitive, and why the effect varies by task family."
            ),
            "",
            "### 8. Limitations and Ethics",
            (
                "- State simulator realism limits, no real wet-lab execution claim, "
                "provider instability, token cost, and the absence of external "
                "realism or human-study evidence in the base package."
            ),
            "",
            "### 9. Conclusion",
            (
                "- Restate the question, summarize the empirical answer, and point "
                "to grounded scientific agents as the broader implication."
            ),
            "",
            "## Experiment Execution Plan",
            "",
            "Run in this order:",
            (
                f"1. `scripted_oracle` on all {len(packet.scenarios)} frozen "
                "scenarios (appendix control)."
            ),
            "2. deterministic appendix matrix for the four primary conditions.",
            "3. robustness main matrix for the four primary conditions, including "
            "`global_one_shot_planner`.",
            "4. `paper-analyze` on the main bundle.",
            "5. `paper-package` and `paper-writeup` on the final cited bundle.",
            "",
            "Keep the paper narrative fixed:",
            "- Main paper: the four primary conditions (three interactive grounded "
            "+ one-shot planner) on the frozen benchmark.",
            "- Sidecar / branch-selection work: future work or separate paper.",
            "",
            "## Section-to-Evidence Map",
            "",
            "- Introduction: no headline numbers except the benchmark scope",
            "- Environment: system diagram + LabVerse loop",
            "- Benchmark Design: taxonomy figure + benchmark overview table",
            "- Experimental Setup: matrix and frozen manifest description",
            "- Results 5.1: overall success/completion table + main result plot",
            "- Results 5.2: safety and blocked-actions figure",
            "- Results 5.3: per-scenario heatmap + scenario breakdown table",
            "- Results 5.4: model interaction figure + model summary table",
            "- Results 5.5: cost-quality figure using prompt/completion token tables",
            "- Failure Analysis: qualitative trace snippets from the bundle or later live runs",
            "",
            "## Result Insertion Checklist",
            "",
            (
                "- Fill the overall success rate for local_structured_interactive_grounded, "
                "local_narrative_interactive_grounded, "
                "global_structured_interactive_grounded, and global_one_shot_planner."
            ),
            ("- Fill the overall completion ratio for the four primary conditions."),
            (
                "- Fill the overall safety-violation and blocked-action means for "
                "the four primary conditions, highlighting how one-shot text "
                "planning handles safety rules it cannot observe enforced."
            ),
            (
                "- Report the two largest local structured gains versus global structured "
                "and the hardest scenario under local structured control."
            ),
            (
                "- Fill the per-model local-structured-minus-global-structured deltas from "
                "`tables/per_model_results.csv`."
            ),
            (
                "- Pull exact paired deltas, CIs, and p-values from "
                "`tables/pairwise_condition_deltas.csv`."
            ),
            (
                "- Do not write a final claim if the readiness notes above say "
                "the bundle is incomplete."
            ),
            "",
            "## Writing Constraints",
            "",
            "- Keep the main paper about local grounding and mechanism, not score-chasing.",
            "- Do not treat the research sidecar as part of the main evidence.",
            (
                "- Treat `efficiency_score` as appendix-only unless explicitly "
                "needed for a reviewer response."
            ),
            (
                "- If this bundle is incomplete, do not write the final empirical "
                "claim yet; use the readiness notes above."
            ),
            "",
        ]
    )


def _render_figure_plan(packet: PaperWriteupPacket) -> str:
    return "\n".join(
        [
            "# Figure Plan",
            "",
            "Use these exported CSV files for the manuscript figures:",
            "",
            "- `tables/overall_condition_results.csv` for the main overall results figure",
            "- `tables/per_scenario_results.csv` for the per-scenario heatmap",
            "- `tables/per_model_results.csv` for the model-interaction figure",
            "- `tables/task_family_results.csv` for family-based subgroup figures",
            "- `tables/paper_role_results.csv` for flagship / boundary / anchor tables",
            "- `tables/scenario_judgment_matrix.csv` for manuscript judgment references",
            "- `figures/taxonomy_results.csv` for spatial/safety taxonomy plots",
            (
                "- `tables/pairwise_condition_deltas.csv` for paired contrast "
                "tables and forest-plot callouts"
            ),
            "",
            "Recommended figure set:",
            "",
            "1. System and condition diagram",
            (
                "- One panel each for local structured interactive grounded, "
                "local narrative interactive grounded, global structured interactive grounded, "
                "and global one-shot planner."
            ),
            "- Show what changes across conditions and what stays fixed.",
            "",
            "2. Scenario taxonomy plot",
            "- Place the frozen scenarios on a spatial-complexity x safety-complexity grid.",
            "",
            "3. Main overall results plot",
            "- Plot success and completion by condition.",
            "- Label paired deltas directly, not only raw bars.",
            "",
            "4. Per-scenario heatmap",
            "- Plot completion or success for each condition x scenario cell.",
            "",
            "5. Safety and blocked-actions plot",
            "- Show safety violations and blocked actions separately.",
            "",
            "6. Model interaction plot",
            "- Plot the local structured advantage within each model.",
            "",
            "7. Cost-quality plot",
            "- Use prompt and completion tokens against success or completion.",
            "",
            "8. Qualitative trace examples",
            "- Use short, readable excerpts only after the quantitative story is complete.",
            "",
            f"Current bundle status: ready_for_main_results = `{packet.ready_for_main_results}`",
        ]
    )


def _render_analysis_plan(packet: PaperWriteupPacket) -> str:
    return "\n".join(
        [
            "# Analysis Plan",
            "",
            "Primary unit of comparison:",
            "- matched `(model, scenario, replication)` cells across conditions",
            "",
            "Primary comparisons:",
            ("- local_structured_interactive_grounded vs local_narrative_interactive_grounded"),
            ("- local_structured_interactive_grounded vs global_structured_interactive_grounded"),
            ("- local_narrative_interactive_grounded vs global_structured_interactive_grounded"),
            "",
            "Primary metrics:",
            "- success",
            "- completion_ratio",
            "- safety_violations",
            "- blocked_actions",
            "- steps_taken",
            "- time_elapsed_seconds",
            "- prompt_tokens",
            "- completion_tokens",
            "",
            "Planned statistical reporting:",
            "- paired mean deltas",
            "- paired bootstrap 95% CIs",
            "- sign-test p-values",
            "- Holm-adjusted p-values",
            "",
            "Subgroup analysis:",
            "- task family",
            "- paper role",
            "- spatial complexity",
            "- safety complexity",
            "- scenario category",
            "- per-model deltas",
            "",
            "Interpretation rules:",
            "- Treat success and completion as the primary headline metrics.",
            "- Treat task-family and paper-role summaries as the main narrative scaffold.",
            "- Treat safety violations and blocked actions as the main mechanism evidence.",
            (
                "- Treat the local-structured vs global-structured contrast as the "
                "main locality/grounding contrast."
            ),
            "- Treat token usage as a cost-quality analysis, not a primary quality metric.",
            "- Keep `efficiency_score` out of the main claims.",
            "",
            f"Current bundle conditions: `{', '.join(packet.conditions)}`",
            f"Current bundle models: `{', '.join(packet.models) if packet.models else 'none'}`",
        ]
    )


def _render_results_checklist(packet: PaperWriteupPacket) -> str:
    local_structured = packet.headline_by_condition.get(
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
        {},
    )
    local_narrative = packet.headline_by_condition.get(
        CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
        {},
    )
    global_structured = packet.headline_by_condition.get(
        CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
        {},
    )
    return "\n".join(
        [
            "# Results Insertion Checklist",
            "",
            "Fill these values into the manuscript once the final cited bundle is ready:",
            "",
            (
                f"- Local structured success: "
                f"{_fmt_percent(local_structured.get('success_rate'))}; "
                f"completion: {_fmt_float(local_structured.get('mean_completion_ratio'))}"
            ),
            (
                f"- Local narrative success: {_fmt_percent(local_narrative.get('success_rate'))}; "
                f"completion: {_fmt_float(local_narrative.get('mean_completion_ratio'))}"
            ),
            (
                f"- Global structured success: "
                f"{_fmt_percent(global_structured.get('success_rate'))}; "
                f"completion: {_fmt_float(global_structured.get('mean_completion_ratio'))}"
            ),
            "- Pull paired deltas and CIs from `tables/pairwise_condition_deltas.csv`.",
            "- Pull per-scenario values from `tables/per_scenario_results.csv`.",
            "- Pull per-model values from `tables/per_model_results.csv`.",
            "- Pull taxonomy subgroup values from `figures/taxonomy_results.csv`.",
            "",
            "Do not finalize the results prose unless:",
            f"- `ready_for_main_results` is `True` (currently `{packet.ready_for_main_results}`)",
            "- the cited bundle is the final frozen bundle for the paper",
            "- the plotted figures match the same bundle directory",
        ]
    )


def generate_paper_writeup(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate derived writeup assets from a frozen paper bundle."""

    bundle_path = Path(bundle_dir)
    target_dir = Path(output_dir) if output_dir is not None else bundle_path / "writeup"
    target_dir.mkdir(parents=True, exist_ok=True)
    analysis_output_dir = target_dir / "analysis" if output_dir is not None else None
    packet = build_writeup_packet(bundle_path, analysis_output_dir=analysis_output_dir)
    tables_dir = target_dir / "tables"
    figures_dir = target_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "writeup_packet.json").write_text(
        json.dumps(packet.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(
        tables_dir / "overall_condition_results.csv",
        ["condition", *PRIMARY_WRITEUP_METRICS],
        [
            [
                condition,
                *[summary.get(metric, "") for metric in PRIMARY_WRITEUP_METRICS],
            ]
            for condition, summary in sorted(packet.headline_by_condition.items())
        ],
    )
    _write_csv(
        tables_dir / "per_scenario_results.csv",
        [
            "scenario",
            "condition",
            "n",
            "success_rate",
            "mean_completion_ratio",
            "mean_safety_violations",
            "mean_blocked_actions",
            "mean_steps_taken",
            "mean_time_elapsed_seconds",
            "mean_prompt_tokens",
            "mean_completion_tokens",
        ],
        [
            [
                row.scenario,
                row.condition,
                row.n,
                row.success_rate,
                row.mean_completion_ratio,
                row.mean_safety_violations,
                row.mean_blocked_actions,
                row.mean_steps_taken,
                row.mean_time_elapsed_seconds,
                row.mean_prompt_tokens,
                row.mean_completion_tokens,
            ]
            for row in packet.scenario_summaries
        ],
    )
    _write_csv(
        tables_dir / "per_model_results.csv",
        [
            "model",
            "condition",
            "n",
            "success_rate",
            "mean_completion_ratio",
            "mean_safety_violations",
            "mean_blocked_actions",
            "mean_prompt_tokens",
            "mean_completion_tokens",
        ],
        [
            [
                row.model,
                row.condition,
                row.n,
                row.success_rate,
                row.mean_completion_ratio,
                row.mean_safety_violations,
                row.mean_blocked_actions,
                row.mean_prompt_tokens,
                row.mean_completion_tokens,
            ]
            for row in packet.model_summaries
        ],
    )
    _write_csv(
        tables_dir / "task_family_results.csv",
        [
            "grouping_axis",
            "grouping_value",
            "condition",
            "n",
            "success_rate",
            "mean_completion_ratio",
            "mean_safety_violations",
            "mean_blocked_actions",
            "mean_prompt_tokens",
            "mean_completion_tokens",
        ],
        [
            [
                row.grouping_axis,
                row.grouping_value,
                row.condition,
                row.n,
                row.success_rate,
                row.mean_completion_ratio,
                row.mean_safety_violations,
                row.mean_blocked_actions,
                row.mean_prompt_tokens,
                row.mean_completion_tokens,
            ]
            for row in packet.task_family_summaries
        ],
    )
    _write_csv(
        tables_dir / "paper_role_results.csv",
        [
            "grouping_axis",
            "grouping_value",
            "condition",
            "n",
            "success_rate",
            "mean_completion_ratio",
            "mean_safety_violations",
            "mean_blocked_actions",
            "mean_prompt_tokens",
            "mean_completion_tokens",
        ],
        [
            [
                row.grouping_axis,
                row.grouping_value,
                row.condition,
                row.n,
                row.success_rate,
                row.mean_completion_ratio,
                row.mean_safety_violations,
                row.mean_blocked_actions,
                row.mean_prompt_tokens,
                row.mean_completion_tokens,
            ]
            for row in packet.paper_role_summaries
        ],
    )
    _write_csv(
        tables_dir / "pairwise_condition_deltas.csv",
        [
            "comparison",
            "metric",
            "n",
            "mean_delta",
            "ci_low",
            "ci_high",
            "sign_test_p",
            "holm_adjusted_p",
            "wins_left",
            "wins_right",
            "ties",
        ],
        [
            [
                row.comparison,
                row.metric,
                row.n,
                row.mean_delta,
                row.ci_low,
                row.ci_high,
                row.sign_test_p,
                row.holm_adjusted_p,
                row.wins_left,
                row.wins_right,
                row.ties,
            ]
            for row in packet.pairwise
        ],
    )
    _write_csv(
        tables_dir / "scenario_judgment_matrix.csv",
        [
            "scenario_name",
            "task_family",
            "paper_role",
            "expected_best_condition",
            "main_claim_support_level",
            "why_this_task_exists",
        ],
        [
            [
                row.scenario_name,
                row.task_family,
                row.paper_role,
                row.expected_best_condition,
                row.main_claim_support_level,
                row.why_this_task_exists,
            ]
            for row in packet.scenario_judgments
        ],
    )
    _write_csv(
        figures_dir / "taxonomy_results.csv",
        [
            "taxonomy_axis",
            "taxonomy_value",
            "condition",
            "n",
            "success_rate",
            "mean_completion_ratio",
            "mean_safety_violations",
            "mean_blocked_actions",
        ],
        [
            [
                row.taxonomy_axis,
                row.taxonomy_value,
                row.condition,
                row.n,
                row.success_rate,
                row.mean_completion_ratio,
                row.mean_safety_violations,
                row.mean_blocked_actions,
            ]
            for row in packet.taxonomy_summaries
        ],
    )

    return target_dir


__all__ = [
    "GroupedConditionSummary",
    "ModelConditionSummary",
    "PaperWriteupPacket",
    "PRIMARY_PAPER_CONDITIONS",
    "ScenarioConditionSummary",
    "ScenarioJudgment",
    "TaxonomySummary",
    "build_writeup_packet",
    "generate_paper_writeup",
]
