"""Static fairness audits for paper-facing benchmark conditions."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labverse.agent.scripted import ScriptedStep
from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
)
from labverse.benchmarks.runner import BenchmarkRunner
from labverse.benchmarks.scenarios.base import DEFAULT_ONE_SHOT_ALLOWED_VERBS
from labverse.models.action import VERB_CATEGORIES, Action
from labverse.paper.manifest import PaperManifest, default_manifest_path, load_paper_manifest
from labverse.paper.probe import build_probe_request

_TASK_CONTRACT_PATTERN = re.compile(
    r"<task_contract>\n(?P<body>.*?)\n</task_contract>",
    re.DOTALL,
)
_SCORER_LEAK_MARKERS = (
    "completion_ratio",
    "Current scored details",
    "<progress_status>",
)


class ConditionFairnessReport(BaseModel):
    """Static fairness audit result for the frozen paper condition surface."""

    manifest_id: str
    valid: bool
    issues: list[str] = Field(default_factory=list)


def condition_fairness_issues(
    manifest: PaperManifest | None = None,
) -> list[str]:
    """Return condition-level fairness issues for the current repo state."""

    manifest = manifest or load_paper_manifest()
    runner = BenchmarkRunner()
    issues: list[str] = []

    if set(DEFAULT_ONE_SHOT_ALLOWED_VERBS) != set(VERB_CATEGORIES):
        issues.append(
            "DEFAULT_ONE_SHOT_ALLOWED_VERBS does not match the canonical action vocabulary."
        )

    if CONDITION_GLOBAL_ONE_SHOT_PLANNER not in manifest.primary_conditions:
        issues.append("global_one_shot_planner is not listed as a primary condition.")
    if "open" not in manifest.tool_schema_parity_note.casefold():
        issues.append("Manifest tool_schema_parity_note does not document the open one-shot plan.")

    for scenario_name in manifest.scenario_names():
        scenario = runner.create_scenario(scenario_name)
        allowed_verbs = scenario.one_shot_allowed_verbs()
        missing_default = set(DEFAULT_ONE_SHOT_ALLOWED_VERBS) - allowed_verbs
        if missing_default:
            issues.append(
                f"{scenario_name}: one-shot allowed verbs omit canonical verbs: "
                + ", ".join(sorted(missing_default))
            )

        oracle_verbs = _scripted_oracle_verbs(scenario.build_scripted_agents())
        missing_oracle_verbs = oracle_verbs - allowed_verbs
        if missing_oracle_verbs:
            issues.append(
                f"{scenario_name}: one-shot allowed verbs omit oracle verbs: "
                + ", ".join(sorted(missing_oracle_verbs))
            )

        local_structured = build_probe_request(
            condition=CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
            scenario=scenario_name,
        )
        local_narrative = build_probe_request(
            condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
            scenario=scenario_name,
        )
        global_structured = build_probe_request(
            condition=CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
            scenario=scenario_name,
        )

        prompt_surfaces = {
            CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED: local_structured,
            CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED: local_narrative,
            CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED: global_structured,
        }
        contracts = {
            condition: _extract_task_contract(
                (request.system_prompt or "") + "\n" + str(request.user_message.get("content", ""))
            )
            for condition, request in prompt_surfaces.items()
        }
        if len(set(contracts.values())) != 1:
            issues.append(f"{scenario_name}: interactive task contracts differ across conditions.")

        for condition, request in prompt_surfaces.items():
            text = "\n".join(
                [
                    request.system_prompt or "",
                    str(request.user_message.get("content", "")),
                ]
            )
            for marker in _SCORER_LEAK_MARKERS:
                if marker in text:
                    issues.append(
                        f"{scenario_name}/{condition}: scorer-derived marker leaked: {marker}"
                    )

        if local_structured.tool_schema != global_structured.tool_schema:
            issues.append(
                f"{scenario_name}: local/global structured first-turn action schemas differ."
            )
        if local_structured.available_action_count != global_structured.available_action_count:
            issues.append(f"{scenario_name}: local/global structured action counts differ.")

        local_user = str(local_structured.user_message.get("content", ""))
        global_user = str(global_structured.user_message.get("content", ""))
        if "<world_state_json>" in local_user:
            issues.append(f"{scenario_name}: local structured prompt includes global world state.")
        if "<world_state_json>" not in global_user:
            issues.append(
                f"{scenario_name}: global structured prompt is missing global world state."
            )

    return issues


def build_condition_fairness_report(
    manifest_path: str | Path | None = None,
) -> ConditionFairnessReport:
    """Build a reviewer-facing static condition fairness report."""

    manifest = load_paper_manifest(manifest_path or default_manifest_path())
    issues = condition_fairness_issues(manifest)
    return ConditionFairnessReport(
        manifest_id=manifest.manifest_id,
        valid=not issues,
        issues=issues,
    )


def _extract_task_contract(text: str) -> str:
    match = _TASK_CONTRACT_PATTERN.search(text)
    return match.group("body") if match else ""


def _scripted_oracle_verbs(agents: list[Any]) -> set[str]:
    verbs: set[str] = set()
    for agent in agents:
        for step in getattr(agent, "_script", []):
            verbs.update(_scripted_step_verbs(step))
    return verbs


def _scripted_step_verbs(step: ScriptedStep) -> set[str]:
    if isinstance(step, Action):
        return {step.verb}
    if callable(step):
        try:
            closure_vars = inspect.getclosurevars(step)
        except TypeError:
            return set()
        verb = closure_vars.nonlocals.get("verb")
        if isinstance(verb, str):
            return {verb}
    return set()


__all__ = [
    "ConditionFairnessReport",
    "build_condition_fairness_report",
    "condition_fairness_issues",
]
