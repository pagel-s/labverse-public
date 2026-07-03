"""Scientific diagnostics for the frozen paper benchmark."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labverse.agent.action_binding import build_action_from_available
from labverse.agent.action_history import ActionTrajectory, prepend_interactive_history_blocks
from labverse.agent.global_structured import format_world_state_summary_json
from labverse.agent.interactive_text import build_narrative_user_message
from labverse.agent.interface import Observation
from labverse.agent.plan_grounding import format_valid_id_inventory_json
from labverse.agent.prompt import (
    TurnStatus,
    build_action_tool,
    build_global_structured_system_prompt,
    build_global_structured_user_message,
    build_local_structured_user_message,
    build_narrative_system_prompt,
    build_system_prompt,
)
from labverse.agent.text_planner import (
    _build_system_prompt as build_one_shot_system_prompt,
)
from labverse.agent.text_planner import (
    _format_full_lab_state,
    _format_initial_context,
)
from labverse.benchmarks.conditions import (
    CONDITION_GLOBAL_ONE_SHOT_PLANNER,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
)
from labverse.benchmarks.experiment import (
    _interactive_team_observation_text,
    _interactive_team_system_prompt,
)
from labverse.benchmarks.paper_agents import (
    TeamAvailableAction,
    _build_joint_action_tool,
    _build_text_only_multi_agent_system_prompt,
    _format_initial_team_state,
    _format_team_state_message,
    build_team_actions_text,
    collect_team_actions,
)
from labverse.benchmarks.runner import BenchmarkRunner, ScenarioExecutionCapture
from labverse.benchmarks.scenarios.base import DEFAULT_ONE_SHOT_ALLOWED_VERBS, BenchmarkScenario
from labverse.gamemaster import GameMaster
from labverse.models.action import Action, AvailableAction
from labverse.paper.fairness import build_condition_fairness_report
from labverse.paper.health import build_pipeline_health_report
from labverse.paper.integrity import validate_scientific_artifacts
from labverse.paper.manifest import (
    REPO_ROOT,
    default_manifest_path,
    load_paper_manifest,
)
from labverse.paper.observability import (
    ScenarioObservabilityEntry,
    load_scenario_observability_matrix,
)
from labverse.paper.runner import PaperRunRecord, load_paper_run_records
from labverse.paper.trace import build_normalized_trace
from labverse.paper.validate import validate_paper_bundle

DEFAULT_SCIENTIFIC_AUDIT_DIR = REPO_ROOT / "paper_audits"
DEFAULT_SCIENTIFIC_AUDIT_JSON = "scientific_audit.json"
DEFAULT_SCIENTIFIC_AUDIT_MD = "scientific_audit.md"
_SCORER_LEAK_MARKERS = ("completion_ratio", "Current scored details", "<progress_status>")
_SYSTEM_BLOCK_TAGS = ("environment", "response_format", "action_contract")
_BLOCK_PATTERN = r"<{tag}>\n(.*?)\n</{tag}>"
_PRIMARY_INTERACTIVE_CONDITIONS = (
    CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED,
    CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
    CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED,
)
_CHALLENGE_LIKE_ROLES = {"flagship", "flagship_secondary", "boundary_case", "challenge"}


class ScenarioMilestoneAudit(BaseModel):
    """One milestone-turn prompt audit summary."""

    label: str
    issues: list[str] = Field(default_factory=list)
    available_action_count: int = 0


class ScenarioScientificAudit(BaseModel):
    """Audit summary for one frozen paper scenario."""

    scenario_name: str
    oracle_success: bool
    observability_contract_ok: bool
    interactive_prompt_parity_ok: bool
    one_shot_prompt_contract_ok: bool
    timed_process_contract_ok: bool
    issues: list[str] = Field(default_factory=list)
    milestones: list[ScenarioMilestoneAudit] = Field(default_factory=list)


class BundleFailureExplanation(BaseModel):
    """Classification and explanation for one non-clean bundle row."""

    row_id: str
    scenario: str
    condition: str
    model: str
    classification: str
    explanation: str
    first_divergence: str | None = None
    run_artifact_dir: str = ""


class ScenarioFailureDossier(BaseModel):
    """Reviewer-facing scenario-level failure dossier for one bundle."""

    scenario_name: str
    representative_success_run: str | None = None
    representative_failure_run: str | None = None
    representative_failure_classification: str | None = None
    representative_failure_explanation: str | None = None
    first_divergence: str | None = None
    visible_information_sufficient: bool = True


class ScientificAuditReport(BaseModel):
    """Full scientific audit report for repo-only or bundle-backed auditing."""

    generated_at: str
    mode: str
    manifest_id: str
    manifest_version: str
    repo_clean: bool
    bundle_dir: str | None = None
    valid: bool = False
    diagnostics_clear: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repo_issues: list[str] = Field(default_factory=list)
    bundle_issues: list[str] = Field(default_factory=list)
    scenarios: list[ScenarioScientificAudit] = Field(default_factory=list)
    bundle_failure_explanations: list[BundleFailureExplanation] = Field(default_factory=list)
    scenario_failure_dossiers: list[ScenarioFailureDossier] = Field(default_factory=list)


def build_scientific_audit_report(
    *,
    bundle_dir: str | Path | None = None,
    manifest_path: str | Path = default_manifest_path(),
    allow_dirty: bool = False,
) -> ScientificAuditReport:
    """Build the scientific diagnostic audit report."""

    manifest = load_paper_manifest(manifest_path)
    observability = load_scenario_observability_matrix(manifest=manifest)

    repo_clean = not _git_dirty()
    repo_issues: list[str] = []
    warnings: list[str] = []
    if not repo_clean and not allow_dirty:
        repo_issues.append(
            "Git worktree is dirty. Commit or stash changes before treating this audit "
            "as citation-grade."
        )
    elif not repo_clean:
        warnings.append("Git worktree is dirty. This audit is debug-only and not citation-grade.")

    repo_issues.extend(validate_scientific_artifacts(manifest=manifest))
    fairness_report = build_condition_fairness_report(manifest_path)
    repo_issues.extend(fairness_report.issues)

    scenario_reports = [
        _audit_scenario(
            scenario_name=scenario_name,
            observability_entry=observability.scenario_map()[scenario_name],
        )
        for scenario_name in manifest.scenario_names()
    ]
    for report in scenario_reports:
        repo_issues.extend(f"{report.scenario_name}: {issue}" for issue in report.issues)

    bundle_issues: list[str] = []
    bundle_failure_explanations: list[BundleFailureExplanation] = []
    scenario_failure_dossiers: list[ScenarioFailureDossier] = []
    bundle_diagnostics_clear = True
    if bundle_dir is not None:
        bundle_path = Path(bundle_dir)
        validation = validate_paper_bundle(bundle_path)
        if not validation.valid:
            bundle_issues.extend(validation.issues)
        health = build_pipeline_health_report(bundle_path)
        if not health.diagnostics_clear:
            bundle_issues.extend(health.blocking_reasons)
        bundle_diagnostics_clear = validation.valid and health.diagnostics_clear
        bundle_failure_explanations = _classify_bundle_failures(
            bundle_path=bundle_path,
            scenario_reports=scenario_reports,
        )
        scenario_failure_dossiers = _build_scenario_failure_dossiers(
            bundle_path=bundle_path,
            failure_explanations=bundle_failure_explanations,
            scenario_reports=scenario_reports,
        )

    blocking_reasons = list(repo_issues)
    if bundle_dir is not None:
        blocking_reasons.extend(bundle_issues)
    valid = not blocking_reasons
    diagnostics_clear = valid and repo_clean and bundle_diagnostics_clear

    return ScientificAuditReport(
        generated_at=datetime.now(UTC).isoformat(),
        mode="bundle" if bundle_dir is not None else "repo",
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.version,
        repo_clean=repo_clean,
        bundle_dir=str(bundle_dir) if bundle_dir is not None else None,
        valid=valid,
        diagnostics_clear=diagnostics_clear,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        repo_issues=repo_issues,
        bundle_issues=bundle_issues,
        scenarios=scenario_reports,
        bundle_failure_explanations=bundle_failure_explanations,
        scenario_failure_dossiers=scenario_failure_dossiers,
    )


def write_scientific_audit_report(
    report: ScientificAuditReport,
    *,
    bundle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist the scientific audit report to JSON and Markdown."""

    if bundle_dir is not None:
        target_dir = Path(bundle_dir) / "health"
    elif output_dir is not None:
        target_dir = Path(output_dir) / "health"
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target_dir = DEFAULT_SCIENTIFIC_AUDIT_DIR / f"audit-{timestamp}" / "health"
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / DEFAULT_SCIENTIFIC_AUDIT_JSON
    md_path = target_dir / DEFAULT_SCIENTIFIC_AUDIT_MD
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    md_path.write_text(_render_scientific_audit_markdown(report))
    return json_path, md_path


def _render_scientific_audit_markdown(report: ScientificAuditReport) -> str:
    lines = [
        "# Scientific Audit",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Mode: `{report.mode}`",
        f"- Manifest: `{report.manifest_id}` v`{report.manifest_version}`",
        f"- Repo clean: `{str(report.repo_clean).lower()}`",
        f"- Valid: `{str(report.valid).lower()}`",
        f"- Diagnostics clear: `{str(report.diagnostics_clear).lower()}`",
    ]
    if report.bundle_dir:
        lines.append(f"- Bundle: `{report.bundle_dir}`")
    lines.append("")
    if report.blocking_reasons:
        lines.extend(["## Blocking Reasons", ""])
        lines.extend(f"- {reason}" for reason in report.blocking_reasons)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend(["## Scenario Summary", ""])
    for scenario in report.scenarios:
        lines.append(
            f"- `{scenario.scenario_name}`: oracle={_yn(scenario.oracle_success)}, "
            f"observability={_yn(scenario.observability_contract_ok)}, "
            f"interactive_parity={_yn(scenario.interactive_prompt_parity_ok)}, "
            f"one_shot={_yn(scenario.one_shot_prompt_contract_ok)}, "
            f"timed_contract={_yn(scenario.timed_process_contract_ok)}"
        )
        for issue in scenario.issues:
            lines.append(f"  issue: {issue}")
    lines.append("")

    if report.bundle_failure_explanations:
        lines.extend(["## Bundle Failure Explanations", ""])
        for failure in report.bundle_failure_explanations:
            lines.append(
                f"- `{failure.row_id}`: `{failure.classification}` - {failure.explanation}"
            )
            if failure.first_divergence:
                lines.append(f"  first_divergence: {failure.first_divergence}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _audit_scenario(
    *,
    scenario_name: str,
    observability_entry: ScenarioObservabilityEntry,
) -> ScenarioScientificAudit:
    runner = BenchmarkRunner()
    scenario = runner.create_scenario(scenario_name)
    oracle_capture = runner.run_scenario_capture(scenario)

    issues: list[str] = []
    oracle_success = bool(
        oracle_capture.score.success and oracle_capture.score.details.get("runner_error") is None
    )
    if not oracle_success:
        issues.append(
            "scripted oracle did not reach full success: "
            f"{oracle_capture.score.details.get('runner_error') or 'unknown failure'}"
        )

    observability_ok = _audit_observability_contract(
        scenario=scenario,
        observability_entry=observability_entry,
        oracle_capture=oracle_capture,
        issues=issues,
    )
    milestone_reports, interactive_ok, timed_ok = _audit_interactive_prompt_milestones(
        scenario=scenario,
        observability_entry=observability_entry,
        issues=issues,
    )
    one_shot_ok = _audit_one_shot_prompt_contract(
        scenario=scenario,
        observability_entry=observability_entry,
        issues=issues,
    )
    return ScenarioScientificAudit(
        scenario_name=scenario_name,
        oracle_success=oracle_success,
        observability_contract_ok=observability_ok,
        interactive_prompt_parity_ok=interactive_ok,
        one_shot_prompt_contract_ok=one_shot_ok,
        timed_process_contract_ok=timed_ok,
        issues=issues,
        milestones=milestone_reports,
    )


def _audit_observability_contract(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    oracle_capture: ScenarioExecutionCapture,
    issues: list[str],
) -> bool:
    ok = True
    details = oracle_capture.score.details
    for criterion in observability_entry.full_success_criteria:
        if criterion.detail_key not in details:
            issues.append(f"observability criterion missing scorer detail {criterion.detail_key}")
            ok = False
            continue
        if details[criterion.detail_key] != criterion.expected_value:
            issues.append(
                f"observability criterion {criterion.detail_key} expected "
                f"{criterion.expected_value!r}, got {details[criterion.detail_key]!r}"
            )
            ok = False
    for timed_process in observability_entry.timed_processes:
        equipment = scenario.configured_world().get_equipment(timed_process.equipment_id)
        duration = equipment.commands[timed_process.command_name].duration_seconds
        if duration is None:
            issues.append(
                f"timed process duration missing for "
                f"{timed_process.equipment_id}.{timed_process.command_name}"
            )
            ok = False
            continue
        if abs(duration - timed_process.expected_duration_seconds) > 1e-9:
            issues.append(
                f"timed process duration drift for "
                f"{timed_process.equipment_id}.{timed_process.command_name}: "
                f"{duration} != {timed_process.expected_duration_seconds}"
            )
            ok = False
    return ok


def _audit_interactive_prompt_milestones(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> tuple[list[ScenarioMilestoneAudit], bool, bool]:
    if scenario.name == "shared_equipment_scheduling":
        return _audit_team_prompt_milestones(
            scenario=scenario,
            observability_entry=observability_entry,
            issues=issues,
        )
    return _audit_single_agent_prompt_milestones(
        scenario=scenario,
        observability_entry=observability_entry,
        issues=issues,
    )


def _audit_single_agent_prompt_milestones(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> tuple[list[ScenarioMilestoneAudit], bool, bool]:
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agent = scenario.build_scripted_agents()[0]
    trajectory = ActionTrajectory(max_entries=50)
    milestones: list[ScenarioMilestoneAudit] = []
    steps_taken = 0
    pending_labels = ["turn_0"]
    saw_timed_start = False
    saw_timed_completion = False
    timed_process_valid = True

    while steps_taken < scenario.get_max_steps():
        if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
            break
        observation = game_master.get_observation(agent.agent_id)
        action_spec = game_master.get_available_actions(agent.agent_id)
        turn_status = TurnStatus(
            scenario_name=scenario.name,
            steps_used=steps_taken,
            max_steps=scenario.get_max_steps(),
        )
        for label in pending_labels:
            milestones.append(
                _audit_single_agent_snapshot(
                    label=label,
                    scenario=scenario,
                    observation=observation,
                    available_actions=action_spec.available_actions,
                    turn_status=turn_status,
                    world_state_json=format_world_state_summary_json(game_master.world.snapshot()),
                    trajectory=trajectory,
                )
            )
        pending_labels = []

        observation_before = observation
        action_spec_before = action_spec
        turn_status_before = turn_status
        world_state_json_before = format_world_state_summary_json(game_master.world.snapshot())
        event_start_index = len(game_master.event_log.events)

        agent.observe(observation)
        action = agent.act(action_spec)

        matched_available = _match_available_action(action, action_spec_before.available_actions)
        if matched_available is None:
            issues.append(
                f"{scenario.name}: scripted oracle action is not buildable from current surface"
            )
        result = game_master.submit_action(
            action,
            observation=observation,
            action_spec=action_spec,
        )
        new_events = game_master.event_log.events[event_start_index:]

        candidate_ref = matched_available.candidate_ref if matched_available is not None else None
        trajectory.record_decision(
            verb=action.verb,
            candidate_ref=candidate_ref,
            actor_id=action.agent_id,
            target_id=action.target_id,
        )
        description = result.outcome_description or result.blocked_reason or ""
        description = description or result.failure_reason or ""
        trajectory.record_last_result(success=result.success, description=description)
        steps_taken += 1

        timed_visibility_issue = _check_timed_process_visibility(
            action=action,
            matched_available=matched_available,
            observability_entry=observability_entry,
        )
        if timed_visibility_issue is not None:
            issues.append(f"{scenario.name}: {timed_visibility_issue}")
            timed_process_valid = False
        if action.verb == "start_equipment":
            saw_timed_start = saw_timed_start or any(
                timed_process.equipment_id == action.target_id
                for timed_process in observability_entry.timed_processes
            )
            if saw_timed_start:
                pending_labels.append("after_first_timed_start")
        if _new_timed_completion_seen(new_events, observability_entry):
            saw_timed_completion = True
            pending_labels.append("after_first_timed_completion")

        if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
            milestones.append(
                _audit_single_agent_snapshot(
                    label="pre_terminal_success",
                    scenario=scenario,
                    observation=observation_before,
                    available_actions=action_spec_before.available_actions,
                    turn_status=turn_status_before,
                    world_state_json=world_state_json_before,
                    trajectory=_trajectory_before_last_step(trajectory),
                )
            )
            break

    if observability_entry.timed_processes and not saw_timed_start:
        issues.append(f"{scenario.name}: oracle never reached the declared timed-process start")
        timed_process_valid = False
    completion_required = any(
        timed_process.completion_event_required
        for timed_process in observability_entry.timed_processes
    )
    if completion_required and not saw_timed_completion:
        issues.append(f"{scenario.name}: oracle never produced the declared timed completion event")
        timed_process_valid = False

    milestone_issues = [issue for milestone in milestones for issue in milestone.issues]
    if milestone_issues:
        issues.extend(f"{scenario.name}: {issue}" for issue in milestone_issues)
    interactive_ok = not milestone_issues
    return milestones, interactive_ok, timed_process_valid


def _audit_team_prompt_milestones(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> tuple[list[ScenarioMilestoneAudit], bool, bool]:
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agents = scenario.build_scripted_agents()
    trajectory = ActionTrajectory(max_entries=50)
    milestones: list[ScenarioMilestoneAudit] = []
    steps_taken = 0
    pending_labels = ["turn_0"]
    timed_process_valid = True
    saw_timed_start = False
    saw_timed_completion = False

    while steps_taken < scenario.get_max_steps():
        if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
            break
        progressed = False
        for agent in agents:
            if steps_taken >= scenario.get_max_steps():
                break
            if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                break
            if getattr(agent, "remaining_steps", 0) <= 0:
                continue

            observations = {
                scripted_agent.agent_id: game_master.get_observation(scripted_agent.agent_id)
                for scripted_agent in agents
            }
            action_specs = {
                scripted_agent.agent_id: game_master.get_available_actions(scripted_agent.agent_id)
                for scripted_agent in agents
            }
            team_actions = collect_team_actions(
                {
                    agent_id: action_spec.available_actions
                    for agent_id, action_spec in action_specs.items()
                }
            )
            turn_status = TurnStatus(
                scenario_name=scenario.name,
                steps_used=steps_taken,
                max_steps=scenario.get_max_steps(),
            )

            for label in pending_labels:
                milestones.append(
                    _audit_team_snapshot(
                        label=label,
                        scenario=scenario,
                        observations=observations,
                        team_actions=team_actions,
                        turn_status=turn_status,
                        world_state_json=format_world_state_summary_json(
                            game_master.world.snapshot()
                        ),
                        trajectory=trajectory,
                    )
                )
            pending_labels = []

            observations_before = observations
            team_actions_before = team_actions
            turn_status_before = turn_status
            world_state_json_before = format_world_state_summary_json(game_master.world.snapshot())
            event_start_index = len(game_master.event_log.events)

            observation = observations[agent.agent_id]
            action_spec = action_specs[agent.agent_id]
            agent.observe(observation)
            action = agent.act(action_spec)
            matched_team_action = _match_team_available_action(action, team_actions_before)
            if matched_team_action is None:
                issues.append(
                    f"{scenario.name}: scripted team action is not buildable from current "
                    "team surface"
                )
            result = game_master.submit_action(
                action,
                observation=observation,
                action_spec=action_spec,
            )
            new_events = game_master.event_log.events[event_start_index:]
            candidate_ref = (
                matched_team_action.candidate_ref if matched_team_action is not None else None
            )
            trajectory.record_decision(
                verb=action.verb,
                candidate_ref=candidate_ref,
                actor_id=action.agent_id,
                target_id=action.target_id,
            )
            description = result.outcome_description or result.blocked_reason or ""
            description = description or result.failure_reason or ""
            trajectory.record_last_result(success=result.success, description=description)
            steps_taken += 1
            progressed = True

            timed_visibility_issue = _check_timed_process_visibility(
                action=action,
                matched_available=matched_team_action.action
                if matched_team_action is not None
                else None,
                observability_entry=observability_entry,
            )
            if timed_visibility_issue is not None:
                issues.append(f"{scenario.name}: {timed_visibility_issue}")
                timed_process_valid = False
            if action.verb == "start_equipment":
                saw_timed_start = saw_timed_start or any(
                    timed_process.equipment_id == action.target_id
                    for timed_process in observability_entry.timed_processes
                )
                if saw_timed_start:
                    pending_labels.append("after_first_timed_start")
            if _new_timed_completion_seen(new_events, observability_entry):
                saw_timed_completion = True
                pending_labels.append("after_first_timed_completion")

            if scenario.evaluate(game_master.event_log, game_master.world.snapshot()).success:
                milestones.append(
                    _audit_team_snapshot(
                        label="pre_terminal_success",
                        scenario=scenario,
                        observations=observations_before,
                        team_actions=team_actions_before,
                        turn_status=turn_status_before,
                        world_state_json=world_state_json_before,
                        trajectory=_trajectory_before_last_step(trajectory),
                    )
                )
                break
        if not progressed:
            break

    if observability_entry.timed_processes and not saw_timed_start:
        issues.append(f"{scenario.name}: oracle never reached the declared timed-process start")
        timed_process_valid = False
    completion_required = any(
        timed_process.completion_event_required
        for timed_process in observability_entry.timed_processes
    )
    if completion_required and not saw_timed_completion:
        issues.append(f"{scenario.name}: oracle never produced the declared timed completion event")
        timed_process_valid = False

    milestone_issues = [issue for milestone in milestones for issue in milestone.issues]
    if milestone_issues:
        issues.extend(f"{scenario.name}: {issue}" for issue in milestone_issues)
    interactive_ok = not milestone_issues
    return milestones, interactive_ok, timed_process_valid


def _audit_single_agent_snapshot(
    *,
    label: str,
    scenario: BenchmarkScenario,
    observation: Observation,
    available_actions: list[AvailableAction],
    turn_status: TurnStatus,
    world_state_json: str,
    trajectory: ActionTrajectory,
) -> ScenarioMilestoneAudit:
    system_prompts = {
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED: build_system_prompt(
            goal=scenario.get_goal_description()
        ),
        CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED: build_narrative_system_prompt(
            goal=scenario.get_goal_description()
        ),
        CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED: build_global_structured_system_prompt(
            goal=scenario.get_goal_description()
        ),
    }
    user_messages = {
        CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED: prepend_interactive_history_blocks(
            build_local_structured_user_message(
                observation,
                available_actions,
                turn_status=turn_status,
            ),
            trajectory,
        ),
        CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED: prepend_interactive_history_blocks(
            build_narrative_user_message(
                observation,
                available_actions,
                turn_status=turn_status,
            ),
            trajectory,
        ),
        CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED: prepend_interactive_history_blocks(
            build_global_structured_user_message(
                observation,
                available_actions,
                world_state_json,
                turn_status=turn_status,
            ),
            trajectory,
        ),
    }
    tool_schema = build_action_tool(available_actions)
    issues: list[str] = []

    task_contracts = [
        _extract_block(text, "task_contract") for text in system_prompts.values()
    ]
    if len(set(task_contracts)) != 1:
        issues.append(f"{label}: interactive task contracts differ")

    for tag in _SYSTEM_BLOCK_TAGS:
        blocks = [_extract_block(text, tag) for text in system_prompts.values()]
        if len(set(blocks)) != 1:
            issues.append(f"{label}: interactive system block {tag} differs across conditions")

    available_blocks = [
        _extract_block(str(message["content"]), "available_actions")
        for message in user_messages.values()
    ]
    if len(set(available_blocks)) != 1:
        issues.append(f"{label}: interactive available_actions blocks differ across conditions")

    turn_status_blocks = [
        _extract_block(str(message["content"]), "turn_status")
        for message in user_messages.values()
    ]
    if len(set(turn_status_blocks)) != 1:
        issues.append(f"{label}: interactive turn_status blocks differ across conditions")

    turn_feedback_blocks = [
        _extract_optional_block(str(message["content"]), "turn_feedback")
        for message in user_messages.values()
    ]
    if len(set(turn_feedback_blocks)) != 1:
        issues.append(f"{label}: interactive turn_feedback blocks differ across conditions")

    trajectory_blocks = [
        _extract_optional_block(str(message["content"]), "trajectory")
        for message in user_messages.values()
    ]
    if len(set(trajectory_blocks)) != 1:
        issues.append(f"{label}: interactive trajectory blocks differ across conditions")

    if "<world_state_json>" in str(
        user_messages[CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: local structured prompt includes world_state_json")
    if "<world_state_json>" in str(
        user_messages[CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: local narrative prompt includes world_state_json")
    if "<world_state_json>" not in str(
        user_messages[CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: global structured prompt is missing world_state_json")

    tool_json = json.dumps(tool_schema, sort_keys=True)
    for condition in _PRIMARY_INTERACTIVE_CONDITIONS:
        if json.dumps(tool_schema, sort_keys=True) != tool_json:
            issues.append(f"{label}: interactive action schemas differ across conditions")
            break
        _check_no_scorer_leak(
            scenario_name=scenario.name,
            condition=condition,
            system_prompt=system_prompts[condition],
            user_content=str(user_messages[condition]["content"]),
            issues=issues,
            label=label,
        )

    return ScenarioMilestoneAudit(
        label=label,
        issues=issues,
        available_action_count=len(available_actions),
    )


def _audit_team_snapshot(
    *,
    label: str,
    scenario: BenchmarkScenario,
    observations: dict[str, Any],
    team_actions: list[TeamAvailableAction],
    turn_status: TurnStatus,
    world_state_json: str,
    trajectory: ActionTrajectory,
) -> ScenarioMilestoneAudit:
    user_messages: dict[str, dict[str, Any]] = {}
    system_prompts: dict[str, str] = {}
    for condition in _PRIMARY_INTERACTIVE_CONDITIONS:
        system_prompts[condition] = _interactive_team_system_prompt(condition)
        observation_text = _interactive_team_observation_text(
            condition=condition,
            observations=observations,
        )
        base_user_message = {
            "role": "user",
            "content": _format_team_state_message(
                goal=scenario.get_goal_description(),
                observations=observation_text,
                team_actions=team_actions,
                last_result_text=None,
                turn_status=turn_status,
                world_state_json=(
                    world_state_json
                    if condition == CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED
                    else None
                ),
            ),
        }
        user_messages[condition] = prepend_interactive_history_blocks(base_user_message, trajectory)
    tool_schema = _build_joint_action_tool(team_actions)
    issues: list[str] = []

    for tag in _SYSTEM_BLOCK_TAGS:
        blocks = [_extract_block(text, tag) for text in system_prompts.values()]
        if len(set(blocks)) != 1:
            issues.append(f"{label}: team system block {tag} differs across conditions")

    task_contracts = [
        _extract_block(str(message["content"]), "task_contract")
        for message in user_messages.values()
    ]
    if len(set(task_contracts)) != 1:
        issues.append(f"{label}: team task contracts differ across conditions")

    action_blocks = [
        _extract_block(str(message["content"]), "available_team_actions")
        for message in user_messages.values()
    ]
    if len(set(action_blocks)) != 1:
        issues.append(f"{label}: available_team_actions blocks differ across conditions")

    turn_status_blocks = [
        _extract_block(str(message["content"]), "turn_status")
        for message in user_messages.values()
    ]
    if len(set(turn_status_blocks)) != 1:
        issues.append(f"{label}: team turn_status blocks differ across conditions")

    turn_feedback_blocks = [
        _extract_optional_block(str(message["content"]), "turn_feedback")
        for message in user_messages.values()
    ]
    if len(set(turn_feedback_blocks)) != 1:
        issues.append(f"{label}: team turn_feedback blocks differ across conditions")

    trajectory_blocks = [
        _extract_optional_block(str(message["content"]), "trajectory")
        for message in user_messages.values()
    ]
    if len(set(trajectory_blocks)) != 1:
        issues.append(f"{label}: team trajectory blocks differ across conditions")

    if "<world_state_json>" in str(
        user_messages[CONDITION_LOCAL_STRUCTURED_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: local structured team prompt includes world_state_json")
    if "<world_state_json>" in str(
        user_messages[CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: local narrative team prompt includes world_state_json")
    if "<world_state_json>" not in str(
        user_messages[CONDITION_GLOBAL_STRUCTURED_INTERACTIVE_GROUNDED]["content"]
    ):
        issues.append(f"{label}: global structured team prompt is missing world_state_json")

    tool_json = json.dumps(tool_schema, sort_keys=True)
    for condition in _PRIMARY_INTERACTIVE_CONDITIONS:
        if json.dumps(tool_schema, sort_keys=True) != tool_json:
            issues.append(f"{label}: team action schemas differ across conditions")
            break
        _check_no_scorer_leak(
            scenario_name=scenario.name,
            condition=condition,
            system_prompt=system_prompts[condition],
            user_content=str(user_messages[condition]["content"]),
            issues=issues,
            label=label,
        )

    return ScenarioMilestoneAudit(
        label=label,
        issues=issues,
        available_action_count=len(team_actions),
    )


def _audit_one_shot_prompt_contract(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> bool:
    if scenario.name == "shared_equipment_scheduling":
        return _audit_team_one_shot_prompt_contract(
            scenario=scenario,
            observability_entry=observability_entry,
            issues=issues,
        )
    return _audit_single_agent_one_shot_prompt_contract(
        scenario=scenario,
        observability_entry=observability_entry,
        issues=issues,
    )


def _audit_single_agent_one_shot_prompt_contract(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> bool:
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agent_id = scenario.build_scripted_agents()[0].agent_id
    observation = game_master.get_observation(agent_id)
    action_spec = game_master.get_available_actions(agent_id)
    allowed_verbs = scenario.one_shot_allowed_verbs()
    system_prompt = build_one_shot_system_prompt(allowed_verbs=allowed_verbs)
    user_sections = [
        f"<task_contract>\n{scenario.get_goal_description()}\n</task_contract>",
    ]
    run_constraints = "\n".join(
        [
            "<run_constraints>",
            f"Max steps: {scenario.get_max_steps()}",
            "Each planned action consumes one turn.",
            "</run_constraints>",
        ]
    )
    user_sections.extend(
        [
            run_constraints,
            f"<world_state_json>\n{_format_full_lab_state(world)}\n</world_state_json>",
            (
                "<valid_id_inventory_json>\n"
                + format_valid_id_inventory_json(
                    world,
                    agent_ids=[agent_id],
                    allowed_verbs=sorted(allowed_verbs),
                )
                + "\n</valid_id_inventory_json>"
            ),
            (
                "<initial_local_context>\n"
                + _format_initial_context(
                    observation,
                    action_spec,
                    allowed_verbs=allowed_verbs,
                )
                + "\n</initial_local_context>"
            ),
        ]
    )
    user_content = "\n\n".join(user_sections)
    ok = True

    if "<task_contract>" not in user_content:
        issues.append(f"{scenario.name}: one-shot planner prompt is missing task_contract")
        ok = False
    if "<world_state_json>" not in user_content:
        issues.append(f"{scenario.name}: one-shot planner prompt is missing world_state_json")
        ok = False
    if "<valid_id_inventory_json>" not in user_content:
        issues.append(
            f"{scenario.name}: one-shot planner prompt is missing valid_id_inventory_json"
        )
        ok = False

    for referenced_id in scenario.task_spec().referenced_ids:
        if referenced_id not in user_content:
            issues.append(
                f"{scenario.name}: one-shot planner prompt does not surface referenced "
                f"id {referenced_id}"
            )
            ok = False

    for criterion in observability_entry.full_success_criteria:
        if criterion.evidence_source == "semantic_note" and "log_observation" not in system_prompt:
            issues.append(
                f"{scenario.name}: one-shot planner prompt omits log_observation while "
                f"{criterion.detail_key} requires a semantic note"
            )
            ok = False

    _check_no_scorer_leak(
        scenario_name=scenario.name,
        condition=CONDITION_GLOBAL_ONE_SHOT_PLANNER,
        system_prompt=system_prompt,
        user_content=user_content,
        issues=issues,
        label="turn_0",
    )
    return ok


def _audit_team_one_shot_prompt_contract(
    *,
    scenario: BenchmarkScenario,
    observability_entry: ScenarioObservabilityEntry,
    issues: list[str],
) -> bool:
    world = scenario.configured_world()
    game_master = GameMaster(world, safety_engine=scenario.build_safety_engine())
    agents = scenario.build_scripted_agents()
    agent_ids = [agent.agent_id for agent in agents]
    observations = {agent_id: game_master.get_observation(agent_id) for agent_id in agent_ids}
    action_text = build_team_actions_text(
        {
            agent_id: game_master.get_available_actions(agent_id).available_actions
            for agent_id in agent_ids
        },
        allowed_verbs=set(DEFAULT_ONE_SHOT_ALLOWED_VERBS),
    )
    system_prompt = _build_text_only_multi_agent_system_prompt(
        allowed_verbs=scenario.one_shot_allowed_verbs(),
    )
    user_content = "\n\n".join(
        [
            f"<agent_ids>\n{json.dumps(agent_ids)}\n</agent_ids>",
            f"<task_contract>\n{scenario.get_goal_description()}\n</task_contract>",
            "\n".join(
                [
                    "<run_constraints>",
                    f"Max steps: {scenario.get_max_steps()}",
                    "Each planned team action consumes one turn.",
                    "</run_constraints>",
                ]
            ),
            (
                "<world_state_json>\n"
                + json.dumps(world.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n</world_state_json>"
            ),
            (
                "<valid_id_inventory_json>\n"
                + format_valid_id_inventory_json(
                    world,
                    agent_ids=agent_ids,
                    allowed_verbs=sorted(scenario.one_shot_allowed_verbs()),
                )
                + "\n</valid_id_inventory_json>"
            ),
            _format_initial_team_state(
                _interactive_team_observation_text(
                    condition=CONDITION_LOCAL_NARRATIVE_INTERACTIVE_GROUNDED,
                    observations=observations,
                ),
                action_text,
            ),
        ]
    )
    ok = True
    for block in (
        "task_contract",
        "world_state_json",
        "valid_id_inventory_json",
        "initial_team_state",
    ):
        if f"<{block}>" not in user_content:
            issues.append(f"{scenario.name}: team one-shot planner prompt is missing {block}")
            ok = False
    for referenced_id in scenario.task_spec().referenced_ids:
        if referenced_id not in user_content:
            issues.append(
                f"{scenario.name}: team one-shot planner prompt does not surface "
                f"referenced id {referenced_id}"
            )
            ok = False
    _check_no_scorer_leak(
        scenario_name=scenario.name,
        condition=CONDITION_GLOBAL_ONE_SHOT_PLANNER,
        system_prompt=system_prompt,
        user_content=user_content,
        issues=issues,
        label="turn_0",
    )
    return ok


def _classify_bundle_failures(
    *,
    bundle_path: Path,
    scenario_reports: list[ScenarioScientificAudit],
) -> list[BundleFailureExplanation]:
    scenario_issue_map = {report.scenario_name: bool(report.issues) for report in scenario_reports}
    records = load_paper_run_records(bundle_path / "results.jsonl")
    explanations: list[BundleFailureExplanation] = []
    oracle_traces = {
        report.scenario_name: _oracle_normalized_trace(report.scenario_name)
        for report in scenario_reports
    }
    for record in records:
        row_id = f"{record.condition}/{record.model}/{record.scenario}/rep_{record.replication}"
        if (
            record.success
            and record.error_taxonomy == "none"
            and record.provider_errors == 0
            and record.artifact_integrity_errors == 0
        ):
            continue
        classification, explanation = _classify_record(record, scenario_issue_map)
        first_divergence = None
        if record.run_artifact_dir:
            run_dir = bundle_path / record.run_artifact_dir
            trace_path = run_dir / "normalized_trace.jsonl"
            if trace_path.exists():
                run_trace = [
                    json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
                ]
                first_divergence = _first_trace_divergence(
                    run_trace,
                    oracle_traces.get(record.scenario, []),
                )
        explanations.append(
            BundleFailureExplanation(
                row_id=row_id,
                scenario=record.scenario,
                condition=record.condition,
                model=record.model,
                classification=classification,
                explanation=explanation,
                first_divergence=first_divergence,
                run_artifact_dir=record.run_artifact_dir,
            )
        )
    return explanations


def _build_scenario_failure_dossiers(
    *,
    bundle_path: Path,
    failure_explanations: list[BundleFailureExplanation],
    scenario_reports: list[ScenarioScientificAudit],
) -> list[ScenarioFailureDossier]:
    records = load_paper_run_records(bundle_path / "results.jsonl")
    by_scenario_failures: dict[str, list[BundleFailureExplanation]] = {}
    for failure in failure_explanations:
        by_scenario_failures.setdefault(failure.scenario, []).append(failure)
    by_scenario_successes: dict[str, list[PaperRunRecord]] = {}
    for record in records:
        if record.success and not record.error:
            by_scenario_successes.setdefault(record.scenario, []).append(record)
    visible_info_ok = {
        report.scenario_name: not bool(report.issues) for report in scenario_reports
    }
    dossiers: list[ScenarioFailureDossier] = []
    for scenario_name in sorted({*visible_info_ok, *by_scenario_failures}):
        success_records = by_scenario_successes.get(scenario_name, [])
        success_run = success_records[0].run_artifact_dir if success_records else None
        failures = by_scenario_failures.get(scenario_name)
        representative_failure: BundleFailureExplanation | None = failures[0] if failures else None
        dossiers.append(
            ScenarioFailureDossier(
                scenario_name=scenario_name,
                representative_success_run=success_run,
                representative_failure_run=(
                    representative_failure.run_artifact_dir
                    if representative_failure is not None
                    else None
                ),
                representative_failure_classification=(
                    representative_failure.classification
                    if representative_failure is not None
                    else None
                ),
                representative_failure_explanation=(
                    representative_failure.explanation
                    if representative_failure is not None
                    else None
                ),
                first_divergence=(
                    representative_failure.first_divergence
                    if representative_failure is not None
                    else None
                ),
                visible_information_sufficient=visible_info_ok.get(scenario_name, True),
            )
        )
    return dossiers


def _classify_record(
    record: PaperRunRecord,
    scenario_issue_map: dict[str, bool],
) -> tuple[str, str]:
    if scenario_issue_map.get(record.scenario, False):
        return (
            "scenario_contract_failure",
            "The repo-level scenario audit found a contract or visibility issue for this scenario.",
        )
    if record.provider_errors > 0 or record.error_taxonomy == "provider_failure":
        return ("provider_failure", "The run recorded a provider or transport failure.")
    if (
        record.artifact_integrity_errors > 0
        or record.error_taxonomy == "artifact_integrity_failure"
    ):
        return (
            "runner_artifact_failure",
            "The run recorded artifact-integrity or replay-output corruption.",
        )
    if record.error_taxonomy == "runner_failure":
        return ("runner_artifact_failure", "The run failed inside the worker or runner layer.")
    if record.grounding_error_count > 0:
        return (
            "grounding_interface_failure",
            "The model output did not bind cleanly to the live action surface.",
        )
    if record.invalid_model_outputs > 0 or record.model_output_format_errors > 0:
        return (
            "prompt_format_failure",
            "The model returned invalid structured output or the wrong response format.",
        )
    if not record.success:
        return (
            "model_behavior_failure",
            "The run remained within the visible contract but did not satisfy the "
            "scenario success criteria.",
        )
    return ("unexpected_clean_row", "The row is non-clean for an uncategorized reason.")


def _first_trace_divergence(
    run_trace: list[dict[str, Any]],
    oracle_trace: list[dict[str, Any]],
) -> str | None:
    if not run_trace:
        return "Run produced no normalized trace rows."
    for index, (run_row, oracle_row) in enumerate(zip(run_trace, oracle_trace), start=1):
        run_signature = (
            run_row.get("actor_id"),
            run_row.get("verb"),
            run_row.get("target_id"),
            run_row.get("success"),
        )
        oracle_signature = (
            oracle_row.get("actor_id"),
            oracle_row.get("verb"),
            oracle_row.get("target_id"),
            oracle_row.get("success"),
        )
        if run_signature != oracle_signature:
            return (
                f"step {index}: observed {run_row.get('verb')} target={run_row.get('target_id')} "
                f"success={run_row.get('success')} vs oracle {oracle_row.get('verb')} "
                f"target={oracle_row.get('target_id')} success={oracle_row.get('success')}"
            )
    if len(run_trace) < len(oracle_trace):
        next_oracle = oracle_trace[len(run_trace)]
        return (
            f"step {len(run_trace) + 1}: run ended before oracle action "
            f"{next_oracle.get('verb')} target={next_oracle.get('target_id')}"
        )
    return None


def _oracle_normalized_trace(scenario_name: str) -> list[dict[str, Any]]:
    runner = BenchmarkRunner()
    capture = runner.run_scenario_capture(scenario_name)
    return build_normalized_trace(capture.step_audits)


def _check_timed_process_visibility(
    *,
    action: Action,
    matched_available: AvailableAction | None,
    observability_entry: ScenarioObservabilityEntry,
) -> str | None:
    if action.verb != "start_equipment":
        return None
    relevant = [
        timed_process
        for timed_process in observability_entry.timed_processes
        if timed_process.equipment_id == action.target_id
    ]
    if not relevant:
        return None
    if matched_available is None:
        return "timed start action was not matched to the visible action surface"
    duration = matched_available.fixed_parameters.get("duration_seconds")
    if duration is None:
        return (
            f"timed action {matched_available.target_id} does not expose fixed duration_seconds "
            "on the visible action surface"
        )
    expected_durations = {timed_process.expected_duration_seconds for timed_process in relevant}
    if duration not in expected_durations:
        return (
            f"timed action {matched_available.target_id} exposes duration_seconds={duration}, "
            f"expected one of {sorted(expected_durations)}"
        )
    return None


def _new_timed_completion_seen(
    new_events: list[Any],
    observability_entry: ScenarioObservabilityEntry,
) -> bool:
    timed_targets = {
        timed_process.equipment_id
        for timed_process in observability_entry.timed_processes
        if timed_process.completion_event_required
    }
    return any(
        getattr(event, "target_id", None) in timed_targets and "completed" in event.description
        for event in new_events
    )


def _trajectory_before_last_step(trajectory: ActionTrajectory) -> ActionTrajectory:
    copied = ActionTrajectory(max_entries=max(len(trajectory.entries), 1))
    for entry in trajectory.entries[:-1]:
        copied.record_decision(
            verb=entry.verb,
            candidate_ref=entry.candidate_ref,
            actor_id=entry.actor_id,
            target_id=entry.target_id,
        )
        if entry.success is not None:
            copied.record_last_result(success=entry.success, description=entry.description)
    return copied


def _match_available_action(
    action: Action,
    available_actions: list[AvailableAction],
) -> AvailableAction | None:
    for available_action in available_actions:
        if available_action.verb != action.verb:
            continue
        if available_action.target_id != action.target_id:
            continue
        try:
            rebuilt = build_action_from_available(
                agent_id=action.agent_id,
                available_action=available_action,
                provided_parameters=action.parameters,
            )
        except ValueError:
            continue
        if all(rebuilt.parameters.get(name) == value for name, value in action.parameters.items()):
            return available_action
    return None


def _match_team_available_action(
    action: Action,
    team_actions: list[TeamAvailableAction],
) -> TeamAvailableAction | None:
    for team_action in team_actions:
        if team_action.agent_id != action.agent_id:
            continue
        available = _match_available_action(action, [team_action.action])
        if available is not None:
            return team_action
    return None


def _extract_block(text: str, tag: str) -> str:
    match = re.search(_BLOCK_PATTERN.format(tag=tag), text, flags=re.S)
    return match.group(1) if match is not None else ""


def _extract_optional_block(text: str, tag: str) -> str:
    return _extract_block(text, tag)


def _check_no_scorer_leak(
    *,
    scenario_name: str,
    condition: str,
    system_prompt: str,
    user_content: str,
    issues: list[str],
    label: str,
) -> None:
    combined = "\n".join([system_prompt, user_content])
    for marker in _SCORER_LEAK_MARKERS:
        if marker in combined:
            issues.append(f"{label}: {scenario_name}/{condition} leaked scorer marker {marker}")


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _yn(value: bool) -> str:
    return "yes" if value else "no"


__all__ = [
    "DEFAULT_SCIENTIFIC_AUDIT_DIR",
    "DEFAULT_SCIENTIFIC_AUDIT_JSON",
    "DEFAULT_SCIENTIFIC_AUDIT_MD",
    "BundleFailureExplanation",
    "ScenarioFailureDossier",
    "ScenarioMilestoneAudit",
    "ScenarioScientificAudit",
    "ScientificAuditReport",
    "build_scientific_audit_report",
    "write_scientific_audit_report",
]
