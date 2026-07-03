"""Tests for the frozen paper runner and artifact bundles."""

from __future__ import annotations

import json
import queue
from pathlib import Path

import pytest

from labverse.benchmarks.experiment import ProviderOptions, ScenarioRunArtifacts
from labverse.benchmarks.scorer import BenchmarkScore
from labverse.paper import (
    analyze_paper_bundle,
    build_paper_bundle_summary,
    build_pipeline_health_report,
    bundle_status,
    load_paper_manifest,
    load_paper_run_records,
    package_paper_bundle,
    preflight_paper_experiment,
    run_paper_experiment,
)
from labverse.paper.runner import (
    _error_taxonomy,
    _infer_provider,
    _run_single_capture_isolated,
    _run_single_capture_subprocess,
)


def test_paper_runner_writes_scripted_oracle_bundle_and_supports_resume(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    assert Path(bundle.results_path).exists()
    records = load_paper_run_records(bundle.results_path)
    assert len(records) == 1
    record = records[0]
    assert record.condition == "scripted_oracle"
    assert record.model == "scripted_oracle"
    assert record.initial_variant_id == "baseline"
    assert len(record.initial_variant_hash) == 64
    assert record.trace_ids
    assert record.event_ids

    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        resume=True,
        allow_dirty=True,
    )
    resumed_records = load_paper_run_records(bundle.results_path)
    assert len(resumed_records) == 1


def test_paper_runner_records_sensitivity_initial_variant_metadata(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
        initial_variant_id="sensitivity_v1",
    )

    metadata = json.loads(Path(bundle.metadata_path).read_text())
    records = load_paper_run_records(bundle.results_path)
    run_meta_path = bundle_dir / records[0].run_artifact_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())

    assert metadata["initial_variant_id"] == "sensitivity_v1"
    assert len(metadata["initial_variant_hash"]) == 64
    assert records[0].initial_variant_id == "sensitivity_v1"
    assert records[0].initial_variant_hash == metadata["initial_variant_hash"]
    assert run_meta["initial_variant_id"] == "sensitivity_v1"


def test_paper_analysis_and_package_write_expected_outputs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["chemical_spill_response"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    analysis_dir = analyze_paper_bundle(bundle.bundle_dir)
    package_dir = package_paper_bundle(bundle.bundle_dir)

    assert (analysis_dir / "report.json").exists()
    report = json.loads((analysis_dir / "report.json").read_text())
    assert report["total_runs"] == 1
    assert (package_dir / "manifest.json").exists()
    assert (package_dir / "analysis" / "condition_summary.csv").exists()
    assert (package_dir / "writeup" / "writeup_packet.json").exists()
    assert (package_dir / "REPRODUCE.md").exists()


def test_paper_runner_defaults_to_all_frozen_manifest_scenarios_for_scripted_oracle(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    manifest = load_paper_manifest()

    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    records = load_paper_run_records(bundle.results_path)
    assert len(records) == len(manifest.scenarios)
    assert {record.scenario for record in records} == set(manifest.scenario_names())


def test_paper_runner_resume_rejects_selection_expansion(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    with pytest.raises(ValueError, match="Cannot resume bundle with selections outside"):
        run_paper_experiment(
            conditions=["scripted_oracle"],
            scenarios=["pcr_setup", "chemical_spill_response"],
            profile_name="deterministic",
            replications=1,
            bundle_dir=bundle_dir,
            resume=True,
            allow_dirty=True,
        )


def test_resume_cleans_up_stale_running_run_meta(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0]
    run_meta_path = bundle_dir / record.run_artifact_dir / "run_meta.json"
    payload = json.loads(run_meta_path.read_text())
    payload["status"] = "running"
    run_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        resume=True,
        allow_dirty=True,
    )

    refreshed = json.loads(run_meta_path.read_text())
    assert refreshed["status"] == "completed"
    assert "recovered_reason" in refreshed


def test_bundle_status_ignores_dead_running_pid(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0]
    run_meta_path = bundle_dir / record.run_artifact_dir / "run_meta.json"
    payload = json.loads(run_meta_path.read_text())
    payload["status"] = "running"
    payload["launcher_pid"] = 99999999
    payload["worker_pid"] = 99999998
    run_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    status = bundle_status(bundle_dir)

    assert status.running_runs == []


def test_paper_runner_rejects_unknown_model_provider(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    try:
        run_paper_experiment(
            models=["mystery-model"],
            conditions=["local_structured_interactive_grounded"],
            scenarios=["pcr_setup"],
            profile_name="deterministic",
            replications=1,
            bundle_dir=bundle_dir,
            allow_dirty=True,
        )
    except ValueError as exc:
        assert "Cannot infer provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model provider inference")


def test_paper_runner_groups_model_backed_tasks_by_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    def fake_subprocess_capture(**kwargs):
        score = BenchmarkScore(
            scenario_name=kwargs["scenario_name"],
            success=True,
            completion_ratio=1.0,
            steps_taken=1,
            time_elapsed_seconds=0.0,
            safety_violations=0,
            blocked_actions=0,
            efficiency_score=1.0,
            details={},
        )
        return (
            type(
                "FakeRunResult",
                (),
                {
                    "score": score,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_time_seconds": 0.01,
                    "error": None,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )(),
            ScenarioRunArtifacts(score=score),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "labverse.paper.runner._run_single_capture_subprocess",
        fake_subprocess_capture,
    )

    bundle = run_paper_experiment(
        models=["gpt-4.1-mini", "ollama/qwen3:14b"],
        conditions=[
            "local_structured_interactive_grounded",
            "global_one_shot_planner",
        ],
        scenarios=["pcr_setup", "chemical_spill_response"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    records = load_paper_run_records(bundle.results_path)
    model_runs = [record.model for record in records]
    model_switches = sum(
        previous != current for previous, current in zip(model_runs, model_runs[1:])
    )

    assert len(records) == 8
    assert model_switches == 1
    assert {record.model for record in records[:4]} == {"gpt-4.1-mini"}
    assert {record.model for record in records[4:]} == {"ollama/qwen3:14b"}


def test_paper_package_rebuilds_missing_summary(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )
    (bundle_dir / "summary.json").unlink()

    package_dir = package_paper_bundle(bundle.bundle_dir)

    assert (bundle_dir / "summary.json").exists()
    assert (package_dir / "summary.json").exists()


def test_paper_runner_writes_health_report_and_run_artifacts(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    records = load_paper_run_records(bundle.results_path)
    run_dir = bundle_dir / records[0].run_artifact_dir
    assert (run_dir / "run_meta.json").exists()
    assert (run_dir / "steps.jsonl").exists()
    assert (run_dir / "normalized_trace.jsonl").exists()
    assert (bundle_dir / "health" / "pipeline_health.json").exists()
    assert (bundle_dir / "health" / "pipeline_health.md").exists()

    report = build_pipeline_health_report(bundle_dir)
    assert report.total_runs == 1
    assert report.bundle_valid is True
    assert report.diagnostics_clear is True


def test_model_bundle_writes_normalized_trace_and_generic_output_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    def fake_subprocess_capture(**kwargs):
        score = BenchmarkScore(
            scenario_name=kwargs["scenario_name"],
            success=False,
            completion_ratio=0.1,
            steps_taken=1,
            time_elapsed_seconds=10.0,
            safety_violations=0,
            blocked_actions=0,
            efficiency_score=1.0,
            details={},
        )
        return (
            type(
                "FakeRunResult",
                (),
                {
                    "condition": kwargs["condition"],
                    "model": kwargs["model"],
                    "provider": kwargs["provider_name"],
                    "scenario": kwargs["scenario_name"],
                    "replication": kwargs["replication"],
                    "score": score,
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "wall_time_seconds": 0.01,
                    "error": None,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )(),
            ScenarioRunArtifacts(
                score=score,
                event_ids=["evt_1"],
                trace_ids=["trace_1"],
                trace_ids_by_agent={"grad_kim": ["trace_1"]},
                step_audits=[
                    {
                        "actor_id": "grad_kim",
                        "actor_name": "fake",
                        "actor_kind": "agent",
                        "prompt_surface_kind": "local_structured_interactive_grounded",
                        "system_prompt": "system",
                        "user_message": {"role": "user", "content": "user"},
                        "tool_schema": {"name": "take_action"},
                        "attempts": [],
                        "resolved_candidate_ref": "ppe_1",
                        "resolved_action_id": "ppe_1",
                        "resolved_verb": "don_ppe",
                        "resolved_target_id": None,
                        "resolved_action_parameters": {"ppe_type": "lab_coat"},
                        "action_result": {
                            "success": True,
                            "status": "succeeded",
                            "description": "Donned lab coat.",
                            "time_elapsed_seconds": 10.0,
                            "warnings": [],
                            "safety_violations": [],
                            "blocked_reason": None,
                            "failure_reason": None,
                        },
                        "trace_ids": ["trace_1"],
                        "event_ids": ["evt_1"],
                    }
                ],
                run_metrics={
                    "invalid_model_outputs": 1,
                    "model_output_retry_count": 1,
                    "model_output_format_errors": 1,
                    "grounding_error_count": 0,
                },
            ),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "labverse.paper.runner._run_single_capture_subprocess",
        fake_subprocess_capture,
    )

    bundle = run_paper_experiment(
        models=["gpt-4.1-mini"],
        conditions=["local_structured_interactive_grounded"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0]
    run_dir = bundle_dir / record.run_artifact_dir
    trace_rows = [
        json.loads(line)
        for line in (run_dir / "normalized_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    run_meta = json.loads((run_dir / "run_meta.json").read_text())

    assert record.invalid_model_outputs == 1
    assert record.model_output_retry_count == 1
    assert trace_rows == [
        {
            "actor_id": "grad_kim",
            "actor_kind": "agent",
            "blocked_reason": None,
            "candidate_ref": "ppe_1",
            "description": "Donned lab coat.",
            "error": None,
            "event_ids": ["evt_1"],
            "failure_reason": None,
            "parameters": {"ppe_type": "lab_coat"},
            "resolved_action_id": "ppe_1",
            "safety_violations": [],
            "schema_version": 1,
            "source_step_index": None,
            "status": "succeeded",
            "step_index": 1,
            "success": True,
            "target_id": None,
            "trace_ids": ["trace_1"],
            "verb": "don_ppe",
            "warnings": [],
        }
    ]
    assert run_meta["normalized_trace_row_count"] == 1
    assert build_pipeline_health_report(bundle_dir).bundle_valid is True


def test_bundle_status_and_health_mark_incomplete_bundle_has_diagnostics(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup", "chemical_spill_response"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    results_path = bundle_dir / "results.jsonl"
    first_line = results_path.read_text().splitlines()[0]
    results_path.write_text(first_line + "\n")
    records = load_paper_run_records(results_path)
    (bundle_dir / "summary.json").write_text(
        json.dumps(build_paper_bundle_summary(records), indent=2, sort_keys=True) + "\n"
    )

    status = bundle_status(bundle_dir)
    health = build_pipeline_health_report(bundle_dir)

    assert status.validation_ok is True
    assert status.diagnostics_clear is False
    assert any("incomplete" in reason.casefold() for reason in status.blocking_reasons)
    assert health.bundle_valid is True
    assert health.diagnostics_clear is False
    assert any("incomplete" in reason.casefold() for reason in health.blocking_reasons)


def test_paper_preflight_requires_clean_repo_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("labverse.paper.runner._git_dirty", lambda: True)
    monkeypatch.setattr("labverse.paper.runner._git_commit_sha", lambda: "deadbeef")

    report = preflight_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
    )

    assert report.valid is False
    assert any("dirty" in issue.casefold() for issue in report.issues)


def test_infer_provider_supports_ollama_prefixed_models() -> None:
    assert _infer_provider("ollama/qwen3:14b") == "ollama"


def test_paper_preflight_accepts_ollama_models_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    report = preflight_paper_experiment(
        models=["ollama/qwen3:14b"],
        conditions=["local_narrative_interactive_grounded"],
        scenarios=["chemical_spill_response"],
        profile_name="deterministic",
        allow_dirty=True,
    )

    assert report.valid is True
    assert report.provider_credentials == {}


def test_bundle_status_uses_same_scale_gate_as_health_report(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record["prompt_tokens"] = 900_000
    record["max_prompt_tokens_single_turn"] = 40_000
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    status = bundle_status(bundle_dir)
    health = build_pipeline_health_report(bundle_dir)

    assert health.diagnostics_clear is False
    assert status.diagnostics_clear is False
    assert status.blocking_reasons == health.blocking_reasons


def test_paper_runner_uses_subprocess_live_capture_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    subprocess_called = {"value": False}

    def fake_subprocess_capture(**kwargs):
        subprocess_called["value"] = True
        return (
            type(
                "FakeRunResult",
                (),
                {
                    "condition": kwargs["condition"],
                    "model": kwargs["model"],
                    "provider": kwargs["provider_name"],
                    "scenario": kwargs["scenario_name"],
                    "replication": kwargs["replication"],
                    "score": BenchmarkScore(
                        scenario_name=kwargs["scenario_name"],
                        success=False,
                        completion_ratio=0.0,
                        steps_taken=0,
                        time_elapsed_seconds=0.0,
                        safety_violations=0,
                        blocked_actions=0,
                        efficiency_score=0.0,
                        details={"runner_error": "synthetic inline capture"},
                    ),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_time_seconds": 0.01,
                    "error": "synthetic inline capture",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )(),
            ScenarioRunArtifacts(
                score=BenchmarkScore(
                    scenario_name=kwargs["scenario_name"],
                    success=False,
                    completion_ratio=0.0,
                    steps_taken=0,
                    time_elapsed_seconds=0.0,
                    safety_violations=0,
                    blocked_actions=0,
                    efficiency_score=0.0,
                    details={"runner_error": "synthetic inline capture"},
                )
            ),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "labverse.paper.runner._run_single_capture_subprocess",
        fake_subprocess_capture,
    )

    run_paper_experiment(
        models=["gpt-4.1-mini"],
        conditions=["local_narrative_interactive_grounded"],
        scenarios=["shared_equipment_scheduling"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    assert subprocess_called["value"] is True


def test_isolated_live_capture_converts_worker_crash_into_failure_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQueue:
        def get_nowait(self) -> dict[str, object]:
            raise queue.Empty

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class FakeProcess:
        exitcode = 139

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def is_alive(self) -> bool:
            return False

    class FakeContext:
        def Queue(self) -> FakeQueue:
            return FakeQueue()

        def Process(self, *args, **kwargs) -> FakeProcess:
            return FakeProcess(*args, **kwargs)

    monkeypatch.setattr(
        "labverse.paper.runner.multiprocessing.get_context",
        lambda _: FakeContext(),
    )

    result, artifacts = _run_single_capture_isolated(
        condition="local_structured_interactive_grounded",
        provider_name="openai",
        model="gpt-4.1-mini",
        scenario_name="pcr_setup",
        replication=1,
        provider_options=None,
    )

    assert result.error is not None
    assert "crashed" in result.error.casefold()
    assert artifacts.score.success is False
    assert "runner_traceback" in artifacts.score.details


def test_isolated_live_capture_converts_worker_timeout_into_failure_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQueue:
        def get_nowait(self) -> dict[str, object]:
            raise queue.Empty

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class FakeProcess:
        exitcode: int | None = None
        terminated = False

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

    class FakeContext:
        def Queue(self) -> FakeQueue:
            return FakeQueue()

        def Process(self, *args, **kwargs) -> FakeProcess:
            return FakeProcess(*args, **kwargs)

    monkeypatch.setattr(
        "labverse.paper.runner.multiprocessing.get_context",
        lambda _: FakeContext(),
    )

    result, artifacts = _run_single_capture_isolated(
        condition="local_structured_interactive_grounded",
        provider_name="openai",
        model="gpt-4.1-mini",
        scenario_name="pcr_setup",
        replication=1,
        provider_options=ProviderOptions(worker_timeout_seconds=1.0),
    )

    assert result.error is not None
    assert "timed out" in result.error.casefold()
    assert artifacts.score.success is False
    assert "runner_traceback" in artifacts.score.details


def test_subprocess_live_capture_converts_process_crash_into_failure_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.pid = 43210
            self.returncode = -11

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            return ("", "segmentation fault")

        def kill(self) -> None:
            return None

    monkeypatch.setattr("labverse.paper.runner.subprocess.Popen", FakeProcess)

    result, artifacts = _run_single_capture_subprocess(
        condition="local_structured_interactive_grounded",
        provider_name="openai",
        model="gpt-4.1-mini",
        scenario_name="pcr_setup",
        replication=1,
        provider_options=ProviderOptions(worker_timeout_seconds=1.0),
    )

    assert result.error is not None
    assert "crashed" in result.error.casefold()
    assert "segmentation fault" in result.error.casefold()
    assert artifacts.score.success is False
    assert artifacts.score.details["worker_pid"] == 43210
    assert artifacts.score.details["exit_code"] == -11
    assert artifacts.score.details["exit_signal"] == "SIGSEGV"
    assert artifacts.score.details["stderr_tail"] == "segmentation fault"
    assert artifacts.score.details["output_file_exists"] is False


def test_subprocess_live_capture_records_crash_diagnostics_without_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.pid = 43211
            self.returncode = -11

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            return ("partial stdout", "")

        def kill(self) -> None:
            return None

    monkeypatch.setattr("labverse.paper.runner.subprocess.Popen", FakeProcess)

    result, artifacts = _run_single_capture_subprocess(
        condition="local_structured_interactive_grounded",
        provider_name="openai",
        model="gpt-4.1-mini",
        scenario_name="pcr_setup",
        replication=1,
        provider_options=ProviderOptions(worker_timeout_seconds=1.0),
    )

    assert result.error is not None
    assert "exit_code=-11" in result.error
    assert artifacts.score.success is False
    assert artifacts.score.details["worker_pid"] == 43211
    assert artifacts.score.details["exit_signal"] == "SIGSEGV"
    assert artifacts.score.details["stdout_tail"] == "partial stdout"
    assert artifacts.score.details["stderr_tail"] == ""
    assert artifacts.score.details["runner_traceback_available"] is False


def test_subprocess_live_capture_uses_lightweight_top_level_worker_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmd: list[str] = []

    class FakeProcess:
        def __init__(self, cmd, *args, **kwargs) -> None:
            del args, kwargs
            nonlocal captured_cmd
            captured_cmd = list(cmd)
            output_path = Path(captured_cmd[captured_cmd.index("--output") + 1])
            payload = {
                "result": {
                    "condition": "local_structured_interactive_grounded",
                    "model": "gpt-4.1-mini",
                    "provider": "openai",
                    "scenario": "pcr_setup",
                    "replication": 1,
                    "score": BenchmarkScore(
                        scenario_name="pcr_setup",
                        success=True,
                        completion_ratio=1.0,
                        steps_taken=1,
                        time_elapsed_seconds=10.0,
                        safety_violations=0,
                        blocked_actions=0,
                        efficiency_score=1.0,
                        details={},
                    ).model_dump(mode="json"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_time_seconds": 0.01,
                    "error": None,
                    "timestamp": "2026-04-21T00:00:00",
                },
                "artifacts": {
                    "score": BenchmarkScore(
                        scenario_name="pcr_setup",
                        success=True,
                        completion_ratio=1.0,
                        steps_taken=1,
                        time_elapsed_seconds=10.0,
                        safety_violations=0,
                        blocked_actions=0,
                        efficiency_score=1.0,
                        details={},
                    ).model_dump(mode="json"),
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "event_ids": [],
                    "trace_ids": [],
                    "trace_ids_by_agent": {},
                    "step_audits": [],
                    "replay_frames": [],
                    "run_metrics": {},
                },
            }
            output_path.write_text(json.dumps(payload))
            self.pid = 43212
            self.returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            return ("", "")

        def kill(self) -> None:
            return None

    monkeypatch.setattr("labverse.paper.runner.subprocess.Popen", FakeProcess)

    result, artifacts = _run_single_capture_subprocess(
        condition="local_structured_interactive_grounded",
        provider_name="openai",
        model="gpt-4.1-mini",
        scenario_name="pcr_setup",
        replication=1,
        provider_options=ProviderOptions(worker_timeout_seconds=1.0),
    )

    assert result.score.success is True
    assert artifacts.score.success is True
    assert "-m" in captured_cmd
    assert "labverse.paper_worker" in captured_cmd
    assert "labverse.paper.cell_runner" not in captured_cmd


def test_error_taxonomy_treats_runner_error_as_runner_failure() -> None:
    score = BenchmarkScore(
        scenario_name="multi_step_protocol",
        success=False,
        completion_ratio=0.5,
        steps_taken=50,
        time_elapsed_seconds=500.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=0.0,
        details={"runner_error": "'str' object is not callable"},
    )
    result = type(
        "FakeRunResult",
        (),
        {
            "score": score,
            "error": "'str' object is not callable",
        },
    )()

    taxonomy = _error_taxonomy(
        result=result,
        condition="global_structured_interactive_grounded",
        scenario="multi_step_protocol",
        run_metrics={},
    )

    assert taxonomy == "runner_failure"


def test_error_taxonomy_treats_invalid_action_retry_loop_as_grounding_failure() -> None:
    score = BenchmarkScore(
        scenario_name="chemical_spill_response",
        success=False,
        completion_ratio=0.2,
        steps_taken=7,
        time_elapsed_seconds=70.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=0.0,
        details={
            "runner_error": (
                "LLM failed to produce a valid action after 3 attempts. "
                "Last error: Unknown candidate_ref 'preview_take_material'. "
                "Available candidate_refs: move_1, observe_1"
            ),
        },
    )
    result = type(
        "FakeRunResult",
        (),
        {
            "score": score,
            "error": (
                "LLM failed to produce a valid action after 3 attempts. "
                "Last error: Unknown candidate_ref 'preview_take_material'. "
                "Available candidate_refs: move_1, observe_1"
            ),
        },
    )()

    taxonomy = _error_taxonomy(
        result=result,
        condition="global_structured_interactive_grounded",
        scenario="chemical_spill_response",
        run_metrics={},
    )

    assert taxonomy == "grounding_failure"


def test_error_taxonomy_treats_one_shot_planner_grounding_failure_as_grounding_failure() -> None:
    score = BenchmarkScore(
        scenario_name="pcr_setup",
        success=False,
        completion_ratio=0.0,
        steps_taken=0,
        time_elapsed_seconds=0.0,
        safety_violations=0,
        blocked_actions=0,
        efficiency_score=0.0,
        details={
            "runner_error": (
                "Planner grounding failure: planned step does not match the current action surface."
            ),
        },
    )
    result = type(
        "FakeRunResult",
        (),
        {
            "score": score,
            "error": (
                "Planner grounding failure: planned step does not match the current action surface."
            ),
        },
    )()

    taxonomy = _error_taxonomy(
        result=result,
        condition="global_one_shot_planner",
        scenario="pcr_setup",
        run_metrics={},
    )

    assert taxonomy == "grounding_failure"


def test_health_reports_diagnostics_when_row_records_runtime_error(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record["error"] = "synthetic runtime error"
    record["error_taxonomy"] = "runner_failure"
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    health = build_pipeline_health_report(bundle_dir)

    assert health.diagnostics_clear is False
    assert any("runtime error" in reason.casefold() for reason in health.blocking_reasons)


def test_paper_analyze_handles_runner_failure_with_empty_artifacts(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record.update(
        {
            "success": False,
            "completion_ratio": 0.0,
            "steps_taken": 0,
            "time_elapsed_seconds": 0.0,
            "efficiency_score": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "wall_time_seconds": 0.01,
            "error": "synthetic isolated worker crash",
            "error_taxonomy": "runner_failure",
            "trace_ids": [],
            "event_ids": [],
            "details": {
                "error": "synthetic isolated worker crash",
                "runner_traceback": None,
                "exit_code": -11,
                "stderr_tail": "",
            },
        }
    )
    run_dir = bundle_dir / record["run_artifact_dir"]
    (run_dir / "steps.jsonl").write_text("")
    (run_dir / "replay.jsonl").write_text("")
    (run_dir / "normalized_trace.jsonl").write_text("")
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    run_meta.update(
        {
            "status": "failed",
            "replay_available": False,
            "replay_frame_count": 0,
            "normalized_trace_available": False,
            "normalized_trace_row_count": 0,
            "details": record["details"],
            "score": {
                "success": False,
                "completion_ratio": 0.0,
                "steps_taken": 0,
                "safety_violations": 0,
                "blocked_actions": 0,
                "time_elapsed_seconds": 0.0,
                "error": "synthetic isolated worker crash",
                "error_taxonomy": "runner_failure",
            },
        }
    )
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    default_dir = analyze_paper_bundle(bundle_dir)
    explicit_dir = analyze_paper_bundle(bundle_dir, output_dir=tmp_path / "analysis-explicit")

    for analysis_dir in (default_dir, explicit_dir):
        assert (analysis_dir / "report.json").exists()
        assert (analysis_dir / "condition_summary.csv").exists()
        assert (analysis_dir / "pairwise_deltas.csv").exists()
        assert (analysis_dir / "task_family_summary.csv").exists()
        assert (analysis_dir / "paper_role_summary.csv").exists()


def test_health_allows_grounding_failure_without_runtime_blocker(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record["success"] = False
    record["completion_ratio"] = 0.5
    record["error"] = (
        "LLM failed to produce a valid action after 3 attempts. "
        "Last error: Unknown candidate_ref 'preview_take_material'. "
        "Available candidate_refs: move_1, observe_1"
    )
    record["error_taxonomy"] = "grounding_failure"
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    health = build_pipeline_health_report(bundle_dir)

    assert health.diagnostics_clear is True
    assert not any("runtime error" in reason.casefold() for reason in health.blocking_reasons)
    assert not any("runner_failure" in reason.casefold() for reason in health.blocking_reasons)


def test_health_normalizes_legacy_runner_failure_grounding_rows(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record["success"] = False
    record["completion_ratio"] = 0.5
    record["error"] = (
        "LLM failed to produce a valid action after 3 attempts. "
        "Last error: Unknown candidate_ref 'preview_take_material'. "
        "Available candidate_refs: move_1, observe_1"
    )
    record["error_taxonomy"] = "runner_failure"
    record["details"] = {
        "runner_error": record["error"],
    }
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    health = build_pipeline_health_report(bundle_dir)

    assert health.diagnostics_clear is True
    assert not any("runtime error" in reason.casefold() for reason in health.blocking_reasons)
    assert not any("runner_failure" in reason.casefold() for reason in health.blocking_reasons)


def test_health_reports_diagnostics_when_row_records_semantic_fallback(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    bundle = run_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        replications=1,
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    record = load_paper_run_records(bundle.results_path)[0].model_dump(mode="json")
    record["semantic_fallback_count"] = 1
    (bundle_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            build_paper_bundle_summary(load_paper_run_records(bundle_dir / "results.jsonl")),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    health = build_pipeline_health_report(bundle_dir)

    assert health.diagnostics_clear is False
    assert any("semantic fallback" in reason.casefold() for reason in health.blocking_reasons)


def test_paper_preflight_persists_preflight_state_json(tmp_path: Path) -> None:
    """preflight_paper_experiment must drop a preflight_state.json next to the
    target bundle when called with an explicit bundle_dir. This anchors the
    drift check in run_paper_experiment so we abort before burning API budget
    if the repo state moves between preflight and launch."""

    bundle_dir = tmp_path / "paper-bundle"
    report = preflight_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )

    assert report.valid is True
    state_path = bundle_dir / "preflight_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["manifest_id"]
    assert state["initial_variant_id"] == "baseline"
    assert len(state["initial_variant_hash"]) == 64
    assert "pcr_setup" in state["scenario_source_hashes"]
    assert "pcr_setup" in state["config_hashes"]
    assert "scripted_oracle" in state["prompt_family_hashes"]
    assert state["scientific_artifact_hashes"]


def test_paper_run_aborts_on_preflight_state_drift(tmp_path: Path) -> None:
    """run_paper_experiment must refuse to launch if preflight_state.json on
    disk disagrees with the current repo state. The user either re-runs
    preflight or explicitly deletes the stale state file."""

    bundle_dir = tmp_path / "paper-bundle"
    preflight = preflight_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        bundle_dir=bundle_dir,
        allow_dirty=True,
    )
    assert preflight.valid is True

    state_path = bundle_dir / "preflight_state.json"
    state = json.loads(state_path.read_text())
    # Poison the persisted scenario-source hash to simulate a post-preflight
    # edit to the scenario source file.
    state["scenario_source_hashes"]["pcr_setup"] = "0" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="drifted"):
        run_paper_experiment(
            conditions=["scripted_oracle"],
            scenarios=["pcr_setup"],
            profile_name="deterministic",
            replications=1,
            bundle_dir=bundle_dir,
            allow_dirty=True,
        )


def test_paper_run_aborts_on_initial_variant_drift(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"
    preflight = preflight_paper_experiment(
        conditions=["scripted_oracle"],
        scenarios=["pcr_setup"],
        profile_name="deterministic",
        bundle_dir=bundle_dir,
        allow_dirty=True,
        initial_variant_id="sensitivity_v1",
    )
    assert preflight.valid is True

    state_path = bundle_dir / "preflight_state.json"
    state = json.loads(state_path.read_text())
    state["initial_variant_hash"] = "0" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="initial_variant_hash drift"):
        run_paper_experiment(
            conditions=["scripted_oracle"],
            scenarios=["pcr_setup"],
            profile_name="deterministic",
            replications=1,
            bundle_dir=bundle_dir,
            allow_dirty=True,
            initial_variant_id="sensitivity_v1",
        )
