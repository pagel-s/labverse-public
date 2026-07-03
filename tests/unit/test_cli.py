"""Tests for the publication-artifact CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from labverse.cli.main import app

RUNNER = CliRunner()


def test_cli_paper_commands_smoke(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "paper-bundle"

    run_result = RUNNER.invoke(
        app,
        [
            "paper-run",
            "--condition",
            "scripted_oracle",
            "--scenario",
            "pcr_setup",
            "--reps",
            "1",
            "--allow-dirty",
            "--output-dir",
            str(bundle_dir),
        ],
    )

    assert run_result.exit_code == 0
    assert "Bundle:" in run_result.stdout

    analyze_result = RUNNER.invoke(app, ["paper-analyze", str(bundle_dir)])
    assert analyze_result.exit_code == 0
    assert "Analysis written to" in analyze_result.stdout

    package_result = RUNNER.invoke(app, ["paper-package", str(bundle_dir)])
    assert package_result.exit_code == 0
    assert "Package written to" in package_result.stdout

    writeup_result = RUNNER.invoke(app, ["paper-writeup", str(bundle_dir)])
    assert writeup_result.exit_code == 0
    assert "Writeup written to" in writeup_result.stdout

    status_result = RUNNER.invoke(app, ["paper-status", str(bundle_dir)])
    assert status_result.exit_code == 0
    status_payload = json.loads(status_result.stdout)
    assert status_payload["completed_runs"] == 1

    inspect_result = RUNNER.invoke(
        app,
        ["paper-inspect", str(bundle_dir), "--tail", "1"],
    )
    assert inspect_result.exit_code == 0
    inspect_payload = json.loads(inspect_result.stdout)
    assert isinstance(inspect_payload, list)

    chat_result = RUNNER.invoke(
        app,
        ["paper-chat-view", str(bundle_dir), "--tail", "1"],
    )
    assert chat_result.exit_code == 0
    assert "[SYSTEM]" in chat_result.stdout or "No recorded steps for run" in chat_result.stdout


def test_cli_paper_preflight_and_compare(tmp_path: Path) -> None:
    left_bundle = tmp_path / "left"
    right_bundle = tmp_path / "right"

    preflight_result = RUNNER.invoke(
        app,
        [
            "paper-preflight",
            "--condition",
            "scripted_oracle",
            "--scenario",
            "pcr_setup",
            "--allow-dirty",
            "--output-dir",
            str(left_bundle),
        ],
    )
    assert preflight_result.exit_code == 0
    preflight_payload = json.loads(preflight_result.stdout)
    assert preflight_payload["issues"] == []

    for bundle_dir in (left_bundle, right_bundle):
        run_result = RUNNER.invoke(
            app,
            [
                "paper-run",
                "--condition",
                "scripted_oracle",
                "--scenario",
                "pcr_setup",
                "--reps",
                "1",
                "--allow-dirty",
                "--output-dir",
                str(bundle_dir),
            ],
        )
        assert run_result.exit_code == 0

    compare_result = RUNNER.invoke(
        app,
        ["paper-compare", str(left_bundle), str(right_bundle)],
    )
    assert compare_result.exit_code == 0
    compare_payload = json.loads(compare_result.stdout)
    assert compare_payload["left_valid"] is True
    assert compare_payload["right_valid"] is True


def test_cli_paper_probe_outputs_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "labverse.cli.main.probe_paper_request",
        lambda **kwargs: type(
            "FakeProbeReport",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "condition": kwargs["condition"],
                    "model": kwargs["model"],
                    "scenario": kwargs["scenario"],
                    "elapsed_seconds": 1.23,
                }
            },
        )(),
    )

    result = RUNNER.invoke(
        app,
        [
            "paper-probe",
            "--model",
            "gpt-4.1-mini",
            "--condition",
            "local_narrative_interactive_grounded",
            "--scenario",
            "shared_equipment_scheduling",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["condition"] == "local_narrative_interactive_grounded"
    assert payload["scenario"] == "shared_equipment_scheduling"


def test_cli_paper_audit_outputs_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "labverse.cli.main.build_scientific_audit_report",
        lambda **kwargs: type(
            "FakeScientificAuditReport",
            (),
            {
                "valid": True,
                "model_dump": lambda self, mode="json": {
                    "mode": "repo",
                    "valid": True,
                    "diagnostics_clear": False,
                    "blocking_reasons": [],
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "labverse.cli.main.write_scientific_audit_report",
        lambda report, **kwargs: (
            tmp_path / "health" / "scientific_audit.json",
            tmp_path / "health" / "scientific_audit.md",
        ),
    )

    result = RUNNER.invoke(
        app,
        [
            "paper-audit",
            "--allow-dirty",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "repo"
    assert payload["valid"] is True
