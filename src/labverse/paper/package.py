"""Packaging helpers for reviewer-facing paper artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from labverse.paper.analysis import analyze_paper_bundle
from labverse.paper.runner import build_paper_bundle_summary, load_paper_run_records
from labverse.paper.validate import assert_valid_paper_bundle
from labverse.paper.writeup import generate_paper_writeup


def package_paper_bundle(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Create a reviewer-facing package directory from a paper bundle."""

    bundle_path = Path(bundle_dir)
    package_dir = Path(output_dir) if output_dir is not None else bundle_path / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    assert_valid_paper_bundle(bundle_path)
    analysis_dir = analyze_paper_bundle(bundle_path)
    writeup_dir = generate_paper_writeup(bundle_path)
    summary_path = bundle_path / "summary.json"
    if not summary_path.exists():
        summary = build_paper_bundle_summary(load_paper_run_records(bundle_path / "results.jsonl"))
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name in [
        "manifest.json",
        "bundle_metadata.json",
        "results.jsonl",
        "summary.json",
    ]:
        shutil.copy2(bundle_path / name, package_dir / name)

    packaged_runs_dir = package_dir / "runs"
    if packaged_runs_dir.exists():
        shutil.rmtree(packaged_runs_dir)
    if (bundle_path / "runs").exists():
        shutil.copytree(bundle_path / "runs", packaged_runs_dir)

    packaged_health_dir = package_dir / "health"
    if packaged_health_dir.exists():
        shutil.rmtree(packaged_health_dir)
    if (bundle_path / "health").exists():
        shutil.copytree(bundle_path / "health", packaged_health_dir)

    packaged_analysis_dir = package_dir / "analysis"
    if packaged_analysis_dir.exists():
        shutil.rmtree(packaged_analysis_dir)
    shutil.copytree(analysis_dir, packaged_analysis_dir)

    packaged_writeup_dir = package_dir / "writeup"
    if packaged_writeup_dir.exists():
        shutil.rmtree(packaged_writeup_dir)
    shutil.copytree(writeup_dir, packaged_writeup_dir)

    reproduce = package_dir / "REPRODUCE.md"
    reproduce.write_text(
        "\n".join(
            [
                "# Reproduce This Paper Bundle",
                "",
                "1. Validate the repo:",
                "",
                "```bash",
                "pytest -q",
                "ruff check src tests",
                "mypy src",
                "cd frontend && npm run build",
                "mkdocs build --strict",
                "```",
                "",
                "2. Re-run the frozen benchmark with the same manifest/profile:",
                "",
                "```bash",
                "labverse paper-run --profile deterministic \\",
                "  --model ollama/gemma4:31b \\",
                "  --model ollama/gemma4:26b \\",
                "  --model ollama/qwen3.5:27b \\",
                "  --model ollama/mistral-small3.2:24b \\",
                "  --condition local_structured_interactive_grounded \\",
                "  --condition local_narrative_interactive_grounded \\",
                "  --condition global_structured_interactive_grounded \\",
                "  --condition global_one_shot_planner \\",
                "  --reps 1 \\",
                "  --output-dir paper_runs/reproduction \\",
                "  --resume",
                "```",
                "",
                "3. Re-analyze the bundle:",
                "",
                "```bash",
                "labverse paper-analyze paper_runs/reproduction",
                "labverse paper-package paper_runs/reproduction",
                "labverse paper-writeup paper_runs/reproduction",
                "```",
                "",
                "This package contains the frozen manifest, bundle metadata, raw results,",
                "derived analysis tables, and writeup table artifacts.",
                "",
            ]
        )
    )
    return package_dir


__all__ = ["package_paper_bundle"]
