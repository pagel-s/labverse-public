"""Publication-artifact CLI entry point."""

from __future__ import annotations

import json

import typer
import uvicorn

from labverse import __version__
from labverse.api import create_app
from labverse.benchmarks.initial_variants import BASELINE_INITIAL_VARIANT_ID
from labverse.paper import (
    ALL_PAPER_CONDITIONS,
    analyze_paper_bundle,
    build_scientific_audit_report,
    bundle_status,
    compare_bundles,
    default_manifest_path,
    generate_paper_writeup,
    inspect_bundle_steps,
    package_paper_bundle,
    preflight_paper_experiment,
    probe_paper_request,
    render_bundle_chat_view,
    run_paper_experiment,
    write_scientific_audit_report,
)

app = typer.Typer(
    name="labverse",
    help="LabVerse publication artifact: reproduce, analyze, and inspect paper runs.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show the LabVerse artifact version."""
    typer.echo(f"labverse {__version__}")


@app.command("paper-serve")
def paper_serve(
    host: str = typer.Option("127.0.0.1", help="Host interface to bind."),
    port: int = typer.Option(8001, help="Port to bind."),
) -> None:
    """Start the paper-bundle inspection API."""
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


@app.command("paper-run")
def paper_run(
    models: list[str] = typer.Option(
        [],
        "--model",
        "-m",
        help="Model names. Required unless only using scripted_oracle.",
    ),
    conditions: list[str] = typer.Option(
        [],
        "--condition",
        "-c",
        help=f"Paper conditions. Supported: {', '.join(ALL_PAPER_CONDITIONS)}.",
    ),
    scenarios: list[str] = typer.Option(
        [],
        "--scenario",
        "-s",
        help="Optional scenario subset from the frozen paper manifest.",
    ),
    profile: str = typer.Option("deterministic", help="Frozen paper profile to use."),
    reps: int | None = typer.Option(None, "--reps", "-r", help="Override profile replications."),
    seed: int = typer.Option(7, help="Random seed for shuffled run order."),
    manifest: str = typer.Option(
        str(default_manifest_path()),
        help="Path to the frozen paper manifest JSON.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Bundle directory. Default: timestamped directory under paper_runs/.",
    ),
    resume: bool = typer.Option(False, help="Resume an existing paper bundle."),
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="Allow running from a dirty git worktree. Intended only for audit/debug runs.",
    ),
    initial_variant: str = typer.Option(
        BASELINE_INITIAL_VARIANT_ID,
        "--initial-variant",
        help="Initial-condition variant from paper/initial_condition_variants.yaml.",
    ),
) -> None:
    """Run the frozen paper benchmark and write an immutable bundle directory."""
    bundle = run_paper_experiment(
        models=models or None,
        conditions=conditions or None,
        scenarios=scenarios or None,
        manifest_path=manifest,
        profile_name=profile,
        replications=reps,
        random_seed=seed,
        bundle_dir=output_dir,
        resume=resume,
        allow_dirty=allow_dirty,
        initial_variant_id=initial_variant,
    )
    typer.echo(f"Bundle: {bundle.bundle_dir}")
    typer.echo(f"Manifest: {bundle.manifest_path}")
    typer.echo(f"Metadata: {bundle.metadata_path}")
    typer.echo(f"Results: {bundle.results_path}")
    typer.echo(f"Summary: {bundle.summary_path}")


@app.command("paper-preflight")
def paper_preflight(
    models: list[str] = typer.Option([], "--model", "-m", help="Model names."),
    conditions: list[str] = typer.Option(
        [],
        "--condition",
        "-c",
        help=f"Paper conditions. Supported: {', '.join(ALL_PAPER_CONDITIONS)}.",
    ),
    scenarios: list[str] = typer.Option(
        [],
        "--scenario",
        "-s",
        help="Optional scenario subset from the frozen paper manifest.",
    ),
    profile: str = typer.Option("deterministic", help="Frozen paper profile to use."),
    reps: int | None = typer.Option(None, "--reps", "-r", help="Override profile replications."),
    seed: int = typer.Option(7, help="Random seed for shuffled run order."),
    manifest: str = typer.Option(str(default_manifest_path()), help="Frozen paper manifest JSON."),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Bundle directory."),
    resume: bool = typer.Option(False, help="Resume an existing paper bundle."),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="Allow a dirty worktree."),
    initial_variant: str = typer.Option(
        BASELINE_INITIAL_VARIANT_ID,
        "--initial-variant",
        help="Initial-condition variant from paper/initial_condition_variants.yaml.",
    ),
) -> None:
    """Validate a frozen paper run configuration before spending model time."""
    report = preflight_paper_experiment(
        models=models or None,
        conditions=conditions or None,
        scenarios=scenarios or None,
        manifest_path=manifest,
        profile_name=profile,
        replications=reps,
        random_seed=seed,
        bundle_dir=output_dir,
        resume=resume,
        allow_dirty=allow_dirty,
        initial_variant_id=initial_variant,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("paper-probe")
def paper_probe(
    model: str = typer.Option(..., "--model", "-m", help="Model name to probe."),
    condition: str = typer.Option(..., "--condition", "-c", help="Paper condition to probe."),
    scenario: str = typer.Option(..., "--scenario", "-s", help="Frozen paper scenario to probe."),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds", help="Provider timeout."),
) -> None:
    """Execute one first-turn provider call for a paper condition/scenario pair."""
    report = probe_paper_request(
        model=model,
        condition=condition,
        scenario=scenario,
        timeout_seconds=timeout_seconds,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("paper-status")
def paper_status(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
) -> None:
    """Show bundle progress and validation state."""
    report = bundle_status(bundle_dir)
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("paper-audit")
def paper_audit(
    bundle_dir: str | None = typer.Argument(
        None,
        help="Optional existing paper bundle directory to audit in bundle mode.",
    ),
    manifest: str = typer.Option(
        str(default_manifest_path()),
        help="Path to the frozen paper manifest JSON.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for repo-only audit mode.",
    ),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="Allow a dirty worktree."),
) -> None:
    """Run the scientific audit for the frozen paper benchmark."""
    report = build_scientific_audit_report(
        bundle_dir=bundle_dir,
        manifest_path=manifest,
        allow_dirty=allow_dirty,
    )
    json_path, md_path = write_scientific_audit_report(
        report,
        bundle_dir=bundle_dir,
        output_dir=output_dir,
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    typer.echo(f"Scientific audit JSON: {json_path}", err=True)
    typer.echo(f"Scientific audit Markdown: {md_path}", err=True)
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("paper-analyze")
def paper_analyze(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Analysis output dir."),
) -> None:
    """Analyze a frozen paper bundle into reproducibility tables and reports."""
    analysis_dir = analyze_paper_bundle(bundle_dir, output_dir=output_dir)
    typer.echo(f"Analysis written to {analysis_dir}")


@app.command("paper-package")
def paper_package(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Package output dir."),
) -> None:
    """Package a frozen paper bundle for reproduction and inspection."""
    package_dir = package_paper_bundle(bundle_dir, output_dir=output_dir)
    typer.echo(f"Package written to {package_dir}")


@app.command("paper-writeup")
def paper_writeup(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Writeup output dir."),
) -> None:
    """Generate writeup tables and derived summaries from a frozen paper bundle."""
    writeup_dir = generate_paper_writeup(bundle_dir, output_dir=output_dir)
    typer.echo(f"Writeup written to {writeup_dir}")


@app.command("paper-inspect")
def paper_inspect(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
    tail: int = typer.Option(10, "--tail", help="Number of latest step records to print."),
) -> None:
    """Print recent recorded step-level audit entries for a bundle."""
    rows = inspect_bundle_steps(bundle_dir, tail=tail)
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@app.command("paper-chat-view")
def paper_chat_view(
    bundle_dir: str = typer.Argument(help="Path to an existing paper bundle directory."),
    run: str | None = typer.Option(None, "--run", help="Optional relative run path."),
    step: int | None = typer.Option(None, "--step", min=1, help="Optional 1-based step index."),
    tail: int = typer.Option(1, "--tail", min=1, help="Tail length when --step is unset."),
    show_tool_schema: bool = typer.Option(
        False,
        "--show-tool-schema",
        help="Include the full action schema JSON for each rendered step.",
    ),
) -> None:
    """Render step audits as a chat-style transcript for human inspection."""
    transcript = render_bundle_chat_view(
        bundle_dir,
        run=run,
        step=step,
        tail=tail,
        show_tool_schema=show_tool_schema,
    )
    typer.echo(transcript)


@app.command("paper-compare")
def paper_compare(
    left_bundle: str = typer.Argument(help="Path to the first paper bundle directory."),
    right_bundle: str = typer.Argument(help="Path to the second paper bundle directory."),
) -> None:
    """Compare two bundles at the condition-summary and integrity level."""
    report = compare_bundles(left_bundle, right_bundle)
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
