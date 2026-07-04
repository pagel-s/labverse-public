# Figures

The core manuscript figures can be recreated from the three frozen public
bundles:

```bash
python -m pip install matplotlib numpy
python scripts/plot_paper_main_results.py
```

This writes:

```text
paper/figures/benchmark_schematic.{pdf,svg,png}
paper/figures/main_success_by_model_planner.{pdf,svg,png}
paper/figures/scenario_heatmap_model_planner.{pdf,svg,png}
paper/figures/failure_taxonomy_by_model_planner.{pdf,svg,png}
paper/figures/sensitivity_initial_conditions.{pdf,svg,png}
```

The script also writes source CSV files for the quantitative panels under
`paper/figures/`.

## Headline Replay Video

The public artifact also includes a short 1080p replay video generated from the
same frozen trace data used by the web inspector:

```bash
python scripts/render_headline_video.py
```

This writes:

```text
paper/videos/headline_labverse_replay.mp4
```

By default the video uses the successful canonical `chemical_spill_response`
trace from `gemma4:31b` under the global one-shot planner. To render another
trace, pass `--bundle`, `--run`, and `--output`.

For manuscript assembly, the restrained journal-style plates are the best
starting point:

```bash
python scripts/plot_paper_journal_figures.py
```

This writes:

```text
paper/figures/journal_main_benchmark.{pdf,svg,png}
paper/figures/journal_failure_taxonomy.{pdf,svg,png}
paper/figures/journal_sensitivity.{pdf,svg,png}
paper/figures/si/journal_si_diagnostics.{pdf,svg,png}
```

Each `journal_*` panel has a matching CSV source file.

The compact camera-ready benchmark plates remain available:

```bash
python scripts/plot_paper_camera_ready.py
```

This writes:

```text
paper/figures/camera_ready_main_benchmark.{pdf,svg,png}
paper/figures/camera_ready_scenario_matrix.{pdf,svg,png}
paper/figures/camera_ready_failure_taxonomy.{pdf,svg,png}
paper/figures/camera_ready_initial_sensitivity.{pdf,svg,png}
```

The `camera_ready_*` CSV files contain the plotted values, including Wilson
confidence intervals for the ranked model-planner success panel.

The more colorful bar-first styled figure family remains available as an
alternative:

```bash
python scripts/plot_paper_styled_results.py
```

This writes the main figures:

```text
paper/figures/styled_main_success_bars.{pdf,svg,png}
paper/figures/styled_scenario_difficulty_bars.{pdf,svg,png}
paper/figures/styled_failure_taxonomy_bars.{pdf,svg,png}
paper/figures/styled_sensitivity_bars.{pdf,svg,png}
```

It also writes supplementary figures:

```text
paper/figures/si/styled_si_completion_efficiency.{pdf,svg,png}
paper/figures/si/styled_si_reliability_counts.{pdf,svg,png}
paper/figures/si/styled_si_runtime_cost.{pdf,svg,png}
paper/figures/si/styled_si_scenario_planner_bars.{pdf,svg,png}
```

Each styled panel has a matching CSV source file.

Scenario labels are intentionally displayed as `Scenario 1` through
`Scenario 14` in frozen-manifest order. The default canonical bundle is
`paper_runs/initial-state-perturbation-2`; the sensitivity figure compares it
against `paper_runs/initial-state-reference` and
`paper_runs/initial-state-perturbation-1`.
