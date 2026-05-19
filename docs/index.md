# LabVerse Publication Artifact

This repository contains the LabVerse simulator, frozen paper benchmark, paper
analysis pipeline, trace viewer, and full-trace bundles used for the
publication artifact and sensitivity checks.

Included bundles:

```text
paper_runs/initial-state-reference        # unperturbed reference state
paper_runs/initial-state-perturbation-1   # nuisance perturbation 1
paper_runs/initial-state-perturbation-2   # nuisance perturbation 2
```

Each contains 224 completed model x planner x scenario runs, including raw
result rows, per-step prompts, model attempts, resolved actions, replay frames,
analysis outputs, health reports, and writeup artifacts.

The three bundles are complete matrix runs under three initial states. The
reference state has no overlay; perturbations 1 and 2 add nuisance-only
materials, samples, and people to test sensitivity to initial conditions.

Expected bundle status:

```text
completed_runs = 224
total_expected_runs = 224
validation_ok = true
```

Some rows record provider/runtime/model failures. Those failures are empirical
results from the benchmark run, not missing artifact files.

Use:

- [Reproduce](reproduce.md) to rerun, resume, analyze, and package the paper benchmark.
- [Inspect Traces](inspect.md) to use the paper viewer.
- [Figures](figures.md) to recreate manuscript and SI figures.
- [Figure Style](figure-style.md) for the publication plot style guide.
- [CLI Reference](cli-reference.md) for supported public commands.
