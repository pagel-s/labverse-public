# Paper Assets

This directory holds frozen, publication-facing assets for the main LabVerse embodiment paper.

Current contents:

- `embodiment_freeze_v3.json`: the frozen 14-scenario benchmark manifest used by `labverse paper-run`

The intended workflow is:

1. freeze the benchmark manifest here
2. run `labverse paper-run`
3. analyze/package the resulting bundle under `paper_runs/`

Use this directory for benchmark freezes and other paper-facing static assets, not for ad hoc
pilot outputs.
