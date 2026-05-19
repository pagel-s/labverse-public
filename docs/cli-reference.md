# CLI Reference

The public CLI is limited to publication-artifact reproduction and inspection.

## Version

```bash
labverse version
```

## Serve The Paper API

```bash
labverse paper-serve --host 127.0.0.1 --port 8000
```

## Run Or Preflight The Paper Matrix

```bash
labverse paper-preflight [OPTIONS]
labverse paper-run [OPTIONS]
```

Common options:

```text
--model TEXT
--condition TEXT
--scenario TEXT
--profile TEXT
--reps INTEGER
--seed INTEGER
--manifest TEXT
--output-dir TEXT
--resume
--allow-dirty
--initial-variant TEXT
```

## Probe One Provider Call

```bash
labverse paper-probe \
  --model ollama/gemma4:31b \
  --condition global_one_shot_planner \
  --scenario pcr_setup
```

## Inspect Existing Bundles

```bash
labverse paper-status paper_runs/initial-state-perturbation-2
labverse paper-audit paper_runs/initial-state-perturbation-2
labverse paper-inspect paper_runs/initial-state-perturbation-2 --tail 10
labverse paper-chat-view paper_runs/initial-state-perturbation-2 --tail 1
```

`paper-audit` is diagnostic and can exit non-zero for empirical bundles that
contain provider, runner, or model failures.

## Analyze, Package, And Compare

```bash
labverse paper-analyze paper_runs/initial-state-perturbation-2
labverse paper-writeup paper_runs/initial-state-perturbation-2
labverse paper-package paper_runs/initial-state-perturbation-2
labverse paper-compare paper_runs/run-a paper_runs/run-b
```
