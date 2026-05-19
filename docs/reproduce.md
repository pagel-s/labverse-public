# Reproduce

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

The paper benchmark can run scripted oracle checks without an LLM provider. Full
local-model reproduction expects Ollama-compatible models reachable through the
configured base URL.

Recommended local-run environment:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11435/v1/
export OLLAMA_THINK=false
export LABVERSE_OLLAMA_KEEP_ALIVE=30s
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
```

Do not reduce context length unless intentionally changing the reproduction
condition.

## Validate Included Bundles

```bash
labverse paper-status paper_runs/initial-state-perturbation-2
labverse paper-status paper_runs/initial-state-reference
labverse paper-status paper_runs/initial-state-perturbation-1
labverse paper-analyze paper_runs/initial-state-perturbation-2
labverse paper-writeup paper_runs/initial-state-perturbation-2
```

The current public validation checks bundle-internal consistency rather than
requiring old bundles to match the current repo's scientific-artifact hashes.
This allows the complete reference and nuisance-perturbation bundles to remain
inspectable together.

`paper-audit` is a diagnostic command for investigating model/provider failure
rows. It may exit non-zero on the included empirical bundles because those
failures are part of the benchmark result, not missing artifact files.

## Rerun The Matrix

Use the frozen deterministic profile and perturbation 2 initial-condition
variant:

```bash
labverse paper-run \
  --profile deterministic \
  --model ollama/gemma4:31b \
  --model ollama/gemma4:26b \
  --model ollama/qwen3.5:27b \
  --model ollama/mistral-small3.2:24b \
  --condition local_structured_interactive_grounded \
  --condition local_narrative_interactive_grounded \
  --condition global_structured_interactive_grounded \
  --condition global_one_shot_planner \
  --reps 1 \
  --initial-variant sensitivity_v2 \
  --output-dir paper_runs/new-reproduction-run
```

If a run is interrupted, restart with the same command plus:

```bash
  --resume
```

Use `--allow-dirty` only for audit/debug runs where the Git cleanliness check is
not part of the claim.

## Analyze And Package A Rerun

```bash
labverse paper-status paper_runs/new-reproduction-run
labverse paper-analyze paper_runs/new-reproduction-run
labverse paper-writeup paper_runs/new-reproduction-run
labverse paper-package paper_runs/new-reproduction-run
```

The package command creates a reviewer-facing copy under
`paper_runs/new-reproduction-run/package` unless `--output-dir` is provided.
