# LabVerse Publication Artifact

LabVerse is a laboratory-simulation benchmark for studying grounded LLM agents
in procedural wet-lab tasks. This public artifact contains the frozen benchmark,
the reproduction pipeline, a paper trace viewer, figure scripts, and full-trace
empirical bundles used for the paper figures and sensitivity checks.

## Replay Preview

<video src="paper/videos/headline_labverse_replay.mp4" controls muted loop playsinline width="100%"></video>

The preview above is generated from a frozen `chemical_spill_response` trace and
shows how LabVerse links model outputs, executable actions, and laboratory state
updates. Regenerate it with:

```bash
python scripts/render_headline_video.py
```

## Included Bundles

```text
paper_runs/initial-state-reference        # unperturbed reference state
paper_runs/initial-state-perturbation-1   # nuisance perturbation 1
paper_runs/initial-state-perturbation-2   # nuisance perturbation 2
```

Each bundle contains 224 model x planner x scenario runs, including prompts,
model attempts, resolved actions, replay frames, analysis outputs, and writeup
artifacts. The perturbation bundles differ only by nuisance initial-condition
overlays; `initial-state-perturbation-2` is the current canonical/default
bundle used by the viewer.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

For the frontend trace viewer:

```bash
cd frontend
npm ci
```

## Inspect Traces

Start the paper API:

```bash
labverse paper-serve --host 127.0.0.1 --port 8001
```

Start the viewer:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The viewer defaults to perturbation 2 and also
provides preset buttons for the reference state and perturbation 1.

## Validate The Artifact

```bash
labverse paper-status paper_runs/initial-state-perturbation-2
labverse paper-status paper_runs/initial-state-reference
labverse paper-status paper_runs/initial-state-perturbation-1
labverse paper-analyze paper_runs/initial-state-perturbation-2
labverse paper-writeup paper_runs/initial-state-perturbation-2
```

Expected status:

```text
completed_runs = 224
total_expected_runs = 224
validation_ok = true
```

Some completed rows record provider/runtime/model failures. Those failures are
part of the empirical result.

## Recreate The Planner Figure

```bash
python scripts/plot_planner_success_matrix.py \
  --bundle-dir paper_runs/initial-state-perturbation-2 \
  --model ollama/gemma4:31b \
  --output-dir paper/figures \
  --stem planner_success_gemma4_31b
```

## Reproduction Docs

The minimal reproduction manual is in `docs/`:

- `docs/reproduce.md`
- `docs/inspect.md`
- `docs/figures.md`
- `docs/cli-reference.md`

## License

This artifact is released under the MIT License.
