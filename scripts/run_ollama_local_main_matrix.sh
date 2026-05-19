#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435/v1/}"
export OLLAMA_THINK="${OLLAMA_THINK:-false}"
export LABVERSE_OLLAMA_KEEP_ALIVE="${LABVERSE_OLLAMA_KEEP_ALIVE:-30s}"

PROFILE="${PROFILE:-deterministic}"
REPS="${REPS:-1}"
GEMMA4_MODEL="${GEMMA4_MODEL:-ollama/gemma4:31b}"
GEMMA4_ALT_MODEL="${GEMMA4_ALT_MODEL:-ollama/gemma4:26b}"
QWEN_MODEL="${QWEN_MODEL:-ollama/qwen3.5:27b}"
MISTRAL_MODEL="${MISTRAL_MODEL:-ollama/mistral-small3.2:24b}"
INITIAL_VARIANT="${INITIAL_VARIANT:-sensitivity_v2}"

OUTPUT_DIR="${1:-/tmp/labverse-ollama-main-matrix-$(date +%Y%m%dT%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

echo "Using OLLAMA_BASE_URL=$OLLAMA_BASE_URL"
echo "Using OLLAMA_THINK=$OLLAMA_THINK"
echo "Using LABVERSE_OLLAMA_KEEP_ALIVE=$LABVERSE_OLLAMA_KEEP_ALIVE"
echo "Recommended Ollama server env: OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1"
echo "Profile=$PROFILE reps=$REPS"
echo "Initial variant=$INITIAL_VARIANT"
echo "Models:"
echo "  $GEMMA4_MODEL"
echo "  $GEMMA4_ALT_MODEL"
echo "  $QWEN_MODEL"
echo "  $MISTRAL_MODEL"
echo "Writing bundle to $OUTPUT_DIR"

MODEL_ARGS=(
  --model "$GEMMA4_MODEL"
  --model "$GEMMA4_ALT_MODEL"
  --model "$QWEN_MODEL"
  --model "$MISTRAL_MODEL"
)

./.venv/bin/labverse paper-run \
  --profile "$PROFILE" \
  "${MODEL_ARGS[@]}" \
  --condition local_structured_interactive_grounded \
  --condition local_narrative_interactive_grounded \
  --condition global_structured_interactive_grounded \
  --condition global_one_shot_planner \
  --reps "$REPS" \
  --initial-variant "$INITIAL_VARIANT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo
echo "Run complete. Inspect with:"
echo "  ./.venv/bin/labverse paper-status $OUTPUT_DIR"
echo "  ./.venv/bin/labverse paper-analyze $OUTPUT_DIR"
