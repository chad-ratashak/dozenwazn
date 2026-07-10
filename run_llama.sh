#!/bin/bash
# Run (or resume) llama validation of all 64,260 candidates
cd "$(dirname "$0")"

# Spotcheck gate — DO NOT REMOVE. A multi-hour run must not start on a model
# that fails the accuracy gate. Run the gate first:
#   uv run python -m src.cli spotcheck -m llama
# Escape hatch for deliberate overrides: DOZENWAZN_SKIP_GATE=1 ./run_llama.sh
if [ ! -f data/spotcheck_llama.PASS ] && [ "${DOZENWAZN_SKIP_GATE:-0}" != "1" ]; then
  echo "REFUSING to start: data/spotcheck_llama.PASS not found."
  echo "Run the accuracy gate first:  uv run python -m src.cli spotcheck -m llama"
  echo "(Deliberate override: DOZENWAZN_SKIP_GATE=1 $0)"
  exit 1
fi

"${UV:-uv}" run python -m src.cli validate -m llama -b 25
