#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Run (or resume) gemma validation of all 64,260 candidates
cd "$(dirname "$0")"

# Spotcheck gate — DO NOT REMOVE. A multi-hour run must not start on a model
# that fails the accuracy gate (the April 2026 gemma run produced degenerate
# output for ~10 hours before anyone noticed). Run the gate first:
#   uv run python -m src.cli spotcheck -m gemma
# Escape hatch for deliberate overrides: DOZENWAZN_SKIP_GATE=1 ./run_gemma.sh
if [ ! -f data/spotcheck_gemma.PASS ] && [ "${DOZENWAZN_SKIP_GATE:-0}" != "1" ]; then
  echo "REFUSING to start: data/spotcheck_gemma.PASS not found."
  echo "Run the accuracy gate first:  uv run python -m src.cli spotcheck -m gemma"
  echo "(Deliberate override: DOZENWAZN_SKIP_GATE=1 $0)"
  exit 1
fi

"${UV:-uv}" run python -m src.cli validate -m gemma -b 25
