#!/bin/bash
# Run (or resume) Gemma 4 E2B validation
cd "$(dirname "$0")"
"${UV:-uv}" run python -m src.cli validate -m gemma -b 25
