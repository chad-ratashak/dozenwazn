#!/bin/bash
# Run (or resume) Llama 3.2 validation
cd "$(dirname "$0")"
"${UV:-uv}" run python -m src.cli validate -m llama -b 25
