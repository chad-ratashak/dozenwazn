#!/usr/bin/env python3
"""Generate per-model CSV templates pre-populated with all verb candidates."""

import csv
import json
import sys
from pathlib import Path

MODELS = ['haiku', 'sonnet', 'opus', 'gpt', 'gemini', 'gemma', 'llama', 'jais']

CSV_HEADERS = [
    'ROOT',
    'ROOT_TYPE',
    'ARITY',
    'PATTERN',
    'VERB',
    'EXISTS?',
    'SINGLE_TOKEN_WORD?',
    'DIALECT_ONLY?',
    'LIST_OF_DIALECTS',
    'DIALECT_MEANINGS',
    'MSA?',
    'MSA_MEANINGS',
]


def generate_csvs(candidates_path: Path, output_dir: Path):
    """Read candidates JSONL and write one CSV per model."""
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))

    output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        csv_path = output_dir / f'{model}.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for c in candidates:
                writer.writerow([
                    c['root'],
                    c['root_type'],
                    c['arity'],
                    c['pattern'],
                    c['verb'],
                    '',  # EXISTS?
                    '',  # SINGLE_TOKEN_WORD?
                    '',  # DIALECT_ONLY?
                    '',  # LIST_OF_DIALECTS
                    '',  # DIALECT_MEANINGS
                    '',  # MSA?
                    '',  # MSA_MEANINGS
                ])
        print(f"  {csv_path} ({len(candidates):,} rows)", file=sys.stderr)

    return len(MODELS)


if __name__ == '__main__':
    candidates = Path('data/candidates.jsonl')
    if not candidates.exists():
        print("Run src/generate.py first to create candidates.jsonl", file=sys.stderr)
        sys.exit(1)
    count = generate_csvs(candidates, Path('data'))
    print(f"\nGenerated {count} CSV files.", file=sys.stderr)
