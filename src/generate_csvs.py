#!/usr/bin/env python3
"""Generate per-model CSV templates pre-populated with all verb candidates.

Safety model: a CSV that contains filled validation rows is never silently
overwritten. The April 2026 gemma run was destroyed by an unconditional
rewrite here — the guards below are the fix. Destructive rewrites require
force=True (the CLI confirms interactively); schema/candidate migrations go
through merge_csv(), which backs up first and carries filled rows by
(ROOT, PATTERN) key. VERB is not a stable key: weak-radical surface fixes
change verb strings, and corrected surfaces are not unique across roots.
"""

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

MODELS = ['haiku', 'sonnet', 'opus', 'gpt', 'gemini', 'gemma', 'llama', 'jais']

# Candidate-identity columns come straight from candidates.jsonl; validation
# columns are filled by validate_ollama or imports. SINGLE_TOKEN_WORD? was
# removed 2026-07: it is a tokenizer property computed deterministically in
# tokenizer_analysis.jsonl, and asking an LLM for it produced garbage.
CANDIDATE_HEADERS = ['ROOT', 'ROOT_TYPE', 'ARITY', 'PATTERN', 'VERB']
VALIDATION_HEADERS = [
    'EXISTS?',
    'DIALECT_ONLY?',
    'LIST_OF_DIALECTS',
    'DIALECT_MEANINGS',
    'MSA?',
    'MSA_MEANINGS',
]
CSV_HEADERS = CANDIDATE_HEADERS + VALIDATION_HEADERS

BACKUP_DIR_NAME = 'backups'


def candidate_key(record: dict) -> tuple[str, str]:
    """Stable join key for a candidate dict or CSV row dict."""
    if 'ROOT' in record:
        return (record['ROOT'], record['PATTERN'])
    return (record['root'], record['pattern'])


def candidate_row(c: dict) -> list:
    """CSV cells for one candidate, validation columns empty."""
    cells = [c['root'], c['root_type'], c['arity'], c['pattern'], c['verb']]
    cells.extend([''] * len(VALIDATION_HEADERS))
    return cells


def load_candidates(candidates_path: Path) -> list[dict]:
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def read_header(csv_path: Path) -> list[str]:
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        return next(csv.reader(f), [])


def assert_csv_schema(csv_path: Path):
    """Abort when a CSV's header doesn't match the current schema.

    Prevents csv.DictWriter from silently corrupting rows written against a
    stale column set (e.g. the pre-2026-07 12-column layout).
    """
    header = read_header(csv_path)
    if header != CSV_HEADERS:
        raise SystemExit(
            f"{csv_path} has a stale header ({len(header)} columns); "
            f"expected {CSV_HEADERS}. Run `python -m src.cli csvs --merge` to migrate."
        )


def count_filled_rows(csv_path: Path) -> int:
    """Number of rows with a non-empty EXISTS? cell."""
    if not csv_path.exists():
        return 0
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if (row.get('EXISTS?') or '').strip())


def _atomic_write(csv_path: Path, rows: list[list]):
    tmp_path = csv_path.with_suffix('.csv.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        writer.writerows(rows)
    tmp_path.rename(csv_path)


def write_scaffold(csv_path: Path, candidates: list[dict]):
    _atomic_write(csv_path, [candidate_row(c) for c in candidates])


def backup_csv(csv_path: Path, tag: str = '', keep: int = 0) -> Path:
    """Copy a CSV into data/backups/ with a timestamp (and optional tag).

    keep > 0 prunes the oldest backups beyond `keep` within the same
    (stem, tag) namespace — premerge/preforce snapshots are never pruned
    by run-time rotation because they carry a different tag.
    """
    backup_dir = csv_path.parent / BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'.{tag}-{ts}.csv' if tag else f'.{ts}.csv'
    dest = backup_dir / (csv_path.stem + suffix)
    shutil.copy2(csv_path, dest)

    if keep > 0:
        pattern = f'{csv_path.stem}.{tag}-*.csv' if tag else f'{csv_path.stem}.*.csv'
        siblings = sorted(backup_dir.glob(pattern))  # timestamp suffix sorts lexicographically
        for stale in siblings[:-keep]:
            stale.unlink()

    return dest


def scaffold_matches(csv_path: Path, candidates: list[dict]) -> bool:
    """True when an existing CSV already has the current header and the
    exact candidate identity rows (so rewriting it would be a no-op)."""
    if read_header(csv_path) != CSV_HEADERS:
        return False
    expected = [
        (c['root'], c['root_type'], str(c['arity']), c['pattern'], c['verb'])
        for c in candidates
    ]
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        actual = [
            (row['ROOT'], row['ROOT_TYPE'], row['ARITY'], row['PATTERN'], row['VERB'])
            for row in reader
        ]
    return actual == expected


def merge_csv(csv_path: Path, candidates: list[dict]) -> tuple[int, int, int]:
    """Rebuild a CSV in canonical candidate order, carrying validation
    columns from the existing file by (ROOT, PATTERN).

    Backs the existing file up first. Returns (carried, added, dropped):
    carried = filled rows preserved, added = candidates with no prior row,
    dropped = filled rows whose key no longer exists in candidates.
    """
    backup = backup_csv(csv_path, tag='premerge')
    print(f"  backup → {backup}", file=sys.stderr)

    old_validation: dict[tuple[str, str], dict] = {}
    old_filled: set[tuple[str, str]] = set()
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = candidate_key(row)
            values = {h: (row.get(h) or '') for h in VALIDATION_HEADERS}
            if any(v.strip() for v in values.values()):
                old_validation[key] = values
                if (row.get('EXISTS?') or '').strip():
                    old_filled.add(key)

    carried = 0
    added = 0
    rows = []
    seen_keys = set()
    for c in candidates:
        key = candidate_key(c)
        seen_keys.add(key)
        cells = [c['root'], c['root_type'], c['arity'], c['pattern'], c['verb']]
        if key in old_validation:
            cells.extend(old_validation[key][h] for h in VALIDATION_HEADERS)
            if key in old_filled:
                carried += 1
        else:
            cells.extend([''] * len(VALIDATION_HEADERS))
            added += 1
        rows.append(cells)

    dropped = len(old_filled - seen_keys)
    if dropped:
        print(
            f"  WARNING: {dropped} filled rows have no matching candidate "
            f"and were dropped (preserved in {backup})",
            file=sys.stderr,
        )

    _atomic_write(csv_path, rows)
    return carried, added, dropped


def generate_csvs(
    candidates_path: Path,
    output_dir: Path,
    force: bool = False,
    merge: bool = False,
) -> int:
    """Write/refresh one CSV per model, never silently destroying data.

    - missing file            → write a fresh scaffold
    - filled rows, merge      → rebuild carrying data by (ROOT, PATTERN)
    - filled rows, force      → overwrite (caller must have confirmed)
    - filled rows, otherwise  → SKIP
    - zero filled rows        → rewrite only if header/candidates differ
    """
    candidates = load_candidates(candidates_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        csv_path = output_dir / f'{model}.csv'

        if not csv_path.exists():
            write_scaffold(csv_path, candidates)
            print(f"  {csv_path} written ({len(candidates):,} rows)", file=sys.stderr)
            continue

        filled = count_filled_rows(csv_path)

        if merge:
            carried, added, dropped = merge_csv(csv_path, candidates)
            print(
                f"  {csv_path} merged (carried {carried}, added {added}, dropped {dropped})",
                file=sys.stderr,
            )
        elif filled > 0:
            if force:
                backup = backup_csv(csv_path, tag='preforce')
                print(f"  backup → {backup}", file=sys.stderr)
                write_scaffold(csv_path, candidates)
                print(
                    f"  {csv_path} OVERWRITTEN, {filled} filled rows destroyed "
                    f"({len(candidates):,} rows)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  SKIP {csv_path} — {filled} filled rows "
                    f"(use --merge to migrate or --force to overwrite)",
                    file=sys.stderr,
                )
        else:
            if scaffold_matches(csv_path, candidates):
                print(f"  {csv_path} up to date, untouched", file=sys.stderr)
            else:
                write_scaffold(csv_path, candidates)
                print(f"  {csv_path} rewritten ({len(candidates):,} rows)", file=sys.stderr)

    return len(MODELS)


if __name__ == '__main__':
    candidates = Path('data/candidates.jsonl')
    if not candidates.exists():
        print("Run src/generate.py first to create candidates.jsonl", file=sys.stderr)
        sys.exit(1)
    count = generate_csvs(candidates, Path('data'))
    print(f"\nProcessed {count} CSV files.", file=sys.stderr)
