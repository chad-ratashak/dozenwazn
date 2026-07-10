#!/usr/bin/env python3
"""Spot-check accuracy gate for LLM validation — the go/no-go before any
multi-hour run.

Samples known-real and known-fake verbs (from data/gold.csv when present,
else the hand-verified seed sets), runs them through the EXACT production
path (same prompt, batching, schema mode), and gates on:

  balanced accuracy ≥ min_balacc  AND  specificity ≥ min_spec
  AND missing-rate ≤ max_missing

The specificity floor is what catches the observed gemma failure mode —
answering "exists: true" for everything scores specificity ≈ 0.

On PASS it writes data/spotcheck_{model}.PASS (the launchers refuse to
start without it); on FAIL any stale marker is removed. Full per-verb
results go to data/spotcheck_{model}_{timestamp}.json either way.

Sound roots only (no weak radicals, no hamza) so surface-rendering issues
can never contaminate the truth labels.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import httpx

from src.stats import fmt_rate, wilson_ci
from src.validate_ollama import (
    OLLAMA_BASE,
    build_batches,
    classify_batch,
    console,
    probe_schema_support,
)

WEAK_OR_HAMZA = set('ءوي')


def _sound_root(root: str) -> bool:
    return not (set(root) & WEAK_OR_HAMZA)


def sample_sets(data_dir: Path, n_real: int, n_fake: int, seed: int):
    """→ (real, fake, source) — lists of {verb, root, pattern} dicts."""
    gold_path = data_dir / 'gold.csv'
    if not gold_path.exists():
        from src.spotcheck_seed import SEED_FAKE, SEED_REAL
        real = [{'verb': v, 'root': r, 'pattern': p} for v, r, p in SEED_REAL]
        fake = [{'verb': v, 'root': r, 'pattern': p} for v, r, p in SEED_FAKE]
        return real[:n_real], fake[:n_fake], 'FALLBACK seed lists (gold.csv missing)'

    import csv
    real_pool: dict[str, list[dict]] = {}
    fake_pool: dict[str, list[dict]] = {}
    with open(gold_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not _sound_root(row['ROOT']) or row['ARITY'] != '3':
                continue
            entry = {'verb': row['VERB'], 'root': row['ROOT'], 'pattern': row['PATTERN']}
            if row['GOLD'] == 'attested_dictionary' and row['ARRAMOOZ'] == 'exact':
                real_pool.setdefault(row['PATTERN'], []).append(entry)
            elif row['GOLD'] == 'unattested':
                fake_pool.setdefault(row['PATTERN'], []).append(entry)

    rng = random.Random(seed)

    def stratified(pool: dict[str, list[dict]], n: int) -> list[dict]:
        patterns = sorted(pool)
        per = max(1, n // max(1, len(patterns)))
        picked = []
        for p in patterns:
            entries = sorted(pool[p], key=lambda e: e['verb'])
            rng.shuffle(entries)
            picked.extend(entries[:per])
        # top up to n from the union
        if len(picked) < n:
            rest = [e for p in patterns for e in pool[p] if e not in picked]
            rng.shuffle(rest)
            picked.extend(rest[: n - len(picked)])
        return picked[:n]

    return stratified(real_pool, n_real), stratified(fake_pool, n_fake), 'gold.csv'


def run_spotcheck(
    model_name: str,
    model_tag: str,
    data_dir: Path,
    n_real: int = 50,
    n_fake: int = 50,
    seed: int = 42,
    batch_size: int = 25,
    min_balacc: float = 0.80,
    min_specificity: float = 0.70,
    max_missing: float = 0.05,
    parallel: int = 1,
) -> bool:
    """Run the gate for one model. Returns True on PASS."""
    real, fake, source = sample_sets(data_dir, n_real, n_fake, seed)
    truth = {e['verb']: True for e in real} | {e['verb']: False for e in fake}
    sample = real + fake
    random.Random(seed).shuffle(sample)  # don't send all-real then all-fake

    console.print(f"\n[bold]Spotcheck: {model_name}[/bold] ({model_tag})")
    console.print(f"  Sample: {len(real)} real + {len(fake)} fake from {source}")

    logger = logging.getLogger(f"spotcheck_{model_name}")
    if not logger.handlers:
        log_dir = data_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        handler = logging.FileHandler(log_dir / f"spotcheck_{model_name}_{ts}.log",
                                      encoding='utf-8')
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    client = httpx.Client(
        base_url=OLLAMA_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )
    from src.validate_ollama import check_model_exists
    if not check_model_exists(model_tag, client):
        client.close()
        console.print(
            f"  [red]NO-GO: model tag '{model_tag}' is not installed on the "
            f"Ollama server — check `ollama list`.[/red]"
        )
        return False
    use_schema = probe_schema_support(model_tag, client, logger)

    predictions: dict[str, bool | None] = {}
    started = time.monotonic()
    try:
        batches = build_batches(sample, batch_size)
        for batch in batches:
            for r in classify_batch(model_tag, batch, client, logger, use_schema=use_schema):
                predictions[r.verb] = r.exists
    finally:
        client.close()
    elapsed = time.monotonic() - started

    tp = sum(1 for e in real if predictions.get(e['verb']) is True)
    fn = sum(1 for e in real if predictions.get(e['verb']) is False)
    tn = sum(1 for e in fake if predictions.get(e['verb']) is False)
    fp = sum(1 for e in fake if predictions.get(e['verb']) is True)
    missing = sum(1 for v in truth if predictions.get(v) is None)

    n_real_answered = tp + fn
    n_fake_answered = tn + fp
    sensitivity = tp / n_real_answered if n_real_answered else 0.0
    specificity = tn / n_fake_answered if n_fake_answered else 0.0
    balacc = (sensitivity + specificity) / 2
    missing_rate = missing / len(truth) if truth else 1.0
    verbs_per_s = len(truth) / elapsed if elapsed > 0 else 0.0

    console.print(f"  Sensitivity: {fmt_rate(tp, n_real_answered)}")
    console.print(f"  Specificity: {fmt_rate(tn, n_fake_answered)}")
    lo, hi = wilson_ci(tp + tn, n_real_answered + n_fake_answered)
    console.print(f"  Balanced accuracy: {balacc:.2%} "
                  f"(overall accuracy 95% CI {lo:.2%}–{hi:.2%})")
    console.print(f"  Missing: {fmt_rate(missing, len(truth))}")
    eta_h = 64_260 / verbs_per_s / 3600 if verbs_per_s else float('inf')
    console.print(f"  Throughput: {verbs_per_s:.2f} verbs/s → full-run ETA ≈ {eta_h:.1f} h")

    passed = (balacc >= min_balacc and specificity >= min_specificity
              and missing_rate <= max_missing)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = data_dir / f"spotcheck_{model_name}_{ts}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name, 'model_tag': model_tag, 'source': source,
            'seed': seed, 'batch_size': batch_size, 'use_schema': use_schema,
            'n_real': len(real), 'n_fake': len(fake),
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'missing': missing,
            'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': balacc, 'missing_rate': missing_rate,
            'verbs_per_s': verbs_per_s, 'elapsed_s': elapsed,
            'thresholds': {'min_balacc': min_balacc, 'min_specificity': min_specificity,
                           'max_missing': max_missing},
            'passed': passed,
            'per_verb': [
                {'verb': v, 'truth': t, 'predicted': predictions.get(v)}
                for v, t in truth.items()
            ],
        }, f, ensure_ascii=False, indent=1)
    console.print(f"  Report: {report_path}")

    marker = data_dir / f"spotcheck_{model_name}.PASS"
    if passed:
        marker.write_text(f"{model_tag} {ts} balacc={balacc:.3f} "
                          f"specificity={specificity:.3f} missing={missing_rate:.3f}\n",
                          encoding='utf-8')
        console.print(f"  [green bold]GO[/green bold] — marker written: {marker}")
    else:
        if marker.exists():
            marker.unlink()
        reasons = []
        if balacc < min_balacc:
            reasons.append(f"balanced accuracy {balacc:.2%} < {min_balacc:.0%}")
        if specificity < min_specificity:
            reasons.append(f"specificity {specificity:.2%} < {min_specificity:.0%}")
        if missing_rate > max_missing:
            reasons.append(f"missing {missing_rate:.2%} > {max_missing:.0%}")
        console.print(f"  [red bold]NO-GO[/red bold]: {'; '.join(reasons)}")
    return passed


def run_bench(model_name: str, model_tag: str, data_dir: Path,
              batch_sizes=(12, 25, 40), parallels=(1, 2, 4),
              n_batches: int = 3, seed: int = 42):
    """Throughput/quality grid: report verbs/s AND missing-rate per config —
    quality degrades with batch size, so pick on both axes."""
    from concurrent.futures import ThreadPoolExecutor

    real, fake, source = sample_sets(data_dir, 60, 60, seed)
    pool_all = real + fake
    random.Random(seed).shuffle(pool_all)

    logger = logging.getLogger(f"bench_{model_name}")
    logger.addHandler(logging.NullHandler())

    console.print(f"\n[bold]Bench: {model_name}[/bold] ({model_tag}) — sample from {source}")
    console.print(f"  {'batch':>6} {'par':>4} {'verbs/s':>8} {'missing':>18}")

    client = httpx.Client(
        base_url=OLLAMA_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )
    use_schema = probe_schema_support(model_tag, client, logger)
    try:
        for bs in batch_sizes:
            for par in parallels:
                batches = build_batches(pool_all, bs)[:n_batches]
                sent = sum(len(b) for b in batches)
                missing = 0
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=par) as pool:
                    futs = [
                        pool.submit(classify_batch, model_tag, b, client, logger,
                                    use_schema)
                        for b in batches
                    ]
                    for fut in futs:
                        missing += sum(1 for r in fut.result() if r.exists is None)
                elapsed = time.monotonic() - started
                rate = sent / elapsed if elapsed else 0.0
                console.print(f"  {bs:>6} {par:>4} {rate:>8.2f} {fmt_rate(missing, sent):>18}")
    finally:
        client.close()
