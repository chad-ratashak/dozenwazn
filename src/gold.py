#!/usr/bin/env python3
"""Build the gold ground-truth column from vendored dictionary/corpus data.

LLM validation measures what models *believe*; this module records what
machine-readable lexica *attest*. Output: data/gold.csv, one row per
candidate, same (ROOT, PATTERN) keying as the model CSVs.

Match ladder (per candidate):
  - arramooz: joined on (root, past-vowel pattern). The bāb-of-conjugation
    column encodes the past middle vowel directly (bāb 1–3 → faʿala,
    4/6 → faʿila, 5 → faʿula), so hollow roots — whose surfaces collapse —
    are still discriminated per root and per vocalization. Level 'exact'
    when the vocalized dictionary surface also equals ours, else 'pattern'.
  - wiktionary: normalized-surface match against the (corrected) candidate
    surface. Level 'exact' when that surface belongs to exactly one root in
    the candidate set, else 'root' (attested, but not attributable to this
    root specifically — e.g. قَالَ matches both قول and قيل).
  - corpus: bare-form frequency, evidence only — never alone "real" beyond
    the attested_corpus_only label. Bare forms are vocalization-ambiguous
    and 2-letter geminate bare forms collide heavily; documented weakness.

GOLD label: attested_dictionary > root_attested_form1 >
attested_corpus_only (freq ≥ threshold) > unattested.
"""

import csv
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

from src.stats import fmt_rate
from src.vendor_common import load_manifest, verify_files
from src.vendor_lexica import MANIFEST_PATH, VENDOR_DIR

GOLD_HEADERS = [
    'ROOT', 'ROOT_TYPE', 'ARITY', 'PATTERN', 'VERB',
    'ARRAMOOZ', 'WIKTIONARY', 'CORPUS_FREQ_BARE', 'GOLD', 'GOLD_GLOSS',
]

# bāb of conjugation → past-tense middle vowel pattern. The six abwāb:
# 1 faʿala/yafʿulu, 2 faʿala/yafʿilu, 3 faʿala/yafʿalu, 4 faʿila/yafʿalu,
# 5 faʿula/yafʿulu, 6 faʿila/yafʿilu.
BAB_TO_PATTERN = {
    '1': 'faʿala', '2': 'faʿala', '3': 'faʿala',
    '4': 'faʿila', '5': 'faʿula', '6': 'faʿila',
}

# Same tashkeel range as tokenizer_check.strip_tashkeel (duplicated to keep
# gold.py import-light — tokenizer_check pulls in tiktoken).
_TASHKEEL = re.compile(r'[ً-ْٰ]')

_HAMZA_CARRIERS = {
    'آ': 'ءا',  # آ → ءا
    'أ': 'ء',        # أ
    'إ': 'ء',        # إ
    'ؤ': 'ء',        # ؤ
    'ئ': 'ء',        # ئ
}


def strip_tashkeel(text: str) -> str:
    return _TASHKEEL.sub('', text)


def normalize_surface(s: str) -> str:
    """NFC + hamza-carrier unification for cross-source surface comparison.

    Different sources seat hamza differently and order combining marks
    differently; NFC fixes mark order, carrier folding fixes seats.
    """
    s = unicodedata.normalize('NFC', s)
    s = s.replace('ٱ', 'ا')  # alif wasla → alif
    s = s.replace('ـ', '')        # tatweel
    for zw in ('‎', '‏', '​', '⁠'):  # directional/zero-width marks
        s = s.replace(zw, '')
    for carrier, repl in _HAMZA_CARRIERS.items():
        s = s.replace(carrier, repl)
    return unicodedata.normalize('NFC', s)


def normalize_root(s: str) -> str:
    """Root radicals with hamza carriers folded to bare ء."""
    out = normalize_surface(strip_tashkeel(s))
    return out


def _verified_dir(name: str) -> Path:
    manifest = load_manifest(MANIFEST_PATH)
    entry = manifest.get(name)
    if entry is None:
        raise SystemExit(
            f"Lexicon '{name}' is not vendored — run `uv run python -m src.cli lexica` first."
        )
    directory = VENDOR_DIR / name
    verify_files(name, directory, entry['files'])
    return directory


def parse_arramooz() -> tuple[set, dict]:
    """→ (set of (root, pattern), dict (root, pattern) → set of normalized surfaces)."""
    directory = _verified_dir('arramooz')
    pairs = set()
    surfaces: dict[tuple[str, str], set] = {}
    with open(directory / 'verb_triliteral.csv', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 3:
                continue
            verb, root, bab = cols[0].strip(), cols[1].strip(), cols[2].strip()
            pattern = BAB_TO_PATTERN.get(bab)
            if pattern is None:
                continue
            key = (normalize_root(root), pattern)
            pairs.add(key)
            surfaces.setdefault(key, set()).add(normalize_surface(verb))
    return pairs, surfaces


def parse_wiktionary() -> dict[str, str]:
    """→ dict normalized vocalized Form-I surface → first gloss.

    Wiktionary page titles (kaikki 'word') are UNVOCALIZED for Arabic; the
    vocalized lemma lives in the forms array tagged 'canonical', usually
    alongside a 'form-i'…'form-x' tag. Only Form I canonicals (or untagged
    ones) are kept — other forms could never match Form I candidates, and
    the tag makes the restriction explicit.
    """
    directory = _verified_dir('wiktionary_ar')
    path = directory / 'kaikki.org-dictionary-Arabic.jsonl.gz'
    verbs: dict[str, str] = {}
    n_entries = 0
    n_surfaces = 0
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if '"verb"' not in line:  # cheap pre-filter before json parse
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get('pos') != 'verb':
                continue
            senses = obj.get('senses') or []
            # Skip pure inflection entries (form-of): they attest the lemma,
            # not this surface as a citation form.
            real_senses = [
                s for s in senses
                if not s.get('form_of') and 'form-of' not in (s.get('tags') or [])
            ]
            if not real_senses:
                continue

            canonicals = []
            for fm in obj.get('forms') or []:
                tags = fm.get('tags') or []
                if 'canonical' not in tags or not fm.get('form'):
                    continue
                form_tags = [t for t in tags if t.startswith('form-')]
                if form_tags and 'form-i' not in form_tags:
                    continue  # Form II–X lemma — can never be a Form I candidate
                canonicals.append(fm['form'])
            if not canonicals:
                continue
            n_entries += 1

            gloss = ''
            for s in real_senses:
                glosses = s.get('glosses') or []
                if glosses:
                    gloss = glosses[0]
                    break
            for surface in canonicals:
                n_surfaces += 1
                verbs.setdefault(normalize_surface(surface), gloss)
    print(f"  wiktionary: {n_entries:,} Form-I verb entries, {n_surfaces:,} canonical "
          f"surfaces → {len(verbs):,} distinct", file=sys.stderr)
    return verbs


def parse_freqlist() -> dict[str, int]:
    """→ dict bare word → corpus count."""
    directory = _verified_dir('freq_opensubtitles')
    freq: dict[str, int] = {}
    with open(directory / 'ar_full.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                freq[parts[0]] = int(parts[1])
    return freq


def build_gold(candidates_path: Path, out_path: Path, freq_threshold: int = 3) -> dict:
    """Write gold.csv; returns the GOLD label counts."""
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    print("Parsing arramooz…", file=sys.stderr)
    arr_pairs, arr_surfaces = parse_arramooz()
    print(f"  arramooz: {len(arr_pairs):,} (root, pattern) attestations", file=sys.stderr)
    print("Parsing wiktionary (streaming ~49MB gz)…", file=sys.stderr)
    wikt = parse_wiktionary()
    print("Parsing frequency list…", file=sys.stderr)
    freq = parse_freqlist()
    print(f"  corpus: {len(freq):,} word types", file=sys.stderr)

    # Surface → roots map decides whether a wiktionary surface hit is
    # attributable to one root ('exact') or shared ('root').
    surface_roots: dict[str, set] = {}
    for c in candidates:
        surface_roots.setdefault(normalize_surface(c['verb']), set()).add(c['root'])

    counts: dict[str, int] = {}
    level_counts = {'arramooz_exact': 0, 'arramooz_pattern': 0,
                    'wikt_exact': 0, 'wikt_root': 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.csv.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(GOLD_HEADERS)
        for c in candidates:
            root_n = normalize_root(c['root'])
            surface_n = normalize_surface(c['verb'])
            key = (root_n, c['pattern'])

            if key in arr_pairs:
                arr_level = 'exact' if surface_n in arr_surfaces.get(key, ()) else 'pattern'
                level_counts[f'arramooz_{arr_level}'] += 1
            else:
                arr_level = ''

            gloss = ''
            if surface_n in wikt:
                wikt_level = 'exact' if len(surface_roots[surface_n]) == 1 else 'root'
                gloss = wikt[surface_n]
                level_counts[f'wikt_{wikt_level}'] += 1
            else:
                wikt_level = ''

            bare = strip_tashkeel(c['verb'])
            corpus_freq = freq.get(bare, 0)

            if arr_level or wikt_level == 'exact':
                gold = 'attested_dictionary'
            elif wikt_level == 'root':
                gold = 'root_attested_form1'
            elif corpus_freq >= freq_threshold:
                gold = 'attested_corpus_only'
            else:
                gold = 'unattested'
            counts[gold] = counts.get(gold, 0) + 1

            writer.writerow([
                c['root'], c['root_type'], c['arity'], c['pattern'], c['verb'],
                arr_level, wikt_level, corpus_freq or '', gold, gloss,
            ])
    tmp_path.rename(out_path)

    n = len(candidates)
    print(f"\nGold summary ({out_path}):", file=sys.stderr)
    for label in ('attested_dictionary', 'root_attested_form1',
                  'attested_corpus_only', 'unattested'):
        print(f"  {label}: {fmt_rate(counts.get(label, 0), n)}", file=sys.stderr)
    print(f"  match levels: {level_counts}", file=sys.stderr)
    return counts
