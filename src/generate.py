#!/usr/bin/env python3
"""
Generate every mathematically possible Arabic Form I verb from all valid
trilateral and quadrilateral root combinations.

Trilateral root types (فعل):
  XYZ  – all three radicals different       (28 × 27 × 26 = 19,656)
  XYY  – 2nd and 3rd identical              (28 × 27 = 756)
  XYX  – 1st and 3rd identical              (28 × 27 = 756)
  Total: 21,168 roots × 3 vowel patterns = 63,504 forms

Quadrilateral root types:
  XYXY – reduplicative (فَعْفَعَ)            (28 × 27 = 756)
  XYZN – all four different (فَعْلَلَ)       (28 × 27 × 26 × 25 = 491,400) [deferred]
  Total implemented: 756 roots × 1 pattern = 756 forms

Grand total (phase 1): 64,260 forms
"""

import json
import sys
from pathlib import Path

from src.models import VerbCandidate

# ── 28 Arabic consonants ─────────────────────────────────────────────
CONSONANTS = [
    '\u0621',  # ء hamza
    '\u0628',  # ب
    '\u062a',  # ت
    '\u062b',  # ث
    '\u062c',  # ج
    '\u062d',  # ح
    '\u062e',  # خ
    '\u062f',  # د
    '\u0630',  # ذ
    '\u0631',  # ر
    '\u0632',  # ز
    '\u0633',  # س
    '\u0634',  # ش
    '\u0635',  # ص
    '\u0636',  # ض
    '\u0637',  # ط
    '\u0638',  # ظ
    '\u0639',  # ع
    '\u063a',  # غ
    '\u0641',  # ف
    '\u0642',  # ق
    '\u0643',  # ك
    '\u0644',  # ل
    '\u0645',  # م
    '\u0646',  # ن
    '\u0647',  # ه
    '\u0648',  # و
    '\u064a',  # ي
]

# ── Diacritics ───────────────────────────────────────────────────────
# Canonical definitions live in morphophonology; re-exported here for
# existing consumers.
from src.morphophonology import (  # noqa: E402
    DAMMA,
    FATHA,
    KASRA,
    SHADDA,
    SUKUN,
    build_form1_past_3ms,
    classify_root,
)
from src import plausibility  # noqa: E402


# ── Root generators ──────────────────────────────────────────────────

def generate_trilateral_roots():
    """Yield all valid trilateral roots as (r1, r2, r3, root_type) tuples."""
    n = len(CONSONANTS)

    # XYZ – all different
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[k], 'XYZ')

    # XYY – r2 == r3, r1 != r2
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[j], 'XYY')

    # XYX – r1 == r3, r2 != r1
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[i], 'XYX')


def generate_quadrilateral_roots_xyxy():
    """Yield all XYXY reduplicative roots as (r1, r2, r3, r4, root_type) tuples."""
    n = len(CONSONANTS)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[i], CONSONANTS[j], 'XYXY')


def generate_quadrilateral_roots_xyzn():
    """Yield all XYZN roots (all four different). Deferred — not called in phase 1."""
    n = len(CONSONANTS)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                for m in range(n):
                    if m == i or m == j or m == k:
                        continue
                    yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[k], CONSONANTS[m], 'XYZN')


# ── Form builders ────────────────────────────────────────────────────

def build_form1_trilateral(r1, r2, r3, middle_vowel=FATHA):
    """Correct Form I past-3ms surface (weak radicals, hamza seating,
    gemination handled in morphophonology)."""
    return build_form1_past_3ms(r1, r2, r3, middle_vowel)


def build_form1_trilateral_naive(r1, r2, r3, middle_vowel=FATHA):
    """The original CaCaCa template render, preserved verbatim.

    Misrenders weak/hamzated roots (قول → قَوَلَ) and gives geminates the
    middle vowel after shadda (مَدِّ). Kept only to fill verb_naive — the
    join key for data produced before the weak-radical fix.
    """
    if r2 == r3:
        return r1 + FATHA + r2 + SHADDA + middle_vowel
    else:
        return r1 + FATHA + r2 + middle_vowel + r3 + FATHA


def build_form1_quadrilateral(r1, r2, r3, r4):
    """
    Build a Form I quadrilateral past tense verb: C1aC2C3aC4a (فَعْلَلَ / فَعْفَعَ)
    """
    return r1 + FATHA + r2 + SUKUN + r3 + FATHA + r4 + FATHA


def extract_root(consonants):
    """Return the bare root consonants (no diacritics) as a string."""
    return ''.join(consonants)


# ── Pattern labels ───────────────────────────────────────────────────

TRILATERAL_VOWELS = [
    (FATHA, 'fa\u02bfala'),   # faʿala
    (KASRA, 'fa\u02bfila'),   # faʿila
    (DAMMA, 'fa\u02bfula'),   # faʿula
]

QUADRILATERAL_PATTERN_XYXY = 'fa\u02bffa\u02bfa'   # faʿfaʿa
QUADRILATERAL_PATTERN_XYZN = 'fa\u02bflala'         # faʿlala


# ── Main generation ──────────────────────────────────────────────────

def generate_all_candidates(include_xyzn=False):
    """Generate all VerbCandidate objects."""

    # Trilateral forms
    for r1, r2, r3, root_type in generate_trilateral_roots():
        root = extract_root([r1, r2, r3])
        weak_class, hamza_pos = classify_root((r1, r2, r3))
        plaus = plausibility.score_root(root, root_type)
        # surface_collapsed is per cell: True when this cell's surface is
        # shared with another vocalization of the same root (hollow and
        # geminate roots collapse fully; lexical exceptions like عَوِرَ leave
        # one cell distinct while the other two still share).
        surfaces = [build_form1_past_3ms(r1, r2, r3, mv) for mv, _ in TRILATERAL_VOWELS]
        for (mid_v, pattern_label), verb in zip(TRILATERAL_VOWELS, surfaces):
            yield VerbCandidate(
                root=root,
                root_type=root_type,
                arity=3,
                pattern=pattern_label,
                verb=verb,
                verb_naive=build_form1_trilateral_naive(r1, r2, r3, mid_v),
                weak_class=weak_class,
                hamza_positions=hamza_pos,
                surface_collapsed=surfaces.count(verb) > 1,
                plausibility=plaus,
            )

    # Quadrilateral XYXY forms (surfaces stay on the naive template — v1
    # is tag-only for quads; see morphophonology module docstring)
    for r1, r2, r3, r4, root_type in generate_quadrilateral_roots_xyxy():
        root = extract_root([r1, r2, r3, r4])
        weak_class, hamza_pos = classify_root((r1, r2, r3, r4))
        verb = build_form1_quadrilateral(r1, r2, r3, r4)
        yield VerbCandidate(
            root=root,
            root_type=root_type,
            arity=4,
            pattern=QUADRILATERAL_PATTERN_XYXY,
            verb=verb,
            verb_naive=verb,
            weak_class=weak_class,
            hamza_positions=hamza_pos,
            surface_collapsed=False,
            plausibility=plausibility.score_root(root, root_type),
        )

    # Quadrilateral XYZN forms (deferred by default)
    if include_xyzn:
        for r1, r2, r3, r4, root_type in generate_quadrilateral_roots_xyzn():
            root = extract_root([r1, r2, r3, r4])
            weak_class, hamza_pos = classify_root((r1, r2, r3, r4))
            verb = build_form1_quadrilateral(r1, r2, r3, r4)
            yield VerbCandidate(
                root=root,
                root_type=root_type,
                arity=4,
                pattern=QUADRILATERAL_PATTERN_XYZN,
                verb=verb,
                verb_naive=verb,
                weak_class=weak_class,
                hamza_positions=hamza_pos,
                surface_collapsed=False,
                plausibility=plausibility.score_root(root, root_type),
            )


def write_candidates_jsonl(output_path: Path, include_xyzn=False):
    """Write all candidates to a JSONL file."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for candidate in generate_all_candidates(include_xyzn=include_xyzn):
            f.write(candidate.model_dump_json() + '\n')
            count += 1
    return count


if __name__ == '__main__':
    output = Path('data/candidates.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = write_candidates_jsonl(output)
    print(f"Generated {count:,} candidates → {output}", file=sys.stderr)
