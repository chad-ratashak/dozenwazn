#!/usr/bin/env python3
"""
Generate every mathematically possible Arabic Form I (فعل) verb
from all valid trilateral root combinations.

Root types allowed:
  XYZ  – all three radicals different
  XYY  – 2nd and 3rd identical, 1st different
  XYX  – 1st and 3rd identical, 2nd different
  (XXY and XXX are excluded)

Form I has three voweling patterns for the past/present:
  faʿala / yafʿulu  (فَعَلَ)  – a-a
  faʿala / yafʿilu  (فَعَلَ)  – a-a (same past, different present)
  faʿila / yafʿalu  (فَعِلَ)  – i-a
  faʿula / yafʿulu  (فَعُلَ)  – u-u

This script generates the three distinct past-tense shapes:
  فَعَلَ  (fa3ala)
  فَعِلَ  (fa3ila)
  فَعُلَ  (fa3ula)

Usage:
  python arabic_form1.py > output.txt
"""

import sys

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
FATHA  = '\u064e'   # َ  a
KASRA  = '\u0650'   # ِ  i
DAMMA  = '\u064f'   # ُ  u
SHADDA = '\u0651'   # ّ  gemination

def build_form1(r1, r2, r3, middle_vowel=FATHA):
    """
    Build a Form I past tense verb: C1aC2vC3a
    where v is the middle vowel (fatha, kasra, or damma).

    If r2 == r3 (XYY root), the final radical gets a shadda
    and the two identical letters are written once: C1aC2ّa
    """
    if r2 == r3:
        # Geminate root (أجوف مضعّف): e.g. ردّ = ر + دّ
        return r1 + FATHA + r2 + SHADDA + middle_vowel
    else:
        return r1 + FATHA + r2 + middle_vowel + r3 + FATHA


def generate_roots():
    """
    Yield all valid trilateral roots as (r1, r2, r3) tuples.
    Types: XYZ, XYY, XYX  (excludes XXY, XXX)
    """
    n = len(CONSONANTS)

    # XYZ – all different
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[k])

    # XYY – r2 == r3, r1 != r2
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[j])

    # XYX – r1 == r3, r2 != r1
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            yield (CONSONANTS[i], CONSONANTS[j], CONSONANTS[i])


def main():
    vowels = [
        (FATHA, 'faʿala'),
        (KASRA, 'faʿila'),
        (DAMMA, 'faʿula'),
    ]

    count = 0
    for r1, r2, r3 in generate_roots():
        for mid_v, _label in vowels:
            word = build_form1(r1, r2, r3, mid_v)
            sys.stdout.write(word + '\n')
            count += 1

    print(f"\n# Total forms generated: {count}", file=sys.stderr)
    # Expected: 21,168 roots × 3 vowel patterns = 63,504


if __name__ == '__main__':
    main()
