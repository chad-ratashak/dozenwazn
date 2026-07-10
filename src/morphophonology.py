"""Arabic Form I past-3ms surface morphophonology.

The generator's original naive template concatenated radicals directly
(CaCaCa), which misrenders every weak or hamzated root: قول came out as
قَوَلَ instead of قَالَ, رمي as رَمَيَ instead of رَمَى. This module produces the
correct dictionary-citation surface (past tense, 3rd person masculine
singular) and classifies each root by its traditional weakness taxonomy.

Scope: Form I past-3ms only. Assimilated roots (initial و/ي) are regular in
this cell — the wāw loss happens in the imperfect (وصل → وَصَلَ, يَصِلُ) which we
do not generate; they are still tagged for downstream analysis. Quadriliteral
surfaces are left on the naive فَعْفَعَ template (tag-only in v1): weak
reduplicative quads are near-unattested and their surface rules are not
standardized enough to encode safely.

Surface dispatch is by radical *shape* (which positions hold و/ي, whether
r2==r3), never by the weak_class label: lafīf roots must reach the defective
branch on their own geometry.
"""

# ── Diacritics (canonical home; generate.py re-exports) ─────────────────
FATHA  = 'َ'   # َ  a
KASRA  = 'ِ'   # ِ  i
DAMMA  = 'ُ'   # ُ  u
SUKUN  = 'ْ'   # ْ  no vowel
SHADDA = 'ّ'   # ّ  gemination

# ── Letters ──────────────────────────────────────────────────────────────
HAMZA        = 'ء'  # ء
ALIF         = 'ا'  # ا
ALIF_MADDA   = 'آ'  # آ
ALIF_HAMZA   = 'أ'  # أ
ALIF_HAMZA_BELOW = 'إ'  # إ
WAW_HAMZA    = 'ؤ'  # ؤ
YA_HAMZA     = 'ئ'  # ئ
ALIF_MAQSURA = 'ى'  # ى
WAW          = 'و'  # و
YA           = 'ي'  # ي

WEAK = {WAW, YA}

# Marker for "the preceding vowel is long ā" in hamza seating context.
LONG_A = 'aa'

WEAK_CLASSES = frozenset({
    'sound', 'geminate',
    'assimilated_w', 'assimilated_y',
    'hollow_w', 'hollow_y',
    'defective_w', 'defective_y',
    'lafif_maqrun',   # C2+C3 weak (روي, قوي, حيي)
    'lafif_mafruq',   # C1+C3 weak (وفي, ولي)
    'weak_12',        # C1+C2 weak (rare)
    'weak_all',       # all three weak (exactly 4 roots)
    'quad_weak',      # arity-4 root containing any weak radical
})


def classify_root(radicals: tuple[str, ...]) -> tuple[str, list[int]]:
    """(weak_class, hamza_positions) for a root.

    hamza_positions is 1-indexed; hamza (مهموز) is orthogonal to weakness
    (معتل) in the traditional taxonomy, so a root like أخذ is
    ('sound', [1]).
    """
    hamza_positions = [i + 1 for i, r in enumerate(radicals) if r == HAMZA]

    if len(radicals) == 4:
        weak_class = 'quad_weak' if any(r in WEAK for r in radicals) else 'sound'
        return weak_class, hamza_positions

    r1, r2, r3 = radicals
    weak_positions = {i + 1 for i, r in enumerate(radicals) if r in WEAK}

    # Non-weak r2==r3 is muḍāʿaf (geminate) regardless of r1: ودد surfaces
    # as وَدَّ through the geminate branch, so a weak initial must not put it
    # in the assimilated (regular-surface) class.
    if r2 == r3 and r2 not in WEAK:
        weak_class = 'geminate'
    elif not weak_positions:
        weak_class = 'sound'
    elif weak_positions == {1}:
        weak_class = 'assimilated_w' if r1 == WAW else 'assimilated_y'
    elif weak_positions == {2}:
        weak_class = 'hollow_w' if r2 == WAW else 'hollow_y'
    elif weak_positions == {3}:
        weak_class = 'defective_w' if r3 == WAW else 'defective_y'
    elif weak_positions == {2, 3}:
        weak_class = 'lafif_maqrun'
    elif weak_positions == {1, 3}:
        weak_class = 'lafif_mafruq'
    elif weak_positions == {1, 2}:
        weak_class = 'weak_12'
    else:
        weak_class = 'weak_all'

    return weak_class, hamza_positions


# Vowel strength for hamza seating: kasra > damma > fatha > long-ā/none.
_VOWEL_STRENGTH = {KASRA: 3, DAMMA: 2, FATHA: 1, LONG_A: 0, None: 0}
_SEAT_FOR_VOWEL = {KASRA: YA_HAMZA, DAMMA: WAW_HAMZA, FATHA: ALIF_HAMZA}


def seat_hamza(prev: str | None, own: str | None) -> str:
    """Orthographic carrier for a hamza radical.

    prev = vowel before the hamza (None word-initially, LONG_A after ā),
    own = the hamza's own vowel. The stronger of the two picks the seat:
    kasra → ئ, damma → ؤ, fatha → أ, neither → ء on the line. Exceptions:
    word-initial hamza ALWAYS seats on alif (أ, or إ under kasra) — never
    ئ/ؤ; and after long ā a fatha does NOT seat on alif — the hamza stays
    on the line (جَاءَ, not جَاأَ); only kasra/damma force a seat there.
    """
    if prev is None:
        return ALIF_HAMZA_BELOW if own == KASRA else ALIF_HAMZA
    if prev == LONG_A:
        return _SEAT_FOR_VOWEL[own] if own in (KASRA, DAMMA) else HAMZA
    strongest = max((prev, own), key=lambda v: _VOWEL_STRENGTH.get(v, 0))
    return _SEAT_FOR_VOWEL.get(strongest, HAMZA)


def normalize_orthography(s: str) -> str:
    """Collapse أَا into آ (madda). Idempotent."""
    return s.replace(ALIF_HAMZA + FATHA + ALIF, ALIF_MADDA)


def _radical(c: str, prev: str | None, own: str | None) -> str:
    """Render one radical, seating hamza by its vowel context."""
    if c == HAMZA:
        return seat_hamza(prev, own)
    return c


# Lexical exceptions where the contraction rules do NOT apply and the
# citation form keeps the weak/geminate radical consonantal (rendered by the
# sound branch). These are lexically conditioned — no rule derives them:
#   - strong-conjugating hollow faʿila verbs (colors/defects): عَوِرَ, حَوِلَ,
#     صَيِدَ, غَيِدَ; hollow faʿula anomaly هَيُؤَ
#   - fakk al-idghām geminate faʿula: لَبُبَ
# Key: (root string, middle vowel). Extend here as more attested exceptions
# surface; keep each entry justified by a dictionary citation.
UNCONTRACTED_EXCEPTIONS = {
    ('عور', KASRA),
    ('حول', KASRA),
    ('صيد', KASRA),
    ('غيد', KASRA),
    ('هيء', DAMMA),
    ('لبب', DAMMA),
}


def build_form1_past_3ms(r1: str, r2: str, r3: str, middle_vowel: str = FATHA) -> str:
    """Correct Form I past-3ms surface for a trilateral root.

    Hollow and geminate roots contract, so several pattern cells of one
    root can share a surface — compute that per root from the returned
    strings (as generate_all_candidates does); it is not a property of a
    single cell.
    """
    # Lexically-conditioned uncontracted citation forms (see above): the
    # weak/geminate radical stays consonantal → plain sound template.
    if (r1 + r2 + r3, middle_vowel) in UNCONTRACTED_EXCEPTIONS:
        c1 = _radical(r1, None, FATHA)
        c2 = _radical(r2, FATHA, middle_vowel)
        c3 = _radical(r3, middle_vowel, FATHA)
        return normalize_orthography(c1 + FATHA + c2 + middle_vowel + c3 + FATHA)

    # Defective: r3 ∈ {و, ي}. Covers defective proper plus lafīf roots —
    # a weak C2 (روي, قوي) stays consonantal here.
    if r3 in WEAK:
        c1 = _radical(r1, None, FATHA)
        if middle_vowel == FATHA:
            c2 = _radical(r2, FATHA, FATHA)
            if r3 == WAW:
                final = ALIF                      # دعو → دَعَا
            elif r2 == YA:
                final = ALIF                      # imlāʾ: no ى after ي (حيي → حَيَا)
            else:
                final = ALIF_MAQSURA              # رمي → رَمَى
            surface = c1 + FATHA + c2 + FATHA + final  # no diacritic on the final alif
        elif middle_vowel == KASRA:
            # Underlying w and y both surface as ي after kasra: رضو → رَضِيَ.
            c2 = _radical(r2, FATHA, KASRA)
            surface = c1 + FATHA + c2 + KASRA + YA + FATHA
        else:  # DAMMA — both surface as و: سرو → سَرُوَ
            c2 = _radical(r2, FATHA, DAMMA)
            surface = c1 + FATHA + c2 + DAMMA + WAW + FATHA
        return normalize_orthography(surface)

    # Hollow: r2 ∈ {و, ي} → long ā for every vocalization (قول → قَالَ,
    # بيع → بَاعَ, خوف/faʿila → خَافَ). The ā carries no diacritic; a final
    # hamza sits on the line after it (جيء → جَاءَ).
    if r2 in WEAK:
        c1 = _radical(r1, None, FATHA)
        c3 = _radical(r3, LONG_A, FATHA)
        surface = c1 + FATHA + ALIF + c3 + FATHA
        return normalize_orthography(surface)

    # Geminate: r2 == r3 → shadda, and the final vowel is ALWAYS fatha in
    # past-3ms (مدّ، ظلّ، مسّ) — the old template's SHADDA + middle_vowel
    # produced invalid مَدِّ/مَدُّ.
    if r2 == r3:
        c1 = _radical(r1, None, FATHA)
        c2 = _radical(r2, FATHA, FATHA)
        surface = c1 + FATHA + c2 + SHADDA + FATHA
        return normalize_orthography(surface)

    # Sound (including assimilated and hamzated roots): plain template with
    # hamza seating per position (أخذ → أَخَذَ, سئم → سَئِمَ, جرؤ → جَرُؤَ).
    c1 = _radical(r1, None, FATHA)
    c2 = _radical(r2, FATHA, middle_vowel)
    c3 = _radical(r3, middle_vowel, FATHA)
    surface = c1 + FATHA + c2 + middle_vowel + c3 + FATHA
    return normalize_orthography(surface)
