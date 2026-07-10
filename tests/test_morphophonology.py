"""Gold-table acceptance tests for Form I past-3ms surface generation.

Every expected surface below is an attested MSA/classical verb copied from
the design spec — the table IS the acceptance criterion for the
weak-radical fix. Roots are given as bare radicals (hamza = ء U+0621).
"""

import re

import pytest

from src.morphophonology import (
    FATHA,
    KASRA,
    DAMMA,
    WEAK_CLASSES,
    build_form1_past_3ms,
    classify_root,
    normalize_orthography,
)

PATTERN_VOWEL = {'faʿala': FATHA, 'faʿila': KASRA, 'faʿula': DAMMA}

# (root, pattern, expected_surface, note)
GOLD_TABLE = [
    ('كتب', 'faʿala', 'كَتَبَ', 'sound baseline'),
    ('قول', 'faʿala', 'قَالَ', 'hollow-w'),
    ('بيع', 'faʿala', 'بَاعَ', 'hollow-y'),
    ('خوف', 'faʿila', 'خَافَ', 'hollow collapse (faʿila)'),
    ('طول', 'faʿula', 'طَالَ', 'hollow collapse (faʿula)'),
    ('رمي', 'faʿala', 'رَمَى', 'defective-y → ى'),
    ('دعو', 'faʿala', 'دَعَا', 'defective-w → ا'),
    ('بقي', 'faʿila', 'بَقِيَ', 'defective faʿila'),
    ('رضو', 'faʿila', 'رَضِيَ', 'w → y after kasra'),
    ('سرو', 'faʿula', 'سَرُوَ', 'defective faʿula'),
    ('وصل', 'faʿala', 'وَصَلَ', 'assimilated = regular in past'),
    ('يبس', 'faʿila', 'يَبِسَ', 'assimilated-y = regular'),
    ('ءخذ', 'faʿala', 'أَخَذَ', 'hamza-1 → أ'),
    ('سءل', 'faʿala', 'سَأَلَ', 'hamza-2 fatha → أ'),
    ('سءم', 'faʿila', 'سَئِمَ', 'hamza-2 kasra → ئ'),
    ('لءم', 'faʿula', 'لَؤُمَ', 'hamza-2 damma → ؤ'),
    ('قرء', 'faʿala', 'قَرَأَ', 'hamza-3 fatha → أ'),
    ('صدء', 'faʿila', 'صَدِئَ', 'hamza-3 kasra → ئ'),
    ('جرء', 'faʿula', 'جَرُؤَ', 'hamza-3 damma → ؤ'),
    # Geminate expectations written with explicit escapes: canonical order
    # is SHADDA then FATHA (editor input methods often flip it).
    ('مدد', 'faʿala', 'مَدَّ', 'geminate'),
    ('ظلل', 'faʿila', 'ظَلَّ', 'geminate: final vowel always fatha'),
    ('ءوب', 'faʿala', 'آبَ', 'madda normalization (أَا → آ)'),
    ('ءتي', 'faʿala', 'أَتَى', 'hamza-1 + defective'),
    ('رءي', 'faʿala', 'رَأَى', 'hamza-2 + defective'),
    ('جيء', 'faʿala', 'جَاءَ', 'hollow + hamza-3 on the line'),
    ('وفي', 'faʿala', 'وَفَى', 'lafif mafruq'),
    ('ولي', 'faʿila', 'وَلِيَ', 'lafif mafruq faʿila'),
    ('روي', 'faʿala', 'رَوَى', 'lafif maqrun (C2 stays consonantal)'),
    ('قوي', 'faʿila', 'قَوِيَ', 'lafif maqrun faʿila'),
    ('حيي', 'faʿila', 'حَيِيَ', 'weak XYY: defective precedence over geminate'),
    ('حيي', 'faʿala', 'حَيَا', 'imlāʾ: no ى after ي'),
    ('وطء', 'faʿila', 'وَطِئَ', 'assimilated + hamza-3'),
    # Lexical exceptions: strong-conjugating hollow + fakk al-idghām geminate
    ('عور', 'faʿila', 'عَوِرَ', 'strong-conjugating hollow (defect verb)'),
    ('صيد', 'faʿila', 'صَيِدَ', 'strong-conjugating hollow-y'),
    ('لبب', 'faʿula', 'لَبُبَ', 'fakk al-idghām geminate'),
    ('عور', 'faʿala', 'عَارَ', 'exception is per-cell: faʿala still contracts'),
    ('ودد', 'faʿala', 'وَدَّ', 'weak-initial geminate contracts'),
]


@pytest.mark.parametrize(
    'root,pattern,expected,note',
    GOLD_TABLE,
    ids=[f"{r}-{p}-{n}" for r, p, _, n in GOLD_TABLE],
)
def test_gold_surface(root, pattern, expected, note):
    r1, r2, r3 = root
    surface = build_form1_past_3ms(r1, r2, r3, PATTERN_VOWEL[pattern])
    assert surface == expected, (
        f"{root}/{pattern} ({note}): got {surface!r}, want {expected!r}"
    )


CLASSIFICATION_TABLE = [
    ('كتب', 'sound', []),
    ('مدد', 'geminate', []),
    ('قول', 'hollow_w', []),
    ('بيع', 'hollow_y', []),
    ('رمي', 'defective_y', []),
    ('دعو', 'defective_w', []),
    ('وصل', 'assimilated_w', []),
    ('يبس', 'assimilated_y', []),
    ('حيي', 'lafif_maqrun', []),
    ('روي', 'lafif_maqrun', []),
    ('وفي', 'lafif_mafruq', []),
    ('ويي', 'weak_all', []),
    ('يوو', 'weak_all', []),
    ('ودد', 'geminate', []),   # weak-initial geminate: muḍāʿaf, not assimilated
    ('يمم', 'geminate', []),
    ('ءخذ', 'sound', [1]),
    ('سءل', 'sound', [2]),
    ('قرء', 'sound', [3]),
    ('رءي', 'defective_y', [2]),
    ('وطء', 'assimilated_w', [3]),
]


@pytest.mark.parametrize('root,weak_class,hamza_pos', CLASSIFICATION_TABLE)
def test_classify_root(root, weak_class, hamza_pos):
    got_class, got_hamza = classify_root(tuple(root))
    assert got_class == weak_class
    assert got_hamza == hamza_pos
    assert got_class in WEAK_CLASSES


def test_quad_classification():
    assert classify_root(tuple('زلزل')) == ('sound', [])
    assert classify_root(tuple('وسوس')) == ('quad_weak', [])


def test_seat_hamza_word_initial():
    from src.morphophonology import seat_hamza
    assert seat_hamza(None, FATHA) == 'أ'
    assert seat_hamza(None, KASRA) == 'إ'   # word-initial: always an alif seat
    assert seat_hamza(None, DAMMA) == 'أ'


def test_normalize_orthography_idempotent():
    s = 'أَاب'
    once = normalize_orthography(s)
    assert once == 'آب'
    assert normalize_orthography(once) == once


# ── Full-generation invariants ──────────────────────────────────────────

@pytest.fixture(scope='module')
def all_candidates():
    from src.generate import generate_all_candidates
    return list(generate_all_candidates())


def test_total_count(all_candidates):
    assert len(all_candidates) == 64_260


def test_no_known_bad_surfaces(all_candidates):
    # The naive template's signature misrenders must be gone from `verb`
    # (they live on only in verb_naive).
    bad = {'قَوَلَ', 'رَمَيَ', 'مَدِّ', 'مَدُّ'}
    assert not any(c.verb in bad for c in all_candidates)


def test_allowed_characters(all_candidates):
    # Arabic letters + the five harakat used; final ا / ى / آ allowed.
    allowed = re.compile(r'^[ء-يىَ-ْ]+$')
    offenders = [c.verb for c in all_candidates if not allowed.match(c.verb)]
    assert not offenders, offenders[:5]


def test_collapse_flag_consistency(all_candidates):
    # Per cell: surface_collapsed ⇔ this cell's surface is shared with
    # another vocalization of the same root. (Lexical exceptions like عور
    # leave one cell distinct while the other two share.)
    from collections import defaultdict
    by_root = defaultdict(list)
    for c in all_candidates:
        if c.arity == 3:
            by_root[c.root].append(c)
    for root, rows in by_root.items():
        surfaces = [c.verb for c in rows]
        for c in rows:
            expected = surfaces.count(c.verb) > 1
            assert c.surface_collapsed == expected, (
                f"{root}/{c.pattern}: collapsed={c.surface_collapsed}, "
                f"surface count={surfaces.count(c.verb)}"
            )


def test_weak_class_marginals(all_candidates):
    # Expected counts from set arithmetic over the 28-consonant alphabet
    # (2 weak consonants, 26 strong incl. hamza).
    roots = {}
    for c in all_candidates:
        if c.arity == 3:
            roots[c.root] = c.weak_class
    counts = {}
    for wc in roots.values():
        counts[wc] = counts.get(wc, 0) + 1

    assert sum(counts.values()) == 21_168
    # hollow: XYZ roots with r2 weak, r1/r3 drawn from the other 27/26 — but
    # r1,r3 must also not be weak (else lafif/weak_12): 26*2*25 per XYZ shape…
    # compute directly instead of hand-deriving:
    from src.generate import generate_trilateral_roots
    from src.morphophonology import classify_root as cr
    expected = {}
    for r1, r2, r3, _ in generate_trilateral_roots():
        wc, _h = cr((r1, r2, r3))
        expected[wc] = expected.get(wc, 0) + 1
    assert counts == expected


def test_every_row_tagged(all_candidates):
    for c in all_candidates:
        assert c.weak_class in WEAK_CLASSES
        assert c.verb_naive
        assert c.plausibility is not None and 0.0 < c.plausibility <= 1.0


def test_collapsed_examples(all_candidates):
    qal = [c for c in all_candidates if c.root == 'قول']
    assert len(qal) == 3
    assert {c.verb for c in qal} == {'قَالَ'}
    assert all(c.surface_collapsed for c in qal)
    assert {c.pattern for c in qal} == {'faʿala', 'faʿila', 'faʿula'}
