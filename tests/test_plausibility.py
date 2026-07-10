"""Tests for the OCP-Place plausibility scorer."""

import itertools

from src import plausibility
from src.plausibility import (
    PLACE_CLASSES,
    pair_penalty,
    score_root,
    sort_candidates_for_validation,
)


def test_place_classes_cover_alphabet_exactly():
    from src.generate import CONSONANTS
    members = list(itertools.chain(*PLACE_CLASSES.values()))
    assert len(members) == len(set(members)) == 28
    assert set(members) == set(CONSONANTS)


def test_clean_root_scores_one():
    # dorsal + coronal_stop + labial — no shared class anywhere.
    assert score_root('كتب', 'XYZ') == 1.0


def test_homorganic_root_scores_low():
    # three labials vs a clean root
    assert score_root('بفم', 'XYZ') < score_root('كتب', 'XYZ')


def test_geminate_exemption():
    # identical C2C3 adds zero penalty: مدد scores like مدب
    # (د and ب share no class with م or each other in slot 12/13).
    assert pair_penalty('د', 'د', '23') == 0.0
    assert score_root('مدد', 'XYY') == 1.0


def test_identity_penalties_ordering():
    # identical C1C3 (XYX) milder than same-class-nonidentical C1C3
    assert pair_penalty('ق', 'ق', '13') < pair_penalty('ق', 'ك', '13')
    # adjacent same-class beats non-adjacent same-class
    assert pair_penalty('ت', 'د', '12') > pair_penalty('ت', 'د', '13')


def test_glide_halving():
    assert pair_penalty('و', 'ي', '12') == 0.5 * pair_penalty('ت', 'د', '12')


def test_xyxy_reduplication_not_penalized():
    # زلزل: base pair ز/ل shares no class → 1.0; the reduplication itself
    # must not add penalties.
    assert score_root('زلزل', 'XYXY') == 1.0


def test_bounds_and_determinism():
    from src.generate import generate_trilateral_roots, extract_root
    scores = {}
    for r1, r2, r3, rt in generate_trilateral_roots():
        root = extract_root([r1, r2, r3])
        s = score_root(root, rt)
        assert 0.0 < s <= 1.0
        scores[root] = s
    # deterministic on re-run
    for root in list(scores)[:100]:
        assert score_root(root, 'XYZ') == scores[root]


def test_sort_preserves_root_grouping():
    candidates = [
        {'root': 'كتب', 'plausibility': 1.0, 'pattern': 'faʿala'},
        {'root': 'بفم', 'plausibility': 0.2, 'pattern': 'faʿala'},
        {'root': 'كتب', 'plausibility': 1.0, 'pattern': 'faʿila'},
        {'root': 'بفم', 'plausibility': 0.2, 'pattern': 'faʿila'},
        {'root': 'قلق', 'plausibility': 0.74, 'pattern': 'faʿala'},
    ]
    ordered = sort_candidates_for_validation(candidates)
    seen_roots = []
    for c in ordered:
        if not seen_roots or seen_roots[-1] != c['root']:
            seen_roots.append(c['root'])
    assert len(seen_roots) == len(set(seen_roots)), "root rows interleaved"
    assert seen_roots[0] == 'كتب' and seen_roots[-1] == 'بفم'


def test_stats_helpers():
    from src.stats import cohen_kappa, fleiss_kappa, wilson_ci
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert cohen_kappa([True, False] * 10, [True, False] * 10) == 1.0
    # Perfect disagreement → negative kappa
    assert cohen_kappa([True] * 5 + [False] * 5, [False] * 5 + [True] * 5) < 0
    # Fleiss on unanimous items = 1.0... (p_e < 1 requires mixed marginals)
    k = fleiss_kappa([(3, 0), (0, 3), (3, 0), (0, 3)])
    assert k == 1.0
