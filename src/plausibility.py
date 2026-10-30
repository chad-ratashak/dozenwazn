# SPDX-License-Identifier: GPL-3.0-or-later
"""Phonotactic plausibility scoring for Arabic root candidates (OCP-Place).

Arabic roots strongly avoid homorganic consonant pairs, gradient by
position: adjacent C1C2 shows the strongest prohibition, C2C3 strong except
for identical-pair geminates (a productive class), non-adjacent C1C3 weaker
(Greenberg 1950, "The patterning of root morphemes in Semitic", Word
6:162–181; formalized as OCP-Place on the root tier by McCarthy 1986 LI /
1994; made gradient via natural-class similarity and observed/expected
co-occurrence by Frisch, Pierrehumbert & Broe 2004, NLLT 22:179–228).

v1 is the class-based categorical version with position weights, pure
stdlib and deterministic. calibrate_from_attested() is the v2 hook: it
derives Frisch-style −log(O/E) pair weights from an attested-root
inventory (the gold lexicon), written to data/plausibility_weights.json;
score_root() prefers that file when present.

The default score takes a handful of discrete values — fine for ranking
and run prioritization, but do not present it as a calibrated probability
until v2 weights exist.
"""

import itertools
import json
import math
from pathlib import Path

# Place-of-articulation classes. Every one of the generator's 28 consonants
# appears exactly once (asserted at import). Debatable memberships (ج as
# coronal affricate, خ/غ in the guttural class, ق dorsal) follow Frisch et
# al. / McCarthy defaults; calibration makes them empirical later.
PLACE_CLASSES = {
    'labial':        ['ب', 'ف', 'م'],
    'coronal_stop':  ['ت', 'د', 'ط', 'ض'],
    'coronal_fric':  ['ث', 'ذ', 'س', 'ز', 'ص', 'ش', 'ظ', 'ج'],
    'coronal_son':   ['ل', 'ر', 'ن'],
    'dorsal':        ['ك', 'ق'],
    'guttural':      ['ء', 'ح', 'خ', 'ع', 'غ', 'ه'],
    'glide':         ['و', 'ي'],
}

_PLACE_OF = {c: place for place, members in PLACE_CLASSES.items() for c in members}
if len(_PLACE_OF) != 28:  # a plain assert would vanish under python -O
    raise RuntimeError("PLACE_CLASSES must cover the 28 consonants exactly once")

# Position weights: adjacency matters (Greenberg's gradient). '14' is the
# quadriliteral distance-3 pair — weaker than the distance-2 '13'.
_SLOT_WEIGHT = {'12': 1.0, '23': 0.8, '13': 0.5, '14': 0.25, 'adj_quad': 0.8}

_WEIGHTS_PATH = Path('data/plausibility_weights.json')
_calibrated_cache: dict | None = None
_calibrated_loaded = False


def place_of(c: str) -> str:
    return _PLACE_OF[c]  # KeyError = caller bug


def _calibrated_weights() -> dict | None:
    global _calibrated_cache, _calibrated_loaded
    if not _calibrated_loaded:
        _calibrated_loaded = True
        if _WEIGHTS_PATH.exists():
            with open(_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                _calibrated_cache = json.load(f)
    return _calibrated_cache


def pair_penalty(ci: str, cj: str, slot: str) -> float:
    """OCP penalty for one radical pair in a positional slot.

    slot ∈ {'12', '23', '13', '14', 'adj_quad'}.

    Identity and glide cases are handled BEFORE any calibrated lookup:
    geminate C2C3 identity is a productive template property, not an OCP
    fact, and calibration must never override its exemption.
    """
    if ci == cj:
        # Identical consonants: C2C3 geminates are a productive attested
        # class — zero penalty, or every XYY root would tank. C1C3 identity
        # (XYX) is rare but attested (قلق-type): mild. C1C2 identity is never
        # generated; keep a defensive strong penalty.
        if slot == '23':
            return 0.0
        if slot in ('13', '14'):
            return 0.3
        return 2.0

    same_class = place_of(ci) == place_of(cj)

    if same_class and place_of(ci) == 'glide':
        # Weak-root co-occurrence is governed by morphophonology (and is
        # already tagged), not by OCP — halve rather than double-punish.
        return _SLOT_WEIGHT[slot] * 0.5

    calibrated = _calibrated_weights()
    if calibrated is not None:
        # Calibration currently covers trilateral slots only; quad slots
        # ('14', 'adj_quad') always use categorical defaults — quad scores
        # are therefore not comparable to calibrated trilateral scores.
        key = f"{place_of(ci)}|{place_of(cj)}|{slot}"
        if key in calibrated:
            return calibrated[key]

    return _SLOT_WEIGHT[slot] if same_class else 0.0


def score_root(root: str, root_type: str) -> float:
    """Plausibility score in (0, 1]: exp(−Σ pair penalties).

    1.0 = no OCP violations; monotone decreasing in violation weight.
    """
    radicals = list(root)
    if len(radicals) == 3:
        total = (
            pair_penalty(radicals[0], radicals[1], '12')
            + pair_penalty(radicals[1], radicals[2], '23')
            + pair_penalty(radicals[0], radicals[2], '13')
        )
    elif root_type == 'XYXY':
        # Reduplication is itself a licit template: the cross-boundary and
        # identity pairs are template-inherent, so only the base C1C2 pair
        # is subject to OCP.
        total = pair_penalty(radicals[0], radicals[1], '12')
    else:  # XYZN — distance-3 pair gets its own weaker slot ('14')
        total = (
            pair_penalty(radicals[0], radicals[1], '12')
            + pair_penalty(radicals[1], radicals[2], 'adj_quad')
            + pair_penalty(radicals[2], radicals[3], 'adj_quad')
            + pair_penalty(radicals[0], radicals[2], '13')
            + pair_penalty(radicals[1], radicals[3], '13')
            + pair_penalty(radicals[0], radicals[3], '14')
        )
    return round(math.exp(-total), 4)


def classify_bin(score: float) -> str:
    """Coarse bin for display/legend use only — never for statistics."""
    if score >= 0.8:
        return 'high'
    if score >= 0.45:
        return 'medium'
    return 'low'


def sort_candidates_for_validation(candidates: list[dict]) -> list[dict]:
    """Order candidates so the most plausible roots are validated first.

    Score is root-level (every pattern row of a root shares it) and the
    tie-break is the root string, so whole-root grouping — which
    build_batches depends on — is preserved.
    """
    return sorted(
        candidates,
        key=lambda c: (-(c.get('plausibility') or 0.0), c['root']),
    )


def calibrate_from_attested(roots, out_path: Path = _WEIGHTS_PATH,
                            alpha: float = 0.5, min_expected: float = 5.0) -> int:
    """v2 hook: derive −log(O/E) pair weights from an attested-root list.

    Input must be bare consonant strings (hamza carriers normalized to ء
    by the caller). Writes {class_i|class_j|slot: weight}. Returns the
    number of attested roots consumed.

    Method notes (Frisch et al. 2004): identical-consonant pairs are
    EXCLUDED from the counts — geminate identity is a template property and
    would bias same-class co-occurrence downward. Joint counts get add-α
    smoothing; class pairs whose expected count is below min_expected are
    NOT emitted, so pair_penalty falls back to the categorical default
    instead of trusting noise. Weights are clamped at 0 — over-represented
    pairs earn no bonus, keeping score_root within (0, 1].
    """
    slot_pairs = {'12': [], '23': [], '13': []}
    n = 0
    for root in roots:
        if len(root) != 3:
            continue
        n += 1
        pos = {'12': (root[0], root[1]), '23': (root[1], root[2]), '13': (root[0], root[2])}
        for slot, (a, b) in pos.items():
            if a == b:
                continue  # identity pairs are template facts, not OCP evidence
            slot_pairs[slot].append((place_of(a), place_of(b)))

    if n == 0:
        raise ValueError("no trilateral roots to calibrate from")

    class_names = list(PLACE_CLASSES)
    n_cells = len(class_names) ** 2
    weights = {}
    for slot, pairs in slot_pairs.items():
        total = len(pairs) + alpha * n_cells
        marg_i: dict[str, int] = {}
        marg_j: dict[str, int] = {}
        joint: dict[tuple[str, str], int] = {}
        for a, b in pairs:
            marg_i[a] = marg_i.get(a, 0) + 1
            marg_j[b] = marg_j.get(b, 0) + 1
            joint[(a, b)] = joint.get((a, b), 0) + 1
        for a, b in itertools.product(class_names, repeat=2):
            expected = marg_i.get(a, 0) * marg_j.get(b, 0) / total if total else 0.0
            if expected < min_expected:
                continue  # not enough evidence — categorical default applies
            oe = (joint.get((a, b), 0) + alpha) / expected
            weights[f"{a}|{b}|{slot}"] = round(max(0.0, -math.log(oe)), 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(weights, f, ensure_ascii=False, indent=1, sort_keys=True)

    if out_path == _WEIGHTS_PATH:
        # Invalidate the module-level cache so scores computed after
        # calibration in the same process pick the new weights up.
        global _calibrated_cache, _calibrated_loaded
        _calibrated_cache = None
        _calibrated_loaded = False
    return n
