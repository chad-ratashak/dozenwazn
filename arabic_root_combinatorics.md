# Arabic Root Combinatorics & Form I Generator

## Trilateral Roots (ثلاثي)

Arabic trilateral roots allow three structural types (but not XXY or XXX):

| Type | Description | Count |
|------|-------------|-------|
| XYZ | All different | 28 × 27 × 26 = 19,656 |
| XYY | 2nd & 3rd identical | 28 × 27 = 756 |
| XYX | 1st & 3rd identical | 28 × 27 = 756 |
| **Total** | | **21,168** |

## Quadrilateral Roots (رباعي)

| Type | Pattern | Description | Count |
|------|---------|-------------|-------|
| XYZN | فَعْلَلَ (faʿlala) | All four different | 28 × 27 × 26 × 25 = 491,400 |
| XYXY | فَعْفَعَ (faʿfaʿa) | Reduplicative | 28 × 27 = 756 |
| **Total** | | | **492,156** |

## Phase 1 Generation (implemented)

The `src/generate.py` script generates:

- **Trilateral Form I** — 21,168 roots × 3 vowel patterns = **63,504 forms**
  - فَعَلَ (faʿala) — fatha
  - فَعِلَ (faʿila) — kasra
  - فَعُلَ (faʿula) — damma
- **Quadrilateral XYXY** — 756 roots × 1 pattern = **756 forms**
  - فَعْفَعَ (faʿfaʿa)

Geminate roots (XYY) are correctly rendered with shadda (e.g. رَدَّ not رَدَدَ).

**Phase 1 total: 64,260 forms**

XYZN quadrilateral roots (491,400) are deferred — available via `--include-xyzn` flag.

---

## Derived Forms (documentation only — not generated)

### Trilateral Derived Forms (II–X)

| Form | Arabic | Pattern | Vowel variants | Example |
|------|--------|---------|---------------|---------|
| I | فَعَلَ | faʿala | 3 (a/i/u) | كَتَبَ (kataba) |
| II | فَعَّلَ | faʿʿala | 1 | عَلَّمَ (ʿallama) |
| III | فَاعَلَ | fāʿala | 1 | قَاتَلَ (qātala) |
| IV | أَفْعَلَ | ʾafʿala | 1 | أَرْسَلَ (ʾarsala) |
| V | تَفَعَّلَ | tafaʿʿala | 1 | تَعَلَّمَ (taʿallama) |
| VI | تَفَاعَلَ | tafāʿala | 1 | تَقَاتَلَ (taqātala) |
| VII | اِنْفَعَلَ | infaʿala | 1 | اِنْكَسَرَ (inkasara) |
| VIII | اِفْتَعَلَ | iftaʿala | 1 | اِجْتَمَعَ (ijtamaʿa) |
| IX | اِفْعَلَّ | ifʿalla | 1 | اِحْمَرَّ (iḥmarra) |
| X | اِسْتَفْعَلَ | istafʿala | 1 | اِسْتَخْرَجَ (istakhraja) |

**Forms per root: 12** (3 for Form I + 1 each for II–X)

**Theoretical max: 21,168 × 12 = 254,016 trilateral forms**

### Quadrilateral Derived Forms (I–IV)

| Form | Arabic | Pattern | Example |
|------|--------|---------|---------|
| I | فَعْلَلَ | faʿlala | دَحْرَجَ (daḥraja) |
| II | تَفَعْلَلَ | tafaʿlala | تَدَحْرَجَ (tadaḥraja) |
| III | اِفْعَلَلَّ | ifʿalalla | اِطْمَأَنَّ (iṭmaʾanna) |
| IV | اِفْعَنْلَلَ | ifʿanlala | اِحْرَنْجَمَ (iḥranjama) |

**Forms per root: 4**

**Theoretical max: 492,156 × 4 = 1,968,624 quadrilateral forms**

### Grand Totals

| Category | Roots | Forms/root | Total forms |
|----------|-------|------------|-------------|
| Trilateral (all forms) | 21,168 | 12 | 254,016 |
| Quadrilateral (all forms) | 492,156 | 4 | 1,968,624 |
| **Grand theoretical total** | **513,324** | | **2,222,640** |

---

## Tokenizer Analysis

Single-token recognition rates for bare (no tashkeel) verb forms, measured
on the **corrected surface forms** (weak-radical morphophonology applied —
hollow قول is measured as قال, not the naive قَوَلَ). Regenerate this table with
`uv run python -m src.cli analyze tokenizers --markdown`.

| Tokenizer | Single-token (bare) | % of n [95% CI] | Fertility | Notes |
|---|---|---|---|---|
| arabert | 16,836/64,260 | 26.20% (95% CI 25.86%–26.54%) | 1.84 | normalizer strips tashkeel (diacriticized hits equal bare hits) |
| aya101 | 1,772/64,260 | 2.76% (95% CI 2.63%–2.89%) | 2.47 | mT5 SentencePiece (Aya-101 uses it unmodified) |
| camelbert | 8,450/64,260 | 13.15% (95% CI 12.89%–13.41%) | 2.04 | WordPiece; lone-[UNK] counted as null, never a hit |
| cl100k_base | 24/64,260 | 0.04% (95% CI 0.03%–0.06%) | 2.98 | GPT-4 |
| jais | 6,116/64,260 | 9.52% (95% CI 9.29%–9.75%) | 2.08 | Arabic-optimized |
| o200k_base | 2,740/64,260 | 4.26% (95% CI 4.11%–4.42%) | 2.29 | GPT-4o/o1 |
| qwen2.5 | 2,969/64,260 | 4.62% (95% CI 4.46%–4.79%) | 2.25 | vocabulary only; see README security model |
| **any tokenizer** | **18,785/64,260** | **29.23% (95% CI 28.88%–29.59%)** | — | — |

Fertility = mean tokens per verb on bare forms (lower is better coverage).

**Weak-radical effect:** switching from the naive CaCaCa template to correct
surfaces raised any-tokenizer coverage from 24.0% to 29.2% — hollow verbs are
72.6% single-token and geminates 79.6% (their contracted bare surfaces like
قال/مد are common short words), versus 20.9% for sound roots.

**Validated-real coverage (the headline):** of consensus-real verbs
(dictionary gold + LLM votes), **72.6% are single tokens in at least one
tokenizer** (6,255/8,611, 95% CI 71.7%–73.6%) — versus 29.2% over all
candidates. Regenerate with `analyze coverage`.

**Diacriticized forms:** the earlier "zero diacriticized single-tokens"
finding holds **only for tokenizers whose normalizer preserves tashkeel**
(0/64,260 across aya101, camelbert, cl100k_base, jais, o200k_base, qwen2.5).
AraBERT's built-in normalizer strips diacritics, so its 16,836 "diacriticized"
hits are bare-form hits in disguise and say nothing about tashkeel-bearing
text.
