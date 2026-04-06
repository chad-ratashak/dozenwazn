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

Single-token recognition rates for bare (no tashkeel) verb forms:

| Tokenizer | Single-token hits | % of 64,260 |
|-----------|-------------------|-------------|
| JAIS-13B (Arabic-optimized) | 5,587 | 8.7% |
| o200k_base (GPT-4o/o1) | 2,209 | 3.4% |
| cl100k_base (GPT-4) | 30 | 0.0% |
| **Any tokenizer** | **5,725** | **8.9%** |

**Key finding:** Zero diacriticized forms are single tokens in any tokenizer. Arabic tashkeel is never part of BPE vocabularies — each diacritical mark becomes its own token.
