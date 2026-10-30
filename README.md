# Dozenwazn (أوزان)

An exhaustive map of the Arabic Form I verb space along two axes: which
phonotactically possible forms are **real words**, and which are **single
tokens** in LLM tokenizers.

The generator enumerates all 64,260 candidates (21,168 trilateral roots ×
3 vocalizations + 756 XYXY quadriliterals) with correct weak-radical
morphophonology (hollow قول → قَالَ, defective رمي → رَمَى, hamza seating,
gemination), tags each root's weakness class and OCP-Place phonotactic
plausibility, measures single-token status across 7 tokenizers, and grounds
"real word" in machine-readable dictionaries before any LLM opinion.

## Headline results (2026-07)

- **29.2%** of all candidates are single tokens in at least one tokenizer
  (18,785/64,260, 95% CI 28.9–29.6%) — but of **validated-real** verbs it is
  **72.6%** (6,255/8,611, 95% CI 71.7–73.6%).
- Correcting weak-radical surfaces raised any-tokenizer coverage from 24.0%
  to 29.2%: hollow verbs are 72.6% single-token, geminates 79.6%, vs 20.9%
  for sound roots. Naive-template numbers undercounted real-verb coverage.
- **7,591 candidates (11.8%)** are dictionary-attested (Arramooz + Wiktionary,
  per-vocalization); 66.9% are unattested — and 4,000 of those unattested
  forms are nonetheless single tokens somewhere: the **neologism map**
  (`data/analysis/neologisms.jsonl`, ranked by phonotactic plausibility).
- Full tokenizer table with CIs and fertility: see
  [arabic_root_combinatorics.md](arabic_root_combinatorics.md).

## Pipeline

```
generate → csvs → vendor / lexica → tokenize → gold → spotcheck → validate → analyze → display
```

| Command (`uv run python -m src.cli …`) | What it does |
|---|---|
| `generate` | All candidates → `data/candidates.jsonl` (`--include-xyzn` for +491K quads) |
| `csvs` | Per-model validation CSV scaffolds; **never destroys filled rows** (`--merge` migrates, `--force` asks) |
| `vendor` | Fetch tokenizer vocab files → `vendor/tokenizers/` (hash-manifested) |
| `lexica` | Fetch dictionary/corpus data → `vendor/lexica/` (hash-manifested) |
| `tokenize` | Token counts per candidate × tokenizer → `data/tokenizer_analysis.jsonl` |
| `gold` | Dictionary/corpus ground truth → `data/gold.csv` |
| `spotcheck` | LLM accuracy gate (GO/NO-GO) before long validation runs |
| `bench` | Batch-size × parallelism throughput grid |
| `validate` | LLM validation via Ollama (resumable, backed-up, gated) |
| `import-xlsx` | Recover validation data from a spreadsheet (dry-run by default) |
| `analyze agreement/coverage/neologisms/tokenizers` | Research outputs → `data/analysis/` |
| `display` / `html` | Terminal stats + interactive canvas → `data/dozenwazn.html` |

Quick start: `uv sync`, then `/launch` (Claude Code) or double-click
`Launch Dozenwazn.command`.

## Security model

Tokenizers and lexica are studied as **inert data files only** — no
`transformers`, no `trust_remote_code`, no network at analysis time. The only
modules allowed to fetch are `vendor_tokenizers.py`/`vendor_lexica.py`
(extension allowlists, commit-pinned where upstream allows); every file's
SHA-256 lives in a committed manifest and is verified before every load.
`vendor/_fallback/arramooz/` is the one committed data copy — a last resort
used only when the fetch fails, and only after it verifies against that same
manifest.

Details and the tokenizer data scope: [CLAUDE.md](CLAUDE.md).

## LLM validation status

No local model currently passes the spot-check gate (balanced accuracy ≥ 0.80
∧ specificity ≥ 0.70 ∧ missing ≤ 5%, on 50 dictionary-real + 50 fake verbs):

| Model | Bal. acc. | Sensitivity | Specificity | ETA (full run) |
|---|---|---|---|---|
| gemma4:e2b | 74.0% | 54% | 94% | ≈ 42 h |
| gemma3:4b | 65.9% | 40% | 92% | ≈ 59 h |
| llama3.2:3b | 59.0% | 62% | 56% | ≈ 75 h |

The everything-is-true failure mode of the April 2026 run is gone (schema
outputs + rewritten prompt); the remaining gap is conservatism on rare-real
verbs. `run_gemma.sh`/`run_llama.sh` refuse to start without a PASS marker
(`DOZENWAZN_SKIP_GATE=1` to override deliberately). JAIS is tokenizer-only:
no official GGUF exists and the transformers path is ruled out by the security
model — revisit only if a first-party GGUF appears.

### Long-run runbook

1. `uv run python -m src.cli spotcheck -m <model>` → needs GO.
2. `uv run python -m src.cli bench -m <model>` → pick `--batch-size`/`--parallel`.
3. If parallel > 1: restart the server with `OLLAMA_NUM_PARALLEL=<n> ollama serve`.
4. `./run_gemma.sh` (or `Run Gemma.command`). Ctrl+C pauses; re-run resumes
   (progress is keyed by (ROOT, PATTERN); snapshots land in `data/backups/`
   every 50 batches).
5. After ≥1 model CSV fills: `analyze agreement` / `analyze coverage`.

## Data files

| File | Contents |
|---|---|
| `data/candidates.jsonl` | 64,260 candidates with weak-class + plausibility tags |
| `data/tokenizer_analysis.jsonl` | Token counts (bare + diacriticized) × 7 tokenizers |
| `data/gold.csv` | Dictionary/corpus ground truth per candidate |
| `data/{model}.csv` × 8 | Per-model LLM validation (haiku: 183 rows recovered) |
| `data/analysis/` | agreement / coverage / ghosts / neologisms / tokenizers artifacts |
| `data/backups/` | Automatic pre-merge/pre-force/run snapshots |
| `data/dozenwazn.html` | Self-contained interactive map (8.7 MB) |

`data/` is gitignored (regenerable); the two `SHA256SUMS.json` manifests are
committed.

## License

Copyright (C) 2026 Chad Ratashak

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

`SPDX-License-Identifier: GPL-3.0-or-later`

### Third-party data

Lexicon and tokenizer data is licensed separately by its own authors and is
fetched, not committed — with one deliberate exception, the Arramooz fallback
copy. Sources, licenses, and attribution: [NOTICE-DATA.md](NOTICE-DATA.md).

- **Arramooz AlWaseet** (Taha Zerrouki) — GPL-2.0-or-later. Verb tables used
  for (root, pattern) attestation. A byte-identical fallback copy is committed
  at `vendor/_fallback/arramooz/`.
- **Wiktionary** Arabic extract via [kaikki.org](https://kaikki.org)
  (wiktextract) — CC BY-SA / GFDL. Fetch-only; not redistributed.
- **FrequencyWords** (OpenSubtitles 2018, hermitdave) — MIT. Fetch-only.

The repository license covers this project's code and documentation. It does
**not** settle the license of the generated dataset — `data/gold.csv` and
`data/dozenwazn.html` carry verbatim Wiktionary glosses, which is a separate
decision to make before publishing either. Scope and options:
[docs/gold-gloss-scope.md](docs/gold-gloss-scope.md).
