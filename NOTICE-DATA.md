# Third-Party Data Notices

Dozenwazn itself is licensed under GPL-3.0-or-later (see `LICENSE`). This file
covers the third-party lexicon and tokenizer data the project depends on,
which is licensed separately by its own authors.

## What this repository does and does not redistribute

Lexicon and tokenizer files are fetched on demand by `src/vendor_lexica.py`
and `src/vendor_tokenizers.py`, pinned by commit where upstream allows, and
verified against the committed SHA-256 manifests before every load. Those
manifests (URLs plus hashes) are the only part of the arrangement normally
committed.

One exception: **`vendor/_fallback/arramooz/` is committed**, and is a
redistribution. The pinned Arramooz URLs resolve through a third-party host,
and if that repository is deleted or force-pushed the project stops building.
The fallback is used only when the network fetch fails, and only after its
contents are verified against the committed manifest. Arramooz's licence
permits this; the attribution below is what that permission requires.

No Wiktionary/kaikki data is redistributed, and none may be added to
`vendor/_fallback/` — see that section for why.

---

## Arramooz AlWaseet

Arabic dictionary for morphological analysis.

Copyright (C) Taha Zerrouki <taha.zerrouki@gmail.com>. Data collected by
Mohamed Kebdani.

- Source: <https://github.com/linuxscout/arramooz>
- Pinned at commit `b055c5dea51b8a9152e7e66a7ca3b6f713fb187d` (2023-02-09)
- Files used: `data/verbs/verb_triliteral.csv`,
  `data/verbs/verb_dic_data-net.csv`
- Redistributed in this repository at `vendor/_fallback/arramooz/`, together
  with the upstream `LICENSE`

### Licence: GPL-2.0-or-later

The upstream repository's `LICENSE` is the bare GPL-2 text and does not by
itself state whether the grant is version-2-only or or-later — its "any later
version" wording sits inside the *How to Apply These Terms* appendix, which is
a template for licensors rather than a grant. GitHub's API reports the
deprecated ambiguous identifier `GPL-2.0`, which does not distinguish the two
either.

The grant is in the file headers. Fifteen source files in the upstream
repository at the pinned commit carry the explicit or-later paragraph,
including the two tools that generate the CSVs used here —
`scripts/verbs/csvdict.py` and `scripts/verbs/align_triverbs.py`:

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

No file in the upstream repository states "version 2 only". Under the
or-later grant, this project takes the material under GPL-3.0-or-later, which
is what makes the fallback copy above redistributable alongside GPL-3 code.

### Modifications

The redistributed files in `vendor/_fallback/arramooz/` are **unmodified** and
byte-identical to the upstream files at the pinned commit; their SHA-256 hashes
are recorded in `vendor/lexica/SHA256SUMS.json`.

Dozenwazn derives a (root, past-vowel-pattern) attestation flag from them by
reading the bāb-of-conjugation column (`src/gold.py`). No Arramooz content is
copied into this project's own source files.

---

## English Wiktionary, via kaikki.org (wiktextract)

- Source: <https://kaikki.org/dictionary/Arabic/> — machine-readable extract of
  English Wiktionary produced by wiktextract (Tatu Ylonen)
- File used: `kaikki.org-dictionary-Arabic.jsonl.gz`
- Attribution is owed to the Wiktionary contributors
- Licence: **CC BY-SA** (per the Wikimedia Terms of Use), additionally
  **GFDL 1.3**

Rebuilt upstream continuously and therefore not commit-pinnable: the fetch
date and SHA-256 are recorded on first use, and a re-fetch that changes the
hash warns loudly.

**Not redistributed, and must not be added to `vendor/_fallback/.`** Committing
CC BY-SA content into this GPL-3 tree would mix two copyleft regimes that do
not compose cleanly in that direction. Keeping it fetch-only is what keeps the
repository's licensing simple.

### Modifications

Form I canonical verb surfaces and their first gloss are extracted and
normalized (`src/gold.py`). Glosses appear verbatim in the generated
`data/gold.csv` and in `data/dozenwazn.html`. Neither is committed — `data/` is
gitignored — but both are publication artifacts: neither can be published
as-is without dropping the glosses or taking on CC BY-SA terms for them.

---

## FrequencyWords — OpenSubtitles 2018, Arabic

Copyright (c) 2016 Hermit Dave. MIT License.

- Source: <https://github.com/hermitdave/FrequencyWords>
- Pinned at commit `525f9b560de45753a5ea01069454e72e9aa541c6`
- File used: `content/2018/ar/ar_full.txt`; the upstream `LICENSE` is fetched
  alongside it
- Not redistributed. MIT would permit it; there is simply no need.

### Modifications

Parsed into a word-to-count map (`src/gold.py`). Not modified.

---

## Tokenizer vocabularies

Vocabulary and merge files for seven tokenizers are fetched by
`src/vendor_tokenizers.py` under the same manifest-verified, data-only model,
and are **not redistributed**.

This matters more here than for the lexica. Several of these tokenizers ship
under non-OSI model licences — the Llama Community License and the Gemma Terms
of Use among them — which restrict redistribution of model artifacts in ways
that are not compatible with simply committing a vocab file to a GPL-3
repository. The Llama 3 and Gemma 2 tokenizers are not vendored yet (their
upstream licences must be accepted first); when they are, **commit the
manifest only, never the vocabulary files.**

See `README.md` for the tokenizer security model and the Qwen data scope.
