#!/usr/bin/env python3
"""Check if generated verbs are single tokens across multiple tokenizers.

Tests both diacriticized (with tashkeel) and bare consonantal forms.
Most Arabic text online is written without short vowels, so BPE tokenizers
are likely trained primarily on undiacriticized text. The bare form is
expected to produce far more single-token hits.

Findings so far:
  - tiktoken (cl100k_base, o200k_base): 0% single-token with tashkeel,
    ~3.4% without (o200k_base only)
  - JAIS (Arabic-optimized, 32K Arabic tokens): common bare roots are
    single tokens; tashkeel explodes every verb into 6-9 tokens since
    each diacritical mark becomes its own token

Security model: open-source tokenizers are loaded ONLY from vendor/tokenizers/
as inert data files via the `tokenizers` library — no transformers, no
trust_remote_code, no network access from this module. Every file is verified
against the committed SHA256SUMS.json manifest before loading; a mismatch
aborts. Run `uv run python -m src.cli vendor` once to fetch the files.
"""

import json
import re
import sys
from pathlib import Path

import tiktoken

from src.vendor_tokenizers import (
    MANIFEST_PATH, TOKENIZER_REGISTRY, VENDOR_DIR, load_manifest, sha256_file,
)

# ── Tiktoken encodings ───────────────────────────────────────────────
TIKTOKEN_ENCODINGS = {
    'cl100k_base': 'GPT-4/GPT-4o (older)',
    'o200k_base': 'GPT-4o/o1 (newer)',
}

# Arabic diacritical marks (tashkeel) Unicode range
TASHKEEL_PATTERN = re.compile(r'[\u064B-\u0652\u0670]')


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritics (short vowels, shadda, sukun, etc.)."""
    return TASHKEEL_PATTERN.sub('', text)


class VendoredTokenizer:
    """Token counter backed by vendored data files. Never executes repo code."""

    _UNK_TOKENS = ('[UNK]', '<unk>')

    def __init__(self, name: str, directory: Path, files: list[str]):
        from tokenizers import Tokenizer

        self.name = name
        if 'tokenizer.json' in files:
            self._tok = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        elif 'vocab.txt' in files:
            # WordPiece vocab without a serialized tokenizer.json (CAMeLBERT).
            # strip_accents=False so tashkeel isn't silently removed.
            from tokenizers.implementations import BertWordPieceTokenizer
            self._tok = BertWordPieceTokenizer(
                str(directory / 'vocab.txt'),
                lowercase=False,
                strip_accents=False,
            )
        else:
            raise ValueError(f"{name}: no loadable tokenizer data in {files}")

        self._unk_id = None
        for unk in self._UNK_TOKENS:
            tid = self._tok.token_to_id(unk)
            if tid is not None:
                self._unk_id = tid
                break

    def count(self, text: str) -> int | None:
        """Token count for text, or None if it collapsed to a lone UNK token.

        A single [UNK] means the tokenizer has no coverage at all — counting
        it as a "single-token hit" would be a false positive.
        """
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if self._unk_id is not None and ids == [self._unk_id]:
            return None
        return len(ids)


def load_vendored_tokenizers() -> dict[str, 'VendoredTokenizer']:
    """Load every vendored tokenizer, verifying file hashes against the manifest.

    Missing files are skipped with a note (run `vendor` to fetch them);
    a hash MISMATCH aborts — that's corruption or tampering, not a soft error.
    """
    manifest = load_manifest()
    if not manifest:
        print(f"  [no vendored tokenizers — run 'uv run python -m src.cli vendor' "
              f"to enable JAIS/Qwen/AraBERT/etc.]", file=sys.stderr)
        return {}

    loaded = {}
    for name, entry in sorted(manifest.items()):
        directory = VENDOR_DIR / name
        files = list(entry.get('files', {}))
        missing = [f for f in files if not (directory / f).exists()]
        if missing:
            print(f"  [{name}: files missing ({', '.join(missing)}) — "
                  f"run vendor to fetch, skipping]", file=sys.stderr)
            continue
        for fname, recorded in entry['files'].items():
            actual = sha256_file(directory / fname)
            if actual != recorded:
                raise RuntimeError(
                    f"SHA-256 mismatch for {directory / fname}: manifest says "
                    f"{recorded[:16]}…, file is {actual[:16]}…. Refusing to load. "
                    f"If you intentionally re-vendored, commit the new manifest; "
                    f"otherwise investigate before proceeding."
                )
        loaded[name] = VendoredTokenizer(name, directory, files)
        print(f"  {name} tokenizer loaded (verified).", file=sys.stderr)
    return loaded


def check_tokenization(candidates_path: Path, output_path: Path):
    """For each candidate, check token counts with and without tashkeel."""
    # Load tiktoken encoders
    tiktoken_encoders = {name: tiktoken.get_encoding(name) for name in TIKTOKEN_ENCODINGS}

    # Load vendored open-source tokenizers (data-only, hash-verified)
    vendored = load_vendored_tokenizers()

    count = 0
    with open(candidates_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            candidate = json.loads(line)
            verb = candidate['verb']
            bare = strip_tashkeel(verb)

            # Tiktoken: diacriticized and bare
            tokens_diac = {}
            tokens_bare = {}
            for enc_name, encoder in tiktoken_encoders.items():
                tokens_diac[enc_name] = len(encoder.encode(verb))
                tokens_bare[enc_name] = len(encoder.encode(bare))

            # Vendored tokenizers: diacriticized and bare
            for name, tok in vendored.items():
                tokens_diac[name] = tok.count(verb)
                tokens_bare[name] = tok.count(bare)

            result = {
                **candidate,
                'verb_bare': bare,
                'tokens_diacriticized': tokens_diac,
                'tokens_bare': tokens_bare,
                'single_token_diacriticized': any(c == 1 for c in tokens_diac.values()),
                'single_token_bare': any(c == 1 for c in tokens_bare.values()),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            count += 1

            if count % 10000 == 0:
                print(f"  Processed {count:,}...", file=sys.stderr)

    return count


def summarize(output_path: Path):
    """Print summary statistics, including fertility (mean tokens per verb)."""
    # Detect all encoder names from first line
    with open(output_path, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
    all_encoders = list(first['tokens_bare'].keys())

    encoder_labels = {
        **TIKTOKEN_ENCODINGS,
        **{name: spec['label'] for name, spec in TOKENIZER_REGISTRY.items()},
    }

    total = 0
    single_diac_by_enc = {name: 0 for name in all_encoders}
    single_bare_by_enc = {name: 0 for name in all_encoders}
    token_sum_bare = {name: 0 for name in all_encoders}
    token_n_bare = {name: 0 for name in all_encoders}
    single_diac_any = 0
    single_bare_any = 0

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            if data['single_token_diacriticized']:
                single_diac_any += 1
            if data['single_token_bare']:
                single_bare_any += 1
            for enc in all_encoders:
                if data['tokens_diacriticized'].get(enc) == 1:
                    single_diac_by_enc[enc] += 1
                bare_count = data['tokens_bare'].get(enc)
                if bare_count == 1:
                    single_bare_by_enc[enc] += 1
                if bare_count is not None:
                    token_sum_bare[enc] += bare_count
                    token_n_bare[enc] += 1

    print(f"\nTokenizer Analysis Summary:", file=sys.stderr)
    print(f"  Total candidates: {total:,}", file=sys.stderr)

    print(f"\n  WITH tashkeel (diacriticized):", file=sys.stderr)
    print(f"    Single-token in any: {single_diac_any:,} ({100*single_diac_any/total:.1f}%)", file=sys.stderr)
    for enc in all_encoders:
        label = encoder_labels.get(enc, enc)
        c = single_diac_by_enc[enc]
        print(f"    {enc} ({label}): {c:,} ({100*c/total:.1f}%)", file=sys.stderr)

    print(f"\n  WITHOUT tashkeel (bare consonants):", file=sys.stderr)
    print(f"    Single-token in any: {single_bare_any:,} ({100*single_bare_any/total:.1f}%)", file=sys.stderr)
    for enc in all_encoders:
        label = encoder_labels.get(enc, enc)
        c = single_bare_by_enc[enc]
        n = token_n_bare[enc]
        fertility = f", fertility {token_sum_bare[enc]/n:.2f} tok/verb" if n else ""
        print(f"    {enc} ({label}): {c:,} ({100*c/total:.1f}%){fertility}", file=sys.stderr)


if __name__ == '__main__':
    candidates = Path('data/candidates.jsonl')
    output = Path('data/tokenizer_analysis.jsonl')

    if not candidates.exists():
        print("Run src/generate.py first to create candidates.jsonl", file=sys.stderr)
        sys.exit(1)

    count = check_tokenization(candidates, output)
    print(f"\nAnalyzed {count:,} candidates → {output}", file=sys.stderr)
    summarize(output)
