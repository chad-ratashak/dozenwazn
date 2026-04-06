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
"""

import json
import re
import sys
from pathlib import Path

import tiktoken

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


def _load_jais_tokenizer():
    """Try to load the JAIS tokenizer. Returns None if unavailable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            'inceptionai/jais-13b',
            trust_remote_code=True,
        )
        return tok
    except Exception as e:
        print(f"  [JAIS tokenizer unavailable: {e}]", file=sys.stderr)
        return None


def check_tokenization(candidates_path: Path, output_path: Path):
    """For each candidate, check token counts with and without tashkeel."""
    # Load tiktoken encoders
    tiktoken_encoders = {name: tiktoken.get_encoding(name) for name in TIKTOKEN_ENCODINGS}

    # Try loading JAIS
    jais_tokenizer = _load_jais_tokenizer()
    has_jais = jais_tokenizer is not None
    if has_jais:
        print("  JAIS tokenizer loaded.", file=sys.stderr)

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

            # JAIS: diacriticized and bare
            if has_jais:
                tokens_diac['jais'] = len(jais_tokenizer.encode(verb, add_special_tokens=False))
                tokens_bare['jais'] = len(jais_tokenizer.encode(bare, add_special_tokens=False))

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
    """Print summary statistics."""
    # Detect all encoder names from first line
    with open(output_path, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
    all_encoders = list(first['tokens_bare'].keys())

    encoder_labels = {
        **TIKTOKEN_ENCODINGS,
        'jais': 'JAIS-13B (Arabic-optimized)',
    }

    total = 0
    single_diac_by_enc = {name: 0 for name in all_encoders}
    single_bare_by_enc = {name: 0 for name in all_encoders}
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
                if data['tokens_bare'].get(enc) == 1:
                    single_bare_by_enc[enc] += 1

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
        print(f"    {enc} ({label}): {c:,} ({100*c/total:.1f}%)", file=sys.stderr)


if __name__ == '__main__':
    candidates = Path('data/candidates.jsonl')
    output = Path('data/tokenizer_analysis.jsonl')

    if not candidates.exists():
        print("Run src/generate.py first to create candidates.jsonl", file=sys.stderr)
        sys.exit(1)

    count = check_tokenization(candidates, output)
    print(f"\nAnalyzed {count:,} candidates → {output}", file=sys.stderr)
    summarize(output)
