# SPDX-License-Identifier: GPL-3.0-or-later
"""Backend tests for VendoredTokenizer (tokenizer.json / vocab.txt / .model).

All tests skip cleanly when the relevant tokenizer isn't vendored yet
(run `uv run python -m src.cli vendor` to fetch). No network access here —
everything loads from vendor/tokenizers/ with hash verification.
"""

import pytest

from src.tokenizer_check import load_vendored_tokenizers
from src.vendor_tokenizers import VENDOR_DIR, load_manifest

WORD = 'كتب'          # common bare triliteral root — should be well covered
GARBAGE = '࢑࢒'  # unassigned Arabic-Extended-B codepoints


@pytest.fixture(scope='module')
def vendored():
    return load_vendored_tokenizers()


def _require(vendored, name):
    if name not in vendored:
        pytest.skip(f"{name} not vendored — run 'uv run python -m src.cli vendor'")
    return vendored[name]


def test_aya101_loads_and_counts(vendored):
    tok = _require(vendored, 'aya101')
    n = tok.count(WORD)
    assert isinstance(n, int) and n >= 1
    # Diacriticized form should also produce a genuine int count
    nd = tok.count('كَتَبَ')
    assert isinstance(nd, int) and nd >= 1


def test_aya101_no_fake_single_token_for_garbage(vendored):
    """mT5 spiece has no byte fallback: garbage must not report a fake 1.

    A lone UNK collapses to None per the count() convention; multiple
    tokens (> 1) are also acceptable — a bare 1 would be a false positive.
    """
    tok = _require(vendored, 'aya101')
    n = tok.count(GARBAGE)
    assert n is None or n > 1


def test_aya101_unk_id_is_set(vendored):
    """mT5 SentencePiece defines <unk> (id 2) — the lone-UNK rule must be armed."""
    tok = _require(vendored, 'aya101')
    assert tok._unk_id is not None and tok._unk_id >= 0


@pytest.mark.parametrize('name', ['arabert', 'jais'])
def test_existing_backends_regression(vendored, name):
    """arabert (tokenizer.json) and jais (tokenizer.json) still count correctly.

    Regression check: VendoredTokenizer.count must match a direct load of the
    same vendored file via the `tokenizers` library.
    """
    tok = _require(vendored, name)
    n = tok.count(WORD)
    assert isinstance(n, int) and n >= 1

    from tokenizers import Tokenizer
    direct = Tokenizer.from_file(str(VENDOR_DIR / name / 'tokenizer.json'))
    expected = len(direct.encode(WORD, add_special_tokens=False).ids)
    assert n == expected


def test_camelbert_wordpiece_backend(vendored):
    """vocab.txt (BertWordPiece) backend still works after the refactor."""
    tok = _require(vendored, 'camelbert')
    n = tok.count(WORD)
    assert isinstance(n, int) and n >= 1


def test_manifest_covers_loaded_tokenizers(vendored):
    manifest = load_manifest()
    for name in vendored:
        assert name in manifest
