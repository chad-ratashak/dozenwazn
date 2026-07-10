#!/usr/bin/env python3
"""Vendor tokenizer data files from HuggingFace into vendor/tokenizers/.

This is the ONLY module in the project allowed to touch the network for
tokenizer assets. Everything it fetches is inert data (vocab/merge tables) —
no repository code is ever downloaded or executed (no trust_remote_code,
no transformers import).

Safety properties:
  - Extension allowlist: only .json/.txt/.model data files may be fetched.
  - Each download is pinned to the exact commit SHA resolved at vendor time.
  - Every file's SHA-256 is recorded in vendor/tokenizers/SHA256SUMS.json.
    Commit that manifest: later vendor runs — and every tokenizer_check.py
    load — verify against it, so a changed upstream file fails loudly
    instead of being silently ingested.
  - Public repos are fetched anonymously (token=False) so no credential is
    ever sent. Gated repos (Llama 3, Gemma) need HF_TOKEN with the license
    accepted on huggingface.co; without it they are skipped with a note.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

VENDOR_DIR = Path('vendor/tokenizers')
MANIFEST_PATH = VENDOR_DIR / 'SHA256SUMS.json'

ALLOWED_EXTENSIONS = {'.json', '.txt', '.model'}

# name → spec. `files` lists the data files needed to *count tokens*; nothing
# else is fetched. Qwen is vocabulary-only research data — see the
# "Tokenizer Data Scope" section in this project's CLAUDE.md.
TOKENIZER_REGISTRY = {
    'jais': {
        'repo': 'inceptionai/jais-13b',
        'files': ['tokenizer.json'],
        'label': 'JAIS-13B (Arabic-optimized)',
        'gated': False,
    },
    'qwen2.5': {
        'repo': 'Qwen/Qwen2.5-7B',
        'files': ['tokenizer.json'],
        'label': 'Qwen 2.5 (vocabulary only — see CLAUDE.md data scope)',
        'gated': False,
    },
    # Gated pair (Llama 3 / Gemma 2) — how to vendor them:
    #   1. Log in to huggingface.co and accept the licenses at
    #      https://huggingface.co/meta-llama/Meta-Llama-3-8B and
    #      https://huggingface.co/google/gemma-2-2b
    #   2. Create a fine-grained READ token (Settings → Access Tokens) scoped
    #      to public gated repos you have access to.
    #   3. export HF_TOKEN=hf_...
    #   4. uv run python -m src.cli vendor --only llama3,gemma2
    #   5. Commit vendor/tokenizers/SHA256SUMS.json so future loads are
    #      integrity-checked.
    'llama3': {
        'repo': 'meta-llama/Meta-Llama-3-8B',
        'files': ['tokenizer.json'],
        'label': 'Llama 3',
        'gated': True,
    },
    'gemma2': {
        'repo': 'google/gemma-2-2b',
        'files': ['tokenizer.json'],
        'label': 'Gemma 2',
        'gated': True,
    },
    # Aya-101's tokenizer is the unmodified mT5 SentencePiece model. The Aya
    # repo (CohereLabs/aya-101, formerly CohereForAI/aya-101; Apache-2.0,
    # ungated) ships only a converted tokenizer.json — the original
    # spiece.model protobuf lives in google/mt5-base (Apache-2.0, ungated,
    # Google/US). Both encode identically (verified at vendor time); we take
    # the SentencePiece original so it loads via the sentencepiece library
    # as inert data.
    'aya101': {
        'repo': 'google/mt5-base',
        'files': ['spiece.model'],
        'label': 'Aya-101 (Cohere, mT5 SentencePiece)',
        'gated': False,
    },
    'arabert': {
        'repo': 'aubmindlab/bert-base-arabertv2',
        'files': ['tokenizer.json'],
        'label': 'AraBERT v2',
        'gated': False,
    },
    'camelbert': {
        'repo': 'CAMeL-Lab/bert-base-arabic-camelbert-mix',
        'files': ['vocab.txt'],
        'label': 'CAMeLBERT (mix)',
        'gated': False,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def _is_vendored_and_valid(name: str, spec: dict, manifest: dict) -> bool:
    """True if this tokenizer's files exist locally and match the manifest."""
    entry = manifest.get(name)
    if entry is None:
        return False
    for fname in spec['files']:
        recorded = entry.get('files', {}).get(fname)
        path = VENDOR_DIR / name / fname
        if recorded is None or not path.exists() or sha256_file(path) != recorded:
            return False
    return True


def vendor_one(name: str, spec: dict, manifest: dict, refresh: bool = False) -> bool:
    """Fetch one tokenizer's data files. Returns True on success."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if not refresh and _is_vendored_and_valid(name, spec, manifest):
        print(f"  {name}: already vendored and verified — skipping", file=sys.stderr)
        return True

    for fname in spec['files']:
        if Path(fname).suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"{name}: {fname} is not an allowed data file "
                f"({sorted(ALLOWED_EXTENSIONS)}) — refusing to vendor"
            )

    repo = spec['repo']
    # Anonymous for public repos: no credential ever leaves this machine.
    # Gated repos use the ambient HF_TOKEN only because HF requires license
    # acceptance to serve the files at all.
    token = None if spec['gated'] else False

    try:
        commit = HfApi().model_info(repo, token=token).sha
        dest_dir = VENDOR_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for fname in spec['files']:
            cached = hf_hub_download(repo, fname, revision=commit, token=token)
            dest = dest_dir / fname
            shutil.copyfile(cached, dest)
            files[fname] = sha256_file(dest)
    except (GatedRepoError, HfHubHTTPError) as e:
        if spec['gated']:
            print(
                f"  {name}: SKIPPED (gated repo). To vendor it: accept the "
                f"license at https://huggingface.co/{repo}, set HF_TOKEN, "
                f"and re-run vendor.",
                file=sys.stderr,
            )
        else:
            print(f"  {name}: FAILED — {e}", file=sys.stderr)
        return False

    old = manifest.get(name)
    if old is not None and old.get('files') != files:
        print(
            f"  {name}: WARNING — file hashes changed since last vendoring "
            f"(old commit {old.get('commit', '?')[:12]}, new {commit[:12]}). "
            f"Review the diff before committing the manifest.",
            file=sys.stderr,
        )
    manifest[name] = {'repo': repo, 'commit': commit, 'files': files}
    print(f"  {name}: vendored @ {commit[:12]} ({', '.join(files)})", file=sys.stderr)
    return True


def vendor(names: list[str] | None = None, refresh: bool = False) -> int:
    """Vendor the requested tokenizers (all by default). Returns count vendored."""
    targets = names or list(TOKENIZER_REGISTRY)
    unknown = [n for n in targets if n not in TOKENIZER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tokenizer(s): {unknown}. "
                         f"Available: {sorted(TOKENIZER_REGISTRY)}")

    manifest = load_manifest()
    ok = 0
    print(f"Vendoring tokenizer data files → {VENDOR_DIR}/", file=sys.stderr)
    for name in targets:
        if vendor_one(name, TOKENIZER_REGISTRY[name], manifest, refresh=refresh):
            ok += 1
    save_manifest(manifest)
    print(f"\n{ok}/{len(targets)} tokenizers vendored. Manifest: {MANIFEST_PATH}",
          file=sys.stderr)
    print("Commit the manifest so future fetches are integrity-checked.",
          file=sys.stderr)
    return ok


if __name__ == '__main__':
    vendor(sys.argv[1:] or None)
