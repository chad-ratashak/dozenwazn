#!/usr/bin/env python3
"""Vendor dictionary/corpus lexicon data into vendor/lexica/.

Mirrors the vendor_tokenizers security pattern: this is the ONLY module
allowed to touch the network for lexicon assets; everything fetched is inert
data (CSV/JSONL/text), hashed into a committed SHA256SUMS.json manifest, and
verified on every load.

Sources (all within the model-origin allowlist; licenses noted for attribution and any
future dataset publication):
  - arramooz  — Arramooz AlWaseet Arabic dictionary (Taha Zerrouki, Algeria),
    GPL-2. verb_triliteral.csv: vocalized Form I past + root + bāb number,
    which encodes the past middle vowel directly. Pinned to a commit SHA.
  - wiktionary_ar — English Wiktionary Arabic extract via kaikki.org
    (wiktextract), CC BY-SA + GFDL. Rebuilt upstream continuously, so no
    commit pinning exists: trust-on-first-use — the hash and fetch date are
    recorded and a re-vendor that changes the hash WARNS loudly.
  - freq_opensubtitles — OpenSubtitles 2018 Arabic frequency list from
    hermitdave/FrequencyWords, MIT. Pinned to a commit SHA. The repo LICENSE
    is vendored alongside for attribution.
"""

import sys
from datetime import datetime
from pathlib import Path

import httpx

from src.vendor_common import load_manifest, save_manifest, sha256_file

VENDOR_DIR = Path('vendor/lexica')
MANIFEST_PATH = VENDOR_DIR / 'SHA256SUMS.json'

ALLOWED_EXTENSIONS = {'.csv', '.xml', '.txt', '.json', '.jsonl', '.gz', '.sql', ''}

_ARRAMOOZ_COMMIT = 'b055c5dea51b8a9152e7e66a7ca3b6f713fb187d'  # 2023-02-09, repo HEAD at vendor-design time
_FREQWORDS_COMMIT = '525f9b560de45753a5ea01069454e72e9aa541c6'

LEXICA_REGISTRY = {
    'arramooz': {
        'urls': {
            'verb_triliteral.csv':
                f'https://raw.githubusercontent.com/linuxscout/arramooz/{_ARRAMOOZ_COMMIT}/data/verbs/verb_triliteral.csv',
            'verb_dic_data-net.csv':
                f'https://raw.githubusercontent.com/linuxscout/arramooz/{_ARRAMOOZ_COMMIT}/data/verbs/verb_dic_data-net.csv',
            'LICENSE':
                f'https://raw.githubusercontent.com/linuxscout/arramooz/{_ARRAMOOZ_COMMIT}/LICENSE',
        },
        'label': 'Arramooz AlWaseet (GPL-2, commit-pinned)',
        'pinned': True,
    },
    'wiktionary_ar': {
        'urls': {
            'kaikki.org-dictionary-Arabic.jsonl.gz':
                'https://kaikki.org/dictionary/Arabic/kaikki.org-dictionary-Arabic.jsonl.gz',
        },
        'label': 'Wiktionary Arabic via kaikki.org (CC BY-SA; TOFU hash)',
        'pinned': False,
    },
    'freq_opensubtitles': {
        'urls': {
            'ar_full.txt':
                f'https://raw.githubusercontent.com/hermitdave/FrequencyWords/{_FREQWORDS_COMMIT}/content/2018/ar/ar_full.txt',
            'LICENSE':
                f'https://raw.githubusercontent.com/hermitdave/FrequencyWords/{_FREQWORDS_COMMIT}/LICENSE',
        },
        'label': 'OpenSubtitles 2018 AR frequency list (MIT, commit-pinned)',
        'pinned': True,
    },
}


def _is_vendored_and_valid(name: str, spec: dict, manifest: dict) -> bool:
    entry = manifest.get(name)
    if entry is None:
        return False
    for fname in spec['urls']:
        recorded = entry.get('files', {}).get(fname)
        path = VENDOR_DIR / name / fname
        if recorded is None or not path.exists() or sha256_file(path) != recorded:
            return False
    return True


def vendor_one(name: str, spec: dict, manifest: dict, refresh: bool = False) -> bool:
    if not refresh and _is_vendored_and_valid(name, spec, manifest):
        print(f"  {name}: already vendored and verified — skipping", file=sys.stderr)
        return True

    for fname in spec['urls']:
        if Path(fname).suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"{name}: {fname} is not an allowed data file — refusing to vendor"
            )

    dest_dir = VENDOR_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=600.0,
                                                write=10.0, pool=10.0),
                          follow_redirects=True) as client:
            for fname, url in spec['urls'].items():
                print(f"  {name}: fetching {fname}…", file=sys.stderr)
                dest = dest_dir / fname
                with client.stream('GET', url) as resp:
                    resp.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + '.tmp')
                    with open(tmp, 'wb') as f:
                        for chunk in resp.iter_bytes(1 << 20):
                            f.write(chunk)
                    tmp.rename(dest)
                files[fname] = sha256_file(dest)
    except httpx.HTTPError as e:
        print(f"  {name}: FAILED — {e}", file=sys.stderr)
        return False

    old = manifest.get(name)
    if old is not None and old.get('files') != files:
        print(
            f"  {name}: WARNING — file hashes changed since last vendoring "
            f"(fetched {old.get('fetched', '?')}). Unpinned sources rebuild "
            f"upstream; review before committing the new manifest.",
            file=sys.stderr,
        )
    manifest[name] = {
        'urls': spec['urls'],
        'pinned': spec['pinned'],
        'fetched': datetime.now().strftime('%Y-%m-%d'),
        'files': files,
    }
    print(f"  {name}: vendored ({', '.join(files)})", file=sys.stderr)

    license_file = dest_dir / 'LICENSE'
    if license_file.exists():
        head = license_file.read_text(encoding='utf-8', errors='replace')[:200]
        print(f"  {name}: LICENSE begins: {head.splitlines()[0] if head else '?'}",
              file=sys.stderr)
    return True


def vendor(names: list[str] | None = None, refresh: bool = False) -> int:
    targets = names or list(LEXICA_REGISTRY)
    unknown = [n for n in targets if n not in LEXICA_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown lexica: {unknown}. Available: {sorted(LEXICA_REGISTRY)}")

    manifest = load_manifest(MANIFEST_PATH)
    ok = 0
    print(f"Vendoring lexicon data files → {VENDOR_DIR}/", file=sys.stderr)
    for name in targets:
        if vendor_one(name, LEXICA_REGISTRY[name], manifest, refresh=refresh):
            ok += 1
    save_manifest(MANIFEST_PATH, manifest)
    print(f"\n{ok}/{len(targets)} lexica vendored. Manifest: {MANIFEST_PATH}",
          file=sys.stderr)
    print("Commit the manifest so future fetches are integrity-checked.",
          file=sys.stderr)
    return ok


if __name__ == '__main__':
    vendor(sys.argv[1:] or None)
