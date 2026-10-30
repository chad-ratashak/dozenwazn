#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vendor dictionary/corpus lexicon data into vendor/lexica/.

Mirrors the vendor_tokenizers security pattern: this is the ONLY module
allowed to touch the network for lexicon assets; everything fetched is inert
data (CSV/JSONL/text), hashed into a committed SHA256SUMS.json manifest, and
verified on every load.

Sources (all within the model-origin allowlist; licenses noted for attribution and any
future dataset publication):
  - arramooz  — Arramooz AlWaseet Arabic dictionary (Taha Zerrouki, Algeria),
    GPL-2.0-or-later. verb_triliteral.csv: vocalized Form I past + root +
    bāb number, which encodes the past middle vowel directly. Pinned to a
    commit SHA, with a committed fallback copy (see FALLBACK_DIR).
  - wiktionary_ar — English Wiktionary Arabic extract via kaikki.org
    (wiktextract), CC BY-SA + GFDL. Rebuilt upstream continuously, so no
    commit pinning exists: trust-on-first-use — the hash and fetch date are
    recorded and a re-vendor that changes the hash WARNS loudly.
  - freq_opensubtitles — OpenSubtitles 2018 Arabic frequency list from
    hermitdave/FrequencyWords, MIT. Pinned to a commit SHA. The repo LICENSE
    is vendored alongside for attribution.

Per-source licences and attribution: NOTICE-DATA.md.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import httpx

from src.vendor_common import (
    load_manifest,
    save_manifest,
    sha256_file,
    verify_files,
)

VENDOR_DIR = Path('vendor/lexica')
MANIFEST_PATH = VENDOR_DIR / 'SHA256SUMS.json'

# In-repo last-resort copies, used only when the network fetch fails. Unlike
# vendor/lexica/, these ARE committed — upstream can vanish or force-push and
# the pinned raw URLs stop resolving. They are never trusted on their own: a
# restored copy is verified against the committed manifest before it counts,
# so the fallback cannot be used to smuggle in different data.
#
# Only redistributable sources belong here. arramooz is GPL-2.0-or-later
# (see NOTICE-DATA.md). The Wiktionary/kaikki extract must NOT be added: it
# is CC BY-SA, and committing it would mix that into this GPL-3 tree.
FALLBACK_DIR = Path('vendor/_fallback')

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
        'label': 'Arramooz AlWaseet (GPL-2.0-or-later, commit-pinned + fallback)',
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


def _restore_from_fallback(name: str, spec: dict, dest_dir: Path,
                           manifest: dict) -> bool:
    """Copy the committed fallback into vendor/lexica/<name>/ after a failed fetch.

    The copy is only accepted if it matches the hashes already recorded in the
    committed manifest, so this cannot become a way to introduce data that the
    manifest never blessed. Returns False when no complete fallback exists (or
    no manifest entry to check it against); raises via verify_files if a
    fallback file is present but wrong.
    """
    src_dir = FALLBACK_DIR / name
    recorded = (manifest.get(name) or {}).get('files')
    if not src_dir.is_dir() or not recorded:
        return False
    if any(not (src_dir / fname).exists() for fname in spec['urls']):
        return False

    for fname in spec['urls']:
        shutil.copyfile(src_dir / fname, dest_dir / fname)
    verify_files(name, dest_dir, {k: v for k, v in recorded.items()
                                  if k in spec['urls']})
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
        print(f"  {name}: fetch FAILED — {e}", file=sys.stderr)
        if _restore_from_fallback(name, spec, dest_dir, manifest):
            print(
                f"  {name}: restored from the committed fallback copy "
                f"({FALLBACK_DIR / name}/) and verified against the manifest — "
                f"upstream is unreachable but the data is byte-identical.",
                file=sys.stderr,
            )
            return True
        print(f"  {name}: no usable fallback copy — giving up", file=sys.stderr)
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
