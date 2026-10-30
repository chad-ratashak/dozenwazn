# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers for vendored-asset manifests (tokenizers, lexica).

Every vendored data file is hashed into a committed SHA256SUMS.json; loaders
verify against it before reading so a changed upstream file fails loudly
instead of being silently ingested.
"""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def verify_files(name: str, directory: Path, recorded_files: dict[str, str]):
    """Verify every recorded file's hash; raise on mismatch, KeyError-style
    FileNotFoundError on absence."""
    for fname, recorded in recorded_files.items():
        path = directory / fname
        if not path.exists():
            raise FileNotFoundError(f"{name}: vendored file missing: {path}")
        actual = sha256_file(path)
        if actual != recorded:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: manifest says {recorded[:16]}…, "
                f"file is {actual[:16]}…. Refusing to load. If you intentionally "
                f"re-vendored, commit the new manifest; otherwise investigate "
                f"before proceeding."
            )
