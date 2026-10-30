# SPDX-License-Identifier: GPL-3.0-or-later
"""Data models for dozenwazn."""

from pydantic import BaseModel


class VerbCandidate(BaseModel):
    root: str
    root_type: str  # XYZ, XYY, XYX, XYXY, XYZN
    arity: int  # 3 or 4
    pattern: str  # faʿala, faʿila, faʿula, faʿfaʿa, faʿlala
    verb: str  # fully diacriticized, correct surface form
    # Fields below default so pre-migration JSONL still parses; weak_class
    # None means "record predates the weak-radical fix", not "sound".
    verb_naive: str | None = None      # old CaCaCa render; join key for legacy data
    weak_class: str | None = None      # see morphophonology.WEAK_CLASSES
    hamza_positions: list[int] = []    # 1-indexed positions of ء radicals
    surface_collapsed: bool = False    # hollow/geminate: 3 patterns → 1 surface
    plausibility: float | None = None  # OCP-Place score, root-level, (0, 1]


class ValidationResult(BaseModel):
    verb: str
    exists: bool | None = None
    dialect_only: bool | None = None
    list_of_dialects: list[str] = []
    dialect_meanings: str = ""
    msa: bool | None = None
    msa_meanings: str = ""
