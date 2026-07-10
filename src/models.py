"""Data models for dozenwazn."""

from pydantic import BaseModel


class VerbCandidate(BaseModel):
    root: str
    root_type: str  # XYZ, XYY, XYX, XYXY, XYZN
    arity: int  # 3 or 4
    pattern: str  # faʿala, faʿila, faʿula, faʿfaʿa, faʿlala
    verb: str  # fully diacriticized


class ValidationResult(BaseModel):
    verb: str
    exists: bool | None = None
    dialect_only: bool | None = None
    list_of_dialects: list[str] = []
    dialect_meanings: str = ""
    msa: bool | None = None
    msa_meanings: str = ""
