"""Paper and span schemas.

Offset semantics (critical, do not change without a migration):
  - Offsets are Python string code-point indices into the *normalized* text.
  - `EvidenceSpan.quote` MUST equal `normalized_text[char_start:char_end]`.
  - PDF page coordinates, if needed, are stored separately in provenance.
"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class Paper(BaseModel):
    paper_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    domain: str | None = None
    source_uri: str | None = None
    access_status: Literal["full_text", "abstract", "unavailable"] = "full_text"
    normalized_text: str
    text_hash: str = ""
    text_version: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        if not self.text_hash:
            object.__setattr__(self, "text_hash", sha256_text(self.normalized_text))


class EvidenceSpan(BaseModel):
    span_id: str
    paper_id: str
    section: str = ""
    paragraph_index: int = 0
    sentence_index: int = 0
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    quote: str

    @model_validator(mode="after")
    def _length_consistency(self) -> "EvidenceSpan":
        if self.char_end <= self.char_start:
            raise ValueError(f"span {self.span_id}: char_end must exceed char_start")
        if len(self.quote) != self.char_end - self.char_start:
            raise ValueError(
                f"span {self.span_id}: quote length {len(self.quote)} != "
                f"offset span {self.char_end - self.char_start}"
            )
        return self


def validate_span_against_document(span: EvidenceSpan, document_text: str) -> None:
    """Raise ValueError if the span's quote does not exactly match the document."""
    if span.char_end > len(document_text):
        raise ValueError(
            f"span {span.span_id}: char_end {span.char_end} > doc length {len(document_text)}"
        )
    actual = document_text[span.char_start:span.char_end]
    if actual != span.quote:
        raise ValueError(
            f"span {span.span_id}: quote mismatch at [{span.char_start}:{span.char_end}]\n"
            f"  recorded: {span.quote!r}\n"
            f"  actual:   {actual!r}"
        )
