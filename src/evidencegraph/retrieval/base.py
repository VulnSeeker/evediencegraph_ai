"""Common retriever interface. All four experimental conditions implement it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class EvidenceItem:
    span_id: str
    paper_id: str
    text: str
    rank: int
    score: float
    char_start: int = -1
    char_end: int = -1
    evidence_type: str = "passage"  # "passage" | "claim"
    provenance: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(
        self, question: str, *, top_k: int, token_budget: int
    ) -> list[EvidenceItem]: ...
