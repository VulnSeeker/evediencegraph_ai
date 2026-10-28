"""Atomic claim and claim-relation schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RelationType = Literal["supports", "contradicts", "qualifies", "refines", "related_to"]


class AtomicClaim(BaseModel):
    claim_id: str
    paper_id: str
    span_ids: list[str] = Field(min_length=1)
    claim_text: str
    claim_type: Literal["result", "method", "hypothesis", "interpretation", "background"] = "result"
    population: str | None = None
    intervention_or_exposure: str | None = None
    outcome: str | None = None
    direction: Literal[
        "increase", "decrease", "no_detectable_effect", "mixed", "unclear"
    ] = "unclear"
    qualifiers: dict = Field(default_factory=dict)
    extraction_version: str = "claims-v0"
    review_status: Literal["unverified", "machine_validated", "human_reviewed"] = "unverified"


class ClaimRelation(BaseModel):
    relation_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = "stub"
    model_version: str = "relations-v0"
    review_status: Literal["unverified", "machine_validated", "human_reviewed"] = "unverified"
