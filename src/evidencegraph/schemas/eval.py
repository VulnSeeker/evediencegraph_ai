"""Question, gold annotation, and answer schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "consistent_evidence",
    "direct_contradiction",
    "contextual_divergence",
    "refuted_statement",
    "insufficient_evidence",
    "qualification",
]


class Question(BaseModel):
    question_id: str
    text: str
    question_type: QuestionType
    top_k: int = 10
    token_budget: int = 4000
    answerability: Literal["answerable", "unanswerable", "ambiguous"] = "answerable"


class GoldAnnotation(BaseModel):
    """Gold labels for one question, produced independently of any system output."""
    question_id: str
    required_evidence_ids: list[str] = Field(default_factory=list)
    gold_answer_points: list[str] = Field(default_factory=list)
    gold_contradictions: list[tuple[str, str]] = Field(default_factory=list)
    annotator_ids: list[str] = Field(default_factory=list)
    annotation_version: str = "gold-v0"
    adjudicated: bool = False
    notes: str = ""


class AnswerClaim(BaseModel):
    claim_text: str
    citation_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "moderate", "high"] = "moderate"


class ContradictionReport(BaseModel):
    claim_a_id: str
    claim_b_id: str
    explanation: str


class Answer(BaseModel):
    question_id: str = ""
    answer: str
    answer_claims: list[AnswerClaim] = Field(default_factory=list)
    contradictions: list[ContradictionReport] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    abstained: bool = False
