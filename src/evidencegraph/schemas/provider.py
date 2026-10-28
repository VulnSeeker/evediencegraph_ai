"""Shared answer generator. One prompt, one schema, all four conditions."""
from __future__ import annotations

import json
from typing import Protocol

from evidencegraph.retrieval.base import EvidenceItem
from evidencegraph.schemas.eval import Answer, AnswerClaim

ANSWER_PROMPT = """You answer a scientific question using ONLY the provided evidence.

Rules:
- Cite individual factual claims using the evidence IDs provided.
- Never invent evidence IDs, paper IDs, or numeric results.
- If the evidence does not support an answer, set abstained=true.
- Report meaningful conflicting findings and note differences in study conditions.
- Return a JSON object matching this schema:
  {"answer": str, "answer_claims": [{"claim_text": str, "citation_ids": [str], "confidence": "low|moderate|high"}], "contradictions": [{"claim_a_id": str, "claim_b_id": str, "explanation": str}], "limitations": [str], "abstained": bool}

Evidence:
{evidence_block}

Question: {question}
"""


def _format_evidence(items: list[EvidenceItem]) -> str:
    lines = []
    for e in items:
        lines.append(f"[{e.span_id}] ({e.paper_id}) {e.text}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: list[EvidenceItem]) -> Answer: ...


class StubGenerator:
    """Deterministic generator used to test the pipeline without an LLM.

    Returns the top passage as a single cited claim. Useful for CI and for
    validating the auditor end-to-end without spending tokens.
    """

    name = "stub"

    def generate(self, question: str, evidence: list[EvidenceItem]) -> Answer:
        if not evidence:
            return Answer(answer="Insufficient evidence.", abstained=True)
        top = evidence[0]
        return Answer(
            answer=f"Top retrieved evidence: {top.text[:200]}",
            answer_claims=[
                AnswerClaim(
                    claim_text=f"Evidence states: {top.text[:120]}",
                    citation_ids=[top.span_id],
                    confidence="low",
                )
            ],
            abstained=False,
        )


class LLMGenerator:
    """OpenAI-compatible generator. Requires `pip install 'evidencegraph[llm]'`."""

    def __init__(self, model: str = "gpt-4o-mini", name: str | None = None):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI()
        self._model = model
        self.name = name or f"llm:{model}"

    def generate(self, question: str, evidence: list[EvidenceItem]) -> Answer:
        prompt = ANSWER_PROMPT.format(
            evidence_block=_format_evidence(evidence), question=question
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        return Answer(
            answer=payload.get("answer", ""),
            answer_claims=[AnswerClaim(**c) for c in payload.get("answer_claims", [])],
            contradictions=payload.get("contradictions", []),
            limitations=payload.get("limitations", []),
            abstained=bool(payload.get("abstained", False)),
        )
