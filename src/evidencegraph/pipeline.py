"""Shared orchestration. All four conditions call this; only the retriever changes."""
from __future__ import annotations

import time
from dataclasses import asdict

from evidencegraph.auditing.citation_validator import CitationAudit, audit_citations
from evidencegraph.generation.provider import Generator
from evidencegraph.retrieval.base import Retriever
from evidencegraph.schemas.eval import Question


def answer_question(
    question: Question,
    retriever: Retriever,
    generator: Generator,
    document_store: dict[str, str],
) -> dict:
    t0 = time.perf_counter()
    evidence = retriever.retrieve(
        question.text, top_k=question.top_k, token_budget=question.token_budget
    )
    t_retrieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = generator.generate(question.text, evidence)
    answer.question_id = question.question_id
    t_generate = time.perf_counter() - t0

    audit: CitationAudit = audit_citations(answer, evidence, document_store)

    return {
        "question_id": question.question_id,
        "retriever": retriever.name,
        "generator": generator.name,
        "evidence": [asdict(e) for e in evidence],
        "answer": answer.model_dump(),
        "audit": audit.model_dump(),
        "latency_s": {"retrieve": t_retrieve, "generate": t_generate},
    }
