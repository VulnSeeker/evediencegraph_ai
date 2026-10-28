"""Three separate checks: existence, span integrity, and (stub) semantic support."""
from __future__ import annotations

from pydantic import BaseModel, Field

from evidencegraph.retrieval.base import EvidenceItem
from evidencegraph.schemas.eval import Answer


class CitationAudit(BaseModel):
    total_citations: int = 0
    resolved_citations: int = 0
    unknown_citation_ids: list[str] = Field(default_factory=list)
    span_mismatches: list[str] = Field(default_factory=list)
    # Semantic support is a separate concern; filled in only by an LLM judge.
    semantic_support_pending: bool = True


def audit_citations(
    answer: Answer,
    evidence: list[EvidenceItem],
    document_store: dict[str, str],
) -> CitationAudit:
    provided = {e.span_id: e for e in evidence}
    audit = CitationAudit()

    for ac in answer.answer_claims:
        for cid in ac.citation_ids:
            audit.total_citations += 1
            if cid not in provided:
                audit.unknown_citation_ids.append(cid)
                continue

            e = provided[cid]
            if e.char_start < 0 or e.char_end < 0:
                continue
            doc = document_store.get(e.paper_id)
            if doc is None:
                audit.span_mismatches.append(f"{cid}: paper {e.paper_id} not in store")
                continue
            if doc[e.char_start:e.char_end] != e.text:
                audit.span_mismatches.append(
                    f"{cid}: stored text does not match document at "
                    f"[{e.char_start}:{e.char_end}]"
                )
                continue
            audit.resolved_citations += 1

    return audit
