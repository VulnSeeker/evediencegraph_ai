from evidencegraph.auditing.citation_validator import audit_citations
from evidencegraph.retrieval.base import EvidenceItem
from evidencegraph.schemas.eval import Answer, AnswerClaim


def _ev(span_id: str, paper_id: str, text: str, start: int, end: int) -> EvidenceItem:
    return EvidenceItem(span_id=span_id, paper_id=paper_id, text=text,
                        rank=0, score=1.0, char_start=start, char_end=end)


def test_unknown_citation_flagged():
    doc = "hello world"
    ev = _ev("S1", "P1", "hello", 0, 5)
    ans = Answer(answer="x", answer_claims=[AnswerClaim(claim_text="c", citation_ids=["S2"])])
    audit = audit_citations(ans, [ev], {"P1": doc})
    assert audit.total_citations == 1
    assert audit.resolved_citations == 0
    assert audit.unknown_citation_ids == ["S2"]


def test_valid_citation_resolves():
    doc = "hello world"
    ev = _ev("S1", "P1", "hello", 0, 5)
    ans = Answer(answer="x", answer_claims=[AnswerClaim(claim_text="c", citation_ids=["S1"])])
    audit = audit_citations(ans, [ev], {"P1": doc})
    assert audit.resolved_citations == 1
    assert not audit.unknown_citation_ids
    assert not audit.span_mismatches


def test_span_mismatch_detected():
    doc = "hello world"
    ev = _ev("S1", "P1", "HELLO", 0, 5)  # deliberately wrong
    ans = Answer(answer="x", answer_claims=[AnswerClaim(claim_text="c", citation_ids=["S1"])])
    audit = audit_citations(ans, [ev], {"P1": doc})
    assert audit.resolved_citations == 0
    assert audit.span_mismatches
