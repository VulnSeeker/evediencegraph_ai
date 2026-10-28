import pytest

from evidencegraph.schemas.source import EvidenceSpan, validate_span_against_document


def test_valid_span():
    doc = "The effect was not significant in this cohort."
    start = doc.index("not significant")
    span = EvidenceSpan(
        span_id="P0001_S0001", paper_id="P0001",
        char_start=start, char_end=start + len("not significant"),
        quote="not significant",
    )
    validate_span_against_document(span, doc)


def test_quote_length_mismatch_rejected():
    with pytest.raises(ValueError, match="quote length"):
        EvidenceSpan(
            span_id="x", paper_id="p", char_start=0, char_end=5, quote="abc"
        )


def test_offset_mismatch_detected():
    doc = "The effect was not significant in this cohort."
    span = EvidenceSpan(
        span_id="x", paper_id="p", char_start=0, char_end=3, quote="The"
    )
    validate_span_against_document(span, doc)
    with pytest.raises(ValueError, match="quote mismatch"):
        validate_span_against_document(span, "XXX" + doc[3:])


def test_out_of_range_rejected():
    doc = "short"
    span = EvidenceSpan(
        span_id="x", paper_id="p", char_start=0, char_end=5, quote="short"
    )
    with pytest.raises(ValueError, match="exceeds doc length|char_end"):
        validate_span_against_document(span, "shor")
