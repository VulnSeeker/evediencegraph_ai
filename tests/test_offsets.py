from evidencegraph.ingestion.normalization import (
    chunk_into_passages, normalize_text, sentence_spans,
)


def test_sentence_spans_roundtrip():
    text = "First sentence. Second one! Third? Fourth."
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == [
        "First sentence.", "Second one!", "Third?", "Fourth."
    ]


def test_chunk_offsets_match_source():
    text = "A. B. C. D. E. F. G. H." * 50
    passages = chunk_into_passages(text, "P0001", target_chars=20)
    assert len(passages) > 1
    for p in passages:
        assert text[p.char_start:p.char_end] == p.text


def test_normalize_idempotent():
    raw = "A\r\n\r\n\r\n\r\nB   C"
    n1 = normalize_text(raw)
    n2 = normalize_text(n1)
    assert n1 == n2
