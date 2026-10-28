"""Text normalization and sentence-aware passage chunking.

Normalization policy (documented, versioned):
  - Unicode NFKC
  - Normalize CRLF -> LF
  - Collapse runs of >2 blank lines to 2
  - Do NOT strip leading/trailing whitespace of the whole doc (offsets are absolute)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) code-point spans for each sentence."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in SENT_BOUNDARY.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


@dataclass
class Passage:
    passage_id: str
    paper_id: str
    text: str
    char_start: int
    char_end: int


def chunk_into_passages(
    text: str,
    paper_id: str,
    target_chars: int = 1200,
) -> list[Passage]:
    """Group sentences into passages of roughly `target_chars` length.

    Offsets are preserved so the passage can be validated against the paper.
    No overlap in v0 — noted in protocol deviations.
    """
    sents = sentence_spans(text)
    if not sents:
        return []
    passages: list[Passage] = []
    i = 0
    idx = 0
    while i < len(sents):
        start = sents[i][0]
        j = i
        while j < len(sents) and (sents[j][1] - start) < target_chars:
            j += 1
        end = sents[j - 1][1] if j > i else sents[i][1]
        passages.append(
            Passage(
                passage_id=f"{paper_id}_pass{idx:04d}",
                paper_id=paper_id,
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
        idx += 1
        i = max(j, i + 1)
    return passages
