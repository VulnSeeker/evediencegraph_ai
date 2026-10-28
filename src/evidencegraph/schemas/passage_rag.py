"""Condition A: vanilla passage RAG over TF-IDF (deterministic, no downloads)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evidencegraph.ingestion.normalization import Passage
from evidencegraph.retrieval.base import EvidenceItem


def _approx_tokens(text: str) -> int:
    return len(text.split())


class PassageRAGRetriever:
    name = "passage_rag"

    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("PassageRAGRetriever requires at least one passage")
        self.passages = passages
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1
        )
        self._matrix = self._vectorizer.fit_transform([p.text for p in passages])

    def retrieve(
        self, question: str, *, top_k: int = 10, token_budget: int = 4000
    ) -> list[EvidenceItem]:
        q = self._vectorizer.transform([question])
        sims = cosine_similarity(q, self._matrix).flatten()
        order = sims.argsort()[::-1]

        items: list[EvidenceItem] = []
        used = 0
        for rank, idx in enumerate(order[: max(top_k * 3, top_k)]):
            p = self.passages[int(idx)]
            cost = _approx_tokens(p.text)
            if items and used + cost > token_budget:
                break
            used += cost
            items.append(
                EvidenceItem(
                    span_id=p.passage_id,
                    paper_id=p.paper_id,
                    text=p.text,
                    rank=len(items),
                    score=float(sims[int(idx)]),
                    char_start=p.char_start,
                    char_end=p.char_end,
                    evidence_type="passage",
                    provenance={"retriever": self.name},
                )
            )
            if len(items) >= top_k:
                break
        return items
