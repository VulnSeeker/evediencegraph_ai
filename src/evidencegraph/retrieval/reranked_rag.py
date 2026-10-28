"""Condition B: passage RAG with cross-encoder reranking.

Design constraints (fairness controls from the blueprint):
- Same `EvidenceItem` return type as Condition A.
- Same final `top_k` and `token_budget` as Condition A.
- Candidate pool is strictly larger than `top_k` (default 50), then reranked.
- Reranker is pinned and injectable, so tests and CI can run without model
  downloads. The default reranker is lazy-loaded only when actually used.
- Candidate-pool rank (from the TF-IDF index) is preserved in `provenance`
  so analyses can separate "the reranker fixed a bad baseline rank" from
  "the baseline was already right".
"""
from __future__ import annotations

from typing import Protocol

from evidencegraph.ingestion.normalization import Passage
from evidencegraph.retrieval.base import EvidenceItem
from evidencegraph.retrieval.passage_rag import PassageRAGRetriever

DEFAULT_RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class _CrossEncoderReranker:
    """Default reranker. `sentence-transformers` is imported lazily so that
    importing this module does not require the `embeddings` extra."""

    def __init__(self, model_id: str = DEFAULT_RERANKER_ID):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model_id = model_id
        self._model = CrossEncoder(model_id)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs)]


class RerankedPassageRAGRetriever:
    name = "reranked_passage_rag"

    def __init__(
        self,
        passages: list[Passage],
        *,
        candidate_pool_size: int = 50,
        reranker: Reranker | None = None,
        reranker_model_id: str = DEFAULT_RERANKER_ID,
    ):
        if not passages:
            raise ValueError("RerankedPassageRAGRetriever requires at least one passage")
        self.passages = passages
        self._base = PassageRAGRetriever(passages)
        self._pool_size = max(candidate_pool_size, 1)
        self._reranker = reranker or _CrossEncoderReranker(reranker_model_id)
        self.reranker_model_id = self._reranker.model_id

    def retrieve(
        self, question: str, *, top_k: int = 10, token_budget: int = 4000
    ) -> list[EvidenceItem]:
        # 1. Pull a larger candidate pool from the *same* TF-IDF index that
        #    Condition A uses. This is the only retrieval step that consumes
        #    the query; everything after is reranking.
        pool = self._base.retrieve(question, top_k=self._pool_size, token_budget=10**12)
        if not pool:
            return []

        # 2. Score every candidate with the pinned reranker.
        scores = self._reranker.score(question, [e.text for e in pool])
        if len(scores) != len(pool):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(pool)} candidates"
            )

        # 3. Take the best candidates subject to the *same* token budget and
        #    final top-k as Condition A.
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        selected: list[EvidenceItem] = []
        used_tokens = 0
        for idx in order:
            if len(selected) >= top_k:
                break
            candidate = pool[idx]
            cost = len(candidate.text.split())
            if selected and used_tokens + cost > token_budget:
                continue  # skip this one; keep scanning lower-ranked items
            used_tokens += cost
            selected.append(
                EvidenceItem(
                    span_id=candidate.span_id,
                    paper_id=candidate.paper_id,
                    text=candidate.text,
                    rank=len(selected),
                    score=float(scores[idx]),
                    char_start=candidate.char_start,
                    char_end=candidate.char_end,
                    evidence_type="passage",
                    provenance={
                        "retriever": self.name,
                        "reranker_model": self.reranker_model_id,
                        "candidate_pool_rank": idx,
                        "candidate_pool_score": candidate.score,
                        "candidate_pool_size": len(pool),
                    },
                )
            )
        return selected
