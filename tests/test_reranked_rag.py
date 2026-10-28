from evidencegraph.ingestion.normalization import Passage
from evidencegraph.retrieval.passage_rag import PassageRAGRetriever
from evidencegraph.retrieval.reranked_rag import RerankedPassageRAGRetriever


class _PreferredMarkerReranker:
    """Mock reranker: scores 1.0 for passages containing `marker`, else 0.0.

    Used to simulate a semantically correct reranker without downloading a
    cross-encoder model. Does not require sentence-transformers.
    """

    def __init__(self, marker: str = "GOLD_MARKER"):
        self.model_id = f"test:preferred-marker:{marker}"
        self._marker = marker

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 if self._marker in t else 0.0 for t in texts]


def _corpus() -> list[Passage]:
    # Distractor contains every query keyword -> high TF-IDF score.
    # Gold uses different vocabulary but is what a real reranker would prefer.
    distractor = Passage(
        passage_id="P_distract",
        paper_id="P1",
        text="query keywords match here completely with lexical overlap.",
        char_start=0,
        char_end=58,
    )
    gold = Passage(
        passage_id="P_gold",
        paper_id="P2",
        text="The GOLD_MARKER sentence is semantically relevant but uses different words.",
        char_start=0,
        char_end=71,
    )
    filler = Passage(
        passage_id="P_fill",
        paper_id="P3",
        text="unrelated filler text.",
        char_start=0,
        char_end=22,
    )
    return [distractor, gold, filler]


def test_reranker_flips_tfidf_ranking():
    passages = _corpus()
    question = "query keywords"

    # Condition A: lexical TF-IDF puts the distractor first.
    a_top = PassageRAGRetriever(passages).retrieve(question, top_k=1)[0]
    assert a_top.span_id == "P_distract"

    # Condition B: the reranker prefers the semantically correct passage.
    reranker = _PreferredMarkerReranker(marker="GOLD_MARKER")
    b_top = RerankedPassageRAGRetriever(
        passages, candidate_pool_size=5, reranker=reranker
    ).retrieve(question, top_k=1)[0]
    assert b_top.span_id == "P_gold"


def test_reranked_rag_preserves_span_metadata():
    passages = _corpus()
    reranker = _PreferredMarkerReranker()
    r = RerankedPassageRAGRetriever(passages, reranker=reranker)
    items = r.retrieve("query keywords", top_k=3)
    for item in items:
        assert item.char_start >= 0
        assert item.char_end > item.char_start
        source = next(p for p in passages if p.passage_id == item.span_id)
        assert item.text == source.text
        assert item.char_start == source.char_start
        assert item.char_end == source.char_end


def test_reranked_rag_records_candidate_pool_rank():
    passages = _corpus()
    reranker = _PreferredMarkerReranker()
    r = RerankedPassageRAGRetriever(passages, candidate_pool_size=5, reranker=reranker)
    items = r.retrieve("query keywords", top_k=3)
    for item in items:
        prov = item.provenance
        assert prov["retriever"] == "reranked_passage_rag"
        assert "candidate_pool_rank" in prov
        assert "candidate_pool_score" in prov
        assert prov["reranker_model"].startswith("test:")


def test_reranked_rag_respects_top_k():
    passages = _corpus()
    reranker = _PreferredMarkerReranker()
    r = RerankedPassageRAGRetriever(passages, reranker=reranker)
    items = r.retrieve("query keywords", top_k=2)
    assert len(items) == 2
    assert [i.rank for i in items] == [0, 1]


def test_reranked_rag_respects_token_budget():
    passages = _corpus()
    reranker = _PreferredMarkerReranker()
    r = RerankedPassageRAGRetriever(passages, reranker=reranker)
    items = r.retrieve("query keywords", top_k=10, token_budget=5)
    # First item always fits (matches Condition A behavior); subsequent items
    # must fit the budget.
    total = sum(len(i.text.split()) for i in items)
    assert total <= max(len(items[0].text.split()), 5) + 1e-9


def test_candidate_pool_must_be_at_least_one():
    passages = _corpus()
    reranker = _PreferredMarkerReranker()
    r = RerankedPassageRAGRetriever(
        passages, candidate_pool_size=0, reranker=reranker
    )
    # Normalizes to 1 rather than crashing.
    assert r.retrieve("query keywords", top_k=1)
