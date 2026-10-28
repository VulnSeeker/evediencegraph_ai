"""Retrieval metrics needed to accept/reject Milestone B.

Only recall@k and MRR are implemented here. Citation precision, contradiction
recall, refuted-claim endorsement rate, and cost accounting are deferred to
Milestone 4.
"""
from __future__ import annotations

import math


def recall_at_k(ranked_span_ids: list[str], gold_span_ids: list[str], k: int) -> float:
    """Fraction of gold evidence found in the first k ranked items.

    Returns NaN when there is no gold evidence for the question (undefined,
    not zero). Callers must filter NaN before aggregating.
    """
    if not gold_span_ids:
        return math.nan
    retrieved = set(ranked_span_ids[:k])
    gold = set(gold_span_ids)
    return len(retrieved & gold) / len(gold)


def mrr(ranked_span_ids: list[str], gold_span_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none found)."""
    gold = set(gold_span_ids)
    for i, sid in enumerate(ranked_span_ids, start=1):
        if sid in gold:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean, ignoring NaN. Returns NaN if no finite values."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)
