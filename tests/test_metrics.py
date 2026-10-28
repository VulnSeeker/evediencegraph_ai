import math

import pytest

from evidencegraph.evaluation.metrics import mean, mrr, recall_at_k


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], ["b", "d"], k=3) == 0.5


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_recall_at_k_none():
    assert recall_at_k(["a", "b"], ["c", "d"], k=2) == 0.0


def test_recall_at_k_undefined_when_no_gold():
    assert math.isnan(recall_at_k(["a"], [], k=1))


def test_mrr_first_hit():
    assert mrr(["a", "b", "c"], ["b"]) == 0.5


def test_mrr_no_hit():
    assert mrr(["a", "b"], ["z"]) == 0.0


def test_mean_ignores_nan():
    assert mean([1.0, math.nan, 2.0]) == 1.5


def test_mean_all_nan():
    assert math.isnan(mean([math.nan, math.nan]))
