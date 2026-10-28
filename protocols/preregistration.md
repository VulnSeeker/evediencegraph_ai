# Pre-registration (v0 — DRAFT)

## Research questions
- **Q1:** Does flat-claim retrieval change sentence-level citation precision
  or refuted-claim endorsement rate relative to passage RAG?
- **Q2:** Do graph relations add a meaningful benefit over the same atomic
  claims without graph traversal, on contradiction recall?
- **Q3:** How does performance differ by question type?

## Conditions
A) passage RAG, B) reranked passage RAG, C) flat-claim retrieval,
D) claim-graph retrieval. All share generator, prompt, token budget, and
answer schema.

## Primary outcome
- Q1: sentence-level citation precision (primary); refuted-claim endorsement
  rate (co-primary, Bonferroni-adjusted).
- Q2: contradiction recall on the pre-labelled contradiction subset.

## Smallest effect size of interest
To be justified from pilot data and prior work (Malmö thesis; Veritas).
**Do not set this margin to make results look favorable.**

## Statistical analysis
Paired comparisons by question; paired bootstrap CIs; equivalence test
(TOST) on the C-vs-D contrast with the registered margin. Non-significance
alone is not evidence of equivalence.

## Deviations log
(append here when any planned element changes)
