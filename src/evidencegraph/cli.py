"""EvidenceGraph AI command-line interface.

Commands
--------
- ingest  : load papers from a directory, normalize, and write a JSONL store
- query   : run one question through a retrieval condition (A or B) with the
            stub generator; used for manual inspection and CI smoke tests

The claim-extraction, graph, and benchmark commands will be added in later
milestones and will follow the same pattern: a thin wrapper that assembles the
right condition and delegates to `evidencegraph.pipeline.answer_question`.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from evidencegraph.generation.provider import StubGenerator
from evidencegraph.ingestion.normalization import chunk_into_passages
from evidencegraph.ingestion.parsers import load_paper
from evidencegraph.pipeline import answer_question
from evidencegraph.retrieval.passage_rag import PassageRAGRetriever
from evidencegraph.schemas.eval import Question


@click.group()
def main() -> None:
    """EvidenceGraph AI — claim-level scientific QA evaluation."""


@main.command()
@click.argument(
    "papers_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("data/processed/papers.jsonl"),
    help="Destination JSONL file for normalized papers.",
)
def ingest(papers_dir: Path, out: Path) -> None:
    """Load every .txt/.pdf in PAPERS_DIR and write normalized papers to JSONL.

    Each paper receives a stable `paper_id` (P0000, P0001, ...) assigned in
    sorted filename order, so re-running ingest on the same directory produces
    the same IDs.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for path in sorted(papers_dir.iterdir()):
            if path.suffix.lower() not in {".txt", ".pdf"}:
                continue
            paper = load_paper(path, paper_id=f"P{n:04d}", title=path.stem)
            f.write(paper.model_dump_json() + "\n")
            n += 1
    click.echo(f"Wrote {n} papers to {out}")


@main.command()
@click.argument("question")
@click.option(
    "--papers",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/papers.jsonl"),
    help="Path to the normalized papers JSONL produced by `ingest`.",
)
@click.option(
    "--condition",
    type=click.Choice(["passage_rag", "reranked_passage_rag"]),
    default="passage_rag",
    help="Retrieval condition to run.",
)
@click.option(
    "--top-k",
    default=5,
    type=int,
    help="Final number of evidence items to return (identical across conditions).",
)
@click.option(
    "--candidate-pool",
    default=50,
    type=int,
    help="Candidate pool size for reranking (Condition B only).",
)
@click.option(
    "--token-budget",
    default=4000,
    type=int,
    help="Context token budget passed to the retriever (identical across conditions).",
)
def query(
    question: str,
    papers: Path,
    condition: str,
    top_k: int,
    candidate_pool: int,
    token_budget: int,
) -> None:
    """Run one question through the chosen retrieval condition.

    Uses the deterministic `StubGenerator`, so no LLM calls are made. This
    command exists for manual inspection and CI smoke tests; benchmark runs
    use `evidencegraph.experiments.run_experiment` (Milestone 4).
    """
    from evidencegraph.schemas.source import Paper

    doc_store: dict[str, str] = {}
    passages = []
    for line in papers.read_text(encoding="utf-8").splitlines():
        p = Paper.model_validate_json(line)
        doc_store[p.paper_id] = p.normalized_text
        passages.extend(chunk_into_passages(p.normalized_text, p.paper_id))

    if condition == "passage_rag":
        retriever = PassageRAGRetriever(passages)
    elif condition == "reranked_passage_rag":
        # Imported lazily: importing the module does not pull in
        # sentence-transformers until the reranker is actually constructed.
        from evidencegraph.retrieval.reranked_rag import (
            RerankedPassageRAGRetriever,
        )

        retriever = RerankedPassageRAGRetriever(
            passages, candidate_pool_size=candidate_pool
        )
    else:  # pragma: no cover - click.Choice rejects anything else
        raise ValueError(f"unknown condition: {condition}")

    q = Question(
        question_id="Q0001",
        text=question,
        question_type="consistent_evidence",
        top_k=top_k,
        token_budget=token_budget,
    )
    result = answer_question(q, retriever, StubGenerator(), doc_store)
    click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
