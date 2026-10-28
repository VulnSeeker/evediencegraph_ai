"""Minimal CLI. Extend with `extract-claims`, `build-graph`, `benchmark` later."""
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
@click.argument("papers_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/processed/papers.jsonl"))
def ingest(papers_dir: Path, out: Path) -> None:
    """Load every .txt/.pdf in PAPERS_DIR and write normalized papers to JSONL."""
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
@click.option("--papers", type=click.Path(exists=True, path_type=Path),
              default=Path("data/processed/papers.jsonl"))
@click.option("--top-k", default=5, type=int)
def query(question: str, papers: Path, top_k: int) -> None:
    """Run one question through the passage-RAG baseline with the stub generator."""
    from evidencegraph.schemas.source import Paper

    doc_store: dict[str, str] = {}
    passages = []
    for line in papers.read_text(encoding="utf-8").splitlines():
        p = Paper.model_validate_json(line)
        doc_store[p.paper_id] = p.normalized_text
        passages.extend(chunk_into_passages(p.normalized_text, p.paper_id))

    retriever = PassageRAGRetriever(passages)
    q = Question(question_id="Q0001", text=question,
                 question_type="consistent_evidence", top_k=top_k)
    result = answer_question(q, retriever, StubGenerator(), doc_store)
    click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
