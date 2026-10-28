"""Paper loading. Supports .txt out of the box and .pdf if pymupdf is installed."""
from __future__ import annotations

from pathlib import Path

from evidencegraph.ingestion.normalization import normalize_text
from evidencegraph.schemas.source import Paper, sha256_text


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PDF support requires `pip install 'evidencegraph[pdf]'`"
        ) from e
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def load_paper(path: Path, paper_id: str, **metadata) -> Paper:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        raw = _read_txt(path)
    elif suffix == ".pdf":
        raw = _read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    normalized = normalize_text(raw)
    return Paper(
        paper_id=paper_id,
        title=metadata.get("title", path.stem),
        doi=metadata.get("doi"),
        year=metadata.get("year"),
        domain=metadata.get("domain"),
        source_uri=metadata.get("source_uri"),
        normalized_text=normalized,
        text_hash=sha256_text(normalized),
    )
