"""Load bundled documents and evaluation questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.types import Document

RAG_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DOCUMENTS_PATH = RAG_DATA_DIR / "documents.jsonl"
EVAL_QUESTIONS_PATH = Path(__file__).parent.parent / "evals" / "eval_questions.jsonl"


def load_documents(path: str | Path | None = None) -> list[Document]:
    """Read the JSONL corpus into validated internal documents."""

    source_path = Path(path) if path else DEFAULT_DOCUMENTS_PATH
    documents: list[Document] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            try:
                documents.append(
                    Document(
                        doc_id=raw["id"],
                        title=raw["title"],
                        content=raw["content"],
                        doc_type=raw["doc_type"],
                        source=raw["source"],
                        metadata=raw.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Invalid document at line {line_number}: missing {exc}") from exc
    if not documents:
        raise ValueError(f"No documents found in {source_path}")
    return documents


def load_eval_questions(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the RAGAS question set (question, ground truth, relevant doc ids)."""

    source_path = Path(path) if path else EVAL_QUESTIONS_PATH
    with source_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
