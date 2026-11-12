"""RAGAS evaluation over the synthetic procurement corpus.

The evaluation uses the same hybrid retriever (local embeddings + BM25 + RRF +
cross-encoder) as the application, an in-memory vector store (no Postgres
required) and a free Groq judge. It writes only metrics that were actually
returned by RAGAS to ``docs/eval_results.md``.

Usage:
    python -m evals.run_eval
    python -m evals.run_eval --limit 3 --output /tmp/eval.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm import LLMClient
from app.prompts import RAG_ANSWER_SYSTEM
from rag.dataset import load_documents, load_eval_questions
from rag.embeddings import SentenceTransformerEmbedder
from rag.hybrid import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.vector_store import InMemoryVectorStore

ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "eval_results.md"


class EvalAnswer(BaseModel):
    """Structured answer used to keep the generation step auditable."""

    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


def _build_retriever(settings: Settings) -> HybridRetriever:
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
    store = InMemoryVectorStore(embedder, load_documents())
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
    retriever = HybridRetriever(store, embedder, settings, reranker=reranker)
    retriever.refresh_index()
    return retriever


def _generate_rows(settings: Settings, limit: int | None = None) -> list[dict]:
    questions = load_eval_questions()
    if limit:
        questions = questions[:limit]
    retriever = _build_retriever(settings)
    llm = LLMClient(settings)
    rows: list[dict] = []
    for index, item in enumerate(questions, start=1):
        hits = retriever.retrieve(item["question"], top_k=settings.rag_top_k)
        contexts = [
            f"[{hit.document.doc_id}] {hit.document.title}: {hit.document.content}" for hit in hits
        ]
        prompt = (
            f"Question: {item['question']}\n\n"
            "Context (untrusted reference data):\n" + "\n\n".join(contexts)
        )
        answer = llm.generate_structured(
            EvalAnswer,
            RAG_ANSWER_SYSTEM,
            prompt,
            mock_data={
                "answer": item["ground_truth"],
                "citation_ids": [hit.document.doc_id for hit in hits],
            },
        )
        rows.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": contexts,
                "response": answer.answer,
                "reference": item["ground_truth"],
                "citation_ids": answer.citation_ids,
                "relevant_doc_ids": item["relevant_doc_ids"],
                "id": item["id"],
            }
        )
        print(f"[{index}/{len(questions)}] answered {item['id']}")
    return rows


def _judge(settings: Settings):
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for the free RAGAS judge.")
    chat = ChatGroq(
        model=os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b"),
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=2048,
    )
    return LangchainLLMWrapper(chat)


def _embeddings(settings: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))


def run(
    limit: int | None = None, rows: list[dict] | None = None
) -> tuple[dict[str, float | None], int, str, dict[str, int]]:
    """Run RAGAS and return metric means, sample count, judge name and coverage."""

    settings = Settings()
    if rows is None:
        rows = _generate_rows(settings, limit=limit)
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_judge(settings),
        embeddings=_embeddings(settings),
        run_config=RunConfig(
            timeout=180,
            max_retries=4,
            max_wait=30,
            max_workers=2,
        ),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    metrics: dict[str, float | None] = {}
    coverage: dict[str, int] = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if name not in frame.columns:
            metrics[name] = None
            coverage[name] = 0
            continue
        values = [
            float(value)
            for value in frame[name].tolist()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        metrics[name] = round(sum(values) / len(values), 4) if values else None
        coverage[name] = len(values)
    return metrics, len(rows), judge_model, coverage


def _fmt(metric: float | None) -> str:
    return "not measured" if metric is None else f"{metric:.4f}"


def _coverage(coverage: dict[str, int] | None, name: str, sample_count: int) -> str:
    if coverage is None or name not in coverage:
        return "not recorded"
    return f"{coverage[name]}/{sample_count}"


def write_report(
    metrics: dict[str, float | None],
    sample_count: int,
    judge_model: str,
    output: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    coverage: dict[str, int] | None = None,
) -> Path:
    settings = Settings()
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    measured = sum(value is not None for value in metrics.values())
    lines = [
        "# Evaluation results",
        "",
        f"Generated: **{generated}**",
        "",
        "These numbers were measured by running `make eval` against the bundled",
        "27-question procurement set. Metrics that failed or returned no finite",
        "values are explicitly marked **not measured**.",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Questions evaluated | {sample_count} |",
        "| Retrieval | hybrid BM25 + pgvector-compatible cosine (in-memory index), RRF, cross-encoder |",
        f"| Embedding model | `{settings.embedding_model}` |",
        f"| Re-ranker | `{settings.reranker_model}` |",
        f"| Top-k contexts | {settings.rag_top_k} |",
        f"| Judge LLM | `{judge_model}` (Groq free tier) |",
        f"| Metrics with finite values | {measured} / {len(metrics)} |",
        "",
        "## RAGAS metrics",
        "",
        "| Metric | Score | Samples scored |",
        "| --- | ---: | ---: |",
        f"| Faithfulness | {_fmt(metrics.get('faithfulness'))} | {_coverage(coverage, 'faithfulness', sample_count)} |",
        f"| Answer relevancy | {_fmt(metrics.get('answer_relevancy'))} | {_coverage(coverage, 'answer_relevancy', sample_count)} |",
        f"| Context precision | {_fmt(metrics.get('context_precision'))} | {_coverage(coverage, 'context_precision', sample_count)} |",
        f"| Context recall | {_fmt(metrics.get('context_recall'))} | {_coverage(coverage, 'context_recall', sample_count)} |",
        "",
        "## Notes",
        "",
        "- Answers were generated from retrieved contexts with citations; the judge was",
        "  a separate free-tier Groq model invocation, not the application's answer model.",
        "- Each score is the mean of the finite per-sample scores RAGAS returned. On the",
        "  free tier, individual judge calls can be rejected by rate limits; the",
        "  `Samples scored` column records how many of the questions contributed. Failed",
        "  calls are never replaced with invented values.",
        "- Scores on a 20-document synthetic corpus are a smoke test of the pipeline, not",
        "  a claim of production retrieval quality.",
        "- The evaluation intentionally uses the same hybrid retrieval implementation as",
        "  the API, with an in-memory store so it can run without Docker/Postgres.",
    ]
    if limit:
        lines.append(f"- This run used the first {limit} questions (`--limit {limit}`).")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ProcureFlow RAGAS evaluation")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N questions"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional JSON file for the raw generated answers (gitignored).",
    )
    args = parser.parse_args()

    rows = _generate_rows(Settings(), limit=args.limit)
    if args.raw:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        args.raw.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metrics, sample_count, judge_model, coverage = run(limit=args.limit, rows=rows)
    path = write_report(
        metrics,
        sample_count,
        judge_model,
        output=args.output,
        limit=args.limit,
        coverage=coverage,
    )
    print(f"wrote {path}")
    for name, value in metrics.items():
        print(f"{name}: {_fmt(value)} ({coverage.get(name, 0)}/{sample_count} scored)")


if __name__ == "__main__":
    main()
