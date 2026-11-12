"""Deterministic parsing and RAG unit tests."""

from agents.orchestrator import deterministic_parse
from rag.bm25 import BM25Index
from rag.dataset import load_documents, load_eval_questions
from rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.types import Document, RetrievalHit, hit_to_citation
from rag.vector_store import InMemoryVectorStore


def test_deterministic_parse_extracts_laptop_request():
    request, tasks, confidence = deterministic_parse("Buy 20 laptops, budget 30k USD")
    assert request.item == "laptops"
    assert request.quantity == 20
    assert request.total_amount_usd == 30000
    assert request.category == "laptops"
    assert [task.agent for task in tasks] == ["policy", "suppliers", "budget"]
    assert confidence > 0.8


def test_deterministic_parse_handles_million_and_missing_amount():
    request, _, confidence = deterministic_parse("Purchase 2 servers, budget $1.5m")
    assert request.quantity == 2
    assert request.total_amount_usd == 1_500_000
    assert confidence > 0.8

    unknown, _, low_confidence = deterministic_parse("Please buy office supplies")
    assert unknown.total_amount_usd == 0
    assert low_confidence < 0.6


def test_corpus_has_expected_documents_and_valid_eval_references():
    documents = load_documents()
    assert len(documents) == 20
    assert {document.doc_type for document in documents} == {"supplier", "policy"}
    doc_ids = {document.doc_id for document in documents}
    questions = load_eval_questions()
    assert 25 <= len(questions) <= 30
    for question in questions:
        assert set(question["relevant_doc_ids"]).issubset(doc_ids)


def test_bm25_and_hybrid_retrieval_return_citations(settings):
    documents = load_documents()
    index = BM25Index(documents)
    hits = index.search("NovaTech laptop warranty", k=3)
    assert hits
    assert any(hit.document.doc_id == "SUP-LAP-001" for hit in hits)

    # Build a hash embedder explicitly for a deterministic, offline retrieval test.
    from rag.embeddings import HashEmbedder

    embedder = HashEmbedder(settings.embedding_dim)
    store = InMemoryVectorStore(embedder, documents)
    retriever = HybridRetriever(store, embedder, settings)
    retriever.refresh_index()
    hybrid_hits = retriever.retrieve("NovaTech laptop", top_k=3, doc_type="supplier")
    assert hybrid_hits
    citation = hit_to_citation(hybrid_hits[0])
    assert citation.doc_id.startswith("SUP-")
    assert citation.doc_id in {hit.document.doc_id for hit in hybrid_hits}


def test_reciprocal_rank_fusion_merges_channels(settings):
    a = Document("A", "Alpha", "alpha", "policy", "test")
    b = Document("B", "Beta", "beta", "supplier", "test")
    channel_one = [
        RetrievalHit(a, 0.9, ("vector",)),
        RetrievalHit(b, 0.8, ("vector",)),
    ]
    channel_two = [RetrievalHit(b, 0.7, ("bm25",))]
    fused = reciprocal_rank_fusion([channel_one, channel_two], rrf_k=60)
    assert [hit.document.doc_id for hit in fused] == ["B", "A"]
    assert set(fused[0].channels) == {"vector", "bm25"}
    assert set(fused[1].channels) == {"vector"}
