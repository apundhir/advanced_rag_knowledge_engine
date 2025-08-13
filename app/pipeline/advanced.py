from __future__ import annotations

from typing import Dict, List, Tuple

from app.llm.providers import generate_answer, generate_hypothetical_document
from app.retrieval.embeddings import EmbeddingModel
from app.ingestion.index import (
    SENTENCE_WINDOW_COLLECTION,
    query_top_k,
    query_top_k_with_embedding,
)
from app.retrieval.rerank import Reranker


def retrieve_with_hyde(question: str, k: int = 8, *, collection_name: str = SENTENCE_WINDOW_COLLECTION) -> List[Tuple[str, Dict[str, str], float]]:
    hyde_text = generate_hypothetical_document(question)
    embedder = EmbeddingModel()
    hyde_vec = embedder.embed_one(hyde_text)
    return query_top_k_with_embedding(hyde_vec, k=k, collection_name=collection_name)


def answer_with_hyde_and_rerank(question: str, k: int = 8, rerank_top_k: int = 5) -> Tuple[str, List[Tuple[str, Dict[str, str], float]]]:
    initial = retrieve_with_hyde(question, k=max(k, rerank_top_k))
    reranker = Reranker()
    reranked = reranker.rerank(question, initial)[:k]
    contexts = [t for t, _m, _s in reranked]
    answer = generate_answer(question, contexts)
    return answer, reranked
