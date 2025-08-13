from __future__ import annotations

from typing import Dict, List, Tuple

from app.ingestion.loaders import chunk_text, load_documents
from app.ingestion.index import index_chunks, query_top_k, index_items, SENTENCE_WINDOW_COLLECTION
from app.ingestion.sentence_window import split_into_sentence_windows
from app.retrieval.embeddings import EmbeddingModel
from app.llm.providers import generate_answer


def ingest_paths(paths: List[str] | None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[int, int]:
    """Load documents from paths and index their chunks.

    Returns:
        Tuple[num_documents, num_chunks]
    """

    docs = load_documents(paths)
    all_chunks: List[Tuple[str, Dict[str, str]]] = []
    for text, meta in docs:
        parts = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for p in parts:
            all_chunks.append((p, meta))
    return index_chunks(all_chunks)


def ingest_sentence_windows(paths: List[str] | None, window_size: int = 2) -> Tuple[int, int]:
    """Index sentence-window documents.

    Embeddings are computed from the center sentence, but stored document text is the full window.
    """

    docs = load_documents(paths)
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    sentences: List[str] = []

    for text, meta in docs:
        windows = split_into_sentence_windows(text, window_size=window_size)
        for window_text, window_meta in windows:
            combined_meta: Dict[str, str] = {**meta, **window_meta}
            documents.append(window_text)
            metadatas.append(combined_meta)
            sentences.append(window_meta["sentence_text"])

    if not documents:
        return 0, 0

    embedder = EmbeddingModel()
    sentence_embeddings = embedder.embed(sentences)
    return index_items(documents, metadatas, embeddings=sentence_embeddings, collection_name=SENTENCE_WINDOW_COLLECTION)


def answer_question(question: str, k: int = 5) -> Tuple[str, List[Tuple[str, Dict[str, str], float]]]:
    retrieved = query_top_k(question, k=k)
    contexts = [t for t, _m, _s in retrieved]
    answer = generate_answer(question, contexts)
    return answer, retrieved


def answer_question_with_collection(question: str, k: int = 5, collection_name: str = "baseline") -> Tuple[str, List[Tuple[str, Dict[str, str], float]]]:
    retrieved = query_top_k(question, k=k, collection_name=collection_name)
    contexts = [t for t, _m, _s in retrieved]
    answer = generate_answer(question, contexts)
    return answer, retrieved
