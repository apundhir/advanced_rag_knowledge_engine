from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.api.types import Documents, Embeddings, IDs, Metadatas

from app.core.config import get_settings
from app.retrieval.embeddings import EmbeddingModel


DEFAULT_BASELINE_COLLECTION = "baseline"
SENTENCE_WINDOW_COLLECTION = "sentence_window"


def get_chroma_collection(name: str = DEFAULT_BASELINE_COLLECTION):
    settings = get_settings()
    persist_dir = Path(settings.db.chroma_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=name)
    return collection


def index_items(
    documents: List[str],
    metadatas: List[Dict[str, str]],
    embeddings: Optional[List[List[float]]] = None,
    *,
    collection_name: str = DEFAULT_BASELINE_COLLECTION,
) -> Tuple[int, int]:
    collection = get_chroma_collection(collection_name)
    if not documents:
        return 0, 0

    ids: IDs = [uuid.uuid4().hex for _ in documents]
    if embeddings is None:
        embedder = EmbeddingModel()
        embeddings = embedder.embed(list(documents))

    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    num_docs = len({m.get("source", str(i)) for i, m in enumerate(metadatas)})
    return num_docs, len(documents)


def index_chunks(chunks: List[Tuple[str, Dict[str, str]]], *, collection_name: str = DEFAULT_BASELINE_COLLECTION) -> Tuple[int, int]:
    documents: Documents = []
    metadatas: Metadatas = []
    for text, meta in chunks:
        documents.append(text)
        metadatas.append(meta)

    return index_items(documents, metadatas, embeddings=None, collection_name=collection_name)


def query_top_k(question: str, k: int = 5, *, collection_name: str = DEFAULT_BASELINE_COLLECTION) -> List[Tuple[str, Dict[str, str], float]]:
    collection = get_chroma_collection(collection_name)
    embedder = EmbeddingModel()
    qvec = embedder.embed_one(question)
    results = collection.query(query_embeddings=[qvec], n_results=k, include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    scored = []
    for text, meta, dist in zip(docs, metas, dists):
        score = float(1.0 / (1.0 + dist)) if dist is not None else None
        scored.append((text, meta, score if score is not None else 0.0))
    return scored


def query_top_k_with_embedding(query_embedding: List[float], k: int = 5, *, collection_name: str = DEFAULT_BASELINE_COLLECTION) -> List[Tuple[str, Dict[str, str], float]]:
    collection = get_chroma_collection(collection_name)
    results = collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    scored = []
    for text, meta, dist in zip(docs, metas, dists):
        score = float(1.0 / (1.0 + dist)) if dist is not None else None
        scored.append((text, meta, score if score is not None else 0.0))
    return scored
