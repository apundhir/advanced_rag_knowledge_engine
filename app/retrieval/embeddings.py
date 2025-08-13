from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingModel:
    """Wrapper around SentenceTransformer for deterministic, simple use."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.astype(float).tolist() for v in vectors]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
