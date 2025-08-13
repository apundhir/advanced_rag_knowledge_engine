from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore


class Reranker:
    """Cross-encoder reranker with graceful fallback.

    If the model cannot be loaded (CI, offline), falls back to a lexical score
    based on simple token overlap.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = None
        try:
            if CrossEncoder is not None:
                self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def score(self, query: str, candidates: Sequence[str]) -> List[float]:
        if self.model is not None:
            pairs = [(query, c) for c in candidates]
            scores = self.model.predict(pairs).tolist()
            return [float(s) for s in scores]
        # Fallback lexical overlap
        q_tokens = set(query.lower().split())
        scores: List[float] = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            overlap = len(q_tokens & c_tokens)
            scores.append(float(overlap))
        return scores

    def rerank(self, query: str, items: Sequence[Tuple[str, dict, float]]) -> List[Tuple[str, dict, float]]:
        texts = [t for t, _m, _s in items]
        scores = self.score(query, texts)
        scored = list(zip(items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored]
