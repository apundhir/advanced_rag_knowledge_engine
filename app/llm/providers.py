from __future__ import annotations

import os
from typing import List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def generate_answer(question: str, contexts: List[str]) -> str:
    """Generate an answer using OpenAI if available, else return a fallback summary.

    This keeps tests deterministic without requiring network access.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    joined = "\n\n".join(contexts)[:8000]

    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a helpful assistant. Answer based ONLY on the provided context.\n\n"
            f"Question: {question}\n\nContext:\n{joined}\n\nAnswer:"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    # Fallback: return the first lines as a pseudo-summary
    if not joined.strip():
        return "No relevant context found."
    preview = joined.splitlines()[:5]
    return " ".join(preview)[:1000]


def generate_hypothetical_document(question: str) -> str:
    """Generate a HyDE-style hypothetical document for a query.

    Uses OpenAI if configured; otherwise, returns a deterministic heuristic expansion.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Write a concise, factual paragraph that would directly answer the question."
            " Avoid speculation and focus on keywords that are likely present in relevant documents.\n\n"
            f"Question: {question}\n\nHypothetical answer:"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=160,
            )
            return resp.choices[0].message.content or question
        except Exception:
            pass

    # Fallback: simple keyword-focused template
    return f"This text describes: {question}. Definitions, key properties, examples, usage, related terms, and context."
