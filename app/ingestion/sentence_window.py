from __future__ import annotations

import re
from typing import Dict, List, Tuple

from syntok.segmenter import process


def _extract_sentences_with_syntok(text: str) -> List[str]:
    sentences: List[str] = []
    for paragraph in process(text):
        for sentence in paragraph:
            # Join token values and fix spacing before punctuation
            s = " ".join(token.value for token in sentence).strip()
            s = re.sub(r"\s+([\.,;:!\?])", r"\1", s)
            if s:
                sentences.append(s)
    return sentences


def _fallback_regex_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_into_sentence_windows(text: str, window_size: int = 2) -> List[Tuple[str, Dict[str, str]]]:
    """Split text into sentences and create a window of surrounding sentences for each.

    Returns a list of tuples (window_text, metadata) where metadata contains:
    - sentence_text: the center sentence
    - s_idx: start sentence index in the original text
    - e_idx: end sentence index in the original text
    """

    if window_size < 0:
        window_size = 0

    sentences = _extract_sentences_with_syntok(text)
    if len(sentences) <= 1:
        sentences = _fallback_regex_split(text)

    windows: List[Tuple[str, Dict[str, str]]] = []
    for i, center in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences) - 1, i + window_size)
        window_text = " ".join(sentences[start : end + 1])
        meta: Dict[str, str] = {
            "sentence_text": center,
            "s_idx": str(start),
            "e_idx": str(end),
        }
        windows.append((window_text, meta))

    return windows
