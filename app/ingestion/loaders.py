from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pypdf import PdfReader


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        content = page.extract_text() or ""
        texts.append(content)
    return "\n".join(texts)


def discover_documents(paths: List[str] | None) -> List[Path]:
    base_paths = paths or [str(Path("data/source_docs").resolve())]
    collected: List[Path] = []
    for base in base_paths:
        p = Path(base)
        if p.is_file():
            collected.append(p)
        elif p.is_dir():
            for ext in (".txt", ".md", ".pdf"):
                collected.extend(p.rglob(f"*{ext}"))
    return collected


def load_documents(paths: List[str] | None) -> List[Tuple[str, Dict[str, str]]]:
    docs: List[Tuple[str, Dict[str, str]]] = []
    for path in discover_documents(paths):
        text = ""
        if path.suffix.lower() in (".txt", ".md"):
            text = _read_text_file(path)
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf_file(path)
        if text.strip():
            docs.append((text, {"source": str(path)}))
    return docs


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
