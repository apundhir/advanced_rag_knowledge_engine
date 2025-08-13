from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for ingestion endpoint."""

    paths: Optional[List[str]] = Field(
        default=None, description="List of files or directories to ingest. If omitted, use data/source_docs."
    )
    chunk_size: int = Field(default=1000, ge=32, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    mode: str = Field(default="baseline", description="baseline or sentence_window")
    window_size: int = Field(default=2, ge=0, le=10, description="Sentence window size on each side")


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=25)
    mode: str = Field(default="baseline", description="baseline or sentence_window")
    use_hyde: bool = Field(default=False)
    use_rerank: bool = Field(default=False)


class RetrievedContext(BaseModel):
    text: str
    source: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    contexts: List[RetrievedContext]
