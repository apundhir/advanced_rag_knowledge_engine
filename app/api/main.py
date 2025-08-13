from __future__ import annotations

import logging
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.health import health_payload
from app.core.logging import configure_logging, new_trace_id, trace_id_ctx
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrievedContext
from app.pipeline.baseline import (
    ingest_paths,
    ingest_sentence_windows,
    answer_question,
    answer_question_with_collection,
)
from app.pipeline.advanced import answer_with_hyde_and_rerank
from app.ingestion.index import SENTENCE_WINDOW_COLLECTION


def create_app() -> FastAPI:
    configure_logging()

    settings = get_settings()
    app = FastAPI(title=settings.app.name, version=settings.app.version)

    @app.middleware("http")
    async def add_trace_id_header(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or new_trace_id()
        # Store in context for log formatter
        trace_id_ctx.set(trace_id)
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        return response

    @app.get("/health", response_class=JSONResponse)
    async def health() -> Dict:
        return JSONResponse(content=health_payload())

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest) -> IngestResponse:
        if req.mode == "sentence_window":
            docs, chunks = ingest_sentence_windows(req.paths, window_size=req.window_size)
        else:
            docs, chunks = ingest_paths(req.paths, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        return IngestResponse(documents_indexed=docs, chunks_indexed=chunks)

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        if req.use_hyde or req.use_rerank:
            answer, retrieved = answer_with_hyde_and_rerank(req.question, k=req.k)
        elif req.mode == "sentence_window":
            answer, retrieved = answer_question_with_collection(req.question, k=req.k, collection_name=SENTENCE_WINDOW_COLLECTION)
        else:
            answer, retrieved = answer_question(req.question, k=req.k)
        contexts = [RetrievedContext(text=t, source=m.get("source"), score=s) for t, m, s in retrieved]
        return QueryResponse(answer=answer, contexts=contexts)

    return app


app = create_app()
