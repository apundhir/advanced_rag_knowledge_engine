from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

from app.api.main import app


@pytest.mark.asyncio
async def test_api_ingest_and_query(tmp_path: Path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    sample = doc_dir / "sample.txt"
    sample.write_text("FastAPI runs on Uvicorn. It is fast and easy to use.")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/ingest", json={"paths": [str(doc_dir)], "chunk_size": 64, "chunk_overlap": 16})
        assert r.status_code == 200
        data = r.json()
        assert data["documents_indexed"] == 1
        assert data["chunks_indexed"] >= 1

        q = await client.post("/query", json={"question": "What runs FastAPI?", "k": 3})
        assert q.status_code == 200
        body = q.json()
        assert "answer" in body
        assert isinstance(body["contexts"], list) and len(body["contexts"]) >= 1
