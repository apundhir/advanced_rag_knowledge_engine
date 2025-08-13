from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

from app.api.main import app


@pytest.mark.asyncio
async def test_api_sentence_window_ingest_and_query(tmp_path: Path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    sample = doc_dir / "sample.txt"
    sample.write_text("Alpha. Beta. Gamma. Delta. Epsilon.")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ingest",
            json={"paths": [str(doc_dir)], "mode": "sentence_window", "window_size": 1},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["documents_indexed"] == 1
        assert data["chunks_indexed"] >= 4

        q = await client.post("/query", json={"question": "What follows Beta?", "k": 3, "mode": "sentence_window"})
        assert q.status_code == 200
        body = q.json()
        assert "answer" in body
        assert isinstance(body["contexts"], list) and len(body["contexts"]) >= 1
