import pytest
from httpx import AsyncClient, ASGITransport

from app.api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "application" in data
        assert data["application"]["status"] == "healthy"
        assert "model" in data and "gpu" in data
