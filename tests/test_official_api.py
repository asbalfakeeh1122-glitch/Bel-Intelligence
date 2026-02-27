import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_root_endpoint(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Bel Intelligence" in response.text


@pytest.mark.anyio
async def test_analyze_validation(client: AsyncClient):
    # Test short text (should fail validation)
    payload = {"text": "Short", "categories": ["A"]}
    response = await client.post("/api/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_chat_validation(client: AsyncClient):
    # Test missing payload
    response = await client.post("/api/chat", json={})
    assert response.status_code == 422
