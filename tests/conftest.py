import os
import sys

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure the root of the project is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app


@pytest.fixture(scope="session", autouse=True)
async def startup_models():
    """World-Tier: Ensure neural models are loaded once per test session."""
    from app.core.nlp_pipeline import nlp_service

    # Load all models for the session
    await nlp_service.load_models()
    return nlp_service


@pytest.fixture(scope="session")
def nlp_service(startup_models):
    """Fixture to provide the already-loaded nlp_service."""
    return startup_models


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
