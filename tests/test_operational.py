import os

import psutil
import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_server_operational(client: AsyncClient):
    """Verify the server is up and responsive."""
    response = await client.get("/")
    assert response.status_code == 200


def test_nlp_models_initialized():
    """Verify that the NLP singleton is initialized."""
    from app.core.nlp_pipeline import nlp_service

    # We check if the instance exists and the basic structure is there.
    # Note: In a unit test context, models might not be fully loaded unless setup runs,
    # but the service object itself should be initialized.
    assert nlp_service is not None
    assert nlp_service._initialized is True


def test_docs_exist():
    """Verify that the overview documentation exists."""
    # Check both relative and absolute paths for robustness in different environments
    doc_path = "docs/overview.md"
    assert os.path.exists(doc_path) or os.path.exists(os.path.join("/app", doc_path))
