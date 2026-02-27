import pytest

from app.core.nlp_pipeline import nlp_service


@pytest.mark.asyncio
async def test_intent_locking_logic():
    """Verify Expert Protocol intent classification."""
    context = "The company reported a 15% increase in revenue for the fiscal year 2023."

    # NUMERIC Intent Detection
    intent_numeric = nlp_service._classify_intent(
        "How much was the revenue increase?", context
    )
    assert intent_numeric == "NUMERIC"

    # FACT Intent Detection
    intent_fact = nlp_service._classify_intent("What did the company report?", context)
    assert intent_fact == "CLASSIFICATION"


@pytest.mark.asyncio
async def test_classification_reranking():
    """Verify Precision Reranking Protocol."""
    text = "Our football team won the championship game last night."
    categories = ["Sports", "Business", "Finance"]

    results = await nlp_service.analyze(text, categories)

    # Sports should be primary
    assert results["primary_domain"] == "Sports"
    assert results["intent"] == "CLASSIFICATION"
    assert len(results["evidence_quotes"]) > 0


@pytest.mark.asyncio
async def test_evidence_verbatim():
    """Verify EVIDENCE intent returns quoted verbatim strings."""
    context = "Project Antigravity is the leading agentic AI framework."
    question = "quote the line about Antigravity"

    res = await nlp_service.ask(context, question)

    assert res["intent"] == "EVIDENCE"
    assert any("Antigravity" in q for q in res["evidence"])
