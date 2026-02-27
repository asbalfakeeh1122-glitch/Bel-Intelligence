import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

# Create a test client
client = TestClient(app)


class TestAPIIntegration(unittest.TestCase):

    # Patch the service exactly where it is used in routes.py
    @patch("app.api.routes.nlp_service.analyze")
    def test_analyze_endpoint(self, mock_analyze):
        # Mock valid response matching AnalyzeResponse schema
        mock_analyze.return_value = {
            "intent": "CLASSIFICATION",
            "primary_domain": "Legal",
            "secondary_domains": ["Finance"],
            "excluded_domains": ["Sports"],
            "reasoning": "Document exhibits legal structures.",
            "evidence_quotes": ["Verbatim quote."],
            "numerical_insights": [],
        }

        payload = {
            "text": "x" * 2500,
            "categories": ["Legal", "Finance"],
        }

        response = client.post("/api/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["primary_domain"], "Legal")

    @patch("app.api.routes.nlp_service.ask")
    def test_chat_endpoint(self, mock_ask):
        # Mock chat response
        mock_ask.return_value = {
            "intent": "FACT",
            "answer": "France",
            "evidence": "Paris is capital of France",
        }

        payload = {"context": "Paris is capital of France", "question": "Where?"}
        response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "France")

    def test_input_validation(self):
        # Test too short input triggers validation error
        # Assuming AnalyzeRequest has a min_length constraint on text
        payload = {"text": "Short", "categories": ["A"]}
        response = client.post("/api/analyze", json=payload)
        # Note: If there's no Pydantic validation for length, this might return 200.
        # But based on original test, it expected 422.
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
