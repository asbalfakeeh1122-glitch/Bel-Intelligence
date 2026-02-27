import unittest
from unittest.mock import MagicMock, patch

from app.core.nlp_pipeline import NLPService


class TestNLPService(unittest.TestCase):
    def setUp(self):
        # We don't want to load provided models, so we mock __new__ or just manually instantiate
        # Since it's a singleton, we need to be careful.
        # For unit testing, we can just manipulate the instance.
        self.service = NLPService()
        self.service.embedding_model = MagicMock()
        self.service.qa_pipeline = MagicMock()
        self.service.executor = None  # We will mock run_in_executor

    def test_chunk_text(self):
        """Test that text is split into chunks with overlap."""
        # Create a text with clearly defined sentences that are long enough (> 30 chars for 4 sentences)
        sentence = "This is a long sentence that exceeds the minimum character limit for chunking. "
        text = sentence * 6

        # sentences_per_chunk=4, overlap=3
        chunks = self.service._chunk_text_recursive(text, chunk_size=300, overlap=50)

        self.assertTrue(len(chunks) >= 2)
        self.assertIn(
            sentence.strip(), chunks[1]
        )  # Sentence should be in both (overlap)

    @patch("app.core.nlp_pipeline.util.cos_sim")
    @patch("app.core.nlp_pipeline.torch")
    def test_rag_flow(self, mock_torch, mock_sim):
        # 1. Mock Bi-Encoder Fast Retrieval
        mock_sim.return_value = [[0.8]]

        # 2. Mock Reranker (Cross-Encoder)
        self.service.cross_encoder = MagicMock()
        self.service.cross_encoder.predict.return_value = [0.99]  # Cross-score

        # Mocking topk return object (Stage 1)
        mock_topk_result = MagicMock()
        mock_topk_result.indices = [MagicMock(item=lambda: 0)]
        mock_topk_result.values = [MagicMock(item=lambda: 0.99)]
        mock_torch.topk.return_value = mock_topk_result

        # 3. Mock QA pipeline to return a real score
        self.service.qa_pipeline.return_value = {
            "answer": "Paris",
            "score": 0.99,
            "start": 0,
            "end": 5,
        }

        # Mock chunking to return at least one chunk so index 0 exists
        with patch.object(
            self.service, "_chunk_text_recursive", return_value=["Paris is capital."]
        ):
            response = self.service._rag_pipeline("Paris is capital.", "Question?")

        self.assertEqual(response["answer"], "Paris")
        self.assertIn("evidence", response)
        self.assertIn("Paris", response["evidence"])


if __name__ == "__main__":
    unittest.main()
