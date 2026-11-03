"""Fast offline tests for FAQ processing and grounded answer generation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from document_processing import (
    EMBEDDING_DIMENSIONS,
    embed_faq_entries,
    load_faq_entries,
    parse_faq_text,
)
from rag_pipeline import (
    API_KEY_ERROR,
    NO_CONTEXT_ANSWER,
    GeminiResponseError,
    MissingAPIKeyError,
    answer_question,
)
from vector_store import FAQSearchResult


class FixedEncoder:
    """Small deterministic encoder stub for tests that need no model download."""

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)


class TravelMateTests(unittest.TestCase):
    def setUp(self):
        self.source = FAQSearchResult(
            question="How do I check my attendance?",
            content="Question: How do I check my attendance?\nAnswer: Review the traveler portal.",
            category="Traveler services",
            source="college_faq.txt",
            distance=0.1,
            similarity=0.9,
        )

    def test_sample_faqs_load_as_question_answer_chunks(self):
        entries = load_faq_entries()
        self.assertEqual(len(entries), 6)
        self.assertIn(entries[0].question, entries[0].content)
        self.assertIn(entries[0].answer, entries[0].content)

    def test_empty_and_malformed_faqs_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_faq_text("")
        with self.assertRaises(ValueError):
            parse_faq_text("Q: Question without an answer")

    def test_embeddings_have_one_384_dimension_vector_per_entry(self):
        entries = load_faq_entries()
        vectors = embed_faq_entries(entries, model=FixedEncoder())
        self.assertEqual(vectors.shape, (len(entries), EMBEDDING_DIMENSIONS))

    def test_rag_answer_uses_retrieved_faq_context(self):
        captured = {}
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: (
                    captured.update(kwargs)
                    or SimpleNamespace(text="Review your attendance in the traveler portal.")
                )
            )
        )

        result = answer_question(
            "Where can I see my attendance?",
            api_key="test-key",
            client=fake_client,
            retriever=lambda question, top_k: [self.source],
        )

        self.assertIn("traveler portal", result.answer)
        self.assertEqual(result.sources, [self.source])
        self.assertIn(self.source.content, captured["contents"])
        self.assertIn("Do not use outside knowledge", captured["contents"])
        self.assertIn("If the context does not provide enough information", captured["contents"])

    def test_empty_retrieval_declines_without_calling_gemini(self):
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: self.fail("Gemini should not be called")
            )
        )
        result = answer_question(
            "When is the semester break?",
            api_key="test-key",
            client=fake_client,
            retriever=lambda question, top_k: [],
        )
        self.assertEqual(result.answer, NO_CONTEXT_ANSWER)
        self.assertEqual(result.sources, [])

    def test_missing_api_key_has_a_clear_message(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(MissingAPIKeyError, API_KEY_ERROR):
                answer_question("Attendance?", retriever=lambda question, top_k: [self.source])

    def test_gemini_error_keeps_sources_available(self):
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
            )
        )
        with self.assertRaises(GeminiResponseError) as caught:
            answer_question(
                "Attendance?",
                api_key="test-key",
                client=fake_client,
                retriever=lambda question, top_k: [self.source],
            )
        self.assertEqual(caught.exception.sources, [self.source])


if __name__ == "__main__":
    unittest.main()
