"""Tests for FAQ single management, traveler questions, notice management, and model fallbacks."""

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from document_processing import (
    FAQEntry,
    add_single_faq_entry,
    delete_single_faq_entry,
    export_faq_csv,
    export_faq_json,
    export_faq_text,
    load_faq_entries,
    parse_faq_csv,
    parse_faq_json,
    parse_faq_text,
    parse_uploaded_faq,
    update_single_faq_entry,
)
from notice_manager import (
    Notice,
    add_notice,
    delete_notice,
    get_active_notices,
    get_pinned_notices,
    load_notices,
    update_notice,
)
from question_manager import (
    SubmittedQuestion,
    delete_submitted_question,
    dismiss_question,
    get_pending_questions,
    load_submitted_questions,
    mark_question_answered,
    save_submitted_questions,
    submit_question,
)
from rag_pipeline import (
    answer_question,
    extract_fallback_answer,
)
from vector_store import FAQSearchResult


class NoticeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_seed_notices_created_on_first_load(self):
        os.remove(self.file_path)
        notices = load_notices(self.file_path)
        self.assertGreaterEqual(len(notices), 3)
        self.assertTrue(any(n.is_pinned for n in notices))

    def test_add_and_retrieve_notice(self):
        add_notice(
            title="Campus Job Fair",
            content="Over 50 companies visiting campus next Wednesday.",
            category="Events",
            priority="Urgent",
            is_pinned=True,
            file_path=self.file_path,
        )
        notices = load_notices(self.file_path)
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].title, "Campus Job Fair")
        self.assertTrue(notices[0].is_pinned)
        self.assertIn("Campus Job Fair", notices[0].searchable_chunk)

    def test_update_notice(self):
        n = add_notice(
            title="Old Title",
            content="Old Content",
            file_path=self.file_path,
        )
        success = update_notice(
            notice_id=n.id,
            title="Updated Title",
            content="Updated Content",
            file_path=self.file_path,
        )
        self.assertTrue(success)
        updated = load_notices(self.file_path)[0]
        self.assertEqual(updated.title, "Updated Title")
        self.assertEqual(updated.content, "Updated Content")

    def test_delete_notice(self):
        n = add_notice(title="To Delete", content="Will be deleted", file_path=self.file_path)
        self.assertEqual(len(load_notices(self.file_path)), 1)
        deleted = delete_notice(n.id, file_path=self.file_path)
        self.assertTrue(deleted)
        self.assertEqual(len(load_notices(self.file_path)), 0)

    def test_pinned_notices_filter(self):
        add_notice(title="Normal Notice", content="Info", is_pinned=False, file_path=self.file_path)
        add_notice(title="Urgent Notice", content="Urgent Info", priority="Urgent", is_pinned=True, file_path=self.file_path)
        pinned = get_pinned_notices(self.file_path)
        self.assertEqual(len(pinned), 1)
        self.assertEqual(pinned[0].title, "Urgent Notice")


class TravelerQuestionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_submit_and_load_question(self):
        # Initialize with empty list
        save_submitted_questions([], self.file_path)
        q = submit_question(
            question="Is there swimming pool membership for travelers?",
            submitted_by="Rohan",
            category="Travel tips",
            file_path=self.file_path,
        )
        self.assertEqual(q.question, "Is there swimming pool membership for travelers?")
        self.assertEqual(q.status, "Pending")

        loaded = load_submitted_questions(self.file_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, q.id)

    def test_answer_and_dismiss_question(self):
        q = submit_question("How do I contact registrar?", file_path=self.file_path)
        success = mark_question_answered(q.id, "Visit building A room 102", file_path=self.file_path)
        self.assertTrue(success)
        loaded = load_submitted_questions(self.file_path)[0]
        self.assertEqual(loaded.status, "Answered")
        self.assertEqual(loaded.answer, "Visit building A room 102")

        dismiss_question(q.id, file_path=self.file_path)
        dismissed = load_submitted_questions(self.file_path)[0]
        self.assertEqual(dismissed.status, "Dismissed")

        delete_submitted_question(q.id, file_path=self.file_path)
        self.assertEqual(len(load_submitted_questions(self.file_path)), 0)


class SingleFAQCRUDTests(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        self.temp_file.write(b"Q: Existing Question\nA: Existing Answer\n")
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_add_single_faq(self):
        add_single_faq_entry(
            question="New Added Q",
            answer="New Added A",
            category="Admissions",
            file_path=self.file_path,
        )
        entries = load_faq_entries(self.file_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1].question, "New Added Q")
        self.assertEqual(entries[1].answer, "New Added A")

    def test_update_single_faq(self):
        success = update_single_faq_entry(
            index=0,
            question="Updated Q",
            answer="Updated A",
            category="Fees & payments",
            file_path=self.file_path,
        )
        self.assertTrue(success)
        entries = load_faq_entries(self.file_path)
        self.assertEqual(entries[0].question, "Updated Q")
        self.assertEqual(entries[0].answer, "Updated A")

    def test_delete_single_faq(self):
        success = delete_single_faq_entry(index=0, file_path=self.file_path)
        self.assertTrue(success)
        entries = load_faq_entries(self.file_path)
        self.assertEqual(len(entries), 0)


class FAQUploadAndExportTests(unittest.TestCase):
    def test_parse_faq_csv(self):
        csv_data = "question,answer,category\nWhat is your name?,TravelMate,General\nWhere is library?,Building C,Travel tips"
        entries = parse_faq_csv(csv_data)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].question, "What is your name?")
        self.assertEqual(entries[0].answer, "TravelMate")
        self.assertEqual(entries[1].category, "Travel tips")

    def test_parse_faq_json(self):
        json_data = json.dumps([
            {"question": "How to register?", "answer": "Go to portal.", "category": "Admissions"},
            {"question": "How to pay fees?", "answer": "Use bank counter.", "category": "Fees & payments"},
        ])
        entries = parse_faq_json(json_data)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].question, "How to register?")
        self.assertEqual(entries[1].answer, "Use bank counter.")

    def test_export_formats(self):
        entries = [
            FAQEntry(question="Q1", answer="A1", category="Admissions"),
            FAQEntry(question="Q2", answer="A2", category="Fees & payments"),
        ]
        txt = export_faq_text(entries)
        self.assertIn("Q: Q1\nA: A1", txt)
        csv_out = export_faq_csv(entries)
        self.assertIn("question,answer,category", csv_out)
        json_out = export_faq_json(entries)
        parsed = json.loads(json_out)
        self.assertEqual(len(parsed), 2)


class PipelineFallbackTests(unittest.TestCase):
    def setUp(self):
        self.source = FAQSearchResult(
            question="Library timing?",
            content="Question: Library timing?\nAnswer: 8 AM to 8 PM Monday to Saturday.",
            category="Travel tips",
            source="college_faq.txt",
            distance=0.1,
            similarity=0.9,
        )

    def test_extractive_fallback_extracts_answer(self):
        fallback = extract_fallback_answer([self.source])
        self.assertIn("8 AM to 8 PM Monday to Saturday.", fallback)

    def test_allow_extractive_fallback_when_gemini_fails(self):
        failing_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Gemini error"))
            )
        )
        ans = answer_question(
            "Library timing?",
            api_key="test-key",
            client=failing_client,
            retriever=lambda q, top_k: [self.source],
            allow_extractive_fallback=True,
        )
        self.assertTrue(ans.is_extractive_fallback)
        self.assertIn("8 AM to 8 PM", ans.answer)


if __name__ == "__main__":
    unittest.main()
