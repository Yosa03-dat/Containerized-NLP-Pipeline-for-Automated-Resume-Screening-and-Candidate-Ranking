"""
Unit tests for SQLite Database Storage and History Tracking (src/database.py).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    init_db,
    save_match_result,
    get_match_history,
    get_match_by_id,
    delete_match_record,
    clear_match_history,
    save_batch_session,
    get_batch_sessions,
    get_batch_candidates,
    get_database_stats,
    get_connection,
)


class TestDatabaseOperations(unittest.TestCase):
    """Tests for SQLite database schema initialization, CRUD actions, and statistics."""

    def setUp(self):
        # Create a temporary database file for each test
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name
        init_db(self.db_path)

    def tearDown(self):
        # Clean up temporary database file
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_init_db_creates_tables(self):
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.assertIn("match_history", tables)
        self.assertIn("batch_sessions", tables)
        self.assertIn("batch_candidates", tables)

    def test_save_and_get_match_result(self):
        mock_result = {
            "candidate_name": "Jordan Lee",
            "contact_info": {"email": "jordan@example.com", "phone": "123-456-7890"},
            "job_title": "Data Scientist",
            "composite_score": 88.5,
            "semantic_score": 85.0,
            "skill_score": 92.0,
            "match_tier": "Exceptional Fit",
            "tier_code": "exceptional",
            "matched_skills": ["python", "pytorch", "sql"],
            "missing_skills": ["docker"],
            "additional_skills": ["git"],
            "recommendations": [{"category": "tools", "suggestion": "Learn Docker"}],
            "category_breakdown": {"programming": {"matched": ["python", "sql"]}},
            "resume_word_count": 450,
            "job_word_count": 320,
        }

        rec_id = save_match_result(mock_result, notes="Interview recommended", db_path=self.db_path)
        self.assertIsInstance(rec_id, int)
        self.assertGreater(rec_id, 0)

        # Retrieve history
        history = get_match_history(db_path=self.db_path)
        self.assertEqual(len(history), 1)

        saved = history[0]
        self.assertEqual(saved["id"], rec_id)
        self.assertEqual(saved["candidate_name"], "Jordan Lee")
        self.assertEqual(saved["candidate_email"], "jordan@example.com")
        self.assertEqual(saved["composite_score"], 88.5)
        self.assertEqual(saved["notes"], "Interview recommended")

        # JSON deserialization checks
        self.assertEqual(saved["matched_skills"], ["python", "pytorch", "sql"])
        self.assertEqual(saved["missing_skills"], ["docker"])
        self.assertEqual(len(saved["recommendations"]), 1)

    def test_get_match_by_id(self):
        mock_result = {
            "candidate_name": "Taylor Swift",
            "job_title": "Audio Analyst",
            "composite_score": 75.0,
            "semantic_score": 70.0,
            "skill_score": 80.0,
            "match_tier": "Strong Fit",
            "tier_code": "strong",
            "matched_skills": ["python"],
        }
        rec_id = save_match_result(mock_result, db_path=self.db_path)

        fetched = get_match_by_id(rec_id, db_path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["candidate_name"], "Taylor Swift")

        # Non-existent ID returns None
        not_found = get_match_by_id(99999, db_path=self.db_path)
        self.assertIsNone(not_found)

    def test_search_and_filter_history(self):
        records = [
            {"candidate_name": "Alice Smith", "job_title": "Data Analyst", "composite_score": 90.0, "semantic_score": 90.0, "skill_score": 90.0, "match_tier": "Exceptional Fit", "tier_code": "exceptional"},
            {"candidate_name": "Bob Jones", "job_title": "DevOps Engineer", "composite_score": 45.0, "semantic_score": 40.0, "skill_score": 50.0, "match_tier": "Low Fit", "tier_code": "low"},
            {"candidate_name": "Charlie Brown", "job_title": "Data Scientist", "composite_score": 72.0, "semantic_score": 70.0, "skill_score": 75.0, "match_tier": "Strong Fit", "tier_code": "strong"},
        ]
        for r in records:
            save_match_result(r, db_path=self.db_path)

        # Search by query
        analyst_matches = get_match_history(search_query="Analyst", db_path=self.db_path)
        self.assertEqual(len(analyst_matches), 1)
        self.assertEqual(analyst_matches[0]["candidate_name"], "Alice Smith")

        # Filter by tier
        low_tier = get_match_history(tier_filter="low", db_path=self.db_path)
        self.assertEqual(len(low_tier), 1)
        self.assertEqual(low_tier[0]["candidate_name"], "Bob Jones")

    def test_delete_and_clear_history(self):
        rec_id = save_match_result(
            {"candidate_name": "To Delete", "job_title": "Temp", "composite_score": 50.0, "semantic_score": 50.0, "skill_score": 50.0, "match_tier": "Moderate Fit", "tier_code": "moderate"},
            db_path=self.db_path
        )
        self.assertTrue(delete_match_record(rec_id, db_path=self.db_path))
        self.assertEqual(len(get_match_history(db_path=self.db_path)), 0)

        # Populate two records and clear all
        save_match_result({"candidate_name": "A", "job_title": "Job", "composite_score": 60.0, "semantic_score": 60.0, "skill_score": 60.0, "match_tier": "Moderate Fit", "tier_code": "moderate"}, db_path=self.db_path)
        save_match_result({"candidate_name": "B", "job_title": "Job", "composite_score": 70.0, "semantic_score": 70.0, "skill_score": 70.0, "match_tier": "Strong Fit", "tier_code": "strong"}, db_path=self.db_path)
        self.assertEqual(len(get_match_history(db_path=self.db_path)), 2)

        clear_match_history(db_path=self.db_path)
        self.assertEqual(len(get_match_history(db_path=self.db_path)), 0)

    def test_save_and_get_batch_session(self):
        session_data = {
            "session_id": "session_test_001",
            "job_title": "Principal ML Engineer",
        }
        candidates = [
            {
                "rank": 1,
                "candidate_name": "Candidate Alpha",
                "email": "alpha@example.com",
                "phone": "555-0001",
                "composite_score": 91.2,
                "semantic_score": 90.0,
                "skill_score": 93.0,
                "match_tier": "Exceptional Fit",
                "tier_code": "exceptional",
                "matched_count": 8,
                "missing_count": 1,
            },
            {
                "rank": 2,
                "candidate_name": "Candidate Beta",
                "email": "beta@example.com",
                "phone": "555-0002",
                "composite_score": 76.5,
                "semantic_score": 75.0,
                "skill_score": 78.0,
                "match_tier": "Strong Fit",
                "tier_code": "strong",
                "matched_count": 6,
                "missing_count": 3,
            },
        ]

        sess_id = save_batch_session(session_data, candidates, db_path=self.db_path)
        self.assertEqual(sess_id, "session_test_001")

        # Get sessions
        sessions = get_batch_sessions(db_path=self.db_path)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "session_test_001")
        self.assertEqual(sessions[0]["top_candidate_name"], "Candidate Alpha")
        self.assertEqual(sessions[0]["total_candidates"], 2)

        # Get session candidates
        cands = get_batch_candidates("session_test_001", db_path=self.db_path)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["rank"], 1)
        self.assertEqual(cands[0]["candidate_name"], "Candidate Alpha")
        self.assertEqual(cands[1]["rank"], 2)
        self.assertEqual(cands[1]["candidate_name"], "Candidate Beta")

    def test_database_stats(self):
        save_match_result(
            {"candidate_name": "A", "job_title": "Job", "composite_score": 80.0, "semantic_score": 80.0, "skill_score": 80.0, "match_tier": "Strong Fit", "tier_code": "strong"},
            db_path=self.db_path
        )
        save_match_result(
            {"candidate_name": "B", "job_title": "Job", "composite_score": 90.0, "semantic_score": 90.0, "skill_score": 90.0, "match_tier": "Exceptional Fit", "tier_code": "exceptional"},
            db_path=self.db_path
        )

        stats = get_database_stats(db_path=self.db_path)
        self.assertEqual(stats["total_single_evaluations"], 2)
        self.assertEqual(stats["average_composite_score"], 85.0)
        self.assertEqual(stats["tier_counts"].get("strong"), 1)
        self.assertEqual(stats["tier_counts"].get("exceptional"), 1)


if __name__ == "__main__":
    unittest.main()
