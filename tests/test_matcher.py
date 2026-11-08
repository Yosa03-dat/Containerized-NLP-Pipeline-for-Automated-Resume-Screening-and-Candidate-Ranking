"""
Unit tests for Skill Gap Analysis, Candidate Scoring, and Batch Ranking (src/matcher.py).
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.matcher import (
    analyze_skill_gap,
    compute_match_score,
    evaluate_resume_against_job,
    rank_candidates_batch,
    TIER_EXCEPTIONAL,
    TIER_STRONG,
    TIER_MODERATE,
)


class TestSkillGapAnalysis(unittest.TestCase):
    """Tests for skill gap identification and domain taxonomy categorization."""

    def test_skill_gap_identical(self):
        skills = ["python", "sql", "tableau"]
        gap = analyze_skill_gap(skills, skills)

        self.assertEqual(gap["matched_skills"], ["python", "sql", "tableau"])
        self.assertEqual(gap["missing_skills"], [])
        self.assertEqual(gap["additional_skills"], [])
        self.assertEqual(gap["skill_coverage"], 1.0)
        self.assertEqual(gap["total_matched_skills"], 3)
        self.assertEqual(gap["total_job_skills"], 3)
        self.assertEqual(gap["recommendations"], [])

    def test_skill_gap_disjoint(self):
        cand_skills = ["docker", "kubernetes", "go"]
        job_skills = ["python", "sql", "snowflake"]

        gap = analyze_skill_gap(cand_skills, job_skills)
        self.assertEqual(gap["matched_skills"], [])
        self.assertEqual(gap["missing_skills"], ["python", "snowflake", "sql"])
        self.assertEqual(gap["additional_skills"], ["docker", "go", "kubernetes"])
        self.assertEqual(gap["skill_coverage"], 0.0)
        self.assertGreater(len(gap["recommendations"]), 0)

    def test_skill_gap_partial(self):
        cand_skills = ["python", "sql", "git", "bash"]
        job_skills = ["python", "sql", "snowflake", "databricks"]

        gap = analyze_skill_gap(cand_skills, job_skills)
        self.assertEqual(gap["matched_skills"], ["python", "sql"])
        self.assertEqual(gap["missing_skills"], ["databricks", "snowflake"])
        self.assertEqual(gap["additional_skills"], ["bash", "git"])
        self.assertEqual(gap["skill_coverage"], 0.5)

    def test_empty_job_skills(self):
        cand_skills = ["python", "sql"]
        gap = analyze_skill_gap(cand_skills, [])

        self.assertEqual(gap["skill_coverage"], 1.0)
        self.assertEqual(gap["missing_skills"], [])

    def test_categorized_dict_inputs(self):
        cand_dict = {
            "programming": ["Python", "SQL"],
            "cloud_bigdata": ["AWS"],
        }
        job_dict = {
            "programming": ["python", "r"],
            "cloud_bigdata": ["aws", "snowflake"],
        }

        gap = analyze_skill_gap(cand_dict, job_dict)
        self.assertIn("python", gap["matched_skills"])
        self.assertIn("aws", gap["matched_skills"])
        self.assertIn("r", gap["missing_skills"])
        self.assertIn("snowflake", gap["missing_skills"])
        self.assertEqual(gap["skill_coverage"], 0.5)

        # Check category breakdown
        prog_cat = gap["category_breakdown"]["programming"]
        self.assertEqual(prog_cat["matched"], ["python"])
        self.assertEqual(prog_cat["missing"], ["r"])
        self.assertEqual(prog_cat["coverage"], 0.5)


class TestScoringAlgorithm(unittest.TestCase):
    """Tests for composite scoring, clamping, weights, and match tier classification."""

    def test_score_calculation(self):
        # 0.8 sim * 0.6 + 0.5 skill * 0.4 = 0.48 + 0.20 = 0.68 -> 68.0%
        res = compute_match_score(0.8, 0.5, weights=(0.6, 0.4))
        self.assertAlmostEqual(res["composite_score"], 68.0, places=1)
        self.assertEqual(res["semantic_score"], 80.0)
        self.assertEqual(res["skill_score"], 50.0)
        self.assertEqual(res["match_tier"], "Moderate Fit")
        self.assertEqual(res["tier_code"], "moderate")

        # 0.8 sim * 0.6 + 0.7 skill * 0.4 = 0.48 + 0.28 = 0.76 -> 76.0% (Strong Fit)
        res_strong = compute_match_score(0.8, 0.7, weights=(0.6, 0.4))
        self.assertAlmostEqual(res_strong["composite_score"], 76.0, places=1)
        self.assertEqual(res_strong["match_tier"], "Strong Fit")


    def test_match_tiers(self):
        res_exceptional = compute_match_score(0.95, 0.90)
        self.assertEqual(res_exceptional["tier_code"], "exceptional")

        res_strong = compute_match_score(0.75, 0.70)
        self.assertEqual(res_strong["tier_code"], "strong")

        res_moderate = compute_match_score(0.60, 0.50)
        self.assertEqual(res_moderate["tier_code"], "moderate")

        res_low = compute_match_score(0.20, 0.20)
        self.assertEqual(res_low["tier_code"], "low")

    def test_clamping_and_bounds(self):
        res = compute_match_score(1.5, -0.5)
        self.assertLessEqual(res["composite_score"], 100.0)
        self.assertGreaterEqual(res["composite_score"], 0.0)
        self.assertEqual(res["semantic_score"], 100.0)
        self.assertEqual(res["skill_score"], 0.0)

    def test_education_bonus(self):
        res_no_bonus = compute_match_score(0.7, 0.7, education_bonus=0.0)
        res_with_bonus = compute_match_score(0.7, 0.7, education_bonus=3.0)
        self.assertEqual(res_with_bonus["composite_score"], res_no_bonus["composite_score"] + 3.0)


class TestSingleEvaluationPipeline(unittest.TestCase):
    """Tests for end-to-end evaluation of a resume against a job posting."""

    def test_evaluate_resume_against_job(self):
        resume_text = (
            "Alex Morgan\n"
            "alex.morgan@example.com | (555) 234-5678\n"
            "github.com/alexmorgan\n\n"
            "Summary: Senior Data Scientist with 6 years experience in Python, SQL, PyTorch, and AWS.\n"
            "Education: Master of Science in Computer Science, Stanford University.\n"
            "Experience: Built machine learning and ETL pipelines using Spark and PostgreSQL."
        )

        job_text = (
            "Job Title: Senior Machine Learning Engineer\n"
            "We are seeking an ML Engineer skilled in Python, PyTorch, SQL, and AWS.\n"
            "Experience with Docker, Kubernetes, and Spark is preferred.\n"
            "Requires Master's degree in Computer Science."
        )

        result = evaluate_resume_against_job(
            resume_input=resume_text,
            job_input=job_text,
            job_title="Senior Machine Learning Engineer",
        )

        self.assertIn("Alex Morgan", result["candidate_name"])
        self.assertEqual(result["contact_info"]["email"], "alex.morgan@example.com")
        self.assertGreater(result["composite_score"], 50.0)
        self.assertIn("python", result["matched_skills"])
        self.assertIn("pytorch", result["matched_skills"])
        self.assertIn("aws", result["matched_skills"])
        self.assertIn("skill_gap", result)
        self.assertIn("recommendations", result)


class TestBatchCandidateRanking(unittest.TestCase):
    """Tests for batch screening, matrix evaluation, and leaderboard ranking."""

    def test_rank_candidates_batch(self):
        candidates = [
            {
                "name": "Jane Data (Strong Fit)",
                "text": "Data Analyst proficient in Python, SQL, Tableau, Power BI, Excel, and Snowflake.",
            },
            {
                "name": "Bob Frontend (Weak Fit)",
                "text": "Frontend React developer skilled in HTML, CSS, JavaScript, UI/UX design, and Figma.",
            },
            {
                "name": "Alice Intermediate (Moderate Fit)",
                "text": "Business Analyst familiar with Excel and basic SQL reporting.",
            },
        ]

        job_text = (
            "Target: Senior Data Analyst.\n"
            "Requirements: Strong proficiency in SQL, Python, Tableau, and Snowflake data warehousing."
        )

        ranked = rank_candidates_batch(
            resumes=candidates,
            job_input=job_text,
            job_title="Senior Data Analyst",
        )

        self.assertEqual(len(ranked), 3)
        # Jane should rank #1
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["candidate_name"], "Jane Data (Strong Fit)")
        # Jane's score should be highest
        self.assertGreater(ranked[0]["composite_score"], ranked[1]["composite_score"])
        self.assertGreater(ranked[1]["composite_score"], ranked[2]["composite_score"])
        # Bob Frontend should be #3
        self.assertEqual(ranked[2]["candidate_name"], "Bob Frontend (Weak Fit)")

    def test_batch_top_k(self):
        candidates = [
            {"name": f"Candidate {i}", "text": f"Skills: Python, SQL {i}"}
            for i in range(5)
        ]
        ranked = rank_candidates_batch(
            resumes=candidates,
            job_input="Looking for Python and SQL developer.",
            top_k=2,
        )
        self.assertEqual(len(ranked), 2)


if __name__ == "__main__":
    unittest.main()
