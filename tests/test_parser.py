"""
Unit tests for Document Parser and Entity Extraction (src/parser.py).
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import (
    normalize_unicode_and_ligatures,
    clean_text,
    clean_job_description,
    extract_contact_info,
    extract_candidate_name,
    extract_education,
    extract_skills,
    parse_resume,
    parse_job_description,
    SKILLS_TAXONOMY,
)


class TestTextNormalization(unittest.TestCase):
    """Tests for unicode, ligatures, and character standardization."""

    def test_ligature_normalization(self):
        corrupted = "We offer great bene\ufb01ts, e\ufb03cient workflows, and pro\ufb01t sharing."
        expected = "We offer great benefits, efficient workflows, and profit sharing."
        self.assertEqual(normalize_unicode_and_ligatures(corrupted), expected)

    def test_html_unescape_and_quotes(self):
        raw = "Data &amp; Analytics &mdash; &ldquo;Machine Learning&rdquo; \u2013 Specialist"
        normalized = normalize_unicode_and_ligatures(raw)
        self.assertIn("Data & Analytics", normalized)
        self.assertIn('"Machine Learning"', normalized)

    def test_clean_text_whitespace_and_bullets(self):
        raw = """
        • Developed machine learning models in Python.
        * Maintained PostgreSQL databases.
        - Deployed microservices to AWS.
        \t\t
        
        
        Key Achievements:
        """
        cleaned = clean_text(raw)
        self.assertNotIn("•", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertIn("Developed machine learning models in Python.", cleaned)
        self.assertIn("Maintained PostgreSQL databases.", cleaned)
        # Check no more than 2 consecutive newlines
        self.assertNotIn("\n\n\n", cleaned)


class TestBoilerplateRemoval(unittest.TestCase):
    """Tests for job description boilerplate removal."""

    def test_strip_eeo_and_benefits(self):
        job_raw = """
        Job Title: Senior Data Analyst
        Responsibilities: Build Power BI dashboards, write complex SQL queries, and perform statistical analysis in Python.
        We are an equal opportunity employer. All qualified applicants will receive consideration without regard to race, color, religion, sex.
        Benefits: 401(k) matching, comprehensive medical, dental, and vision insurance.
        Apply at https://careers.example.com/jobs/123.
        """
        cleaned = clean_job_description(job_raw)
        self.assertIn("Power BI dashboards", cleaned)
        self.assertIn("complex SQL queries", cleaned)
        self.assertNotIn("equal opportunity employer", cleaned.lower())
        self.assertNotIn("401(k)", cleaned.lower())
        self.assertNotIn("https://careers.example.com", cleaned)


class TestContactInfoExtraction(unittest.TestCase):
    """Tests for candidate name and contact information extraction."""

    def test_extract_contact_info(self):
        resume_sample = """
        Jane Doe
        Email: jane.doe@example.com | Phone: (555) 123-4567
        LinkedIn: https://www.linkedin.com/in/janedoe-data
        GitHub: github.com/janedoe
        Portfolio: https://janedoe.dev
        """
        contact = extract_contact_info(resume_sample)
        self.assertEqual(contact["name"], "Jane Doe")
        self.assertEqual(contact["email"], "jane.doe@example.com")
        self.assertEqual(contact["phone"], "(555) 123-4567")
        self.assertEqual(contact["linkedin"], "https://www.linkedin.com/in/janedoe-data")
        self.assertEqual(contact["github"], "https://github.com/janedoe")
        self.assertIn("https://janedoe.dev", contact["urls"])


class TestEducationExtraction(unittest.TestCase):
    """Tests for academic degrees and education history extraction."""

    def test_extract_bachelor_and_master(self):
        resume_sample = """
        EDUCATION
        Master of Science in Data Science - Stanford University, 2022
        Bachelor of Science in Computer Science - University of California, Berkeley, 2020
        """
        edu = extract_education(resume_sample)
        self.assertGreaterEqual(len(edu), 2)
        degrees = [e["degree"] for e in edu]
        fields = [e["field"] for e in edu]
        years = [e["year"] for e in edu]

        self.assertIn("Master", degrees)
        self.assertIn("Bachelor", degrees)
        self.assertIn("Data Science", fields)
        self.assertIn("Computer Science", fields)
        self.assertIn("2022", years)
        self.assertIn("2020", years)


class TestSkillsExtraction(unittest.TestCase):
    """Tests for domain skill taxonomy matching."""

    def test_extract_skills_multi_category(self):
        text = """
        Proficient in Python, SQL, and R.
        Experienced with Tableau, Power BI, and Excel.
        Cloud architecture on AWS and Snowflake with Apache Spark for ETL data pipelines.
        PostgreSQL and MongoDB databases.
        Built Machine Learning models with Scikit-Learn, Regression, and A/B Testing.
        Tools: Git, Docker, Jira, Agile methodologies. Also know C++ and C#.
        """
        res = extract_skills(text)
        all_skills = res["all_skills"]

        # Programming
        self.assertIn("python", all_skills)
        self.assertIn("sql", all_skills)
        self.assertIn("r", all_skills)
        self.assertIn("c++", all_skills)
        self.assertIn("c#", all_skills)

        # BI
        self.assertIn("tableau", all_skills)
        self.assertIn("power bi", all_skills)
        self.assertIn("excel", all_skills)

        # Cloud / Big Data
        self.assertIn("aws", all_skills)
        self.assertIn("snowflake", all_skills)
        self.assertIn("spark", all_skills)

        # Databases
        self.assertIn("postgresql", all_skills)
        self.assertIn("mongodb", all_skills)

        # Analytics / ML
        self.assertIn("machine learning", all_skills)
        self.assertIn("scikit-learn", all_skills)
        self.assertIn("a/b testing", all_skills)

        # Data engineering
        self.assertIn("etl", all_skills)
        self.assertIn("data pipelines", all_skills)

        # Tools
        self.assertIn("git", all_skills)
        self.assertIn("docker", all_skills)

        # Check category mappings
        by_cat = res["skills_by_category"]
        self.assertIn("programming", by_cat)
        self.assertIn("bi_visualization", by_cat)
        self.assertIn("cloud_bigdata", by_cat)
        self.assertIn("databases", by_cat)
        self.assertIn("analytics_ml", by_cat)
        self.assertIn("data_engineering", by_cat)
        self.assertIn("tools_methodologies", by_cat)
        self.assertGreaterEqual(res["skill_count"], 15)


class TestHighLevelFacades(unittest.TestCase):
    """Tests for parse_resume and parse_job_description."""

    def test_parse_resume_string(self):
        resume_text = """
        Alex Rivera
        Email: alex.rivera@example.com | Phone: (415) 555-9876
        LinkedIn: linkedin.com/in/alexrivera-ml
        GitHub: github.com/alexrivera

        Education:
        Bachelor of Science in Statistics, Columbia University, 2021

        Skills:
        Python, SQL, Tableau, AWS, Docker, Machine Learning, ETL

        Experience:
        Data Analyst at FinTech Corp. Built automated ETL pipelines and visual dashboards.
        """
        result = parse_resume(resume_text)
        self.assertEqual(result["candidate_name"], "Alex Rivera")
        self.assertEqual(result["contact_info"]["email"], "alex.rivera@example.com")
        self.assertEqual(result["contact_info"]["phone"], "(415) 555-9876")
        self.assertGreaterEqual(len(result["education"]), 1)
        self.assertIn("python", result["skills"]["all_skills"])
        self.assertIn("sql", result["skills"]["all_skills"])
        self.assertGreater(result["word_count"], 20)

    def test_parse_job_description(self):
        job_desc = """
        Lead Data Scientist
        We are seeking a Lead Data Scientist to build predictive models and lead analytics.
        Requirements:
        - 5+ years experience with Python, PyTorch, SQL, and Docker.
        - Experience with AWS, Snowflake, and Machine Learning algorithms.
        - Excellent communication and Agile experience.
        Equal Opportunity Employer. 401(k) and medical benefits included.
        """
        result = parse_job_description(job_desc, job_title="Lead Data Scientist")
        self.assertEqual(result["job_title"], "Lead Data Scientist")
        self.assertIn("python", result["required_skills"]["all_skills"])
        self.assertIn("pytorch", result["required_skills"]["all_skills"])
        self.assertIn("aws", result["required_skills"]["all_skills"])
        self.assertNotIn("equal opportunity employer", result["cleaned_text"].lower())
        self.assertTrue(result["dense_profile"].startswith("Job Title: Lead Data Scientist."))
        self.assertIn("Required Skills:", result["dense_profile"])


class TestPdfExtraction(unittest.TestCase):
    """Tests for PDF text extraction using pypdf/pdfplumber."""

    def test_extract_text_from_pdf_stream(self):
        import io
        from src.parser import extract_text_from_pdf

        # Valid minimal PDF 1.4 text stream
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 85 >>\nstream\n"
            b"BT\n/F1 12 Tf\n72 712 Td\n(John Doe - Python and SQL Developer at Acme Corp) Tj\nET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000241 00000 n \n"
            b"0000000375 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n448\n%%EOF\n"
        )
        stream = io.BytesIO(minimal_pdf)
        extracted = extract_text_from_pdf(stream, method="auto")
        self.assertIn("John Doe", extracted)
        self.assertIn("Python", extracted)
        self.assertIn("SQL", extracted)


if __name__ == "__main__":
    unittest.main()

