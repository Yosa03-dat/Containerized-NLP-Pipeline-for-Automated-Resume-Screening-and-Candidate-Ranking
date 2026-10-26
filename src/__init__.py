"""
Semantic Resume & Job Matcher
Source package initialization.
"""

from src.parser import (
    extract_text_from_pdf,
    clean_text,
    clean_job_description,
    normalize_unicode_and_ligatures,
    extract_contact_info,
    extract_candidate_name,
    extract_education,
    extract_skills,
    parse_resume,
    parse_job_description,
    get_spacy_nlp,
    SKILLS_TAXONOMY,
)

__all__ = [
    "extract_text_from_pdf",
    "clean_text",
    "clean_job_description",
    "normalize_unicode_and_ligatures",
    "extract_contact_info",
    "extract_candidate_name",
    "extract_education",
    "extract_skills",
    "parse_resume",
    "parse_job_description",
    "get_spacy_nlp",
    "SKILLS_TAXONOMY",
]
