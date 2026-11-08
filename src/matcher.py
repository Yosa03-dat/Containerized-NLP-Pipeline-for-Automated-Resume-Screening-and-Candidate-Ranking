"""
Skill Gap Analysis, Candidate Scoring, and Batch Ranking Module.

This module provides production-grade components for:
1. Analyzing technical skill overlap and identifying critical skill gaps across domain taxonomies.
2. Computing transparent, weighted composite candidate scores balancing semantic similarity and keyword coverage.
3. Orchestrating end-to-end evaluation pipelines comparing candidate resumes against target job descriptions.
4. Performing high-throughput batch candidate screening, ranking, and leaderboard generation.
"""

import io
import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import numpy as np

from src.parser import (
    SKILLS_TAXONOMY,
    parse_resume,
    parse_job_description,
    clean_text,
)
from src.embedder import (
    embed_text,
    embed_batch,
    compute_cosine_similarity,
    build_resume_embedding_text,
    get_embedding_model,
)

logger = logging.getLogger(__name__)

# Default scoring weights: 60% semantic similarity, 40% hard skill coverage
DEFAULT_SEMANTIC_WEIGHT = 0.60
DEFAULT_SKILL_WEIGHT = 0.40

# Match tier classification thresholds
TIER_EXCEPTIONAL = 85.0
TIER_STRONG = 70.0
TIER_MODERATE = 50.0


# ===========================================================================
# 1. Skill Normalization & Extraction Utilities
# ===========================================================================

def _flatten_skill_collection(
    skills_input: Union[Dict[str, Any], List[str], Set[str], None]
) -> Set[str]:
    """
    Normalizes arbitrary skill representations into a clean set of lowercase strings.

    Args:
        skills_input: Categorized dict, list, set, or None.

    Returns:
        Set of lowercase, trimmed skill strings.
    """
    if not skills_input:
        return set()

    flat_set: Set[str] = set()

    if isinstance(skills_input, dict):
        if "all_skills" in skills_input and isinstance(skills_input["all_skills"], (list, tuple, set)):
            for s in skills_input["all_skills"]:
                if isinstance(s, str) and s.strip():
                    flat_set.add(s.strip().lower())
            return flat_set

        if "skills_by_category" in skills_input and isinstance(skills_input["skills_by_category"], dict):
            return _flatten_skill_collection(skills_input["skills_by_category"])

        for val in skills_input.values():
            if isinstance(val, (list, tuple, set)):
                for s in val:
                    if isinstance(s, str) and s.strip():
                        flat_set.add(s.strip().lower())
            elif isinstance(val, str) and val.strip():
                flat_set.add(val.strip().lower())
            elif isinstance(val, dict):
                flat_set.update(_flatten_skill_collection(val))
    elif isinstance(skills_input, (list, tuple, set)):
        for item in skills_input:
            if isinstance(item, str) and item.strip():
                flat_set.add(item.strip().lower())
            elif isinstance(item, dict):
                flat_set.update(_flatten_skill_collection(item))

    return flat_set


# ===========================================================================
# 2. Skill Gap Analysis Engine
# ===========================================================================

def analyze_skill_gap(
    resume_skills: Union[Dict[str, Any], List[str], Set[str], None],
    job_skills: Union[Dict[str, Any], List[str], Set[str], None],
    taxonomy: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Compares candidate skills against required job posting skills.

    Calculates matched skills, missing skills, additional skills, overall coverage ratio,
    and granular category-by-category breakdowns across domain taxonomies.

    Args:
        resume_skills: Skills found in candidate resume.
        job_skills: Skills required by target job posting.
        taxonomy: Optional custom domain taxonomy dictionary. Defaults to SKILLS_TAXONOMY.

    Returns:
        Structured dictionary containing:
        - 'matched_skills': List of overlapping skills (sorted)
        - 'missing_skills': List of required skills missing from candidate (sorted)
        - 'additional_skills': Candidate skills beyond job requirements (sorted)
        - 'skill_coverage': Float ratio [0.0, 1.0]
        - 'total_job_skills': Total count of required skills
        - 'total_matched_skills': Count of matched skills
        - 'category_breakdown': Detailed dict per domain category
        - 'recommendations': Actionable advice for candidate upskilling
    """
    active_taxonomy = taxonomy or SKILLS_TAXONOMY
    cand_set = _flatten_skill_collection(resume_skills)
    job_set = _flatten_skill_collection(job_skills)

    matched = sorted(list(job_set.intersection(cand_set)))
    missing = sorted(list(job_set.difference(cand_set)))
    additional = sorted(list(cand_set.difference(job_set)))

    total_job_skills = len(job_set)
    total_matched = len(matched)

    if total_job_skills > 0:
        skill_coverage = round(total_matched / total_job_skills, 4)
    else:
        # If no explicit technical skills were detected in the job, default to 1.0
        skill_coverage = 1.0

    # Categorized breakdown across taxonomy
    category_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat_name, cat_skills in active_taxonomy.items():
        cat_set = {s.lower() for s in cat_skills}
        req_in_cat = sorted(list(job_set.intersection(cat_set)))
        cand_in_cat = sorted(list(cand_set.intersection(cat_set)))
        matched_in_cat = sorted(list(set(req_in_cat).intersection(cand_in_cat)))
        missing_in_cat = sorted(list(set(req_in_cat).difference(cand_in_cat)))

        cat_total = len(req_in_cat)
        cat_cov = round(len(matched_in_cat) / cat_total, 4) if cat_total > 0 else 1.0

        category_breakdown[cat_name] = {
            "required": req_in_cat,
            "candidate": cand_in_cat,
            "matched": matched_in_cat,
            "missing": missing_in_cat,
            "coverage": cat_cov,
            "has_requirements": cat_total > 0,
        }

    # Generate prioritized recommendations for candidate upskilling
    recommendations: List[Dict[str, Any]] = []
    if missing:
        # Prioritize missing skills by categories that have the highest deficit
        sorted_cats = sorted(
            category_breakdown.items(),
            key=lambda item: len(item[1]["missing"]),
            reverse=True,
        )
        for cat_name, cat_data in sorted_cats:
            if cat_data["missing"]:
                recommendations.append({
                    "category": cat_name,
                    "missing_skills": cat_data["missing"],
                    "impact": "High" if len(cat_data["missing"]) >= 2 else "Medium",
                    "suggestion": (
                        f"Acquiring or highlighting experience with {', '.join(cat_data['missing'])} "
                        f"would substantially strengthen alignment for {cat_name.replace('_', ' ').title()}."
                    ),
                })

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_skills": additional,
        "skill_coverage": skill_coverage,
        "total_job_skills": total_job_skills,
        "total_matched_skills": total_matched,
        "category_breakdown": category_breakdown,
        "recommendations": recommendations,
    }


# ===========================================================================
# 3. Composite Candidate Scoring Algorithm
# ===========================================================================

def compute_match_score(
    semantic_similarity: float,
    skill_coverage: float,
    weights: Tuple[float, float] = (DEFAULT_SEMANTIC_WEIGHT, DEFAULT_SKILL_WEIGHT),
    education_bonus: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes a weighted composite match score combining dense semantic vector
    similarity and hard skill coverage percentage.

    Composite Score Formula:
        Composite = (w_sem * clamp(semantic_similarity) + w_skill * clamp(skill_coverage)) * 100 + bonus

    Args:
        semantic_similarity: Cosine similarity score [0.0, 1.0].
        skill_coverage: Keyword overlap ratio [0.0, 1.0].
        weights: Tuple of (semantic_weight, skill_weight). Defaults to (0.6, 0.4).
        education_bonus: Optional bonus points [0.0, 5.0] for matching advanced degrees.

    Returns:
        Dictionary containing:
        - 'composite_score': Final percentage [0.0, 100.0] rounded to 1 decimal
        - 'semantic_score': Semantic similarity percentage [0.0, 100.0]
        - 'skill_score': Skill overlap percentage [0.0, 100.0]
        - 'match_tier': Classification string ('Exceptional Fit', 'Strong Fit', etc.)
        - 'tier_code': Standardized code ('exceptional', 'strong', 'moderate', 'low')
        - 'explanation': Human-readable summary rationale
    """
    w_sem, w_skill = weights
    total_w = w_sem + w_skill
    if total_w <= 0:
        w_sem, w_skill = DEFAULT_SEMANTIC_WEIGHT, DEFAULT_SKILL_WEIGHT
        total_w = 1.0
    w_sem_norm = w_sem / total_w
    w_skill_norm = w_skill / total_w

    clamped_sim = max(0.0, min(1.0, float(semantic_similarity)))
    clamped_cov = max(0.0, min(1.0, float(skill_coverage)))

    raw_composite = (w_sem_norm * clamped_sim + w_skill_norm * clamped_cov) * 100.0 + education_bonus
    composite_score = round(max(0.0, min(100.0, raw_composite)), 1)
    semantic_score = round(clamped_sim * 100.0, 1)
    skill_score = round(clamped_cov * 100.0, 1)

    if composite_score >= TIER_EXCEPTIONAL:
        match_tier = "Exceptional Fit"
        tier_code = "exceptional"
        explanation = (
            "Outstanding alignment across both contextual project experience and required technical competencies."
        )
    elif composite_score >= TIER_STRONG:
        match_tier = "Strong Fit"
        tier_code = "strong"
        explanation = (
            "Strong overall match. Candidate demonstrates key core qualifications with minor skill or domain gaps."
        )
    elif composite_score >= TIER_MODERATE:
        match_tier = "Moderate Fit"
        tier_code = "moderate"
        explanation = (
            "Partial alignment. Candidate possesses transferable concepts but lacks several critical technical skills."
        )
    else:
        match_tier = "Low Fit"
        tier_code = "low"
        explanation = (
            "Limited overlap. Candidate background does not closely align with the target posting requirements."
        )

    return {
        "composite_score": composite_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "match_tier": match_tier,
        "tier_code": tier_code,
        "weights": {"semantic": round(w_sem_norm, 2), "skill": round(w_skill_norm, 2)},
        "explanation": explanation,
    }


# ===========================================================================
# 4. End-to-End Single Resume Evaluation Pipeline
# ===========================================================================

def evaluate_resume_against_job(
    resume_input: Union[str, bytes, io.BytesIO, Dict[str, Any]],
    job_input: Union[str, Dict[str, Any]],
    job_title: Optional[str] = None,
    weights: Tuple[float, float] = (DEFAULT_SEMANTIC_WEIGHT, DEFAULT_SKILL_WEIGHT),
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the complete evaluation workflow for a single candidate resume against
    a target job description.

    Steps:
    1. Parses resume input (PDF stream/bytes, file path, or raw text) via src.parser.
    2. Parses job description input (raw string or existing dictionary).
    3. Builds dense representation text and computes 384-d semantic embeddings.
    4. Computes cosine similarity between candidate and job profile.
    5. Conducts skill gap analysis across domain taxonomies.
    6. Calculates composite match score, tier, and actionable recommendations.

    Args:
        resume_input: PDF bytes/stream/path, raw text string, or pre-parsed resume dict.
        job_input: Raw job description string or pre-parsed job dict.
        job_title: Optional title for the job (defaults to extracted or 'Target Role').
        weights: Scoring weights (semantic_weight, skill_weight).
        model: Optional pre-loaded SentenceTransformer model instance.

    Returns:
        Comprehensive evaluation dictionary ready for UI display and database logging.
    """
    # Step 1: Parse candidate resume
    if isinstance(resume_input, dict) and "cleaned_text" in resume_input:
        parsed_resume = resume_input
    else:
        parsed_resume = parse_resume(resume_input)

    # Step 2: Parse job description
    resolved_title = job_title
    if isinstance(job_input, dict) and "cleaned_text" in job_input:
        parsed_job = job_input
        if not resolved_title:
            resolved_title = parsed_job.get("job_title") or parsed_job.get("title") or "Target Role"
    else:
        parsed_job = parse_job_description(str(job_input), job_title=resolved_title)
        if not resolved_title:
            resolved_title = parsed_job.get("job_title") or "Target Role"

    resolved_title = resolved_title or "Target Role"

    # Step 3: Compute dense embeddings
    resume_embed_text = build_resume_embedding_text(parsed_resume)
    job_embed_text = parsed_job.get("dense_profile") or parsed_job.get("cleaned_text") or ""

    resume_vector = embed_text(resume_embed_text, model=model, normalize=True)
    job_vector = embed_text(job_embed_text, model=model, normalize=True)

    # Step 4: Semantic similarity calculation
    sim_score = compute_cosine_similarity(resume_vector, job_vector)

    # Step 5: Skill gap analysis
    resume_skills = parsed_resume.get("skills") or parsed_resume.get("categorized_skills") or {}
    job_skills = parsed_job.get("required_skills") or parsed_job.get("categorized_skills") or {}
    skill_gap = analyze_skill_gap(resume_skills, job_skills)

    # Education bonus calculation (small bonus for Master's or PhD)
    education_bonus = 0.0
    edu_list = parsed_resume.get("education", [])
    for edu in edu_list:
        deg = str(edu.get("degree_level", "")).lower()
        if "doctorate" in deg or "phd" in deg:
            education_bonus = max(education_bonus, 3.0)
        elif "master" in deg:
            education_bonus = max(education_bonus, 1.5)

    # Step 6: Scoring & tier computation
    scores = compute_match_score(
        semantic_similarity=sim_score,
        skill_coverage=skill_gap["skill_coverage"],
        weights=weights,
        education_bonus=education_bonus,
    )

    candidate_name = parsed_resume.get("candidate_name") or "Anonymous Candidate"
    contact_info = parsed_resume.get("contact_info", {})

    return {
        "candidate_name": candidate_name,
        "contact_info": contact_info,
        "education": edu_list,
        "job_title": resolved_title,
        "composite_score": scores["composite_score"],
        "semantic_score": scores["semantic_score"],
        "skill_score": scores["skill_score"],
        "match_tier": scores["match_tier"],
        "tier_code": scores["tier_code"],
        "explanation": scores["explanation"],
        "skill_gap": skill_gap,
        "matched_skills": skill_gap["matched_skills"],
        "missing_skills": skill_gap["missing_skills"],
        "additional_skills": skill_gap["additional_skills"],
        "skill_coverage": skill_gap["skill_coverage"],
        "category_breakdown": skill_gap["category_breakdown"],
        "recommendations": skill_gap["recommendations"],
        "resume_word_count": parsed_resume.get("word_count", 0),
        "job_word_count": parsed_job.get("word_count", 0),
        "parsed_resume": parsed_resume,
        "parsed_job": parsed_job,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# ===========================================================================
# 5. Batch Candidate Screening & Ranking Engine
# ===========================================================================

def rank_candidates_batch(
    resumes: List[Union[Dict[str, Any], Tuple[str, str], str]],
    job_input: Union[str, Dict[str, Any]],
    job_title: Optional[str] = None,
    top_k: Optional[int] = None,
    weights: Tuple[float, float] = (DEFAULT_SEMANTIC_WEIGHT, DEFAULT_SKILL_WEIGHT),
    model: Optional[Any] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Performs high-throughput ranking across multiple candidate resumes against a target job.

    Optimized for bulk screening by:
    1. Batch parsing candidate inputs into structured profiles.
    2. Batch vectorizing all candidate profiles in parallel matrix operations via embed_batch().
    3. Performing matrix cosine similarity operations against the target job vector.
    4. Evaluating skill coverage, assigning ranks, and returning a sorted candidate leaderboard.

    Args:
        resumes: List of candidate inputs (can be dicts with 'name' and 'text', or raw text strings).
        job_input: Target job description string or parsed job dictionary.
        job_title: Optional title of the position.
        top_k: Optional limit on the number of returned top candidates.
        weights: Tuple of (semantic_weight, skill_weight).
        model: Optional cached SentenceTransformer model instance.
        batch_size: Embedding batch size for transformer inference.

    Returns:
        List of ranked candidate dictionaries in descending order of composite score,
        with rank positions and score breakdowns.
    """
    if not resumes:
        return []

    # Step 1: Parse target job description
    resolved_title = job_title
    if isinstance(job_input, dict) and "cleaned_text" in job_input:
        parsed_job = job_input
        if not resolved_title:
            resolved_title = parsed_job.get("job_title") or parsed_job.get("title") or "Target Role"
    else:
        parsed_job = parse_job_description(str(job_input), job_title=resolved_title)
        if not resolved_title:
            resolved_title = parsed_job.get("job_title") or "Target Role"

    job_skills = parsed_job.get("required_skills") or parsed_job.get("categorized_skills") or {}
    job_embed_text = parsed_job.get("dense_profile") or parsed_job.get("cleaned_text") or ""
    job_vector = embed_text(job_embed_text, model=model, normalize=True)

    # Step 2: Ingest & parse all candidates
    parsed_candidates: List[Dict[str, Any]] = []
    dense_texts: List[str] = []

    for idx, item in enumerate(resumes):
        if isinstance(item, dict):
            name = item.get("name") or item.get("candidate_name") or f"Candidate {idx + 1}"
            raw_text = item.get("text") or item.get("resume_text") or item.get("raw_text") or ""
            if "cleaned_text" in item and ("skills" in item or "categorized_skills" in item):
                parsed_c = item
            else:
                parsed_c = parse_resume(raw_text)
                if name:
                    parsed_c["candidate_name"] = name
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            name, raw_text = item[0], item[1]
            parsed_c = parse_resume(raw_text)
            parsed_c["candidate_name"] = str(name)
        elif isinstance(item, str):
            name = f"Candidate {idx + 1}"
            parsed_c = parse_resume(item)
            parsed_c["candidate_name"] = name
        else:
            name = f"Candidate {idx + 1}"
            parsed_c = parse_resume(str(item))
            parsed_c["candidate_name"] = name

        parsed_candidates.append(parsed_c)
        dense_texts.append(build_resume_embedding_text(parsed_c))

    # Step 3: Batch compute dense vector embeddings
    candidate_vectors = embed_batch(
        dense_texts,
        model=model,
        batch_size=batch_size,
        normalize=True,
    )

    # Step 4: Compute similarities and evaluate each candidate
    ranked_candidates: List[Dict[str, Any]] = []

    for idx, (cand, cand_vec) in enumerate(zip(parsed_candidates, candidate_vectors)):
        sim = compute_cosine_similarity(cand_vec, job_vector)
        cand_skills = cand.get("skills") or cand.get("categorized_skills") or {}
        gap = analyze_skill_gap(cand_skills, job_skills)

        # Education bonus
        edu_bonus = 0.0
        for edu in cand.get("education", []):
            deg = str(edu.get("degree_level", "")).lower()
            if "doctorate" in deg or "phd" in deg:
                edu_bonus = max(edu_bonus, 3.0)
            elif "master" in deg:
                edu_bonus = max(edu_bonus, 1.5)

        scores = compute_match_score(
            semantic_similarity=sim,
            skill_coverage=gap["skill_coverage"],
            weights=weights,
            education_bonus=edu_bonus,
        )

        cand_name = cand.get("candidate_name") or f"Candidate {idx + 1}"
        contact = cand.get("contact_info", {})

        ranked_candidates.append({
            "candidate_name": cand_name,
            "email": contact.get("email"),
            "phone": contact.get("phone"),
            "composite_score": scores["composite_score"],
            "semantic_score": scores["semantic_score"],
            "skill_score": scores["skill_score"],
            "match_tier": scores["match_tier"],
            "tier_code": scores["tier_code"],
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
            "additional_skills": gap["additional_skills"],
            "matched_count": len(gap["matched_skills"]),
            "missing_count": len(gap["missing_skills"]),
            "skill_coverage": gap["skill_coverage"],
            "category_breakdown": gap["category_breakdown"],
            "recommendations": gap["recommendations"],
            "explanation": scores["explanation"],
            "word_count": cand.get("word_count", 0),
        })

    # Step 5: Sort by composite score descending (secondary key: semantic_score)
    ranked_candidates.sort(
        key=lambda x: (x["composite_score"], x["semantic_score"]),
        reverse=True,
    )

    # Step 6: Assign rankings
    for rank_idx, cand in enumerate(ranked_candidates, start=1):
        cand["rank"] = rank_idx

    if top_k is not None and top_k > 0:
        return ranked_candidates[:top_k]

    return ranked_candidates
