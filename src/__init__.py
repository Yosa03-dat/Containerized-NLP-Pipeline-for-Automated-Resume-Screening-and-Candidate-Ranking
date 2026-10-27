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

from src.embedder import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM,
    DEFAULT_CHROMA_DIR,
    get_optimal_device,
    get_embedding_model,
    embed_text,
    embed_batch,
    compute_cosine_similarity,
    compute_similarity_matrix,
    compute_text_similarity,
    build_resume_embedding_text,
    embed_resume,
    embed_job,
    ChromaVectorStore,
    index_semantic_chunks_csv,
    index_job_descriptions_csv,
)

__all__ = [
    # Parser module
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
    # Embedder module
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIM",
    "DEFAULT_CHROMA_DIR",
    "get_optimal_device",
    "get_embedding_model",
    "embed_text",
    "embed_batch",
    "compute_cosine_similarity",
    "compute_similarity_matrix",
    "compute_text_similarity",
    "build_resume_embedding_text",
    "embed_resume",
    "embed_job",
    "ChromaVectorStore",
    "index_semantic_chunks_csv",
    "index_job_descriptions_csv",
]
