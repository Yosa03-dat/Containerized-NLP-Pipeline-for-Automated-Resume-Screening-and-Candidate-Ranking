"""
Document Parsing and NLP Entity Extraction Module.

This module provides production-grade utilities for:
1. Extracting raw text from PDF resumes using pdfplumber / pypdf (with auto-fallback).
2. Normalizing unicode ligatures, typographical artifacts, and standardizing whitespace.
3. Extracting candidate contact information (email, phone, LinkedIn, GitHub, portfolio links).
4. Extracting structured education history (degrees, institutions, fields of study, years).
5. Extracting technical skills categorized across domain taxonomies via regex & spaCy NER.
6. Generating high-level parsed resume and job description profiles for downstream embedding.
"""

import io
import re
import html
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Configure logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PDF Ingestion Libraries (Graceful Fallback)
# ---------------------------------------------------------------------------
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

# ---------------------------------------------------------------------------
# Optional spaCy Library (Graceful Fallback)
# ---------------------------------------------------------------------------
try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None
    Language = None


# ---------------------------------------------------------------------------
# Domain Skill Taxonomy (7 Strategic Technical Domains)
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "programming": [
        "python", "r", "sql", "t-sql", "pl/sql", "sas", "scala", "java",
        "vba", "c++", "c#", "bash", "shell", "rust", "go", "golang",
        "javascript", "typescript", "html", "css", "matlab", "ruby", "perl",
        "php", "swift", "kotlin", "julia"
    ],
    "bi_visualization": [
        "tableau", "power bi", "looker", "qlik", "qlikview", "qliksense",
        "excel", "ssrs", "ssis", "ssas", "quicksight", "metabase",
        "microstrategy", "d3.js", "seaborn", "matplotlib", "plotly",
        "google data studio", "looker studio", "cognos", "alteryx"
    ],
    "cloud_bigdata": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "snowflake", "databricks", "spark", "apache spark", "pyspark",
        "hadoop", "redshift", "bigquery", "kafka", "airflow", "emr",
        "s3", "ec2", "lambda", "synapse", "athena", "glue", "cloudera",
        "flink", "hive", "sqoop", "presto", "trino"
    ],
    "databases": [
        "postgresql", "mysql", "oracle", "sql server", "mongodb",
        "dynamodb", "redis", "cassandra", "neo4j", "elasticsearch",
        "sqlite", "couchdb", "mariadb", "teradata", "firebase", "rds"
    ],
    "analytics_ml": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "statistics", "statistical analysis", "predictive modeling",
        "regression", "classification", "clustering", "hypothesis testing",
        "a/b testing", "time series", "time series forecasting", "data mining",
        "neural networks", "llm", "large language models", "transformers",
        "random forest", "xgboost", "lightgbm", "svm", "pca", "tensorflow",
        "keras", "pytorch", "scikit-learn", "genai", "generative ai",
        "langchain", "hugging face", "vector embeddings", "fine-tuning"
    ],
    "data_engineering": [
        "etl", "elt", "data warehousing", "data modeling", "data architecture",
        "data governance", "data quality", "data pipelines", "dbt",
        "data lake", "lakehouse", "star schema", "snowflake schema",
        "dimensional modeling", "master data management", "data catalog",
        "data lineage", "delta lake", "iceberg"
    ],
    "tools_methodologies": [
        "git", "github", "gitlab", "jira", "confluence", "agile", "scrum",
        "kanban", "ci/cd", "docker", "kubernetes", "linux", "unix", "api",
        "rest api", "graphql", "fastapi", "flask", "django", "postman",
        "jenkins", "terraform", "ansible", "mlops", "devops"
    ]
}

# Precompile regex patterns for skill matching
COMPILED_SKILL_PATTERNS: Dict[str, Tuple[str, re.Pattern]] = {}
for _category, _skills in SKILLS_TAXONOMY.items():
    for _skill in _skills:
        # Escape special regex chars (+, #, etc.) while building word boundaries
        _escaped = re.escape(_skill)
        # Use negative lookbehind and lookahead to match exact word tokens
        _pattern = re.compile(r"(?<![a-zA-Z0-9])" + _escaped + r"(?![a-zA-Z0-9])", re.IGNORECASE)
        COMPILED_SKILL_PATTERNS[_skill] = (_category, _pattern)


# ---------------------------------------------------------------------------
# Precompiled Regex Patterns for Entities & Boilerplate
# ---------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE
)

# Standard international and domestic phone number formats
PHONE_PATTERN = re.compile(
    r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?",
    re.IGNORECASE
)
SIMPLE_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
)

LINKEDIN_PATTERN = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|profile)\/([A-Za-z0-9_\-\.]+)\/?",
    re.IGNORECASE
)

GITHUB_PATTERN = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?github\.com\/([A-Za-z0-9_\-\.]+)\/?",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
    re.IGNORECASE
)

# Education degree patterns
DEGREE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Doctorate", re.compile(r"\b(ph\.?d\.?|doctor of philosophy|doctorate)\b", re.IGNORECASE)),
    ("Master", re.compile(r"\b(m\.?s\.?|m\.?a\.?|m\.?sc\.?|m\.?tech\.?|m\.?e\.?|mba|m\.?b\.?a\.?|master(?:'s)?(?: of (?:science|arts|business administration|technology|engineering))?)\b", re.IGNORECASE)),
    ("Bachelor", re.compile(r"\b(b\.?s\.?|b\.?a\.?|b\.?sc\.?|b\.?tech\.?|b\.?e\.?|b\.?com\.?|bba|b\.?b\.?a\.?|bachelor(?:'s)?(?: of (?:science|arts|business administration|technology|engineering|commerce))?)\b", re.IGNORECASE)),
    ("Associate", re.compile(r"\b(associate(?:'s)?(?: degree| of (?:science|arts))?|a\.?s\.?|a\.?a\.?)\b", re.IGNORECASE)),
    ("Diploma", re.compile(r"\b(diploma|post graduate diploma|pg diploma|certification)\b", re.IGNORECASE)),
]

EDUCATION_FIELDS = [
    "computer science", "data science", "software engineering", "information technology",
    "statistics", "mathematics", "applied mathematics", "physics", "electrical engineering",
    "electronics", "mechanical engineering", "civil engineering", "business analytics",
    "information systems", "management information systems", "economics", "finance",
    "accounting", "business administration", "artificial intelligence", "machine learning",
    "computational biology", "bioinformatics", "data analytics"
]
COMPILED_EDU_FIELDS = [
    (field, re.compile(r"\b" + re.escape(field) + r"\b", re.IGNORECASE))
    for field in EDUCATION_FIELDS
]

YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Job description boilerplate patterns
EEO_PATTERNS = [
    r"equal opportunity employer[^\.]*(\.|$)",
    r"affirmative action[^\.]*(\.|$)",
    r"all qualified applicants will receive consideration[^\.]*(\.|$)",
    r"race,\s*color,\s*religion,\s*sex[^\.]*(\.|$)",
    r"veteran status[^\.]*(\.|$)",
    r"disability status[^\.]*(\.|$)",
    r"gender identity[^\.]*(\.|$)",
    r"sexual orientation[^\.]*(\.|$)",
    r"we celebrate diversity[^\.]*(\.|$)",
    r"eeo\s*statement[^\.]*(\.|$)",
    r"we do not discriminate[^\.]*(\.|$)",
]

BENEFITS_PATTERNS = [
    r"401\s*\(?\s*k\s*\)?\s*(matching|plan)?[^\.]*(\.|$)",
    r"medical,\s*dental,\s*(and\s*)?vision[^\.]*(\.|$)",
    r"health insurance[^\.]*(\.|$)",
    r"paid time off[^\.]*(\.|$)",
    r"tuition reimbursement[^\.]*(\.|$)",
    r"wellness program[^\.]*(\.|$)",
    r"life insurance[^\.]*(\.|$)",
    r"parental leave[^\.]*(\.|$)",
]

# Common typographical ligatures and non-standard whitespace/bullets
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00a0": " ",  # Non-breaking space
    "\u2013": "-",  # En-dash
    "\u2014": "-",  # Em-dash
    "\u2018": "'",  # Left single quote
    "\u2019": "'",  # Right single quote
    "\u201c": '"',  # Left double quote
    "\u201d": '"',  # Right double quote
    "\u2022": " ",  # Bullet
    "\u25cf": " ",  # Black circle
    "\u25aa": " ",  # Black small square
    "\u25b6": " ",  # Black right-pointing triangle
    "\u2713": " ",  # Checkmark
    "\u2714": " ",  # Heavy checkmark
    "\u2605": " ",  # Black star
}

# Global spaCy NLP model cache (singleton)
_SPACY_NLP_CACHE: Dict[str, Any] = {}


# ===========================================================================
# 1. Text Normalization & Cleaning Utilities
# ===========================================================================

def normalize_unicode_and_ligatures(text: str) -> str:
    """
    Normalizes unicode ligatures, HTML entities, smart quotes, dashes, and accents.

    Converts typographic fi/fl ligatures (e.g., \\ufb01 -> 'fi') and decomposed
    accents into clean ASCII equivalents, preventing [UNK] token corruption in
    transformer tokenizers.

    Args:
        text: Raw input string.

    Returns:
        Normalized, clean ASCII string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g., &amp;, &lt;, &gt;)
    text = html.unescape(text)

    # 2. Targeted ligature and typographic character replacements
    for char, replacement in LIGATURE_MAP.items():
        if char in text:
            text = text.replace(char, replacement)

    # 3. NFKD decomposition to separate base letters from diacritics
    normalized = unicodedata.normalize("NFKD", text)

    # 4. Re-encode to ASCII, ignoring unencodable characters
    clean_ascii = normalized.encode("ascii", "ignore").decode("utf-8")

    return clean_ascii


def clean_text(
    text: str,
    remove_bullets: bool = True,
    preserve_case: bool = True
) -> str:
    """
    Cleans raw resume or job description text.

    Steps:
    1. Normalizes unicode ligatures and special quotes/dashes.
    2. Removes non-printable / control characters.
    3. Strips list bullets and redundant formatting symbols.
    4. Standardizes whitespace (consecutive spaces/tabs collapsed to single space,
       excessive newlines collapsed to double newlines).

    Args:
        text: Raw text to clean.
        remove_bullets: If True, strips bullet points and line markers.
        preserve_case: If True, preserves original casing; otherwise converts to lowercase.

    Returns:
        Cleaned, standardized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize ligatures and unicode
    text = normalize_unicode_and_ligatures(text)

    # Remove non-printable control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Optionally remove common bullet symbols at start of lines or within text
    if remove_bullets:
        text = re.sub(r"(?m)^\s*[\*\-\•\▪\●\★\○\■\➢\❖\+]+\s*", " ", text)
        text = re.sub(r"[\*\•\▪\●\★\○\■\➢\❖]", " ", text)

    # Normalize horizontal whitespace (spaces, tabs)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3 or more consecutive newlines into 2 (preserving paragraph structure)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Standardize individual line padding
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    if not preserve_case:
        text = text.lower()

    return text


def clean_job_description(text: str) -> str:
    """
    Cleans job description text by stripping recruiter boilerplate,
    EEO legal statements, corporate benefits disclaimers, and URLs.

    This ensures that vector embeddings and similarity computations focus
    strictly on required skills, qualifications, and core job responsibilities.

    Args:
        text: Raw job description text.

    Returns:
        Cleaned job description with boilerplate removed.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize ligatures and casing
    text = normalize_unicode_and_ligatures(text)

    # Remove URLs and email addresses
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    # Remove EEO legal statements
    for pat in EEO_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Remove benefits disclaimers
    for pat in BENEFITS_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Standardize whitespace and bullets
    return clean_text(text, remove_bullets=True, preserve_case=True)


# ===========================================================================
# 2. PDF Text Extraction
# ===========================================================================

def extract_text_from_pdf(
    pdf_source: Union[str, Path, io.BytesIO, bytes],
    method: str = "auto"
) -> str:
    """
    Extracts raw text from an uploaded PDF resume or file path.

    Supports:
    - File path (str or Path)
    - File-like buffer (io.BytesIO or Streamlit UploadedFile)
    - Raw bytes

    Features automatic fallback:
    Tries 'pdfplumber' first (which preserves visual layout and tabular skills).
    If pdfplumber fails or is unavailable, falls back to 'pypdf' / 'PyPDF2'.

    Args:
        pdf_source: PDF file path, BytesIO stream, or raw bytes.
        method: 'auto', 'pdfplumber', or 'pypdf'.

    Returns:
        Extracted raw text string from all pages combined.

    Raises:
        ValueError: If pdf_source is invalid or cannot be parsed.
        RuntimeError: If neither pdfplumber nor pypdf is installed.
    """
    # Check library availability
    if pdfplumber is None and PdfReader is None:
        raise RuntimeError(
            "Neither 'pdfplumber' nor 'pypdf' / 'PyPDF2' is installed. "
            "Please install at least one: pip install pypdf pdfplumber"
        )

    # Convert bytes to BytesIO if necessary
    if isinstance(pdf_source, bytes):
        stream = io.BytesIO(pdf_source)
    elif isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found at path: {pdf_source}")
        stream = str(path)
    elif hasattr(pdf_source, "read"):
        # File-like object (e.g. io.BytesIO or Streamlit UploadedFile)
        # Ensure pointer is at beginning
        try:
            pdf_source.seek(0)
        except Exception:
            pass
        stream = pdf_source
    else:
        raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

    raw_text = ""

    # Strategy 1: pdfplumber (preferred for accurate multi-column layout extraction)
    if method in ("auto", "pdfplumber") and pdfplumber is not None:
        try:
            # For stream objects, ensure we pass open file or stream
            with pdfplumber.open(stream) as pdf:
                pages_text = []
                for idx, page in enumerate(pdf.pages):
                    page_content = page.extract_text(layout=False) or ""
                    if page_content.strip():
                        pages_text.append(page_content)
                raw_text = "\n\n".join(pages_text).strip()
                if raw_text:
                    logger.info("Successfully extracted PDF text using pdfplumber (%d pages)", len(pdf.pages))
                    return raw_text
        except Exception as e:
            logger.warning("pdfplumber extraction failed or returned empty: %s. Attempting fallback.", str(e))
            # Reset stream pointer for fallback if seekable
            if hasattr(stream, "seek"):
                try:
                    stream.seek(0)
                except Exception:
                    pass

    # Strategy 2: pypdf / PyPDF2 (fast, lightweight fallback)
    if (method in ("auto", "pypdf", "pypdf2") or not raw_text) and PdfReader is not None:
        try:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    logger.warning("Encrypted PDF could not be decrypted with empty password: %s", str(e))

            pages_text = []
            for idx, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                if page_content.strip():
                    pages_text.append(page_content)
            raw_text = "\n\n".join(pages_text).strip()
            if raw_text:
                logger.info("Successfully extracted PDF text using pypdf (%d pages)", len(reader.pages))
                return raw_text
        except Exception as e:
            logger.error("pypdf extraction failed: %s", str(e))

    if not raw_text:
        logger.warning("Extracted text is empty. PDF may be a scanned image or corrupted.")

    return raw_text


# ===========================================================================
# 3. spaCy Pipeline Loader & Caching
# ===========================================================================

def get_spacy_nlp(
    model_name: str = "en_core_web_sm",
    add_skill_ruler: bool = True
) -> Optional[Any]:
    """
    Loads and caches a spaCy NLP pipeline (singleton pattern).

    Gracefully falls back to a blank English model ('spacy.blank("en")') if the
    specified pre-trained model is not downloaded. Optionally attaches an
    EntityRuler to detect technical skills as 'SKILL' named entities.

    Args:
        model_name: Name of spaCy model (e.g., 'en_core_web_sm').
        add_skill_ruler: If True, attaches skill entity ruler to pipeline.

    Returns:
        Loaded spaCy Language pipeline, or None if spaCy is not installed.
    """
    if spacy is None:
        logger.warning("spaCy is not installed. NLP entity extraction will use regex-only fallback.")
        return None

    cache_key = f"{model_name}_{add_skill_ruler}"
    if cache_key in _SPACY_NLP_CACHE:
        return _SPACY_NLP_CACHE[cache_key]

    nlp = None
    # 1. Try loading specified pre-trained model
    try:
        nlp = spacy.load(model_name)
        logger.info("Loaded pre-trained spaCy model '%s'", model_name)
    except Exception as e:
        logger.info("Model '%s' not found (%s). Falling back to spacy.blank('en').", model_name, str(e))
        try:
            nlp = spacy.blank("en")
            # Add sentencizer to blank pipeline for sentence segmentation
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        except Exception as inner_e:
            logger.error("Failed to initialize blank spaCy pipeline: %s", str(inner_e))
            return None

    # 2. Attach EntityRuler for technical skills
    if add_skill_ruler and nlp is not None:
        try:
            # Check if entity_ruler already exists
            if "entity_ruler" not in nlp.pipe_names:
                # Add before ner if ner exists, else at end
                if "ner" in nlp.pipe_names:
                    ruler = nlp.add_pipe("entity_ruler", before="ner")
                else:
                    ruler = nlp.add_pipe("entity_ruler")

                patterns = []
                for category, skills in SKILLS_TAXONOMY.items():
                    for skill in skills:
                        patterns.append({
                            "label": "SKILL",
                            "pattern": [{"LOWER": token} for token in skill.split()],
                            "id": category
                        })
                ruler.add_patterns(patterns)
                logger.info("Configured spaCy EntityRuler with %d skill patterns", len(patterns))
        except Exception as e:
            logger.warning("Could not attach EntityRuler to spaCy pipeline: %s", str(e))

    _SPACY_NLP_CACHE[cache_key] = nlp
    return nlp


# ===========================================================================
# 4. Contact Information Extraction
# ===========================================================================

def extract_candidate_name(text: str, nlp: Optional[Any] = None) -> Optional[str]:
    """
    Heuristically extracts the candidate's name from resume text.

    Uses:
    1. spaCy PERSON named entities found in the header (first 500 characters).
    2. Fallback heuristic: First non-empty, title-cased line without emails,
       phones, or common resume header labels.

    Args:
        text: Raw or cleaned resume text.
        nlp: Optional spaCy pipeline instance.

    Returns:
        Extracted candidate name string, or None if not confidently identified.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # 1. spaCy PERSON entity check in top 600 characters
    if nlp is not None:
        try:
            header_doc = nlp(text[:600])
            for ent in header_doc.ents:
                if ent.label_ == "PERSON":
                    clean_name = ent.text.strip()
                    # Validate: 2-4 tokens, reasonable character length, no digits or @
                    tokens = clean_name.split()
                    if (
                        1 < len(tokens) <= 4
                        and len(clean_name) <= 40
                        and not any(char.isdigit() for char in clean_name)
                        and "@" not in clean_name
                    ):
                        return clean_name
        except Exception as e:
            logger.debug("spaCy PERSON extraction error: %s", str(e))

    # 2. Rule-based heuristic fallback: inspect first 5 non-empty lines
    blacklist_terms = {
        "resume", "curriculum", "vitae", "cv", "summary", "profile",
        "contact", "education", "experience", "skills", "projects",
        "phone", "email", "address", "linkedin", "github", "page",
        "candidate", "objective"
    }

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:6]:
        # Ignore lines with email, phone or URL
        if "@" in line or re.search(r"\d{3}", line) or "http" in line.lower() or "www." in line.lower():
            continue

        # Strip commas or titles (e.g. "John Doe, PMP")
        candidate = line.split(",")[0].strip()
        tokens = candidate.split()

        # Typical names are 2 to 4 words, each capitalized, with no digits
        if 2 <= len(tokens) <= 4:
            is_valid = True
            for tok in tokens:
                # Remove period if middle initial
                tok_clean = tok.rstrip(".")
                if not tok_clean.isalpha() or tok.lower() in blacklist_terms:
                    is_valid = False
                    break
            if is_valid:
                return candidate.title()

    return None


def extract_contact_info(
    text: str,
    nlp: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extracts contact details from resume text including email, phone number,
    LinkedIn URL, GitHub URL, portfolio links, and candidate name.

    Args:
        text: Raw or cleaned resume text.
        nlp: Optional spaCy pipeline instance.

    Returns:
        Dictionary containing:
        - 'name': Candidate name (or None)
        - 'email': Primary email address (or None)
        - 'phone': Primary phone number (or None)
        - 'linkedin': Full LinkedIn URL or handle (or None)
        - 'github': Full GitHub URL or handle (or None)
        - 'urls': List of all discovered URLs
    """
    if not isinstance(text, str):
        return {
            "name": None, "email": None, "phone": None,
            "linkedin": None, "github": None, "urls": []
        }

    # Normalize text for regex search
    norm_text = normalize_unicode_and_ligatures(text)

    # 1. Extract Email
    email_matches = EMAIL_PATTERN.findall(norm_text)
    primary_email = email_matches[0].lower() if email_matches else None

    # 2. Extract Phone
    primary_phone = None
    simple_phones = SIMPLE_PHONE_PATTERN.findall(norm_text)
    if simple_phones:
        # Standardize phone string
        primary_phone = simple_phones[0].strip()
    else:
        phone_matches = PHONE_PATTERN.findall(norm_text)
        if phone_matches:
            # Reconstruct first valid match
            match = phone_matches[0]
            digits = "".join([m for m in match if m])
            if len(digits) >= 10:
                primary_phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"

    # 3. Extract LinkedIn
    linkedin_match = LINKEDIN_PATTERN.search(norm_text)
    primary_linkedin = None
    if linkedin_match:
        handle = linkedin_match.group(1).rstrip("/")
        primary_linkedin = f"https://www.linkedin.com/in/{handle}"

    # 4. Extract GitHub
    github_match = GITHUB_PATTERN.search(norm_text)
    primary_github = None
    if github_match:
        handle = github_match.group(1).rstrip("/")
        primary_github = f"https://github.com/{handle}"

    # 5. Extract all external URLs
    raw_urls = URL_PATTERN.findall(norm_text)
    clean_urls = list(dict.fromkeys([u.rstrip("/.,;)") for u in raw_urls]))

    # 6. Extract Candidate Name
    candidate_name = extract_candidate_name(text, nlp=nlp)

    return {
        "name": candidate_name,
        "email": primary_email,
        "phone": primary_phone,
        "linkedin": primary_linkedin,
        "github": primary_github,
        "urls": clean_urls
    }


# ===========================================================================
# 5. Education Extraction
# ===========================================================================

def extract_education(
    text: str,
    nlp: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extracts structured education history from resume text.

    Detects:
    - Degree Level (Doctorate, Master, Bachelor, Associate, Diploma)
    - Field of Study (e.g. Computer Science, Statistics, Economics)
    - Graduation Year (e.g. 2021)
    - Academic Institution (via spaCy ORG entity or keyword heuristics)
    - Raw matching text snippet

    Args:
        text: Raw or cleaned resume text.
        nlp: Optional spaCy pipeline instance.

    Returns:
        List of dictionaries with structured education records.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    norm_text = normalize_unicode_and_ligatures(text)
    lines = norm_text.split("\n")

    education_records: List[Dict[str, Any]] = []
    seen_degrees: Set[str] = set()

    # Pre-extract ORG entities if spaCy is available
    org_entities: List[str] = []
    if nlp is not None:
        try:
            doc = nlp(norm_text)
            org_entities = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "ORG" and any(
                    kw in ent.text.lower()
                    for kw in ["univ", "college", "institute", "school", "academy", "polytechnic"]
                )
            ]
        except Exception as e:
            logger.debug("spaCy ORG extraction error: %s", str(e))

    # Examine text in 3-line sliding windows to catch degree + field + institution across lines
    for i in range(len(lines)):
        window = " ".join(lines[i:min(i + 3, len(lines))])

        for degree_name, degree_regex in DEGREE_PATTERNS:
            degree_match = degree_regex.search(window)
            if not degree_match:
                continue

            # Detect matching field of study in window
            matched_field = None
            for field_name, field_regex in COMPILED_EDU_FIELDS:
                if field_regex.search(window):
                    matched_field = field_name.title()
                    break

            # Detect graduation year in window
            year_match = YEAR_PATTERN.search(window)
            grad_year = year_match.group(1) if year_match else None

            # Detect institution
            institution = None
            # Check spaCy ORG matches first
            for org in org_entities:
                if org.lower() in window.lower():
                    institution = org
                    break

            # Heuristic institution detection if spaCy didn't catch it
            if not institution:
                inst_match = re.search(
                    r"\b([A-Z][A-Za-z\s&]+(?:University|College|Institute|School|Academy))\b",
                    window
                )
                if inst_match:
                    institution = inst_match.group(1).strip()

            dedup_key = f"{degree_name}_{matched_field}_{grad_year}"
            if dedup_key not in seen_degrees:
                seen_degrees.add(dedup_key)
                education_records.append({
                    "degree": degree_name,
                    "field": matched_field or "General / Unspecified",
                    "institution": institution or "Not specified",
                    "year": grad_year,
                    "snippet": window.strip()
                })

    return education_records


# ===========================================================================
# 6. Technical Skills Extraction
# ===========================================================================

def extract_skills(
    text: str,
    nlp: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extracts technical skills from text using precompiled regex taxonomy
    and optional spaCy EntityRuler detection.

    Categorizes found skills into 7 key technical domains:
    - programming
    - bi_visualization
    - cloud_bigdata
    - databases
    - analytics_ml
    - data_engineering
    - tools_methodologies

    Args:
        text: Raw or cleaned text string.
        nlp: Optional spaCy pipeline instance.

    Returns:
        Dictionary containing:
        - 'all_skills': Alphabetically sorted list of all unique skills found
        - 'skills_by_category': Dict mapping domain category to list of skills
        - 'skill_count': Total count of unique skills detected
        - 'categories_present': List of categories with at least one match
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "all_skills": [],
            "skills_by_category": {},
            "skill_count": 0,
            "categories_present": []
        }

    norm_text = normalize_unicode_and_ligatures(text).lower()

    found_skills: Set[str] = set()
    category_map: Dict[str, List[str]] = {cat: [] for cat in SKILLS_TAXONOMY.keys()}

    # 1. Fast regex taxonomy matching
    for skill_name, (category, pattern) in COMPILED_SKILL_PATTERNS.items():
        if pattern.search(norm_text):
            clean_name = skill_name.replace(r"\+", "+").replace(r"\#", "#")
            found_skills.add(clean_name)
            category_map[category].append(clean_name)

    # 2. Supplementary spaCy NER extraction if available
    if nlp is not None:
        try:
            doc = nlp(norm_text)
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    clean_ent = ent.text.strip().lower()
                    if clean_ent in COMPILED_SKILL_PATTERNS:
                        cat = COMPILED_SKILL_PATTERNS[clean_ent][0]
                        if clean_ent not in found_skills:
                            found_skills.add(clean_ent)
                            category_map[cat].append(clean_ent)
        except Exception as e:
            logger.debug("spaCy skill extraction error: %s", str(e))

    # Clean and sort category results
    populated_categories = {}
    for cat, skills in category_map.items():
        if skills:
            populated_categories[cat] = sorted(list(set(skills)))

    return {
        "all_skills": sorted(list(found_skills)),
        "skills_by_category": populated_categories,
        "skill_count": len(found_skills),
        "categories_present": sorted(list(populated_categories.keys()))
    }


# ===========================================================================
# 7. High-Level Resume & Job Description Facades
# ===========================================================================

def parse_resume(
    source: Union[str, Path, io.BytesIO, bytes],
    nlp: Optional[Any] = None
) -> Dict[str, Any]:
    """
    End-to-end resume parser. Ingests a PDF (path, stream, or bytes) or raw text,
    extracts all content, standardizes formatting, and extracts structured entities.

    Args:
        source: PDF file path, BytesIO object, raw bytes, or plain text string.
        nlp: Optional spaCy Language instance.

    Returns:
        Structured dictionary containing:
        - 'raw_text': Extracted raw string
        - 'cleaned_text': Normalized, standardized text
        - 'candidate_name': Extracted name (or None)
        - 'contact_info': Email, phone, LinkedIn, GitHub, links
        - 'education': List of structured degree/institution records
        - 'skills': Categorized technical competencies
        - 'word_count': Total word count of cleaned text
        - 'char_count': Total character count of cleaned text
    """
    # 1. Determine input type and extract raw text
    if isinstance(source, str) and not source.lower().endswith(".pdf") and "\n" in source:
        # Direct raw text provided
        raw_text = source
    else:
        # PDF document provided
        raw_text = extract_text_from_pdf(source, method="auto")

    # 2. Standardize and clean text
    cleaned = clean_text(raw_text, remove_bullets=True, preserve_case=True)

    # 3. Extract contact info & name
    contact = extract_contact_info(cleaned, nlp=nlp)

    # 4. Extract education history
    education = extract_education(cleaned, nlp=nlp)

    # 5. Extract categorized technical skills
    skills_data = extract_skills(cleaned, nlp=nlp)

    words = cleaned.split()

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "candidate_name": contact.get("name"),
        "contact_info": contact,
        "education": education,
        "skills": skills_data,
        "word_count": len(words),
        "char_count": len(cleaned)
    }


def parse_job_description(
    text: str,
    job_title: Optional[str] = None,
    nlp: Optional[Any] = None
) -> Dict[str, Any]:
    """
    End-to-end job description parser. Cleans raw description, strips recruiter
    boilerplate, extracts required technical skills, and builds a dense semantic
    profile optimized for transformer embeddings.

    Args:
        text: Raw job description string.
        job_title: Optional job title (e.g., 'Senior Data Analyst').
        nlp: Optional spaCy Language instance.

    Returns:
        Structured dictionary containing:
        - 'job_title': Resolved job title
        - 'cleaned_text': Text with boilerplate stripped
        - 'required_skills': Categorized technical skills
        - 'dense_profile': Token-budget-optimized summary (< 180 words)
        - 'word_count': Word count of cleaned text
    """
    cleaned = clean_job_description(text)
    skills_data = extract_skills(cleaned, nlp=nlp)

    # Resolve job title if not explicitly provided
    resolved_title = job_title or "Job Opening"
    if not job_title and cleaned:
        first_line = cleaned.split("\n")[0]
        if len(first_line.split()) <= 6:
            resolved_title = first_line.title()

    # Generate dense semantic profile (Title + Key Skills + First 120 words of description)
    skills_str = ", ".join(skills_data["all_skills"]) if skills_data["all_skills"] else "general technical skills"
    desc_snippet = " ".join(cleaned.split()[:120])
    dense_profile = (
        f"Job Title: {resolved_title}. "
        f"Required Skills: {skills_str}. "
        f"Responsibilities & Scope: {desc_snippet}"
    )

    return {
        "job_title": resolved_title,
        "cleaned_text": cleaned,
        "required_skills": skills_data,
        "dense_profile": dense_profile,
        "word_count": len(cleaned.split())
    }
