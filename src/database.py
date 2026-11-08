"""
SQLite Database Storage and Candidate Audit Logging Module.

This module provides thread-safe persistence and audit trail tracking for:
1. Individual candidate match evaluations (scores, contact info, skill gaps, timestamps).
2. Batch candidate screening sessions and candidate leaderboards.
3. Historical analytics, performance metrics, and database statistics.
"""

import os
import json
import sqlite3
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Default SQLite database location (overrideable via SQLITE_DB_PATH env var)
DEFAULT_DB_FILE = os.environ.get("SQLITE_DB_PATH", "history.db")


def get_db_path(custom_path: Optional[Union[str, Path]] = None) -> str:
    """
    Resolves the SQLite database file path.

    Args:
        custom_path: Optional custom path.

    Returns:
        Resolved file path string.
    """
    if custom_path:
        return str(custom_path)
    return str(Path(DEFAULT_DB_FILE).resolve())


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Establishes an SQLite database connection configured with Row factory
    and foreign key constraints enabled.

    Args:
        db_path: Optional database file path.

    Returns:
        sqlite3.Connection object.
    """
    resolved_path = get_db_path(db_path)
    parent_dir = Path(resolved_path).parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initializes database schema, creating tables and indexes if they do not exist.

    Args:
        db_path: Optional database file path.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # Table 1: Individual candidate evaluations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    candidate_name TEXT,
                    candidate_email TEXT,
                    candidate_phone TEXT,
                    job_title TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    semantic_score REAL NOT NULL,
                    skill_score REAL NOT NULL,
                    match_tier TEXT NOT NULL,
                    tier_code TEXT,
                    matched_skills TEXT,       -- JSON array of strings
                    missing_skills TEXT,       -- JSON array of strings
                    additional_skills TEXT,    -- JSON array of strings
                    recommendations TEXT,      -- JSON array of objects
                    category_breakdown TEXT,   -- JSON object
                    resume_word_count INTEGER,
                    job_word_count INTEGER,
                    notes TEXT
                );
            """)

            # Table 2: Batch screening sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    job_title TEXT NOT NULL,
                    total_candidates INTEGER NOT NULL,
                    top_candidate_name TEXT,
                    top_score REAL,
                    average_score REAL
                );
            """)

            # Table 3: Batch candidates breakdown
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    candidate_name TEXT NOT NULL,
                    candidate_email TEXT,
                    candidate_phone TEXT,
                    composite_score REAL NOT NULL,
                    semantic_score REAL NOT NULL,
                    skill_score REAL NOT NULL,
                    match_tier TEXT NOT NULL,
                    tier_code TEXT,
                    matched_count INTEGER DEFAULT 0,
                    missing_count INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES batch_sessions(session_id) ON DELETE CASCADE
                );
            """)

            # Indexes for high-speed search and retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON match_history(timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_candidate ON match_history(candidate_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_job ON match_history(job_title);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_session ON batch_candidates(session_id);")

        logger.info("SQLite database schema initialized at: %s", get_db_path(db_path))
    finally:
        conn.close()


# ===========================================================================
# 1. Single Evaluation CRUD Operations
# ===========================================================================

def save_match_result(
    result: Dict[str, Any],
    notes: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Persists an individual candidate evaluation record into SQLite.

    Args:
        result: Dictionary produced by evaluate_resume_against_job().
        notes: Optional user notes or recruiter comments.
        db_path: Optional database file path.

    Returns:
        The auto-generated integer primary key ID of the inserted record.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    # Safely extract contact info
    contact = result.get("contact_info", {})
    email = contact.get("email") if isinstance(contact, dict) else None
    phone = contact.get("phone") if isinstance(contact, dict) else None

    # JSON serialization
    matched_json = json.dumps(result.get("matched_skills", []))
    missing_json = json.dumps(result.get("missing_skills", []))
    additional_json = json.dumps(result.get("additional_skills", []))
    recs_json = json.dumps(result.get("recommendations", []))
    cat_breakdown_json = json.dumps(result.get("category_breakdown", {}))

    timestamp = result.get("timestamp") or datetime.datetime.utcnow().isoformat()

    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO match_history (
                    timestamp, candidate_name, candidate_email, candidate_phone,
                    job_title, composite_score, semantic_score, skill_score,
                    match_tier, tier_code, matched_skills, missing_skills,
                    additional_skills, recommendations, category_breakdown,
                    resume_word_count, job_word_count, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                result.get("candidate_name", "Anonymous Candidate"),
                email,
                phone,
                result.get("job_title", "Target Role"),
                float(result.get("composite_score", 0.0)),
                float(result.get("semantic_score", 0.0)),
                float(result.get("skill_score", 0.0)),
                result.get("match_tier", "Unknown"),
                result.get("tier_code", "low"),
                matched_json,
                missing_json,
                additional_json,
                recs_json,
                cat_breakdown_json,
                int(result.get("resume_word_count", 0)),
                int(result.get("job_word_count", 0)),
                notes,
            ))
            record_id = cursor.lastrowid
            logger.info("Saved match evaluation record #%d into SQLite", record_id)
            return record_id
    finally:
        conn.close()


def get_match_history(
    limit: int = 50,
    offset: int = 0,
    search_query: Optional[str] = None,
    tier_filter: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves a list of recent evaluation records with JSON deserialization.

    Args:
        limit: Max number of records to return.
        offset: Offset for pagination.
        search_query: Optional search keyword to filter by candidate or job title.
        tier_filter: Optional filter by tier code ('exceptional', 'strong', etc.).
        db_path: Optional database file path.

    Returns:
        List of match records formatted as dictionaries.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    query = "SELECT * FROM match_history WHERE 1=1"
    params: List[Any] = []

    if search_query and search_query.strip():
        query += " AND (candidate_name LIKE ? OR job_title LIKE ? OR candidate_email LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard, wildcard])

    if tier_filter and tier_filter.strip():
        query += " AND tier_code = ?"
        params.append(tier_filter.strip().lower())

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        records: List[Dict[str, Any]] = []

        for row in rows:
            rec = dict(row)
            # Safely unpack JSON fields
            for json_field in ["matched_skills", "missing_skills", "additional_skills", "recommendations", "category_breakdown"]:
                if rec.get(json_field):
                    try:
                        rec[json_field] = json.loads(rec[json_field])
                    except Exception:
                        pass
            records.append(rec)

        return records
    finally:
        conn.close()


def get_match_by_id(
    match_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific evaluation record by its primary key ID.

    Args:
        match_id: Record ID.
        db_path: Optional database file path.

    Returns:
        Record dict if found, None otherwise.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM match_history WHERE id = ?", (match_id,))
        row = cursor.fetchone()
        if not row:
            return None

        rec = dict(row)
        for json_field in ["matched_skills", "missing_skills", "additional_skills", "recommendations", "category_breakdown"]:
            if rec.get(json_field):
                try:
                    rec[json_field] = json.loads(rec[json_field])
                except Exception:
                    pass
        return rec
    finally:
        conn.close()


def delete_match_record(
    match_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Deletes an evaluation record by ID.

    Args:
        match_id: Record ID to remove.
        db_path: Optional database file path.

    Returns:
        True if deleted, False otherwise.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM match_history WHERE id = ?", (match_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_match_history(db_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Deletes all records from match_history.

    Args:
        db_path: Optional database file path.

    Returns:
        True upon success.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM match_history;")
        return True
    finally:
        conn.close()


# ===========================================================================
# 2. Batch Screening Sessions & Candidate Operations
# ===========================================================================

def save_batch_session(
    session_data: Dict[str, Any],
    candidate_results: List[Dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Persists a batch screening session and its ranked candidate rows.

    Args:
        session_data: Session metadata (session_id, job_title, total_candidates, etc.).
        candidate_results: List of candidate dicts produced by rank_candidates_batch().
        db_path: Optional database file path.

    Returns:
        The session_id string.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    import uuid
    session_id = session_data.get("session_id") or f"batch_{uuid.uuid4().hex[:12]}"
    timestamp = session_data.get("timestamp") or datetime.datetime.utcnow().isoformat()
    job_title = session_data.get("job_title", "Target Role")
    total_cand = len(candidate_results)

    top_cand = candidate_results[0].get("candidate_name") if candidate_results else None
    top_score = float(candidate_results[0].get("composite_score", 0.0)) if candidate_results else 0.0
    avg_score = round(
        sum(float(c.get("composite_score", 0.0)) for c in candidate_results) / total_cand, 1
    ) if total_cand > 0 else 0.0

    try:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO batch_sessions (
                    session_id, timestamp, job_title, total_candidates,
                    top_candidate_name, top_score, average_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                timestamp,
                job_title,
                total_cand,
                top_cand,
                top_score,
                avg_score,
            ))

            # Insert candidates
            for cand in candidate_results:
                conn.execute("""
                    INSERT INTO batch_candidates (
                        session_id, rank, candidate_name, candidate_email,
                        candidate_phone, composite_score, semantic_score,
                        skill_score, match_tier, tier_code, matched_count, missing_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    int(cand.get("rank", 1)),
                    cand.get("candidate_name", "Anonymous Candidate"),
                    cand.get("email"),
                    cand.get("phone"),
                    float(cand.get("composite_score", 0.0)),
                    float(cand.get("semantic_score", 0.0)),
                    float(cand.get("skill_score", 0.0)),
                    cand.get("match_tier", "Unknown"),
                    cand.get("tier_code", "low"),
                    int(cand.get("matched_count", 0)),
                    int(cand.get("missing_count", 0)),
                ))

        logger.info("Saved batch session '%s' with %d candidates", session_id, total_cand)
        return session_id
    finally:
        conn.close()


def get_batch_sessions(
    limit: int = 50,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves recent batch screening sessions.

    Args:
        limit: Max sessions to retrieve.
        db_path: Optional database file path.

    Returns:
        List of session metadata dictionaries.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM batch_sessions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_batch_candidates(
    session_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all ranked candidates belonging to a batch session.

    Args:
        session_id: Batch session ID.
        db_path: Optional database file path.

    Returns:
        List of candidate ranking rows.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM batch_candidates WHERE session_id = ? ORDER BY rank ASC",
            (session_id,)
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# ===========================================================================
# 3. Analytics & Statistics
# ===========================================================================

def get_database_stats(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Computes summary metrics for historical screening activity.

    Args:
        db_path: Optional database file path.

    Returns:
        Dictionary of stats including total evaluations, average scores, and tier breakdown.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        # Total single evaluations
        cursor = conn.execute("SELECT COUNT(*), AVG(composite_score) FROM match_history")
        hist_count, hist_avg = cursor.fetchone()
        hist_count = hist_count or 0
        hist_avg = round(hist_avg, 1) if hist_avg is not None else 0.0

        # Tier counts
        cursor = conn.execute(
            "SELECT tier_code, COUNT(*) FROM match_history GROUP BY tier_code"
        )
        tier_counts = {row[0]: row[1] for row in cursor.fetchall() if row[0]}

        # Batch counts
        cursor = conn.execute("SELECT COUNT(*) FROM batch_sessions")
        batch_sessions_count = cursor.fetchone()[0] or 0

        cursor = conn.execute("SELECT COUNT(*) FROM batch_candidates")
        batch_cand_count = cursor.fetchone()[0] or 0

        return {
            "total_single_evaluations": hist_count,
            "average_composite_score": hist_avg,
            "tier_counts": tier_counts,
            "total_batch_sessions": batch_sessions_count,
            "total_batch_candidates": batch_cand_count,
            "total_candidates_evaluated": hist_count + batch_cand_count,
        }
    finally:
        conn.close()
