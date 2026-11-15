"""
Dense Vector Embedding Engine and Vector Database Storage Module.

This module provides production-grade components for:
1. Initializing and caching SentenceTransformer models ('all-MiniLM-L6-v2').
2. Generating normalized 384-dimensional dense vector embeddings for text, resumes, and job descriptions.
3. Calculating pairwise cosine similarities and similarity matrices.
4. Managing ChromaDB collections for persistent or in-memory vector similarity retrieval.
5. Ingesting preprocessed semantic chunks and job profiles into vector collections.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "models/chroma_db")

# ---------------------------------------------------------------------------
# Optional Library Ingestion (Graceful Fallback)
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

# Global model cache (singleton pattern)
_MODEL_CACHE: Dict[str, Any] = {}


# ===========================================================================
# 1. Device and Model Initialization
# ===========================================================================

def get_optimal_device() -> str:
    """
    Determines the fastest available compute device (CUDA GPU, Apple MPS, or CPU).

    Returns:
        Device identifier string: 'cuda', 'mps', or 'cpu'.
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None
) -> Any:
    """
    Retrieves or lazily initializes a cached SentenceTransformer model instance.
    Utilizes a singleton cache to avoid redundant model loading across application calls.

    Args:
        model_name: Hugging Face model identifier (default: 'all-MiniLM-L6-v2').
        device: Target compute device ('cuda', 'cpu', 'mps'). If None, auto-detected.

    Returns:
        SentenceTransformer model instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        RuntimeError: If model failed to load.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "The 'sentence-transformers' library is required but not installed. "
            "Please install it using 'pip install sentence-transformers'."
        )

    resolved_device = device or get_optimal_device()
    cache_key = f"{model_name}_{resolved_device}"

    if cache_key not in _MODEL_CACHE:
        logger.info(
            "Initializing SentenceTransformer model '%s' on device '%s'...",
            model_name, resolved_device
        )
        try:
            model = SentenceTransformer(model_name, device=resolved_device)
            _MODEL_CACHE[cache_key] = model
            logger.info("Successfully loaded and cached model '%s'.", model_name)
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer model '%s': %s", model_name, str(exc))
            raise RuntimeError(f"Could not load embedding model '{model_name}': {exc}") from exc

    return _MODEL_CACHE[cache_key]


# ===========================================================================
# 2. Vector Embedding Computation
# ===========================================================================

def embed_text(
    text: str,
    model: Optional[Any] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Encodes a single text string into a dense 1D vector embedding.

    Args:
        text: Input text to encode.
        model: Optional pre-loaded SentenceTransformer instance. If None, uses default cached model.
        normalize: If True, normalizes vector to unit length (L2 norm = 1.0) so dot product == cosine similarity.

    Returns:
        1D NumPy array of shape (384,) with float32 values.
    """
    if not isinstance(text, str) or not text.strip():
        # Return zero vector for empty/invalid inputs
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    encoder = model or get_embedding_model()
    embedding = encoder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )

    return np.asarray(embedding, dtype=np.float32)


def embed_batch(
    texts: List[str],
    model: Optional[Any] = None,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Encodes a list of text strings into a 2D dense vector embedding matrix.

    Args:
        texts: List of strings to encode.
        model: Optional pre-loaded SentenceTransformer instance.
        batch_size: Number of texts to process in parallel per forward pass.
        normalize: If True, normalizes each vector to unit length.
        show_progress_bar: Whether to display a tqdm progress bar during encoding.

    Returns:
        2D NumPy array of shape (len(texts), 384) with float32 values.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    # Sanitize inputs (ensure all entries are strings)
    sanitized_texts = [str(t) if t is not None else "" for t in texts]

    encoder = model or get_embedding_model()
    embeddings = encoder.encode(
        sanitized_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar
    )

    return np.asarray(embeddings, dtype=np.float32)


# ===========================================================================
# 3. Similarity Metrics & Comparison
# ===========================================================================

def compute_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray
) -> float:
    """
    Calculates the cosine similarity score between two vector embeddings.
    If the vectors are already L2 normalized, returns their dot product.
    Otherwise computes dot(u, v) / (||u|| * ||v||).

    Args:
        vec_a: First 1D vector embedding.
        vec_b: Second 1D vector embedding.

    Returns:
        Cosine similarity float in range [-1.0, 1.0].
    """
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # If already approximately unit length, use direct dot product
    if abs(norm_a - 1.0) < 1e-4 and abs(norm_b - 1.0) < 1e-4:
        sim = float(np.dot(a, b))
    else:
        sim = float(np.dot(a, b) / (norm_a * norm_b))

    # Clamp numerical boundary precision
    return max(-1.0, min(1.0, sim))


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculates pairwise cosine similarity matrix between queries and a corpus.

    Args:
        query_embeddings: 2D array of shape (Q, D) or 1D array of shape (D,).
        corpus_embeddings: 2D array of shape (C, D).

    Returns:
        2D NumPy array of shape (Q, C) with similarity scores.
    """
    q = np.asarray(query_embeddings, dtype=np.float32)
    c = np.asarray(corpus_embeddings, dtype=np.float32)

    if q.ndim == 1:
        q = q.reshape(1, -1)
    if c.ndim == 1:
        c = c.reshape(1, -1)

    if q.shape[0] == 0 or c.shape[0] == 0:
        return np.empty((q.shape[0], c.shape[0]), dtype=np.float32)

    # Normalize vectors
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c, axis=1, keepdims=True)

    q_norms[q_norms == 0.0] = 1.0
    c_norms[c_norms == 0.0] = 1.0

    q_normed = q / q_norms
    c_normed = c / c_norms

    # Dot product matrix
    sim_matrix = np.dot(q_normed, c_normed.T)
    return np.clip(sim_matrix, -1.0, 1.0)


def compute_text_similarity(
    text_a: str,
    text_b: str,
    model: Optional[Any] = None
) -> float:
    """
    End-to-end convenience function to encode two text strings and return their
    cosine similarity score.

    Args:
        text_a: First text string.
        text_b: Second text string.
        model: Optional pre-loaded SentenceTransformer model.

    Returns:
        Cosine similarity score float in range [-1.0, 1.0].
    """
    vec_a = embed_text(text_a, model=model, normalize=True)
    vec_b = embed_text(text_b, model=model, normalize=True)
    return compute_cosine_similarity(vec_a, vec_b)


# ===========================================================================
# 4. Resume & Job Description Embedding Profiles
# ===========================================================================

def build_resume_embedding_text(parsed_resume: Dict[str, Any]) -> str:
    """
    Synthesizes a structured, token-dense representation of a parsed resume
    specifically designed for embedding comparison against job descriptions.

    Args:
        parsed_resume: Dictionary output from `src.parser.parse_resume`.

    Returns:
        Formatted summary string prioritizing skills, education, and experience.
    """
    candidate_name = parsed_resume.get("candidate_name") or "Candidate"
    
    # Skills string
    skills_data = parsed_resume.get("skills", {})
    all_skills = skills_data.get("all_skills", [])
    skills_str = ", ".join(all_skills) if all_skills else "general technical background"

    # Education string
    education_entries = parsed_resume.get("education", [])
    edu_summaries = []
    for edu in education_entries[:3]:
        degree = edu.get("degree", "")
        field = edu.get("field", "")
        inst = edu.get("institution", "")
        summary_part = f"{degree} in {field}" if field else degree
        if inst:
            summary_part = f"{summary_part} from {inst}"
        if summary_part.strip():
            edu_summaries.append(summary_part.strip())
    edu_str = "; ".join(edu_summaries) if edu_summaries else "unspecified degree"

    # Cleaned body preview
    cleaned_text = parsed_resume.get("cleaned_text", "")
    body_snippet = " ".join(cleaned_text.split()[:200])

    return (
        f"Candidate: {candidate_name}. "
        f"Key Skills: {skills_str}. "
        f"Education: {edu_str}. "
        f"Professional Background: {body_snippet}"
    )


def embed_resume(
    parsed_resume: Dict[str, Any],
    model: Optional[Any] = None
) -> np.ndarray:
    """
    Generates a dense vector embedding for a parsed resume dictionary.

    Args:
        parsed_resume: Dictionary from `src.parser.parse_resume`.
        model: Optional SentenceTransformer model.

    Returns:
        1D normalized NumPy vector embedding.
    """
    dense_text = build_resume_embedding_text(parsed_resume)
    return embed_text(dense_text, model=model, normalize=True)


def embed_job(
    parsed_job: Dict[str, Any],
    model: Optional[Any] = None
) -> np.ndarray:
    """
    Generates a dense vector embedding for a parsed job description dictionary.

    Args:
        parsed_job: Dictionary from `src.parser.parse_job_description`.
        model: Optional SentenceTransformer model.

    Returns:
        1D normalized NumPy vector embedding.
    """
    dense_profile = parsed_job.get("dense_profile")
    if not dense_profile:
        # Fallback to cleaned text snippet if dense_profile is not present
        title = parsed_job.get("job_title", "Job Opening")
        cleaned = parsed_job.get("cleaned_text", "")
        dense_profile = f"Job Title: {title}. Requirements: {' '.join(cleaned.split()[:180])}"

    return embed_text(dense_profile, model=model, normalize=True)


# ===========================================================================
# 5. ChromaDB Vector Database Storage Manager
# ===========================================================================

class ChromaVectorStore:
    """
    Production wrapper for ChromaDB vector database interactions.
    Supports persistent disk storage or in-memory ephemeral storage,
    batch document ingestion, and cosine-similarity nearest neighbor search.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        embedding_model: Optional[Any] = None
    ):
        """
        Initializes the ChromaDB client.

        Args:
            persist_dir: Directory path for on-disk persistence (default: 'models/chroma_db').
            in_memory: If True, uses ephemeral in-memory storage (useful for tests and volatile sessions).
            embedding_model: Optional SentenceTransformer instance used to generate embeddings.
        """
        if chromadb is None:
            raise ImportError(
                "The 'chromadb' library is required but not installed. "
                "Please install it using 'pip install chromadb'."
            )

        self.in_memory = in_memory
        self.embedding_model = embedding_model

        if self.in_memory:
            logger.info("Initializing in-memory Ephemeral ChromaDB client...")
            self.client = chromadb.EphemeralClient()
            self.persist_dir = None
        else:
            resolved_dir = Path(persist_dir or DEFAULT_CHROMA_DIR)
            resolved_dir.mkdir(parents=True, exist_ok=True)
            self.persist_dir = str(resolved_dir)
            logger.info("Initializing persistent ChromaDB client at '%s'...", self.persist_dir)
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Retrieves an existing collection or creates a new one configured with cosine distance.

        Args:
            collection_name: Name of the collection (e.g., 'job_descriptions', 'resumes').
            metadata: Optional collection-level metadata.

        Returns:
            ChromaDB Collection instance.
        """
        collection_metadata = metadata or {}
        # Set cosine space if not already defined
        if "hnsw:space" not in collection_metadata:
            collection_metadata["hnsw:space"] = "cosine"

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata
        )

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 128
    ) -> int:
        """
        Adds text documents, metadata, and embeddings to a collection in batches.
        If embeddings are not provided, they are automatically generated using `embed_batch`.

        Args:
            collection_name: Target collection name.
            documents: List of text content for each record.
            ids: Unique identifiers for each document.
            metadatas: Optional list of metadata dictionaries.
            embeddings: Optional precomputed 2D float32 NumPy array of embeddings.
            batch_size: Ingestion batch size.

        Returns:
            Number of documents successfully indexed.
        """
        if not documents:
            return 0

        if len(documents) != len(ids):
            raise ValueError(f"Mismatch: received {len(documents)} documents and {len(ids)} IDs.")

        collection = self.get_or_create_collection(collection_name)
        total_items = len(documents)

        # Compute embeddings if not provided
        if embeddings is None:
            logger.info("Computing embeddings for %d documents...", total_items)
            embeddings = embed_batch(
                documents,
                model=self.embedding_model,
                batch_size=min(batch_size, 64),
                normalize=True
            )

        # Sanitize metadatas (ChromaDB requires non-empty dicts of primitives: str, int, float, bool)
        clean_metadatas = None
        if metadatas is not None:
            clean_metadatas = []
            for meta in metadatas:
                sanitized_meta = {}
                for k, v in (meta or {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    elif isinstance(v, (list, tuple)):
                        sanitized_meta[k] = ", ".join(str(item) for item in v)
                    elif v is None:
                        sanitized_meta[k] = ""
                    else:
                        sanitized_meta[k] = str(v)
                # ChromaDB requires metadata dicts to be non-empty if passed
                if not sanitized_meta:
                    sanitized_meta["_has_metadata"] = False
                clean_metadatas.append(sanitized_meta)

        # Batch upsert to ChromaDB
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            batch_docs = documents[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()

            upsert_kwargs = {
                "ids": batch_ids,
                "documents": batch_docs,
                "embeddings": batch_embeddings
            }
            if clean_metadatas is not None:
                upsert_kwargs["metadatas"] = clean_metadatas[i:end_idx]

            collection.upsert(**upsert_kwargs)

        logger.info(
            "Successfully upserted %d documents into collection '%s'.",
            total_items, collection_name
        )
        return total_items

    def query(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a nearest-neighbor vector similarity search against a collection.

        Args:
            collection_name: Collection to search.
            query_text: Plain text query (embedded automatically if query_embedding not passed).
            query_embedding: Precomputed 1D vector embedding.
            n_results: Maximum number of top matches to return.
            where: Filter dict on metadata fields (e.g., {"chunk_type": "dense_profile"}).
            where_document: Filter dict on document content.

        Returns:
            List of result dictionaries containing 'id', 'document', 'metadata',
            'distance', and 'similarity' (1.0 - cosine_distance).
        """
        collection = self.get_or_create_collection(collection_name)
        total_docs = collection.count()
        if total_docs == 0:
            return []

        limit = min(n_results, total_docs)

        # Resolve query embedding
        if query_embedding is not None:
            resolved_embedding = np.asarray(query_embedding, dtype=np.float32).flatten().tolist()
            results = collection.query(
                query_embeddings=[resolved_embedding],
                n_results=limit,
                where=where,
                where_document=where_document
            )
        elif query_text is not None:
            vec = embed_text(query_text, model=self.embedding_model, normalize=True).tolist()
            results = collection.query(
                query_embeddings=[vec],
                n_results=limit,
                where=where,
                where_document=where_document
            )
        else:
            raise ValueError("Must provide either 'query_text' or 'query_embedding'.")

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else ["" for _ in ids]
        metas = results["metadatas"][0] if results.get("metadatas") else [{} for _ in ids]
        distances = results["distances"][0] if results.get("distances") else [0.0 for _ in ids]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # For cosine space: distance = 1.0 - cosine_similarity
            # Therefore similarity = 1.0 - distance (bounded to [0.0, 1.0])
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            formatted_results.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": float(dist),
                "similarity": similarity,
                "match_percentage": round(similarity * 100.0, 2)
            })

        return formatted_results

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a single document and its metadata by ID.

        Args:
            collection_name: Target collection.
            doc_id: Unique document identifier.

        Returns:
            Dictionary with document details or None if not found.
        """
        collection = self.get_or_create_collection(collection_name)
        result = collection.get(ids=[doc_id])
        if result and result.get("ids") and len(result["ids"]) > 0:
            doc = result["documents"][0] if result.get("documents") else ""
            meta = result["metadatas"][0] if result.get("metadatas") else {}
            return {"id": doc_id, "document": doc, "metadata": meta}
        return None

    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """
        Deletes documents by their IDs from a collection.

        Args:
            collection_name: Target collection.
            ids: List of document IDs to delete.

        Returns:
            True if deletion was executed.
        """
        if not ids:
            return False
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=ids)
        return True

    def count(self, collection_name: str) -> int:
        """
        Returns the total number of items stored in a collection.

        Args:
            collection_name: Target collection.

        Returns:
            Total count integer.
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def reset_collection(self, collection_name: str) -> bool:
        """
        Deletes and recreates an empty collection.

        Args:
            collection_name: Collection to reset.

        Returns:
            True if collection was reset.
        """
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        self.get_or_create_collection(collection_name)
        return True

    def list_collections(self) -> List[str]:
        """
        Lists all collections currently present in the vector store.

        Returns:
            List of collection names.
        """
        collections = self.client.list_collections()
        # Collections can be objects with .name attribute or string names depending on Chroma version
        names = []
        for col in collections:
            if hasattr(col, "name"):
                names.append(col.name)
            else:
                names.append(str(col))
        return names


# ===========================================================================
# 6. Bulk Dataset Indexing Utilities
# ===========================================================================

def index_semantic_chunks_csv(
    csv_path: Union[str, Path] = "data/processed/semantic_chunks.csv",
    vector_store: Optional[ChromaVectorStore] = None,
    collection_name: str = "job_descriptions",
    batch_size: int = 128
) -> int:
    """
    Ingests preprocessed semantic chunks from CSV into ChromaDB.
    Reads chunk_id, job_id, job_title, chunk_type, and chunk_text.

    Args:
        csv_path: Path to semantic_chunks.csv.
        vector_store: Optional ChromaVectorStore instance. If None, creates persistent store.
        collection_name: Destination collection name.
        batch_size: Batch size for embedding and ingestion.

    Returns:
        Total number of chunks successfully indexed.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic chunks CSV file not found at: {path}")

    logger.info("Reading semantic chunks from '%s'...", path)
    df = pd.read_csv(path)
    required_cols = {"chunk_id", "job_id", "job_title", "chunk_type", "chunk_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Semantic chunks CSV is missing required columns: {missing}")

    # Drop null text rows
    df = df.dropna(subset=["chunk_text", "chunk_id"]).copy()

    ids = df["chunk_id"].astype(str).tolist()
    documents = df["chunk_text"].astype(str).tolist()
    metadatas = df[["job_id", "job_title", "chunk_type"]].to_dict(orient="records")

    store = vector_store or ChromaVectorStore()
    indexed_count = store.add_documents(
        collection_name=collection_name,
        documents=documents,
        ids=ids,
        metadatas=metadatas,
        batch_size=batch_size
    )

    logger.info("Successfully indexed %d semantic chunks into '%s'.", indexed_count, collection_name)
    return indexed_count


def index_job_descriptions_csv(
    csv_path: Union[str, Path] = "data/processed/cleaned_job_descriptions.csv",
    vector_store: Optional[ChromaVectorStore] = None,
    collection_name: str = "job_dense_profiles",
    batch_size: int = 128
) -> int:
    """
    Ingests cleaned job descriptions (using their token-dense profiles) into ChromaDB.

    Args:
        csv_path: Path to cleaned_job_descriptions.csv.
        vector_store: Optional ChromaVectorStore instance.
        collection_name: Destination collection name.
        batch_size: Ingestion batch size.

    Returns:
        Total number of jobs indexed.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned job descriptions CSV not found at: {path}")

    df = pd.read_csv(path)
    required_cols = {"job_id", "Job Title", "dense_profile"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Job descriptions CSV is missing required columns: {missing}")

    df = df.dropna(subset=["dense_profile", "job_id"]).copy()

    ids = df["job_id"].astype(str).tolist()
    documents = df["dense_profile"].astype(str).tolist()

    metadatas = []
    for _, row in df.iterrows():
        meta = {
            "job_id": str(row["job_id"]),
            "job_title": str(row.get("Job Title", "Job Opening")),
            "skills": str(row.get("skills_str", "")),
            "skill_count": int(row.get("skill_count", 0)),
            "word_count": int(row.get("clean_word_count", 0)),
            "chunk_type": "dense_profile"
        }
        metadatas.append(meta)

    store = vector_store or ChromaVectorStore()
    indexed_count = store.add_documents(
        collection_name=collection_name,
        documents=documents,
        ids=ids,
        metadatas=metadatas,
        batch_size=batch_size
    )

    logger.info("Successfully indexed %d dense job profiles into '%s'.", indexed_count, collection_name)
    return indexed_count
