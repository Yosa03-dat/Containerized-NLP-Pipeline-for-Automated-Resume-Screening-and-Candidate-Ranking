"""
Unit tests for Dense Vector Embedding Engine and ChromaDB Vector Store (src/embedder.py).
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedder import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM,
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
)


class TestEmbeddingModelAndDevice(unittest.TestCase):
    """Tests for device detection and model singleton initialization."""

    def test_get_optimal_device(self):
        device = get_optimal_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_model_initialization(self):
        model = get_embedding_model()
        self.assertIsNotNone(model)
        # Ensure singleton returns the identical object
        model_again = get_embedding_model()
        self.assertIs(model, model_again)


class TestVectorEmbeddings(unittest.TestCase):
    """Tests for vector encoding, normalization, and dimensional consistency."""

    def test_embed_text_shape_and_norm(self):
        text = "Experienced Senior Data Analyst skilled in Python, SQL, and Power BI."
        vec = embed_text(text, normalize=True)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (EMBEDDING_DIM,))
        self.assertEqual(vec.dtype, np.float32)

        # L2 norm of normalized vector should be approximately 1.0
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_embed_empty_and_whitespace_text(self):
        empty_vec = embed_text("")
        self.assertEqual(empty_vec.shape, (EMBEDDING_DIM,))
        self.assertTrue(np.all(empty_vec == 0.0))

        whitespace_vec = embed_text("   \n\t  ")
        self.assertEqual(whitespace_vec.shape, (EMBEDDING_DIM,))
        self.assertTrue(np.all(whitespace_vec == 0.0))

    def test_embed_batch(self):
        texts = [
            "Data Analyst with SQL and Tableau experience.",
            "Cloud Engineer specializing in AWS and Kubernetes.",
            "Financial Analyst focused on financial modeling."
        ]
        matrix = embed_batch(texts, batch_size=2, normalize=True)

        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (3, EMBEDDING_DIM))
        self.assertEqual(matrix.dtype, np.float32)

        # Each row should have unit norm
        norms = np.linalg.norm(matrix, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_embed_batch_empty(self):
        matrix = embed_batch([])
        self.assertEqual(matrix.shape, (0, EMBEDDING_DIM))


class TestSimilarityMetrics(unittest.TestCase):
    """Tests for cosine similarity and pairwise similarity matrix calculations."""

    def test_identical_text_similarity(self):
        text = "Machine Learning Engineer with PyTorch and NLP experience."
        sim = compute_text_similarity(text, text)
        self.assertAlmostEqual(sim, 1.0, places=3)

    def test_semantic_relevance_ordering(self):
        anchor = "Data Scientist proficient in Python and Machine Learning algorithms."
        related = "ML Engineer building predictive neural network models using Python."
        unrelated = "Pastry chef baking artisanal French sourdough bread and croissants."

        sim_related = compute_text_similarity(anchor, related)
        sim_unrelated = compute_text_similarity(anchor, unrelated)

        self.assertGreater(sim_related, sim_unrelated)
        self.assertGreater(sim_related, 0.4)
        self.assertLess(sim_unrelated, 0.3)

    def test_compute_cosine_similarity_edge_cases(self):
        zero_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        random_vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
        
        # Zero vector similarity should be 0.0
        self.assertEqual(compute_cosine_similarity(zero_vec, random_vec), 0.0)

    def test_compute_similarity_matrix(self):
        queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        corpus = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        matrix = compute_similarity_matrix(queries, corpus)
        self.assertEqual(matrix.shape, (2, 3))
        # query 0 should have similarity 1.0 with corpus 0
        self.assertAlmostEqual(matrix[0, 0], 1.0, places=3)
        # query 0 should have similarity 0.0 with corpus 1
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=3)


class TestResumeJobEmbeddingProfiles(unittest.TestCase):
    """Tests for high-level resume and job description embedding integration."""

    def test_build_resume_embedding_text(self):
        sample_resume = {
            "candidate_name": "Sarah Connor",
            "skills": {
                "all_skills": ["python", "sql", "aws", "docker"],
                "skills_by_category": {"programming": ["python", "sql"]}
            },
            "education": [
                {"degree": "Bachelor of Science", "field": "Computer Science", "institution": "MIT"}
            ],
            "cleaned_text": "Experienced software engineer with 5 years building backend microservices."
        }
        dense_str = build_resume_embedding_text(sample_resume)
        self.assertIn("Sarah Connor", dense_str)
        self.assertIn("python", dense_str)
        self.assertIn("MIT", dense_str)
        self.assertIn("Bachelor of Science", dense_str)

    def test_embed_resume_and_job(self):
        sample_resume = {
            "candidate_name": "John Doe",
            "skills": {"all_skills": ["python", "sql"]},
            "education": [],
            "cleaned_text": "Data analyst with SQL experience."
        }
        sample_job = {
            "job_title": "Data Analyst",
            "dense_profile": "Job Title: Data Analyst. Required Skills: python, sql. Responsibilities: analyze data."
        }

        resume_vec = embed_resume(sample_resume)
        job_vec = embed_job(sample_job)

        self.assertEqual(resume_vec.shape, (EMBEDDING_DIM,))
        self.assertEqual(job_vec.shape, (EMBEDDING_DIM,))

        sim = compute_cosine_similarity(resume_vec, job_vec)
        self.assertGreater(sim, 0.5)


class TestChromaVectorStore(unittest.TestCase):
    """Tests for ChromaDB vector store operations using ephemeral in-memory storage."""

    def setUp(self):
        # Initialize an isolated in-memory vector store for testing
        self.store = ChromaVectorStore(in_memory=True)
        self.collection_name = "test_jobs"
        self.store.reset_collection(self.collection_name)

    def test_add_and_query_documents(self):
        docs = [
            "Senior Data Analyst with expertise in Tableau, Power BI, and SQL.",
            "Lead DevOps Engineer managing Kubernetes clusters and Terraform on AWS.",
            "Marketing Specialist managing Google Ads and email marketing campaigns."
        ]
        ids = ["job_1", "job_2", "job_3"]
        metas = [
            {"job_id": "1", "title": "Data Analyst", "domain": "Analytics"},
            {"job_id": "2", "title": "DevOps Engineer", "domain": "Infrastructure"},
            {"job_id": "3", "title": "Marketing Specialist", "domain": "Marketing"}
        ]

        # Ingest documents
        count_added = self.store.add_documents(
            collection_name=self.collection_name,
            documents=docs,
            ids=ids,
            metadatas=metas
        )
        self.assertEqual(count_added, 3)
        self.assertEqual(self.store.count(self.collection_name), 3)

        # Query using a semantically related query string
        query = "Looking for a candidate with strong SQL, business intelligence, and Tableau skills."
        results = self.store.query(
            collection_name=self.collection_name,
            query_text=query,
            n_results=2
        )

        self.assertEqual(len(results), 2)
        top_match = results[0]
        self.assertEqual(top_match["id"], "job_1")
        self.assertEqual(top_match["metadata"]["title"], "Data Analyst")
        self.assertGreater(top_match["similarity"], 0.4)
        self.assertGreater(top_match["match_percentage"], 40.0)

    def test_get_document(self):
        self.store.add_documents(
            collection_name=self.collection_name,
            documents=["Sample text"],
            ids=["doc_100"],
            metadatas=[{"dept": "Engineering"}]
        )
        doc = self.store.get_document(self.collection_name, "doc_100")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["id"], "doc_100")
        self.assertEqual(doc["metadata"]["dept"], "Engineering")

        # Non-existent doc
        missing = self.store.get_document(self.collection_name, "non_existent")
        self.assertIsNone(missing)

    def test_delete_and_reset(self):
        self.store.add_documents(
            collection_name=self.collection_name,
            documents=["A", "B", "C"],
            ids=["id_a", "id_b", "id_c"]
        )
        self.assertEqual(self.store.count(self.collection_name), 3)

        # Delete single doc
        self.store.delete_documents(self.collection_name, ["id_a"])
        self.assertEqual(self.store.count(self.collection_name), 2)

        # Reset collection
        self.store.reset_collection(self.collection_name)
        self.assertEqual(self.store.count(self.collection_name), 0)


if __name__ == "__main__":
    unittest.main()
