# 📄 Containerized NLP Pipeline for Automated Resume Screening and Candidate Ranking
### *Semantic Resume & Job Matcher*

An end-to-end, production-grade Natural Language Processing (NLP) and Information Retrieval system that screens and matches candidate resumes against target job descriptions using dense semantic vector embeddings, Named Entity Recognition (NER), domain-specific skill extraction, and vector similarity search—fully containerized using Docker and Docker Compose.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Technical Stack](#-technical-stack)
3. [Repository Structure](#-repository-structure)
4. [Raw Dataset Audit & Deep Diagnostics](#-raw-dataset-audit--deep-diagnostics)
   - [Overview & Schema](#overview--schema)
   - [Key Anomalies & Outliers Discovered](#key-anomalies--outliers-discovered)
   - [The Token Window Limitation & Why Preprocessing Matters](#the-token-window-limitation--why-preprocessing-matters)
5. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
   - [Pipeline Stages](#pipeline-stages)
   - [Skill Extraction Taxonomy](#skill-extraction-taxonomy)
   - [Dual-Strategy Semantic Chunking](#dual-strategy-semantic-chunking)
6. [Processed Dataset Artifacts](#-processed-dataset-artifacts)
7. [Getting Started & Local Execution](#-getting-started--local-execution)
8. [Next Steps in the Implementation Roadmap](#-next-steps-in-the-implementation-roadmap)

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Browser                       │
└                            ▲                            │
                             │ HTTP / UI                  │
┌────────────────────────────▼────────────────────────────┐
│                    Streamlit / FastAPI                  │
│                (Web Interface & API Layer)              │
└─────────┬───────────────────────────────────┬───────────┘
          │ Parse PDF                         │ Compute Similarity
          ▼                                   ▼
┌───────────────────┐                 ┌───────────────────┐
│  spaCy / PyPDF    │                 │ Sentence-Transformers │
│  (Text Extractor) │                 │  (all-MiniLM-L6-v2) │
└───────────────────┘                 └─────────┬─────────┘
                                                │ Vector Search
                                      ┌─────────▼─────────┐
                                      │  ChromaDB / FAISS │
                                      │  (Vector Database)│
                                      └───────────────────┘
```

The application provides a two-page interactive Streamlit dashboard:
- **Match Dashboard:** Upload an individual candidate resume (PDF), compare it against a target job description, and view an instant semantic match score percentage, matched skills, and recommended missing skill gaps.
- **Batch Candidate Ranking:** Ingest a batch of candidate resumes (CSV/PDFs), rank them against a target job description using vector similarity, and export candidate leaderboards.

---

## 💻 Technical Stack

- **Language & Runtime:** Python 3.10+
- **Core NLP & Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), `PyTorch`, `Hugging Face transformers`, `spaCy`
- **Document Ingestion:** `PyPDF` / `pdfplumber`
- **Vector Database:** `ChromaDB` / `FAISS`
- **Relational Storage:** `SQLite` (for audit history, job postings, and scoring logs)
- **User Interface:** `Streamlit` with `Plotly` interactive charts
- **Data Preprocessing & Analysis:** `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`, `Jupyter Notebook`
- **DevOps & Containerization:** `Docker`, `Docker Compose`, multi-stage builds

---

## 📁 Repository Structure

```
Containerized-NLP-Pipeline-for-Automated-Resume-Screening-and-Candidate-Ranking/
├── data/
│   ├── raw/                             # Original raw datasets
│   │   ├── data.csv                     # Raw job postings (521 rows, 2 columns)
│   │   ├── data_01.csv                  # Byte-identical copy of data.csv
│   │   └── .gitkeep
│   └── processed/                       # Cleaned, normalized, and chunked outputs
│       ├── cleaned_job_descriptions.csv # 517 validated, cleaned job records
│       ├── job_skills_extracted.csv     # Extracted technical skills per job
│       ├── semantic_chunks.csv          # 3,284 semantic chunks for vector embeddings
│       └── .gitkeep
├── models/                              # Cached embedding models & vector stores
│   └── .gitkeep
├── notebooks/                           # Jupyter Notebooks for EDA & Preprocessing
│   └── 01_data_preprocessing.ipynb      # Complete, fully-executed preprocessing pipeline
├── src/                                 # Production application modules
│   ├── __init__.py
│   ├── parser.py                        # Document parsing & text extraction
│   ├── embedder.py                      # Vector embeddings & vector store queries
│   ├── matcher.py                       # Skill gap analysis & scoring algorithms
│   └── database.py                      # SQLite history tracking & CRUD operations
├── app.py                               # Streamlit interactive application entrypoint
├── Dockerfile                           # Optimized multi-stage Docker build
├── docker-compose.yml                   # Multi-container orchestration
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 🔍 Raw Dataset Audit & Deep Diagnostics

### Overview & Schema
The raw dataset located in `data/raw/data.csv` contains scraped job descriptions:
- **Total Records:** 521 rows
- **Columns:** 2 (`Job Title`, `Description`)
- **Memory Footprint:** ~8.3 KB (metadata), 2.41 MB (on-disk CSV)
- **Duplicate Files:** `data/raw/data_01.csv` was verified through byte-level comparison to be an **exact 1:1 duplicate** of `data.csv`.

### Key Anomalies & Outliers Discovered

During deep exploratory data analysis, several critical dataset quirks and anomalies were uncovered:

1. **Degenerate Placeholder & Short Records:**
   - **Row 18:** `Job Title: "Data Analyst"`, `Description: "na"` (1 word, placeholder string).
   - **Row 248:** `Job Title: "Business Analyst"`, `Description: "role business analyst location brooklyn, nyhybrid 3 day onsite"` (9 words, location snippet only).
   - **Row 283:** 18-word summary snippet without any listed skills or qualifications.
   - **Row 349:** 20-word recruiter rate-card header (`"usc,gc,h4 job title data analyst 1 client cisco duration 12 months c2c rate 55hr..."`).
   - *Resolution:* All records with `< 30 words` are flagged and filtered out of the active retrieval store.

2. **Typographical Unicode Ligatures (The "fi/fl" Bug):**
   - The raw text contains Unicode ligatures such as `\ufb01` ("fi"), `\ufb02` ("fl"), and `\u0153` ("oe").
   - Words like **`benefits`**, **`profit`**, **`efficient`**, and **`significant`** appear in raw text as `bene\ufb01ts`, `pro\ufb01t`, `e\ufb03cient`.
   - *Impact:* Standard regex and subword tokenizers treat these ligatures as unknown tokens (`[UNK]`), corrupting keyword extraction and embedding representations.
   - *Resolution:* Decomposed through explicit character mapping and `unicodedata.normalize('NFKD', text)`.

3. **High Proportion of Non-Semantic Boilerplate:**
   - **58.2%** (303/521 rows) contain Equal Employment Opportunity (EEO) legal statements.
   - **59.1%** (308/521 rows) contain employee benefits disclaimers (401(k), dental, vision, PTO).
   - **31.5%** (164/521 rows) mention salary ranges, compensation figures, or hourly rates.
   - **82 rows** contain external application URLs (`www...`).
   - *Impact:* Boilerplate dilutes vector cosine similarity away from technical qualifications.
   - *Resolution:* Targeted regex stripping eliminates legal and insurance boilerplate while preserving core job requirements.

4. **Multi-lingual Postings:**
   - 4 job postings (Rows 56, 161, 278, 356) contain French sections or are predominantly in French (e.g. Montreal-based roles).

### The Token Window Limitation & Why Preprocessing Matters
- The dense embedding model (`sentence-transformers/all-MiniLM-L6-v2`) has a maximum context window of **256 word pieces** (~180-200 words).
- The raw dataset has an average length of **650 words** (4,601 characters), with the longest descriptions reaching **2,146 words** (14,638 characters).
- **Over 89%** of raw job postings exceed the transformer's single-pass token budget.
- Feeding raw job descriptions directly into the model results in **truncating 60-75% of the document**, silently dropping technical requirements and qualifications positioned in the middle or bottom of the posting.
- Preprocessing and **semantic chunking** are therefore mandatory to ensure complete skill coverage.

---

## ⚙️ Data Preprocessing Pipeline

The complete preprocessing pipeline is implemented and documented in [`notebooks/01_data_preprocessing.ipynb`](notebooks/01_data_preprocessing.ipynb).

```
Raw CSV Ingestion
       │
       ▼
Quality Filtering (Drop < 30 word stubs)
       │
       ▼
Unicode & Ligature Normalization (NFKD + fi/fl map)
       │
       ▼
Boilerplate & Noise Stripping (EEO, Benefits, URLs)
       │
       ▼
Domain Skill Extraction & Categorization (Regex Taxonomy)
       │
       ▼
Dual-Strategy Semantic Chunking (Dense Profiles + Sliding Windows)
       │
       ▼
Export Processed Datasets (data/processed/)
```

### Skill Extraction Taxonomy
The pipeline extracts technical competencies across 7 key domain categories:
- **Programming & Scripting:** Python, R, SQL, T-SQL, PL/SQL, SAS, Scala, Java, C++, C#, Bash, VBA
- **BI & Visualization:** Tableau, Power BI, Looker, Qlik, Excel, SSRS, SSIS, SSAS, QuickSight, Metabase
- **Cloud & Big Data:** AWS, Azure, GCP, Snowflake, Databricks, Spark, Hadoop, Redshift, BigQuery, Kafka, Airflow
- **Databases:** PostgreSQL, MySQL, Oracle, SQL Server, MongoDB, DynamoDB, Redis, Cassandra
- **Analytics, Statistics & ML:** Machine Learning, Statistics, Predictive Modeling, Regression, Hypothesis Testing, A/B Testing, Deep Learning, NLP, Time Series
- **Data Engineering & Governance:** ETL, ELT, Data Modeling, Data Warehousing, Data Governance, Data Quality, Data Pipelines
- **Methodologies & Tools:** Git, Jira, Agile, Scrum, CI/CD, Docker

### Top In-Demand Skills Discovered in Dataset
1. **SQL:** 73.1% of postings
2. **Excel:** 58.0% of postings
3. **Tableau:** 50.7% of postings
4. **Power BI:** 48.9% of postings
5. **Python:** 44.5% of postings
6. **ETL:** 29.8% of postings
7. **R:** 23.4% of postings
8. **Data Modeling:** 20.9% of postings
9. **Statistics:** 18.2% of postings
10. **AWS:** 17.0% of postings

### Dual-Strategy Semantic Chunking
To reconcile long documents with `all-MiniLM-L6-v2`'s 256-token limit:
1. **Dense Profile Chunk:** A curated, high-information-density passage composed of `Job Title + Extracted Skills + First 120 words of Responsibilities` (strictly constrained under 180 words). This chunk provides a high-level representation for fast candidate ranking.
2. **Sliding Window Body Chunks:** Cleaned job descriptions are chunked into 150-word passages with a 30-word overlap to ensure fine-grained vector similarity search across all sections of a job posting.

---

## 📦 Processed Dataset Artifacts

The preprocessing notebook outputs three ready-to-use artifacts in `data/processed/`:

| Artifact | File Path | Record Count | Description |
|---|---|---|---|
| **Cleaned Job Descriptions** | `data/processed/cleaned_job_descriptions.csv` | 517 rows | Validated jobs with normalized text, word counts, skill lists, and dense profile strings. |
| **Extracted Skills** | `data/processed/job_skills_extracted.csv` | 517 rows | Structured mapping of skills per job title, including categorized JSON dictionaries. |
| **Semantic Chunks** | `data/processed/semantic_chunks.csv` | 3,284 chunks | Dense profiles (517) + sliding window chunks (2,767) formatted for ChromaDB/FAISS ingestion. |

---

## 🚀 Getting Started & Local Execution

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Setup
Clone the repository and install the initial data science and NLP dependencies:

```bash
# Clone repository
git clone https://github.com/Yosa03-dat/Containerized-NLP-Pipeline-for-Automated-Resume-Screening-and-Candidate-Ranking.git
cd Containerized-NLP-Pipeline-for-Automated-Resume-Screening-and-Candidate-Ranking

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Preprocessing Notebook
Launch JupyterLab or VS Code to explore and execute the step-by-step pipeline:

```bash
jupyter lab notebooks/01_data_preprocessing.ipynb
```

The notebook is pre-executed with all statistical summaries, anomaly diagnostics, and Matplotlib/Seaborn visualization plots embedded.

---

## 🗺️ Next Steps in the Implementation Roadmap

- [x] **Phase 1: Raw Data Exploration & Diagnostic Audit**
- [x] **Phase 2: Text Preprocessing, Ligature Fixing & Semantic Chunking Notebook**
- [x] **Phase 3: Repository Documentation & Architecture Spec (`README.md`)**
- [x] **Phase 4: Parsing Layer (`src/parser.py`)** — PDF text extraction (PyPDF/pdfplumber) and spaCy NER candidate contact/skills extraction.
- [ ] **Phase 5: Embedding & Vector Search Engine (`src/embedder.py`)** — Sentence-Transformers (`all-MiniLM-L6-v2`) vector store setup with ChromaDB.
- [ ] **Phase 6: Skill Gap Matcher & Scoring Algorithm (`src/matcher.py`)** — Cosine similarity computation combined with weighted keyword overlap.
- [ ] **Phase 7: SQLite Database Layer (`src/database.py`)** — Search logging, candidate ranking history, and session management.
- [ ] **Phase 8: Multi-Page Streamlit Dashboard (`app.py`)** — Single resume evaluation and batch candidate ranking interfaces.
- [ ] **Phase 9: Docker Containerization (`Dockerfile`, `docker-compose.yml`)** — Multi-stage slim container build with persistent volume mounts.
