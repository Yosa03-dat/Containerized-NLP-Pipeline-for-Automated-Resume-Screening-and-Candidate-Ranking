"""
Semantic Resume & Job Matcher — Production Dashboard.

Pages:
  1. Match Dashboard  — single-candidate PDF/text evaluation with skill gap analysis.
  2. Batch Ranking    — multi-candidate cohort screening and leaderboard.
  3. Audit History    — SQLite-backed evaluation records with analytics.
  4. System Explorer  — architecture diagram, skill taxonomy, and runtime diagnostics.
"""

import io
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.parser import (
    extract_text_from_pdf,
    clean_text,
    clean_job_description,
    parse_resume,
    parse_job_description,
    SKILLS_TAXONOMY,
)
from src.embedder import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM,
    get_optimal_device,
    get_embedding_model,
    embed_text,
    compute_cosine_similarity,
)
from src.matcher import (
    analyze_skill_gap,
    compute_match_score,
    evaluate_resume_against_job,
    rank_candidates_batch,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_SKILL_WEIGHT,
)
from src.database import (
    init_db,
    save_match_result,
    get_match_history,
    get_match_by_id,
    delete_match_record,
    clear_match_history,
    save_batch_session,
    get_batch_sessions,
    get_batch_candidates,
    get_database_stats,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RecruitIQ — AI Resume Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ---------------------------------------------------------------------------
# Global CSS — premium dark-accented design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
  --brand:      #6366F1;
  --brand-dark: #4F46E5;
  --brand-light:#EEF2FF;
  --success:    #10B981;
  --warning:    #F59E0B;
  --danger:     #EF4444;
  --neutral:    #64748B;
  --surface:    #FFFFFF;
  --surface-2:  #F8FAFC;
  --border:     #E2E8F0;
  --text:       #0F172A;
  --text-muted: #64748B;
  --radius:     12px;
  --shadow-sm:  0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
  --shadow-md:  0 4px 6px -1px rgba(0,0,0,.1), 0 2px 4px -1px rgba(0,0,0,.06);
  --shadow-lg:  0 10px 15px -3px rgba(0,0,0,.1), 0 4px 6px -2px rgba(0,0,0,.05);
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
  border-right: none;
}
[data-testid="stSidebar"] * { color: #E0E7FF !important; }
[data-testid="stSidebar"] .stRadio label { color: #C7D2FE !important; }
[data-testid="stSidebar"] .stSlider label { color: #A5B4FC !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebarContent"] .stAlert { background: rgba(255,255,255,0.08); }

/* Sidebar brand block */
.sidebar-brand {
  display: flex; align-items: center; gap: 12px;
  padding: 0 0 20px 0;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  margin-bottom: 20px;
}
.sidebar-brand-icon {
  width: 44px; height: 44px; border-radius: 10px;
  background: linear-gradient(135deg, #818CF8, #6366F1);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; flex-shrink: 0;
}
.sidebar-brand-text h2 { color: #FFFFFF !important; font-size: 1.1rem; font-weight: 700; margin: 0; }
.sidebar-brand-text p  { color: #A5B4FC !important; font-size: 0.72rem; margin: 0; }

/* Sidebar nav pills */
.stRadio [data-testid="stMarkdownContainer"] p { font-weight: 500; }
div[role="radiogroup"] label {
  border-radius: 8px !important;
  padding: 8px 12px !important;
  margin-bottom: 4px !important;
  transition: background .15s;
}
div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.1); }

/* Sidebar stat card */
.stat-pill {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 12px 16px;
  margin-top: 4px;
}
.stat-pill .sp-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 6px;
}
.stat-pill .sp-row:last-child { margin-bottom: 0; }
.stat-pill .sp-label { font-size: 0.72rem; color: #A5B4FC !important; }
.stat-pill .sp-val   { font-size: 0.85rem; font-weight: 600; color: #FFFFFF !important; }

/* ── Page header ── */
.page-hero {
  background: linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 100%);
  border: 1px solid #E0E7FF;
  border-radius: var(--radius);
  padding: 28px 32px;
  margin-bottom: 24px;
  position: relative; overflow: hidden;
}
.page-hero::before {
  content: ''; position: absolute; top: -40px; right: -40px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(99,102,241,.12) 0%, transparent 70%);
}
.page-hero h1 {
  font-size: 1.8rem; font-weight: 800; color: var(--text);
  margin: 0 0 6px; letter-spacing: -0.5px;
}
.page-hero p {
  font-size: 0.95rem; color: var(--text-muted); margin: 0; max-width: 680px;
}
.page-hero .hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--brand); color: white;
  font-size: 0.72rem; font-weight: 600; letter-spacing: .05em;
  padding: 4px 10px; border-radius: 20px; margin-bottom: 10px;
}

/* ── Section card ── */
.section-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 16px;
}
.section-card-title {
  font-size: 0.8rem; font-weight: 700; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .08em;
  margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.section-card-title span { color: var(--brand); font-size: 1rem; }

/* ── KPI metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  position: relative; overflow: hidden;
  transition: box-shadow .2s, transform .2s;
}
.kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.kpi-card-accent { position: absolute; top: 0; left: 0; height: 3px; width: 100%; border-radius: var(--radius) var(--radius) 0 0; }
.kpi-card .kpi-icon { font-size: 1.5rem; margin-bottom: 10px; }
.kpi-card .kpi-val  { font-size: 2rem; font-weight: 800; color: var(--text); line-height: 1; margin-bottom: 4px; }
.kpi-card .kpi-label { font-size: 0.75rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: .06em; }
.kpi-card .kpi-sub   { font-size: 0.72rem; color: var(--success); font-weight: 600; margin-top: 6px; }

/* ── Tier badge ── */
.tier-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: 20px;
  font-size: 0.82rem; font-weight: 700; letter-spacing: .02em;
}
.tier-exceptional { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
.tier-strong      { background: #EEF2FF; color: #3730A3; border: 1px solid #C7D2FE; }
.tier-moderate    { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.tier-low         { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }

/* ── Result banner ── */
.result-banner {
  background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
  border-radius: var(--radius);
  padding: 24px 28px;
  margin-bottom: 20px;
  color: white;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;
}
.result-banner-left h2 { color: white; font-size: 1.3rem; font-weight: 700; margin: 0 0 4px; }
.result-banner-left p  { color: #C7D2FE; font-size: 0.85rem; margin: 0; }
.result-score-ring {
  width: 80px; height: 80px; border-radius: 50%;
  background: rgba(255,255,255,0.12);
  border: 3px solid rgba(255,255,255,0.3);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.result-score-ring .score-num { font-size: 1.5rem; font-weight: 800; color: white; line-height: 1; }
.result-score-ring .score-pct { font-size: 0.65rem; color: #C7D2FE; font-weight: 500; }

/* ── Skill badges ── */
.skill-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px; border-radius: 20px;
  font-size: 0.78rem; font-weight: 600; margin: 3px;
  transition: transform .15s;
}
.skill-badge:hover { transform: scale(1.05); }
.badge-matched { background: #ECFDF5; color: #065F46; border: 1px solid #6EE7B7; }
.badge-missing { background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }
.badge-extra   { background: #EEF2FF; color: #3730A3; border: 1px solid #A5B4FC; }

/* ── Skills panel ── */
.skills-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.skills-panel-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px;
}
.skills-panel-title { font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }
.skills-panel-count {
  font-size: 0.72rem; font-weight: 700;
  padding: 2px 8px; border-radius: 10px;
}
.count-matched { background: #ECFDF5; color: #065F46; }
.count-missing { background: #FEF2F2; color: #991B1B; }
.count-extra   { background: #EEF2FF; color: #3730A3; }
.skills-empty { font-size: 0.82rem; color: var(--text-muted); font-style: italic; padding: 8px 0; }

/* ── Recommendation card ── */
.rec-card {
  border-left: 3px solid var(--warning);
  background: #FFFBEB;
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 14px 18px;
  margin-bottom: 10px;
}
.rec-card.high { border-color: var(--danger); background: #FEF2F2; }
.rec-card.low  { border-color: var(--success); background: #ECFDF5; }
.rec-card-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--text-muted); margin-bottom: 4px; }
.rec-card-text  { font-size: 0.88rem; color: var(--text); }

/* ── Rank card (batch leaderboard) ── */
.rank-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 10px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .15s;
}
.rank-card:hover { box-shadow: var(--shadow-md); }
.rank-num {
  width: 38px; height: 38px; border-radius: 50%;
  background: var(--brand-light); color: var(--brand-dark);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.95rem; font-weight: 800; flex-shrink: 0;
}
.rank-num.gold   { background: #FEF9C3; color: #854D0E; }
.rank-num.silver { background: #F1F5F9; color: #475569; }
.rank-num.bronze { background: #FEF3C7; color: #92400E; }
.rank-info { flex: 1; min-width: 0; }
.rank-name  { font-size: 0.95rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-email { font-size: 0.75rem; color: var(--text-muted); }
.rank-scores { display: flex; gap: 10px; align-items: center; flex-shrink: 0; }
.score-pill {
  text-align: center; padding: 4px 10px; border-radius: 8px;
  font-size: 0.75rem; min-width: 52px;
}
.sp-composite { background: #EEF2FF; color: #3730A3; font-weight: 800; }
.sp-semantic  { background: #ECFDF5; color: #065F46; font-weight: 600; }
.sp-skill     { background: #FFFBEB; color: #92400E; font-weight: 600; }

/* ── Candidate metadata card ── */
.meta-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px;
}
.meta-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.meta-row:last-child { margin-bottom: 0; }
.meta-icon { font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
.meta-label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.meta-value { font-size: 0.88rem; color: var(--text); font-weight: 500; }

/* ── Audit table tweaks ── */
.stDataFrame { border-radius: var(--radius); overflow: hidden; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; background: var(--surface-2);
  border-radius: 10px; padding: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px; padding: 8px 18px;
  font-weight: 600; font-size: 0.85rem;
  color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface) !important;
  color: var(--brand-dark) !important;
  box-shadow: var(--shadow-sm);
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: 0.9rem !important;
  padding: 12px 24px !important; letter-spacing: .02em !important;
  box-shadow: 0 4px 14px rgba(99,102,241,.4) !important;
  transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 6px 20px rgba(99,102,241,.55) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  border-radius: 10px !important; font-weight: 600 !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  border: 2px dashed #C7D2FE !important;
  border-radius: var(--radius) !important;
  background: #EEF2FF !important;
  padding: 12px !important;
}

/* ── Divider ── */
.fancy-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 20px 0;
}

/* ── Empty state ── */
.empty-state {
  text-align: center; padding: 48px 24px;
  background: var(--surface-2); border-radius: var(--radius);
  border: 1px dashed var(--border);
}
.empty-state .es-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state .es-title { font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.empty-state .es-text  { font-size: 0.88rem; color: var(--text-muted); }

/* ── Taxonomy domain card ── */
.domain-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 10px;
  box-shadow: var(--shadow-sm);
}
.domain-card-header {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;
}
.domain-card-title { font-size: 0.88rem; font-weight: 700; color: var(--text); }
.domain-count-badge {
  font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 10px;
  background: var(--brand-light); color: var(--brand-dark);
}

/* ── Arch diagram box ── */
.arch-box {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px; font-family: 'Courier New', monospace;
  font-size: 0.75rem; color: var(--text); line-height: 1.6; overflow-x: auto;
}

/* ── Progress bar override ── */
.stProgress > div > div { background-color: var(--brand) !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_preprocessed_jobs() -> pd.DataFrame:
    csv_path = Path("data/processed/cleaned_job_descriptions.csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Could not read cleaned_job_descriptions.csv: %s", e)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return get_embedding_model()
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        return None


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
SAMPLE_RESUMES = {
    "Senior Data Scientist": (
        "Sarah Jenkins\n"
        "sarah.jenkins@example.com | (555) 789-1234 | San Francisco, CA\n"
        "linkedin.com/in/sarahjenkins-ds | github.com/sjenkins-data\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Lead Data Scientist with 7+ years of experience designing scalable machine learning pipelines, "
        "predictive modeling systems, and deep learning architectures in cloud environments. Proven expertise in "
        "Python, PyTorch, SQL, Snowflake, and AWS. Spearheaded enterprise LLM fine-tuning and predictive analytics "
        "driving $4.2M in annual operational savings.\n\n"
        "EDUCATION\n"
        "Master of Science in Computer Science, Stanford University (2018)\n"
        "Bachelor of Science in Statistics, UC Berkeley (2016)\n\n"
        "TECHNICAL SKILLS\n"
        "Languages & Frameworks: Python, SQL, R, PyTorch, Scikit-Learn, TensorFlow, Bash\n"
        "Cloud & Big Data: AWS (S3, EC2, Lambda), Snowflake, Apache Spark, Databricks, Kafka, Airflow\n"
        "Databases & Tools: PostgreSQL, Redis, Docker, Git, CI/CD, MLflow, Tableau\n\n"
        "EXPERIENCE\n"
        "Principal Machine Learning Engineer | DataCore Systems (2021 - Present)\n"
        "- Engineered end-to-end production ML recommendation pipelines processing 200M+ events daily.\n"
        "- Architected real-time feature store using Snowflake, Spark, and Redis.\n"
        "- Deployed microservices using Docker, FastAPI, and Kubernetes on AWS EKS."
    ),
    "BI & Reporting Analyst": (
        "Michael Chang\n"
        "mchang.bi@example.com | (555) 456-7890 | Chicago, IL\n"
        "linkedin.com/in/mchang-bi\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Senior Business Intelligence Analyst with 5 years experience transforming complex corporate data into "
        "executive dashboards and automated reports. Advanced proficiency in SQL, Power BI, Tableau, Excel, SSIS.\n\n"
        "EDUCATION\n"
        "Bachelor of Science in Information Systems, University of Illinois (2019)\n\n"
        "TECHNICAL SKILLS\n"
        "BI & Visualization: Power BI, Tableau, Excel (Advanced VBA, Power Query), SSRS, Looker\n"
        "Databases & ETL: SQL, T-SQL, PostgreSQL, SQL Server, SSIS, Data Modeling, Star Schema\n"
        "Methodologies: Agile, Scrum, Jira, Git"
    ),
    "Junior Python Developer": (
        "David Miller\n"
        "david.m@example.com | (555) 123-9876 | Austin, TX\n\n"
        "SUMMARY\n"
        "Recent Computer Science graduate with strong fundamentals in Python, Git, HTML, and CSS. "
        "Built academic projects utilizing Flask, SQLite, and REST APIs.\n\n"
        "EDUCATION\n"
        "Bachelor of Science in Computer Science, University of Texas (2023)\n\n"
        "TECHNICAL SKILLS\n"
        "Python, JavaScript, HTML, CSS, Git, Linux, SQLite"
    ),
}

DEMO_BATCH_CANDIDATES = [
    {"name": "Sarah Jenkins (Senior DS)",      "email": "sarah.jenkins@example.com", "text": SAMPLE_RESUMES["Senior Data Scientist"]},
    {"name": "Michael Chang (BI Lead)",         "email": "mchang.bi@example.com",     "text": SAMPLE_RESUMES["BI & Reporting Analyst"]},
    {"name": "David Miller (Junior Dev)",       "email": "david.m@example.com",       "text": SAMPLE_RESUMES["Junior Python Developer"]},
    {"name": "Elena Rostova (Data Engineer)",   "email": "elena.r@example.com",       "text": (
        "Elena Rostova | Data Engineer\nEmail: elena.rostova@example.com\n"
        "Expert in building distributed data pipelines using Python, SQL, Apache Spark, Airflow, "
        "Kafka, Snowflake, AWS, Docker, and Kubernetes. Master of Science in Data Engineering."
    )},
    {"name": "Carlos Gomez (Product Analyst)",  "email": "carlos.g@example.com",      "text": (
        "Carlos Gomez | Product Analyst\nEmail: carlos.g@example.com\n"
        "Experienced in product analytics, A/B testing, SQL, Tableau, Excel, and cohort analysis. "
        "Bachelor of Business Administration."
    )},
    {"name": "Lucas Vance (Frontend Specialist)", "email": "lucas.v@example.com",     "text": (
        "Lucas Vance | Frontend UI Engineer\nEmail: lucas.v@example.com\n"
        "Proficient in React, JavaScript, TypeScript, CSS, HTML, Tailwind, and Figma. "
        "Bachelor of Fine Arts in Web Design."
    )},
]


# ---------------------------------------------------------------------------
# Helper: tier → CSS class
# ---------------------------------------------------------------------------
_TIER_CSS = {
    "exceptional": "tier-exceptional",
    "strong":      "tier-strong",
    "moderate":    "tier-moderate",
    "low":         "tier-low",
}
_TIER_ICON = {
    "exceptional": "🏆",
    "strong":      "✅",
    "moderate":    "🔶",
    "low":         "⚠️",
}

def tier_badge_html(tier_code: str, tier_label: str) -> str:
    css  = _TIER_CSS.get(tier_code, "tier-low")
    icon = _TIER_ICON.get(tier_code, "⚠️")
    return f'<span class="tier-badge {css}">{icon} {tier_label}</span>'

def kpi_card_html(icon, value, label, accent, sub="") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-card-accent" style="background:{accent}"></div>
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-label">{label}</div>
      {sub_html}
    </div>"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-brand-icon">⚡</div>
      <div class="sidebar-brand-text">
        <h2>RecruitIQ</h2>
        <p>AI-Powered Resume Screener</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🎯  Match Dashboard",
            "📊  Batch Screening",
            "📜  Audit History",
            "🔬  System Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**⚙️ Scoring Weights**")
    semantic_weight = st.slider(
        "Semantic Similarity",
        min_value=0.0, max_value=1.0,
        value=float(DEFAULT_SEMANTIC_WEIGHT), step=0.05,
        help="Proportion of score from dense transformer embedding cosine similarity.",
    )
    skill_weight = round(1.0 - semantic_weight, 2)
    st.caption(f"Keyword Coverage weight: **{skill_weight:.2f}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    stats = get_database_stats()
    st.markdown(f"""
    <div class="stat-pill">
      <div class="sp-row">
        <span class="sp-label">📋 Single Evals</span>
        <span class="sp-val">{stats['total_single_evaluations']}</span>
      </div>
      <div class="sp-row">
        <span class="sp-label">🗂️ Batch Sessions</span>
        <span class="sp-val">{stats['total_batch_sessions']}</span>
      </div>
      <div class="sp-row">
        <span class="sp-label">👥 Total Candidates</span>
        <span class="sp-val">{stats['total_candidates_evaluated']}</span>
      </div>
      <div class="sp-row">
        <span class="sp-label">📈 Avg Score</span>
        <span class="sp-val">{stats['average_composite_score']}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    dev = get_optimal_device()
    dev_icon = "🔥" if dev == "cuda" else ("🍎" if dev == "mps" else "💻")
    st.caption(f"{dev_icon} Compute: **{dev.upper()}** · Model: `{DEFAULT_MODEL_NAME}`")

jobs_df = load_preprocessed_jobs()


# ===========================================================================
# PAGE 1 — Match Dashboard
# ===========================================================================
if page == "🎯  Match Dashboard":

    st.markdown("""
    <div class="page-hero">
      <div class="hero-badge">🎯 Single Candidate Evaluation</div>
      <h1>Candidate Match Dashboard</h1>
      <p>Upload a resume PDF or paste raw text, select a job description, and get an instant
         semantic match score, hard-skill gap breakdown, and AI-generated upskilling recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("""<div class="section-card-title"><span>📄</span> STEP 1 — Candidate Resume</div>""", unsafe_allow_html=True)
        resume_tab_pdf, resume_tab_text = st.tabs(["📤 Upload PDF", "✍️ Paste / Select Text"])
        resume_input_data: Optional[Union[bytes, str]] = None
        candidate_source_name = "Custom Resume"

        with resume_tab_pdf:
            uploaded_pdf = st.file_uploader(
                "Drop candidate resume here (PDF)",
                type=["pdf"], label_visibility="collapsed",
            )
            if uploaded_pdf is not None:
                resume_input_data = uploaded_pdf.read()
                candidate_source_name = uploaded_pdf.name
                st.success(f"✅ Loaded: **{uploaded_pdf.name}** ({len(resume_input_data)//1024} KB)")

        with resume_tab_text:
            preset_choice = st.selectbox("Sample candidate profile:", list(SAMPLE_RESUMES.keys()))
            if st.button("Load Sample →", key="btn_load_sample"):
                st.session_state["pasted_resume_text"] = SAMPLE_RESUMES[preset_choice]
            default_text = st.session_state.get("pasted_resume_text", SAMPLE_RESUMES["Senior Data Scientist"])
            pasted_text = st.text_area("Paste resume text:", value=default_text, height=240, label_visibility="collapsed")
            if uploaded_pdf is None and pasted_text.strip():
                resume_input_data = pasted_text
                candidate_source_name = preset_choice

    with col_right:
        st.markdown("""<div class="section-card-title"><span>💼</span> STEP 2 — Target Job Description</div>""", unsafe_allow_html=True)
        job_tab_ds, job_tab_custom = st.tabs(["🗄️ Dataset (517 Jobs)", "✍️ Custom Posting"])
        job_input_data = ""
        target_job_title = "Target Position"

        with job_tab_ds:
            if not jobs_df.empty:
                job_titles = jobs_df["Job Title"].tolist()
                selected_title_idx = st.selectbox(
                    "Pick a job from dataset:",
                    range(len(job_titles)),
                    format_func=lambda i: f"{job_titles[i]}  (#{jobs_df.iloc[i]['job_id']})",
                    label_visibility="collapsed",
                )
                selected_row = jobs_df.iloc[selected_title_idx]
                target_job_title = selected_row["Job Title"]
                job_input_data   = selected_row["clean_description"]
                skills_preview   = selected_row.get("skills_str", "—")
                st.caption(f"📌 Required skills: `{skills_preview[:120]}{'...' if len(str(skills_preview))>120 else ''}`")
            else:
                st.warning("No preprocessed dataset found in `data/processed/`. Use Custom Posting tab.")

        with job_tab_custom:
            custom_title = st.text_input("Job Title:", value="Senior Machine Learning Engineer")
            default_job_desc = (
                "We are seeking a Senior Machine Learning Engineer to design and deploy scalable ML pipelines. "
                "The ideal candidate must be highly proficient in Python, SQL, PyTorch, and AWS cloud architecture. "
                "Experience with Snowflake, Spark, Docker, and CI/CD pipelines is strongly preferred. "
                "Candidate will lead predictive modeling initiatives and collaborate with cross-functional data teams."
            )
            custom_desc = st.text_area("Job Description:", value=default_job_desc, height=200, label_visibility="collapsed")
            if jobs_df.empty or st.checkbox("Use this custom posting (overrides dataset)", value=False):
                job_input_data  = custom_desc
                target_job_title = custom_title

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    eval_btn = st.button(
        "⚡  Evaluate Candidate — Compute Embeddings & Analyze Skill Gaps",
        type="primary", use_container_width=True,
    )

    if eval_btn:
        if not resume_input_data:
            st.error("⚠️ Please provide a resume by uploading a PDF or pasting text.")
        elif not job_input_data or not job_input_data.strip():
            st.error("⚠️ Please provide a job description.")
        else:
            with st.spinner("🔄 Computing 384-d dense embeddings and running skill gap analysis…"):
                try:
                    result = evaluate_resume_against_job(
                        resume_input=resume_input_data,
                        job_input=job_input_data,
                        job_title=target_job_title,
                        weights=(semantic_weight, skill_weight),
                    )
                    st.session_state["last_eval_result"] = result
                except Exception as ex:
                    st.error(f"Evaluation failed: {ex}")
                    logger.exception("Evaluation error")

    # ── Results ──
    if "last_eval_result" in st.session_state:
        res = st.session_state["last_eval_result"]
        tier_code  = res.get("tier_code", "low")
        tier_label = res.get("match_tier", "Low Fit")

        # Result banner
        st.markdown(f"""
        <div class="result-banner">
          <div class="result-banner-left">
            <h2>👤 {res['candidate_name']}  →  {res['job_title']}</h2>
            <p>{res.get('explanation', '')}</p>
            <div style="margin-top:10px">{tier_badge_html(tier_code, tier_label)}</div>
          </div>
          <div class="result-score-ring">
            <div class="score-num">{res['composite_score']}</div>
            <div class="score-pct">MATCH %</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        matched_len = len(res["matched_skills"])
        total_req   = res["skill_gap"]["total_job_skills"]
        coverage_pct = round(matched_len / max(total_req, 1) * 100)

        st.markdown(f"""
        <div class="kpi-grid">
          {kpi_card_html("🎯", f"{res['composite_score']}%", "Composite Score",   "#6366F1", f"Semantic×{semantic_weight}  +  Skill×{skill_weight}")}
          {kpi_card_html("🧠", f"{res['semantic_score']}%", "Semantic Alignment", "#10B981")}
          {kpi_card_html("🔧", f"{res['skill_score']}%",   "Hard-Skill Coverage", "#F59E0B")}
          {kpi_card_html("✅", f"{matched_len} / {total_req}", "Skills Matched",  "#6366F1", f"{coverage_pct}% coverage")}
        </div>
        """, unsafe_allow_html=True)

        # Charts
        c_gauge, c_bar = st.columns(2, gap="large")

        with c_gauge:
            score_val = res["composite_score"]
            gauge_color = "#10B981" if score_val >= 85 else "#6366F1" if score_val >= 70 else "#F59E0B" if score_val >= 50 else "#EF4444"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                domain={"x": [0, 1], "y": [0, 1]},
                number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
                title={"text": "Composite Match Score", "font": {"size": 14, "color": "#64748B"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1", "dtick": 25},
                    "bar":  {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "#F8FAFC",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "#FEE2E2"},
                        {"range": [50, 70], "color": "#FEF9C3"},
                        {"range": [70, 85], "color": "#DBEAFE"},
                        {"range": [85, 100],"color": "#DCFCE7"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_bar:
            cat_breakdown = res["category_breakdown"]
            cats      = [c.replace("_", " ").title() for c in cat_breakdown]
            coverages = [round(cat_breakdown[c]["coverage"] * 100, 1) for c in cat_breakdown]
            colors    = ["#10B981" if v >= 70 else "#6366F1" if v >= 40 else "#EF4444" for v in coverages]
            fig_bar = go.Figure(go.Bar(
                x=coverages, y=cats, orientation="h",
                marker_color=colors,
                text=[f"{v}%" for v in coverages],
                textposition="outside",
                textfont={"size": 11, "family": "Inter"},
            ))
            fig_bar.update_layout(
                title={"text": "Domain-Level Coverage Breakdown", "font": {"size": 14, "color": "#64748B"}},
                xaxis={"range": [0, 115], "showgrid": False, "visible": False},
                yaxis={"automargin": True},
                height=280,
                margin=dict(l=10, r=60, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Skill panels
        col_matched, col_missing, col_extra = st.columns(3, gap="medium")

        with col_matched:
            matched = res["matched_skills"]
            st.markdown(f"""
            <div class="skills-panel">
              <div class="skills-panel-header">
                <span class="skills-panel-title" style="color:#065F46">✅ Matched Skills</span>
                <span class="skills-panel-count count-matched">{len(matched)}</span>
              </div>
              <div>
                {"".join(f'<span class="skill-badge badge-matched">✓ {s}</span>' for s in matched)
                 if matched else '<div class="skills-empty">No exact skill matches detected.</div>'}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_missing:
            missing = res["missing_skills"]
            st.markdown(f"""
            <div class="skills-panel">
              <div class="skills-panel-header">
                <span class="skills-panel-title" style="color:#991B1B">❌ Missing Skills</span>
                <span class="skills-panel-count count-missing">{len(missing)}</span>
              </div>
              <div>
                {"".join(f'<span class="skill-badge badge-missing">✗ {s}</span>' for s in missing)
                 if missing else '<div class="skills-empty" style="color:#065F46">🎉 Full skill coverage achieved!</div>'}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_extra:
            extra = res["additional_skills"][:18]
            st.markdown(f"""
            <div class="skills-panel">
              <div class="skills-panel-header">
                <span class="skills-panel-title" style="color:#3730A3">➕ Additional Skills</span>
                <span class="skills-panel-count count-extra">{len(res['additional_skills'])}</span>
              </div>
              <div>
                {"".join(f'<span class="skill-badge badge-extra">+ {s}</span>' for s in extra)
                 if extra else '<div class="skills-empty">No extra keywords detected.</div>'}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Upskilling recommendations
        st.markdown("""<div class="section-card-title"><span>💡</span> STRATEGIC UPSKILLING RECOMMENDATIONS</div>""", unsafe_allow_html=True)
        recs = res.get("recommendations", [])
        if recs:
            for rec in recs:
                impact = rec.get("impact", "medium").lower()
                impact_class = "high" if impact == "high" else ("low" if impact == "low" else "")
                impact_label = rec["category"].replace("_", " ").title()
                st.markdown(f"""
                <div class="rec-card {impact_class}">
                  <div class="rec-card-label">{impact_label} · {impact.upper()} Priority</div>
                  <div class="rec-card-text">{rec['suggestion']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-card low">
              <div class="rec-card-label">Assessment Complete</div>
              <div class="rec-card-text">🎯 No critical skill gaps detected. This candidate strongly aligns with the role requirements.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Candidate metadata
        with st.expander("🔍 Extracted Candidate Profile & Education Details", expanded=False):
            m1, m2 = st.columns(2, gap="large")
            with m1:
                st.markdown("<div class='meta-card'>", unsafe_allow_html=True)
                contact = res.get("contact_info", {})
                for icon, label, val in [
                    ("👤", "Name",       res.get("candidate_name", "—")),
                    ("📧", "Email",      contact.get("email") or "—"),
                    ("📱", "Phone",      contact.get("phone") or "—"),
                    ("💼", "LinkedIn",   contact.get("linkedin") or "—"),
                    ("🐙", "GitHub",     contact.get("github") or "—"),
                    ("📝", "Word Count", f"{res.get('resume_word_count', 0)} words"),
                ]:
                    st.markdown(f"""
                    <div class="meta-row">
                      <span class="meta-icon">{icon}</span>
                      <div>
                        <div class="meta-label">{label}</div>
                        <div class="meta-value">{val}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with m2:
                edus = res.get("education", [])
                if edus:
                    for edu in edus:
                        deg   = edu.get("degree", "")
                        field = f" in {edu['field']}" if edu.get("field") else ""
                        inst  = edu.get("institution", "")
                        yr    = edu.get("year", "")
                        st.markdown(f"""
                        <div class="meta-card" style="margin-bottom:8px">
                          <div class="meta-row">
                            <span class="meta-icon">🎓</span>
                            <div>
                              <div class="meta-label">{deg}{field}</div>
                              <div class="meta-value">{inst}{' · ' + yr if yr else ''}</div>
                            </div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="skills-empty">No formal degree detected via NER.</div>', unsafe_allow_html=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Save & Export
        notes = st.text_input("📝 Recruiter Notes (optional)", placeholder="e.g., Schedule phone screen for next Thursday…")
        c_save, c_dl = st.columns([1, 1])
        with c_save:
            if st.button("💾  Save to Audit Database", use_container_width=True):
                rec_id = save_match_result(res, notes=notes)
                st.success(f"Saved to database — Record **#{rec_id}**")
        with c_dl:
            report_json = json.dumps(res, indent=2, default=str)
            st.download_button(
                "📥  Download Match Report (JSON)",
                data=report_json,
                file_name=f"match_{res['candidate_name'].replace(' ', '_').lower()}.json",
                mime="application/json",
                use_container_width=True,
            )


# ===========================================================================
# PAGE 2 — Batch Screening
# ===========================================================================
elif page == "📊  Batch Screening":

    st.markdown("""
    <div class="page-hero">
      <div class="hero-badge">📊 Bulk Candidate Evaluation</div>
      <h1>Batch Screening & Leaderboard</h1>
      <p>Ingest a cohort of candidate resumes, batch-encode dense vector embeddings,
         and automatically rank every candidate against a target job posting.</p>
    </div>
    """, unsafe_allow_html=True)

    col_b_left, col_b_right = st.columns(2, gap="large")

    with col_b_left:
        st.markdown("""<div class="section-card-title"><span>👥</span> STEP 1 — Candidate Cohort</div>""", unsafe_allow_html=True)
        batch_source = st.radio(
            "Input method:", ["🎭  Demo Cohort (6 candidates)", "📄  Upload CSV File"],
            horizontal=True, label_visibility="collapsed",
        )
        batch_candidates_data: List[Dict[str, Any]] = []

        if "Demo" in batch_source:
            st.markdown(f"""
            <div class="stat-pill" style="background:#EEF2FF; border-color:#C7D2FE;">
              <div class="sp-row">
                <span class="sp-label" style="color:#4338CA !important">👥 Candidates loaded</span>
                <span class="sp-val" style="color:#3730A3 !important">{len(DEMO_BATCH_CANDIDATES)}</span>
              </div>
              <div class="sp-row">
                <span class="sp-label" style="color:#4338CA !important">🏷️ Profiles</span>
                <span class="sp-val" style="color:#3730A3 !important; font-size:.75rem !important">
                  DS · BI · Dev · DE · PA · FE
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            batch_candidates_data = DEMO_BATCH_CANDIDATES
        else:
            uploaded_csv = st.file_uploader("Upload CSV file:", type=["csv"], label_visibility="collapsed")
            if uploaded_csv is not None:
                try:
                    cand_df = pd.read_csv(uploaded_csv)
                    st.dataframe(cand_df.head(3), use_container_width=True)
                    nc = st.selectbox("Name column:", cand_df.columns, index=0)
                    tc = st.selectbox("Resume text column:", cand_df.columns,
                                      index=1 if len(cand_df.columns) > 1 else 0)
                    for _, r in cand_df.iterrows():
                        batch_candidates_data.append({"name": str(r[nc]), "text": str(r[tc])})
                    st.success(f"✅ Parsed **{len(batch_candidates_data)}** candidate records.")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")

    with col_b_right:
        st.markdown("""<div class="section-card-title"><span>💼</span> STEP 2 — Target Job</div>""", unsafe_allow_html=True)
        if not jobs_df.empty:
            b_job_idx = st.selectbox(
                "Select job posting:",
                range(len(jobs_df)),
                format_func=lambda i: f"{jobs_df.iloc[i]['Job Title']}  (#{jobs_df.iloc[i]['job_id']})",
                label_visibility="collapsed",
            )
            b_sel = jobs_df.iloc[b_job_idx]
            b_target_title = b_sel["Job Title"]
            b_target_desc  = b_sel["clean_description"]
            skills_preview = b_sel.get("skills_str", "—")
            st.caption(f"📌 Skills: `{str(skills_preview)[:120]}`")
        else:
            b_target_title = st.text_input("Job Title:", value="Senior Data Analyst")
            b_target_desc  = st.text_area("Job Description:", value="Looking for Senior Data Analyst with Python, SQL, Tableau, Power BI, Snowflake.", height=160, label_visibility="collapsed")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    top_k = st.slider("Return top K candidates:", min_value=1,
                      max_value=max(10, len(batch_candidates_data) or 10),
                      value=min(6, len(batch_candidates_data) or 6))

    run_batch = st.button(
        "🚀  Run Batch Vector Screening — Build Leaderboard",
        type="primary", use_container_width=True,
    )

    if run_batch:
        if not batch_candidates_data:
            st.error("No candidate data loaded. Use the demo cohort or upload a CSV.")
        else:
            with st.spinner(f"🔄 Embedding {len(batch_candidates_data)} candidates with `{DEFAULT_MODEL_NAME}`…"):
                ranked = rank_candidates_batch(
                    resumes=batch_candidates_data,
                    job_input=b_target_desc,
                    job_title=b_target_title,
                    top_k=top_k,
                    weights=(semantic_weight, skill_weight),
                )
                st.session_state["last_ranked_batch"]     = ranked
                st.session_state["last_batch_job_title"] = b_target_title

    if "last_ranked_batch" in st.session_state:
        ranked       = st.session_state["last_ranked_batch"]
        target_title = st.session_state.get("last_batch_job_title", "Target Role")

        st.markdown(f"""
        <div class="page-hero" style="padding:18px 24px; margin-top:8px; margin-bottom:16px;">
          <div class="hero-badge">🏆 Results</div>
          <h1 style="font-size:1.3rem">Leaderboard — {target_title}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Rank cards
        rank_colors = ["gold", "silver", "bronze"]
        for c in ranked:
            rk   = c["rank"]
            rk_cls = rank_colors[rk - 1] if rk <= 3 else ""
            tcode  = c.get("tier_code", "low")
            st.markdown(f"""
            <div class="rank-card">
              <div class="rank-num {rk_cls}">#{rk}</div>
              <div class="rank-info">
                <div class="rank-name">{c['candidate_name']}</div>
                <div class="rank-email">
                  {tier_badge_html(tcode, c['match_tier'])}
                  &nbsp; ✅ {c.get('matched_count',0)} matched · ❌ {c.get('missing_count',0)} missing
                </div>
                <div style="margin-top:6px; font-size:.75rem; color:#64748B">
                  Skills: {', '.join(c.get('matched_skills',[])[:5])}{'…' if len(c.get('matched_skills',[])) > 5 else ''}
                </div>
              </div>
              <div class="rank-scores">
                <div class="score-pill sp-composite">{c['composite_score']}%<br><span style="font-size:.65rem;font-weight:400">Composite</span></div>
                <div class="score-pill sp-semantic">{c['semantic_score']}%<br><span style="font-size:.65rem;font-weight:400">Semantic</span></div>
                <div class="score-pill sp-skill">{c['skill_score']}%<br><span style="font-size:.65rem;font-weight:400">Skills</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Scatter plot
        plot_df = pd.DataFrame([{
            "Candidate":           c["candidate_name"],
            "Semantic Alignment (%)": c["semantic_score"],
            "Skill Coverage (%)":  c["skill_score"],
            "Composite Score (%)": c["composite_score"],
            "Tier":                c["match_tier"],
        } for c in ranked])

        tier_colors = {
            "Exceptional Fit": "#10B981",
            "Strong Fit":      "#6366F1",
            "Moderate Fit":    "#F59E0B",
            "Low Fit":         "#EF4444",
        }
        fig_scatter = px.scatter(
            plot_df,
            x="Semantic Alignment (%)", y="Skill Coverage (%)",
            size="Composite Score (%)", color="Tier",
            hover_name="Candidate", text="Candidate",
            title="Strategic Fit Matrix — Semantic Context vs Technical Skill Coverage",
            color_discrete_map=tier_colors,
            range_x=[0, 110], range_y=[0, 110],
            size_max=42,
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=11)
        fig_scatter.add_hline(y=50, line_dash="dot", line_color="#CBD5E1", line_width=1.5)
        fig_scatter.add_vline(x=50, line_dash="dot", line_color="#CBD5E1", line_width=1.5)
        fig_scatter.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC",
            font={"family": "Inter"},
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
            xaxis={"gridcolor": "#E2E8F0"},
            yaxis={"gridcolor": "#E2E8F0"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Save & Export
        c_save_b, c_dl_b = st.columns(2)
        with c_save_b:
            if st.button("💾  Save Session to Database", use_container_width=True):
                sid = save_batch_session({"job_title": target_title}, ranked)
                st.success(f"Session saved — ID `{sid}`")
        with c_dl_b:
            csv_export = pd.DataFrame(ranked).to_csv(index=False)
            st.download_button(
                "📥  Export Leaderboard (CSV)",
                data=csv_export,
                file_name=f"ranked_{target_title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ===========================================================================
# PAGE 3 — Audit History
# ===========================================================================
elif page == "📜  Audit History":

    st.markdown("""
    <div class="page-hero">
      <div class="hero-badge">📜 Audit Trail</div>
      <h1>Evaluation History & Analytics</h1>
      <p>Review all historical candidate evaluations persisted in SQLite, filter by tier or keyword,
         inspect individual records, and export full audit logs.</p>
    </div>
    """, unsafe_allow_html=True)

    db_stats = get_database_stats()
    tier_counts = db_stats.get("tier_counts", {})

    st.markdown(f"""
    <div class="kpi-grid">
      {kpi_card_html("📋", db_stats['total_single_evaluations'], "Single Evaluations", "#6366F1")}
      {kpi_card_html("🗂️", db_stats['total_batch_sessions'],     "Batch Sessions",     "#10B981")}
      {kpi_card_html("👥", db_stats['total_candidates_evaluated'],"Total Candidates",  "#F59E0B")}
      {kpi_card_html("📈", f"{db_stats['average_composite_score']}%", "Avg Match Score", "#6366F1")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns([3, 1], gap="medium")
    with f1:
        search_kw = st.text_input("🔍 Search by name, job title, or email…", placeholder="e.g. 'Sarah' or 'Data Analyst'", label_visibility="collapsed")
    with f2:
        tier_opt = st.selectbox("Filter by tier:", ["All Tiers", "Exceptional", "Strong", "Moderate", "Low"], label_visibility="collapsed")

    resolved_tier = None if tier_opt == "All Tiers" else tier_opt.lower()
    history_records = get_match_history(limit=200, search_query=search_kw, tier_filter=resolved_tier)

    if history_records:
        history_df = pd.DataFrame([{
            "ID":          r["id"],
            "Date":        r["timestamp"][:10],
            "Time":        r["timestamp"][11:19],
            "Candidate":   r["candidate_name"],
            "Job Title":   r["job_title"],
            "Composite %": r["composite_score"],
            "Semantic %":  r["semantic_score"],
            "Skill %":     r["skill_score"],
            "Tier":        r["match_tier"],
            "Notes":       r.get("notes") or "",
        } for r in history_records])

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Composite %": st.column_config.ProgressColumn("Composite %", min_value=0, max_value=100, format="%.1f%%"),
                "Semantic %":  st.column_config.ProgressColumn("Semantic %",  min_value=0, max_value=100, format="%.1f%%"),
                "Skill %":     st.column_config.ProgressColumn("Skill %",     min_value=0, max_value=100, format="%.1f%%"),
            },
        )

        # Charts
        if len(history_df) >= 2:
            ch1, ch2 = st.columns(2, gap="large")
            with ch1:
                fig_hist = px.histogram(
                    history_df, x="Composite %", color="Tier", nbins=12,
                    title="Match Score Distribution",
                    color_discrete_map={
                        "Exceptional Fit": "#10B981",
                        "Strong Fit":      "#6366F1",
                        "Moderate Fit":    "#F59E0B",
                        "Low Fit":         "#EF4444",
                    },
                    barmode="overlay",
                )
                fig_hist.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC",
                    font={"family": "Inter"},
                    legend={"orientation": "h", "y": -0.3},
                    xaxis={"gridcolor": "#E2E8F0"}, yaxis={"gridcolor": "#E2E8F0"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with ch2:
                if tier_counts:
                    tc_df = pd.DataFrame(list(tier_counts.items()), columns=["Tier", "Count"])
                    tc_df["Tier"] = tc_df["Tier"].str.title() + " Fit"
                    fig_pie = px.pie(
                        tc_df, values="Count", names="Tier",
                        title="Tier Distribution",
                        color="Tier",
                        color_discrete_map={
                            "Exceptional Fit": "#10B981",
                            "Strong Fit":      "#6366F1",
                            "Moderate Fit":    "#F59E0B",
                            "Low Fit":         "#EF4444",
                        },
                        hole=0.42,
                    )
                    fig_pie.update_layout(
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Inter"},
                        legend={"orientation": "h", "y": -0.2},
                    )
                    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Record inspector
        with st.expander("🔎 Deep-Inspect an Evaluation Record"):
            sel_id = st.selectbox("Select record:", [r["id"] for r in history_records])
            detail = get_match_by_id(sel_id)
            if detail:
                st.json(detail)

        # Actions
        a1, a2 = st.columns(2)
        with a1:
            if st.button("🗑️  Clear Entire History", type="secondary", use_container_width=True):
                clear_match_history()
                st.warning("History cleared. Reload the page to refresh.")
        with a2:
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                "📥  Export Full Audit Log (CSV)",
                data=csv_data,
                file_name="evaluation_audit_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="es-icon">📭</div>
          <div class="es-title">No Evaluation Records Found</div>
          <div class="es-text">Run a candidate evaluation on the Match Dashboard, then save it to the database to see it here.</div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 4 — System Explorer
# ===========================================================================
elif page == "🔬  System Explorer":

    st.markdown("""
    <div class="page-hero">
      <div class="hero-badge">🔬 Technical Reference</div>
      <h1>System Architecture & NLP Explorer</h1>
      <p>Inspect the end-to-end NLP pipeline architecture, 7-domain skill taxonomy, and
         live runtime diagnostics including compute device and model metadata.</p>
    </div>
    """, unsafe_allow_html=True)

    t_arch, t_tax, t_diag = st.tabs(["🏛️ Architecture", "📚 Skill Taxonomy", "⚡ Runtime Diagnostics"])

    with t_arch:
        st.markdown("""
        #### End-to-End Pipeline
        """)
        st.markdown("""
        <div class="arch-box">
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                       Streamlit Web Interface                            │
        │    Match Dashboard  │  Batch Screening  │  Audit History  │  Explorer   │
        └────────────┬─────────────────────────────────────────┬────────────────── ┘
                     │  Parse PDF / Clean Text                 │  Compute Embeddings
                     ▼                                         ▼
        ┌────────────────────────────────┐       ┌─────────────────────────────────┐
        │       Document Parser          │       │     Sentence-Transformers        │
        │  · Dual-engine PDF ingestion   │       │  · all-MiniLM-L6-v2 (384-d)     │
        │    (pdfplumber → pypdf)        │       │  · Cosine similarity scoring     │
        │  · Unicode/ligature fixing     │       │  · Dense profile chunking        │
        │  · spaCy NER (en_core_web_sm) │       │  · CUDA / MPS / CPU auto-detect  │
        └────────────┬───────────────────┘       └───────────────┬─────────────────┘
                     │  Parsed metadata                          │  Dense vectors
                     └───────────────────────┬───────────────────┘
                                             ▼
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                    Skill Gap & Ranking Matcher                           │
        │  · 7-Domain technical taxonomy comparison (regex + spaCy EntityRuler)   │
        │  · Composite scoring  =  semantic_weight × sim + skill_weight × overlap │
        │  · Upskilling recommendations by domain and impact priority             │
        │  · Batch ranking with leaderboard and strategic fit matrix              │
        └──────────────────────────────────────────┬───────────────────────────────┘
                                                   │  Persist results
                                                   ▼
                              ┌────────────────────────────────┐
                              │         SQLite Database         │
                              │  · match_history (single evals) │
                              │  · batch_sessions & candidates  │
                              └────────────────────────────────┘
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### ⚠️ Token Window Bottleneck & Our Solution

        | Metric | Value |
        |--------|-------|
        | Model max context window | **256 tokens** (~180 words) |
        | Average raw job posting  | **650 words** |
        | Longest raw job posting  | **2,146 words** |
        | Postings exceeding token budget | **89%** |

        **Dense Profile Chunks** solve this: `Job Title + Extracted Skills + First 120 words of Responsibilities`
        packs the highest-signal content under the 256-token limit, so 100% of required skills are vectorized.
        """)

    with t_tax:
        st.markdown("#### 7-Domain Technical Skill Taxonomy")
        st.caption("Extracted via word-boundary regex patterns and spaCy EntityRuler patterns.")

        domain_icons = {
            "programming":          "💻",
            "bi_visualization":     "📊",
            "cloud_bigdata":        "☁️",
            "databases":            "🗄️",
            "analytics_ml":         "🤖",
            "data_engineering":     "🔧",
            "tools_methodologies":  "🛠️",
        }
        domain_colors = {
            "programming":         "#6366F1",
            "bi_visualization":    "#0EA5E9",
            "cloud_bigdata":       "#10B981",
            "databases":           "#F59E0B",
            "analytics_ml":        "#8B5CF6",
            "data_engineering":    "#EF4444",
            "tools_methodologies": "#64748B",
        }

        for domain_name, skills_list in SKILLS_TAXONOMY.items():
            icon   = domain_icons.get(domain_name, "📌")
            color  = domain_colors.get(domain_name, "#6366F1")
            label  = domain_name.replace("_", " ").title()
            with st.expander(f"{icon}  {label}  —  {len(skills_list)} keywords", expanded=False):
                chips = "".join(
                    f'<span class="skill-badge badge-extra" style="border-color:{color}20; color:{color}; background:{color}10;">{s}</span>'
                    for s in skills_list
                )
                st.markdown(chips, unsafe_allow_html=True)

    with t_diag:
        st.markdown("#### Live Runtime Diagnostics")
        dev = get_optimal_device()
        dev_icon  = "🔥 NVIDIA CUDA" if dev == "cuda" else ("🍎 Apple MPS" if dev == "mps" else "💻 CPU")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Compute Device",    dev_icon)
        d2.metric("Embedding Model",   DEFAULT_MODEL_NAME)
        d3.metric("Vector Dimensions", f"{EMBEDDING_DIM}-d")
        d4.metric("Job Dataset Size",  f"{len(jobs_df)} jobs" if not jobs_df.empty else "Not loaded")

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        col_diag_l, col_diag_r = st.columns(2, gap="large")
        with col_diag_l:
            st.markdown("**Pipeline Configuration**")
            st.json({
                "embedding_model":     DEFAULT_MODEL_NAME,
                "vector_dimensions":   EMBEDDING_DIM,
                "compute_device":      dev,
                "default_sem_weight":  DEFAULT_SEMANTIC_WEIGHT,
                "default_skill_weight": DEFAULT_SKILL_WEIGHT,
            })
        with col_diag_r:
            st.markdown("**Storage Paths**")
            st.json({
                "sqlite_database":   str(Path(os.getenv("SQLITE_DB_PATH",  "history.db")).resolve()),
                "chroma_vector_db":  str(Path(os.getenv("CHROMA_PERSIST_DIR", "models/chroma_db")).resolve()),
                "hf_model_cache":    str(Path(os.getenv("HF_HOME", "models/hf_home")).resolve()),
            })
