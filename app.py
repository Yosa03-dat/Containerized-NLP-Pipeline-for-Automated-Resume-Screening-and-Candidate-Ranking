"""
Semantic Resume & Job Matcher — Production Dashboard.

Pages:
  1. Match Dashboard  — single-candidate PDF/text evaluation with skill gap analysis.
  2. Batch Screening  — multi-candidate cohort screening and leaderboard.
  3. Audit History    — SQLite-backed evaluation records with analytics.
  4. System Explorer  — architecture diagram, skill taxonomy, and runtime diagnostics.

Features:
  - Adaptive Dark & Light themes (Manual toggle + Auto / System preference).
  - Unbreakable sidebar controls (collapsedControl remains fully accessible on all devices).
  - Fully mobile-responsive layouts (KPI grids, score cards, banners, charts).
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
# Theme state management
# ---------------------------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "🌓 Auto"

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
# Sidebar UI
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

    # Theme Switcher
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#A5B4FC; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;'>Appearance</div>", unsafe_allow_html=True)
    theme_selection = st.radio(
        "Theme Mode",
        ["🌓 Auto", "☀️ Light", "🌙 Dark"],
        horizontal=True,
        index=["🌓 Auto", "☀️ Light", "🌙 Dark"].index(st.session_state["theme_mode"]),
        key="theme_mode_selector",
        label_visibility="collapsed",
    )
    st.session_state["theme_mode"] = theme_selection

    st.markdown("<hr style='margin:14px 0;'>", unsafe_allow_html=True)

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

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#A5B4FC; text-transform:uppercase; letter-spacing:0.05em;'>Scoring Parameters</div>", unsafe_allow_html=True)
    semantic_weight = st.slider(
        "Semantic Similarity Weight",
        min_value=0.0, max_value=1.0,
        value=float(DEFAULT_SEMANTIC_WEIGHT), step=0.05,
        help="Proportion of score calculated from 384-d dense transformer vector cosine similarity.",
    )
    skill_weight = round(1.0 - semantic_weight, 2)
    st.caption(f"Keyword Coverage weight: **{skill_weight:.2f}**")

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
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
        <span class="sp-label">👥 Total Screened</span>
        <span class="sp-val">{stats['total_candidates_evaluated']}</span>
      </div>
      <div class="sp-row">
        <span class="sp-label">📈 Avg Score</span>
        <span class="sp-val">{stats['average_composite_score']}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    dev = get_optimal_device()
    dev_icon = "🔥" if dev == "cuda" else ("🍎" if dev == "mps" else "💻")
    st.caption(f"{dev_icon} Device: **{dev.upper()}** · `{DEFAULT_MODEL_NAME}`")

jobs_df = load_preprocessed_jobs()

# Determine active theme styling state
active_theme = st.session_state["theme_mode"]
is_dark_explicit = active_theme == "🌙 Dark"
is_light_explicit = active_theme == "☀️ Light"

# ---------------------------------------------------------------------------
# Global CSS Injection (Adaptive Dark/Light & Mobile Responsive)
# ---------------------------------------------------------------------------
dark_tokens = """
  --brand:              #818CF8;
  --brand-dark:         #6366F1;
  --brand-light:        rgba(99, 102, 241, 0.18);
  --brand-border:       #4338CA;
  --surface:            #1E293B;
  --surface-2:          #0F172A;
  --surface-hover:      #334155;
  --border:             #334155;
  --border-subtle:      #1E293B;
  --text:               #F8FAFC;
  --text-muted:         #94A3B8;
  --hero-bg:            linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
  --hero-border:        #312E81;
  --badge-matched-bg:   #064E3B;
  --badge-matched-text: #6EE7B7;
  --badge-matched-bdr:  #047857;
  --badge-missing-bg:   #7F1D1D;
  --badge-missing-text: #FCA5A5;
  --badge-missing-bdr:  #991B1B;
  --badge-extra-bg:     #1E1B4B;
  --badge-extra-text:   #A5B4FC;
  --badge-extra-bdr:    #4338CA;
  --tier-exc-bg:        #064E3B;
  --tier-exc-text:      #A7F3D0;
  --tier-exc-bdr:       #047857;
  --tier-str-bg:        #1E1B4B;
  --tier-str-text:      #C7D2FE;
  --tier-str-bdr:       #3730A3;
  --tier-mod-bg:        #78350F;
  --tier-mod-text:      #FDE68A;
  --tier-mod-bdr:       #92400E;
  --tier-low-bg:        #7F1D1D;
  --tier-low-text:      #FECACA;
  --tier-low-bdr:       #991B1B;
  --rec-high-bg:        #450A0A;
  --rec-med-bg:         #451A03;
  --rec-low-bg:         #064E3B;
  --shadow-sm:          0 1px 3px rgba(0,0,0,.4);
  --shadow-md:          0 4px 6px -1px rgba(0,0,0,.5);
  --shadow-lg:          0 10px 15px -3px rgba(0,0,0,.6);
"""

light_tokens = """
  --brand:              #6366F1;
  --brand-dark:         #4F46E5;
  --brand-light:        #EEF2FF;
  --brand-border:       #C7D2FE;
  --surface:            #FFFFFF;
  --surface-2:          #F8FAFC;
  --surface-hover:      #F1F5F9;
  --border:             #E2E8F0;
  --border-subtle:      #F1F5F9;
  --text:               #0F172A;
  --text-muted:         #64748B;
  --hero-bg:            linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 100%);
  --hero-border:        #E0E7FF;
  --badge-matched-bg:   #ECFDF5;
  --badge-matched-text: #065F46;
  --badge-matched-bdr:  #6EE7B7;
  --badge-missing-bg:   #FEF2F2;
  --badge-missing-text: #991B1B;
  --badge-missing-bdr:  #FCA5A5;
  --badge-extra-bg:     #EEF2FF;
  --badge-extra-text:   #3730A3;
  --badge-extra-bdr:    #A5B4FC;
  --tier-exc-bg:        #ECFDF5;
  --tier-exc-text:      #065F46;
  --tier-exc-bdr:       #A7F3D0;
  --tier-str-bg:        #EEF2FF;
  --tier-str-text:      #3730A3;
  --tier-str-bdr:       #C7D2FE;
  --tier-mod-bg:        #FFFBEB;
  --tier-mod-text:      #92400E;
  --tier-mod-bdr:       #FDE68A;
  --tier-low-bg:        #FEF2F2;
  --tier-low-text:      #991B1B;
  --tier-low-bdr:       #FECACA;
  --rec-high-bg:        #FEF2F2;
  --rec-med-bg:         #FFFBEB;
  --rec-low-bg:         #ECFDF5;
  --shadow-sm:          0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
  --shadow-md:          0 4px 6px -1px rgba(0,0,0,.1), 0 2px 4px -1px rgba(0,0,0,.06);
  --shadow-lg:          0 10px 15px -3px rgba(0,0,0,.1), 0 4px 6px -2px rgba(0,0,0,.05);
"""

if is_dark_explicit:
    theme_css = f":root, [data-testid='stAppViewContainer'], .stApp {{ {dark_tokens} }}"
elif is_light_explicit:
    theme_css = f":root, [data-testid='stAppViewContainer'], .stApp {{ {light_tokens} }}"
else:
    # Auto: default light + media query for dark
    theme_css = f"""
    :root, [data-testid='stAppViewContainer'], .stApp {{ {light_tokens} }}
    @media (prefers-color-scheme: dark) {{
      :root, [data-testid='stAppViewContainer'], .stApp {{ {dark_tokens} }}
    }}
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Dynamic Theme Tokens */
{theme_css}

:root {{
  --success: #10B981;
  --warning: #F59E0B;
  --danger:  #EF4444;
  --radius:  12px;
}}

/* Base Elements */
html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
.stApp {{
  background-color: var(--surface-2) !important;
  color: var(--text) !important;
}}
.main .block-container {{
  padding: 1.5rem 2rem 3.5rem;
  max-width: 1400px;
}}

/* ── CRITICAL: Restore & Enhance Sidebar Collapse Button ── */
/* Never hide header completely — keeps sidebar toggle button active! */
#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
header {{
  background: transparent !important;
  z-index: 99 !important;
}}
[data-testid="stHeader"] {{
  background: transparent !important;
  z-index: 99 !important;
}}

/* Floating button when sidebar is collapsed */
[data-testid="collapsedControl"] {{
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 999999 !important;
  top: 14px !important;
  left: 14px !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 6px !important;
  box-shadow: var(--shadow-md) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
[data-testid="collapsedControl"]:hover {{
  background: var(--brand-light) !important;
  border-color: var(--brand) !important;
  transform: scale(1.06) !important;
}}
[data-testid="collapsedControl"] svg {{
  fill: var(--text) !important;
  stroke: var(--text) !important;
  width: 22px !important;
  height: 22px !important;
}}

/* Sidebar internal collapse button */
[data-testid="stSidebarCollapseButton"] {{
  display: flex !important;
  visibility: visible !important;
}}
[data-testid="stSidebarCollapseButton"] svg {{
  fill: #E0E7FF !important;
}}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.1) !important;
}}
[data-testid="stSidebar"] * {{
  color: #E0E7FF !important;
}}
[data-testid="stSidebar"] .stRadio label {{
  color: #C7D2FE !important;
}}
[data-testid="stSidebar"] hr {{
  border-color: rgba(255,255,255,0.15) !important;
}}
[data-testid="stSidebarContent"] .stAlert {{
  background: rgba(255,255,255,0.08);
}}

/* Sidebar brand block */
.sidebar-brand {{
  display: flex; align-items: center; gap: 12px;
  padding: 0 0 16px 0;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  margin-bottom: 16px;
}}
.sidebar-brand-icon {{
  width: 42px; height: 42px; border-radius: 10px;
  background: linear-gradient(135deg, #818CF8, #6366F1);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; flex-shrink: 0;
  box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}}
.sidebar-brand-text h2 {{ color: #FFFFFF !important; font-size: 1.15rem; font-weight: 700; margin: 0; }}
.sidebar-brand-text p  {{ color: #A5B4FC !important; font-size: 0.72rem; margin: 0; }}

/* Sidebar stat pill */
.stat-pill {{
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 12px 14px;
}}
.stat-pill .sp-row {{
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 6px;
}}
.stat-pill .sp-row:last-child {{ margin-bottom: 0; }}
.stat-pill .sp-label {{ font-size: 0.72rem; color: #A5B4FC !important; }}
.stat-pill .sp-val   {{ font-size: 0.82rem; font-weight: 700; color: #FFFFFF !important; }}

/* ── Page Hero Header ── */
.page-hero {{
  background: var(--hero-bg);
  border: 1px solid var(--hero-border);
  border-radius: var(--radius);
  padding: 24px 28px;
  margin-bottom: 22px;
  position: relative; overflow: hidden;
  box-shadow: var(--shadow-sm);
}}
.page-hero::before {{
  content: ''; position: absolute; top: -40px; right: -40px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(99,102,241,.15) 0%, transparent 70%);
}}
.page-hero h1 {{
  font-size: 1.7rem; font-weight: 800; color: var(--text);
  margin: 0 0 6px; letter-spacing: -0.5px;
}}
.page-hero p {{
  font-size: 0.92rem; color: var(--text-muted); margin: 0; max-width: 720px;
  line-height: 1.5;
}}
.page-hero .hero-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--brand); color: white;
  font-size: 0.72rem; font-weight: 700; letter-spacing: .05em;
  padding: 4px 10px; border-radius: 20px; margin-bottom: 8px;
}}

/* ── Section Titles ── */
.section-card-title {{
  font-size: 0.8rem; font-weight: 700; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .08em;
  margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}}
.section-card-title span {{ color: var(--brand); font-size: 1rem; }}

/* ── KPI Metric Cards ── */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 20px;
}}
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
  position: relative; overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;
}}
.kpi-card:hover {{
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}}
.kpi-card-accent {{
  position: absolute; top: 0; left: 0; height: 3px; width: 100%;
}}
.kpi-card .kpi-icon {{ font-size: 1.4rem; margin-bottom: 8px; }}
.kpi-card .kpi-val  {{ font-size: 1.85rem; font-weight: 800; color: var(--text); line-height: 1.1; margin-bottom: 4px; }}
.kpi-card .kpi-label {{ font-size: 0.73rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.kpi-card .kpi-sub   {{ font-size: 0.72rem; color: var(--success); font-weight: 600; margin-top: 6px; }}

/* ── Match Result Banner ── */
.result-banner {{
  background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: var(--radius);
  padding: 22px 26px;
  margin-bottom: 20px;
  color: white;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;
  box-shadow: var(--shadow-md);
}}
.result-banner-left h2 {{ color: white; font-size: 1.25rem; font-weight: 700; margin: 0 0 4px; }}
.result-banner-left p  {{ color: #C7D2FE; font-size: 0.85rem; margin: 0; line-height: 1.4; }}
.result-score-ring {{
  width: 76px; height: 76px; border-radius: 50%;
  background: rgba(255,255,255,0.12);
  border: 3px solid rgba(255,255,255,0.3);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  flex-shrink: 0;
}}
.result-score-ring .score-num {{ font-size: 1.45rem; font-weight: 800; color: white; line-height: 1; }}
.result-score-ring .score-pct {{ font-size: 0.65rem; color: #C7D2FE; font-weight: 600; letter-spacing: 0.05em; }}

/* ── Tier Badges ── */
.tier-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 20px;
  font-size: 0.8rem; font-weight: 700; letter-spacing: .02em;
}}
.tier-exceptional {{ background: var(--tier-exc-bg); color: var(--tier-exc-text); border: 1px solid var(--tier-exc-bdr); }}
.tier-strong      {{ background: var(--tier-str-bg); color: var(--tier-str-text); border: 1px solid var(--tier-str-bdr); }}
.tier-moderate    {{ background: var(--tier-mod-bg); color: var(--tier-mod-text); border: 1px solid var(--tier-mod-bdr); }}
.tier-low         {{ background: var(--tier-low-bg); color: var(--tier-low-text); border: 1px solid var(--tier-low-bdr); }}

/* ── Skills Panels & Badges ── */
.skills-panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
  height: 100%;
}}
.skills-panel-header {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px;
}}
.skills-panel-title {{ font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }}
.skills-panel-count {{
  font-size: 0.72rem; font-weight: 700;
  padding: 2px 8px; border-radius: 10px;
}}
.count-matched {{ background: var(--badge-matched-bg); color: var(--badge-matched-text); }}
.count-missing {{ background: var(--badge-missing-bg); color: var(--badge-missing-text); }}
.count-extra   {{ background: var(--badge-extra-bg);   color: var(--badge-extra-text); }}
.skills-empty {{ font-size: 0.82rem; color: var(--text-muted); font-style: italic; padding: 6px 0; }}

.skill-badge {{
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px; border-radius: 18px;
  font-size: 0.77rem; font-weight: 600; margin: 3px;
  transition: transform .15s ease;
}}
.skill-badge:hover {{ transform: scale(1.05); }}
.badge-matched {{ background: var(--badge-matched-bg); color: var(--badge-matched-text); border: 1px solid var(--badge-matched-bdr); }}
.badge-missing {{ background: var(--badge-missing-bg); color: var(--badge-missing-text); border: 1px solid var(--badge-missing-bdr); }}
.badge-extra   {{ background: var(--badge-extra-bg);   color: var(--badge-extra-text);   border: 1px solid var(--badge-extra-bdr); }}

/* ── Recommendations ── */
.rec-card {{
  border-left: 3.5px solid var(--warning);
  background: var(--rec-med-bg);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 14px 18px;
  margin-bottom: 10px;
}}
.rec-card.high {{ border-color: var(--danger); background: var(--rec-high-bg); }}
.rec-card.low  {{ border-color: var(--success); background: var(--rec-low-bg); }}
.rec-card-label {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--text-muted); margin-bottom: 4px; }}
.rec-card-text  {{ font-size: 0.88rem; color: var(--text); line-height: 1.45; }}

/* ── Leaderboard Rank Cards ── */
.rank-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 10px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .2s ease, transform .15s ease;
}}
.rank-card:hover {{
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}}
.rank-num {{
  width: 36px; height: 36px; border-radius: 50%;
  background: var(--brand-light); color: var(--brand);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.92rem; font-weight: 800; flex-shrink: 0;
}}
.rank-num.gold   {{ background: #FEF9C3; color: #854D0E; }}
.rank-num.silver {{ background: #F1F5F9; color: #475569; }}
.rank-num.bronze {{ background: #FEF3C7; color: #92400E; }}
.rank-info {{ flex: 1; min-width: 0; }}
.rank-name  {{ font-size: 0.94rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.rank-email {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }}
.rank-scores {{ display: flex; gap: 8px; align-items: center; flex-shrink: 0; }}
.score-pill {{
  text-align: center; padding: 4px 8px; border-radius: 8px;
  font-size: 0.74rem; min-width: 54px;
}}
.sp-composite {{ background: var(--brand-light); color: var(--brand); font-weight: 800; }}
.sp-semantic  {{ background: var(--badge-matched-bg); color: var(--badge-matched-text); font-weight: 600; }}
.sp-skill     {{ background: var(--badge-missing-bg); color: var(--badge-missing-text); font-weight: 600; }}

/* ── Candidate Metadata Card ── */
.meta-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}}
.meta-row {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }}
.meta-row:last-child {{ margin-bottom: 0; }}
.meta-icon {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}
.meta-label {{ font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
.meta-value {{ font-size: 0.88rem; color: var(--text); font-weight: 500; word-break: break-all; }}

/* ── Tab & Button Styling ── */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; background: var(--surface);
  border-radius: 10px; padding: 4px;
  border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 8px; padding: 7px 16px;
  font-weight: 600; font-size: 0.84rem;
  color: var(--text-muted) !important;
}}
.stTabs [aria-selected="true"] {{
  background: var(--brand-light) !important;
  color: var(--brand) !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: 0.88rem !important;
  padding: 11px 22px !important;
  box-shadow: 0 4px 14px rgba(99,102,241,.35) !important;
  transition: all .2s ease !important;
}}
.stButton > button[kind="primary"]:hover {{
  box-shadow: 0 6px 20px rgba(99,102,241,.5) !important;
  transform: translateY(-1px) !important;
}}
.stButton > button[kind="secondary"] {{
  border-radius: 10px !important; font-weight: 600 !important;
}}

/* ── Drop Zone & Divider ── */
[data-testid="stFileUploader"] {{
  border: 2px dashed var(--brand-border) !important;
  border-radius: var(--radius) !important;
  background: var(--brand-light) !important;
  padding: 10px !important;
}}
.fancy-divider {{
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 20px 0;
}}
.empty-state {{
  text-align: center; padding: 40px 20px;
  background: var(--surface); border-radius: var(--radius);
  border: 1px dashed var(--border);
}}
.empty-state .es-icon {{ font-size: 2.8rem; margin-bottom: 10px; }}
.empty-state .es-title {{ font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }}
.empty-state .es-text  {{ font-size: 0.85rem; color: var(--text-muted); }}

.arch-box {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px; font-family: 'Courier New', monospace;
  font-size: 0.74rem; color: var(--text); line-height: 1.55; overflow-x: auto;
}}

/* ── Streamlit Form Input Styling for Light/Dark ── */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}}

/* ── MOBILE RESPONSIVENESS OVERRIDES ── */
@media (max-width: 992px) {{
  .kpi-grid {{
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 12px !important;
  }}
}}

@media (max-width: 768px) {{
  .main .block-container {{
    padding: 1rem 0.75rem 2.5rem !important;
  }}
  .page-hero {{
    padding: 16px 18px !important;
    margin-bottom: 16px !important;
  }}
  .page-hero h1 {{
    font-size: 1.35rem !important;
  }}
  .page-hero p {{
    font-size: 0.82rem !important;
  }}
  .kpi-grid {{
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
  }}
  .kpi-card {{
    padding: 12px 14px !important;
  }}
  .kpi-card .kpi-val {{
    font-size: 1.45rem !important;
  }}
  .result-banner {{
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    padding: 18px !important;
  }}
  .result-score-ring {{
    margin-top: 8px !important;
  }}
  .rank-card {{
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 10px !important;
    padding: 14px !important;
  }}
  .rank-scores {{
    justify-content: space-between !important;
    width: 100% !important;
  }}
  .score-pill {{
    flex: 1 !important;
  }}
  .skills-panel {{
    margin-bottom: 12px !important;
  }}
}}

@media (max-width: 480px) {{
  .kpi-grid {{
    grid-template-columns: 1fr !important;
  }}
  .rank-scores {{
    flex-direction: row !important;
    gap: 6px !important;
  }}
  .score-pill {{
    font-size: 0.7rem !important;
    min-width: 0 !important;
  }}
}}
</style>
""", unsafe_allow_html=True)


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

# Chart theme constants
plotly_font_color = "#F8FAFC" if is_dark_explicit else "#0F172A"
plotly_grid_color = "#334155" if is_dark_explicit else "#E2E8F0"
plotly_bg_color   = "rgba(0,0,0,0)"


# ===========================================================================
# PAGE 1 — Match Dashboard
# ===========================================================================
if page == "🎯  Match Dashboard":

    st.markdown("""
    <div class="page-hero">
      <div class="hero-badge">🎯 Single Candidate Evaluation</div>
      <h1>Candidate Match Dashboard</h1>
      <p>Upload a resume PDF or paste text, select target job requirements, and get an instant
         dense semantic match score, hard-skill gap breakdown, and targeted upskilling recommendations.</p>
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
            if st.button("Load Sample Profile →", key="btn_load_sample"):
                st.session_state["pasted_resume_text"] = SAMPLE_RESUMES[preset_choice]
            default_text = st.session_state.get("pasted_resume_text", SAMPLE_RESUMES["Senior Data Scientist"])
            pasted_text = st.text_area("Paste resume text:", value=default_text, height=230, label_visibility="collapsed")
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
            custom_desc = st.text_area("Job Description:", value=default_job_desc, height=190, label_visibility="collapsed")
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
          {kpi_card_html("🎯", f"{res['composite_score']}%", "Composite Score",   "#6366F1", f"Semantic×{semantic_weight:.2f} + Skill×{skill_weight:.2f}")}
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
                number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
                title={"text": "Composite Match Score", "font": {"size": 14, "color": plotly_font_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": plotly_grid_color, "dtick": 25, "tickfont": {"color": plotly_font_color}},
                    "bar":  {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],   "color": "rgba(239, 68, 68, 0.2)"},
                        {"range": [50, 70],  "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [70, 85],  "color": "rgba(99, 102, 241, 0.2)"},
                        {"range": [85, 100], "color": "rgba(16, 185, 129, 0.2)"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor=plotly_bg_color,
                font={"family": "Inter", "color": plotly_font_color},
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
                textfont={"size": 11, "family": "Inter", "color": plotly_font_color},
            ))
            fig_bar.update_layout(
                title={"text": "Domain-Level Coverage Breakdown", "font": {"size": 14, "color": plotly_font_color}},
                xaxis={"range": [0, 115], "showgrid": False, "visible": False},
                yaxis={"automargin": True, "tickfont": {"color": plotly_font_color}},
                height=280,
                margin=dict(l=10, r=60, t=50, b=10),
                paper_bgcolor=plotly_bg_color,
                plot_bgcolor=plotly_bg_color,
                font={"family": "Inter", "color": plotly_font_color},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Skill panels
        col_matched, col_missing, col_extra = st.columns(3, gap="medium")

        with col_matched:
            matched = res["matched_skills"]
            st.markdown(f"""
            <div class="skills-panel">
              <div class="skills-panel-header">
                <span class="skills-panel-title" style="color:#10B981">✅ Matched Skills</span>
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
                <span class="skills-panel-title" style="color:#EF4444">❌ Missing Skills</span>
                <span class="skills-panel-count count-missing">{len(missing)}</span>
              </div>
              <div>
                {"".join(f'<span class="skill-badge badge-missing">✗ {s}</span>' for s in missing)
                 if missing else '<div class="skills-empty" style="color:#10B981">🎉 Full skill coverage achieved!</div>'}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_extra:
            extra = res["additional_skills"][:18]
            st.markdown(f"""
            <div class="skills-panel">
              <div class="skills-panel-header">
                <span class="skills-panel-title" style="color:#818CF8">➕ Additional Skills</span>
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
        notes = st.text_input("📝 Recruiter Notes (optional)", placeholder="e.g., Schedule technical interview next week…")
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
            <div class="stat-pill" style="background:var(--brand-light); border-color:var(--brand-border);">
              <div class="sp-row">
                <span class="sp-label" style="color:var(--text-muted) !important">👥 Cohort Size</span>
                <span class="sp-val" style="color:var(--brand) !important">{len(DEMO_BATCH_CANDIDATES)} Candidates</span>
              </div>
              <div class="sp-row">
                <span class="sp-label" style="color:var(--text-muted) !important">🏷️ Profiles</span>
                <span class="sp-val" style="color:var(--text) !important; font-size:.76rem !important">
                  Data Scientist · BI Lead · Python Dev · Data Eng · Analyst · Frontend
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
                <div style="margin-top:6px; font-size:.75rem; color:var(--text-muted)">
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
            size_max=38,
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=11, textfont_color=plotly_font_color)
        fig_scatter.add_hline(y=50, line_dash="dot", line_color=plotly_grid_color, line_width=1.5)
        fig_scatter.add_vline(x=50, line_dash="dot", line_color=plotly_grid_color, line_width=1.5)
        fig_scatter.update_layout(
            height=460,
            paper_bgcolor=plotly_bg_color,
            plot_bgcolor=plotly_bg_color,
            font={"family": "Inter", "color": plotly_font_color},
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
            xaxis={"gridcolor": plotly_grid_color, "tickfont": {"color": plotly_font_color}},
            yaxis={"gridcolor": plotly_grid_color, "tickfont": {"color": plotly_font_color}},
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
                    paper_bgcolor=plotly_bg_color,
                    plot_bgcolor=plotly_bg_color,
                    font={"family": "Inter", "color": plotly_font_color},
                    legend={"orientation": "h", "y": -0.3},
                    xaxis={"gridcolor": plotly_grid_color, "tickfont": {"color": plotly_font_color}},
                    yaxis={"gridcolor": plotly_grid_color, "tickfont": {"color": plotly_font_color}},
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
                        paper_bgcolor=plotly_bg_color,
                        font={"family": "Inter", "color": plotly_font_color},
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
        #### End-to-End Pipeline Architecture
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
                    f'<span class="skill-badge badge-extra" style="border-color:{color}40; color:{color}; background:{color}15;">{s}</span>'
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
                "embedding_model":      DEFAULT_MODEL_NAME,
                "vector_dimensions":    EMBEDDING_DIM,
                "compute_device":       dev,
                "default_sem_weight":   DEFAULT_SEMANTIC_WEIGHT,
                "default_skill_weight": DEFAULT_SKILL_WEIGHT,
            })
        with col_diag_r:
            st.markdown("**Storage Paths**")
            st.json({
                "sqlite_database":   str(Path(os.getenv("SQLITE_DB_PATH",  "history.db")).resolve()),
                "chroma_vector_db":  str(Path(os.getenv("CHROMA_PERSIST_DIR", "models/chroma_db")).resolve()),
                "hf_model_cache":    str(Path(os.getenv("HF_HOME", "models/hf_home")).resolve()),
            })
