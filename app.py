"""
Semantic Resume & Job Matcher - Streamlit Production Dashboard.

This interactive web application provides:
1. Match Dashboard: PDF resume parsing, semantic embedding alignment, and skill gap analysis.
2. Batch Candidate Ranking: Multi-candidate CSV screening, automated leaderboard, and strategic fit charts.
3. Match History & Audit Logs: SQLite persistence, recruiter notes, and evaluation analytics.
4. Architecture & Taxonomy Explorer: Inspection of dense embedding models and 7-domain technical taxonomy.
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

# Import pipeline modules
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

# Set page configuration
st.set_page_config(
    page_title="Semantic Resume & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .skill-badge-matched {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin: 3px;
        border: 1px solid #BCF0DA;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin: 3px;
        border: 1px solid #FBD5D5;
    }
    .skill-badge-extra {
        display: inline-block;
        background-color: #E1EFFE;
        color: #1E429F;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin: 3px;
        border: 1px solid #C3DDFD;
    }
    .tier-badge-exceptional {
        color: #046C4E;
        background: #DEF7EC;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .tier-badge-strong {
        color: #1A56DB;
        background: #E1EFFE;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .tier-badge-moderate {
        color: #C27803;
        background: #FEF08A;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .tier-badge-low {
        color: #C81E1E;
        background: #FDE8E8;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Cached Resource Loaders
# ===========================================================================

@st.cache_data
def load_preprocessed_jobs() -> pd.DataFrame:
    """Loads cleaned job descriptions artifact if present."""
    csv_path = Path("data/processed/cleaned_job_descriptions.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            logger.warning("Could not read cleaned_job_descriptions.csv: %s", e)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    """Initializes and caches the SentenceTransformer embedding model."""
    try:
        return get_embedding_model()
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        return None


# ===========================================================================
# Sample Candidate & Job Presets
# ===========================================================================

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
        "executive dashboards and automated reports. Advanced proficiency in SQL, Power BI, Tableau, Excel, and SSIS.\n\n"
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
        "Built academic projects utilizing Flask, SQLite, and REST APIs. Seeking an entry-level development role.\n\n"
        "EDUCATION\n"
        "Bachelor of Science in Computer Science, University of Texas (2023)\n\n"
        "TECHNICAL SKILLS\n"
        "Python, JavaScript, HTML, CSS, Git, Linux, SQLite"
    )
}

DEMO_BATCH_CANDIDATES = [
    {
        "name": "Sarah Jenkins (Senior DS)",
        "email": "sarah.jenkins@example.com",
        "text": SAMPLE_RESUMES["Senior Data Scientist"],
    },
    {
        "name": "Michael Chang (BI Lead)",
        "email": "mchang.bi@example.com",
        "text": SAMPLE_RESUMES["BI & Reporting Analyst"],
    },
    {
        "name": "David Miller (Junior Dev)",
        "email": "david.m@example.com",
        "text": SAMPLE_RESUMES["Junior Python Developer"],
    },
    {
        "name": "Elena Rostova (Data Engineer)",
        "email": "elena.r@example.com",
        "text": (
            "Elena Rostova | Data Engineer\n"
            "Email: elena.rostova@example.com\n"
            "Expert in building distributed data pipelines using Python, SQL, Apache Spark, Airflow, "
            "Kafka, Snowflake, AWS, Docker, and Kubernetes. Master of Science in Data Engineering."
        ),
    },
    {
        "name": "Carlos Gomez (Product Analyst)",
        "email": "carlos.g@example.com",
        "text": (
            "Carlos Gomez | Product Analyst\n"
            "Email: carlos.g@example.com\n"
            "Experienced in product analytics, A/B testing, SQL, Tableau, Excel, and user retention cohort analysis. "
            "Bachelor of Business Administration."
        ),
    },
    {
        "name": "Lucas Vance (Frontend Specialist)",
        "email": "lucas.v@example.com",
        "text": (
            "Lucas Vance | Frontend UI Engineer\n"
            "Email: lucas.v@example.com\n"
            "Proficient in React, JavaScript, TypeScript, CSS, HTML, Tailwind, and Figma UI prototyping. "
            "Bachelor of Fine Arts in Web Design."
        ),
    },
]


# ===========================================================================
# Sidebar Navigation & Settings
# ===========================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/resume.png", width=64)
st.sidebar.title("Semantic Matcher")
st.sidebar.caption("AI-Powered Resume Screening & Skill Gap Retrieval")

page = st.sidebar.radio(
    "Navigation",
    [
        "🎯 Match Dashboard",
        "📊 Batch Candidate Ranking",
        "📜 Match History & Analytics",
        "ℹ️ System Architecture & Explorer",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scoring Weights")
semantic_weight = st.sidebar.slider(
    "Semantic Similarity Weight",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_SEMANTIC_WEIGHT),
    step=0.05,
    help="Weight assigned to dense transformer contextual embedding similarity."
)
skill_weight = round(1.0 - semantic_weight, 2)
st.sidebar.info(f"Skill Keyword Overlap Weight: **{skill_weight}**")

st.sidebar.markdown("---")
stats = get_database_stats()
st.sidebar.caption(
    f"📊 **Database Telemetry:**\n"
    f"- Saved Matches: `{stats['total_single_evaluations']}`\n"
    f"- Batch Sessions: `{stats['total_batch_sessions']}`\n"
    f"- Avg Score: `{stats['average_composite_score']}%`"
)

# Load jobs dataset
jobs_df = load_preprocessed_jobs()


# ===========================================================================
# PAGE 1: Match Dashboard (Single Candidate)
# ===========================================================================

if page == "🎯 Match Dashboard":
    st.markdown('<div class="main-header">🎯 Candidate Match & Skill Gap Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a candidate resume PDF or paste raw text, match against target job requirements, and diagnose technical skill gaps in real time.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Candidate Resume")
        resume_tab1, resume_tab2 = st.tabs(["📄 Upload PDF", "✍️ Paste / Select Text"])

        resume_input_data: Optional[Union[bytes, str]] = None
        candidate_source_name = "Custom Resume"

        with resume_tab1:
            uploaded_pdf = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
            if uploaded_pdf is not None:
                resume_input_data = uploaded_pdf.read()
                candidate_source_name = uploaded_pdf.name

        with resume_tab2:
            preset_choice = st.selectbox("Or choose a pre-loaded sample candidate:", list(SAMPLE_RESUMES.keys()))
            if st.button("Load Sample Candidate", key="btn_load_sample"):
                st.session_state["pasted_resume_text"] = SAMPLE_RESUMES[preset_choice]

            default_text = st.session_state.get("pasted_resume_text", SAMPLE_RESUMES["Senior Data Scientist"])
            pasted_text = st.text_area("Candidate Resume Content", value=default_text, height=260)
            if uploaded_pdf is None and pasted_text.strip():
                resume_input_data = pasted_text
                candidate_source_name = preset_choice

    with col_right:
        st.subheader("2. Target Job Description")
        job_tab1, job_tab2 = st.tabs(["💼 Preprocessed Dataset (517 Jobs)", "✍️ Custom Job Posting"])

        job_input_data = ""
        target_job_title = "Target Position"

        with job_tab1:
            if not jobs_df.empty:
                job_titles = jobs_df["Job Title"].tolist()
                selected_title_idx = st.selectbox(
                    "Select a benchmark job from dataset:",
                    range(len(job_titles)),
                    format_func=lambda i: f"{job_titles[i]} (ID: {jobs_df.iloc[i]['job_id']})"
                )
                selected_row = jobs_df.iloc[selected_title_idx]
                target_job_title = selected_row["Job Title"]
                job_input_data = selected_row["clean_description"]
                st.caption(f"**Required Skills in Dataset:** `{selected_row.get('skills_str', 'None')}`")
            else:
                st.warning("Preprocessed dataset not detected in data/processed/. Please use custom job posting.")

        with job_tab2:
            custom_title = st.text_input("Job Title", value="Senior Machine Learning Engineer")
            default_job_desc = (
                "We are seeking a Senior Machine Learning Engineer to design and deploy scalable ML pipelines. "
                "The ideal candidate must be highly proficient in Python, SQL, PyTorch, and AWS cloud architecture. "
                "Experience with Snowflake, Spark, Docker, and CI/CD pipelines is strongly preferred. "
                "Candidate will lead predictive modeling initiatives and collaborate with cross-functional data teams."
            )
            custom_desc = st.text_area("Job Description Requirements", value=default_job_desc, height=220)
            if jobs_df.empty or st.checkbox("Use custom job posting instead of dataset", value=False):
                job_input_data = custom_desc
                target_job_title = custom_title

    st.markdown("---")
    eval_btn = st.button("⚡ Evaluate Candidate & Analyze Skill Gaps", type="primary", use_container_width=True)

    if eval_btn:
        if not resume_input_data:
            st.error("Please provide a resume by uploading a PDF or pasting text.")
        elif not job_input_data or not job_input_data.strip():
            st.error("Please provide a job description.")
        else:
            with st.spinner("Parsing documents, computing 384-d dense embeddings, and executing skill gap engine..."):
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

    if "last_eval_result" in st.session_state:
        res = st.session_state["last_eval_result"]

        # Score Banner & Tier Badge
        tier = res["match_tier"]
        tier_class = f"tier-badge-{res.get('tier_code', 'low')}"

        st.markdown(f"### Evaluation Result: **{res['candidate_name']}** → **{res['job_title']}**")
        st.markdown(f"**Match Classification:** <span class='{tier_class}'>{tier}</span>", unsafe_allow_html=True)
        st.caption(res["explanation"])

        # Metric Cards Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val' style='color:#1E40AF;'>{res['composite_score']}%</div>"
                f"<div class='metric-label'>Composite Match Score</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val' style='color:#059669;'>{res['semantic_score']}%</div>"
                f"<div class='metric-label'>Semantic Alignment</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val' style='color:#D97706;'>{res['skill_score']}%</div>"
                f"<div class='metric-label'>Hard Skill Coverage</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m4:
            matched_len = len(res['matched_skills'])
            total_req = res['skill_gap']['total_job_skills']
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val' style='color:#4B5563;'>{matched_len} / {total_req}</div>"
                f"<div class='metric-label'>Skills Matched</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("####")

        # Two Columns: Gauge Chart & Radar Chart
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["composite_score"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Composite Match Index", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2563EB"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 50], 'color': '#FEE2E2'},
                        {'range': [50, 70], 'color': '#FEF9C3'},
                        {'range': [70, 85], 'color': '#DBEAFE'},
                        {'range': [85, 100], 'color': '#DCFCE7'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': res["composite_score"]
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_chart2:
            # Domain radar / bar chart
            cat_breakdown = res["category_breakdown"]
            cats = [c.replace("_", " ").title() for c in cat_breakdown.keys()]
            coverages = [round(cat_breakdown[c]["coverage"] * 100, 1) for c in cat_breakdown.keys()]

            fig_bar = px.bar(
                x=coverages,
                y=cats,
                orientation='h',
                labels={'x': 'Coverage %', 'y': 'Technical Domain'},
                title="Skill Domain Coverage Breakdown",
                color=coverages,
                color_continuous_scale="Blues",
                range_x=[0, 100]
            )
            fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Skill Chips Display
        col_matched, col_missing, col_extra = st.columns(3)

        with col_matched:
            st.markdown(f"##### ✅ Matched Skills ({len(res['matched_skills'])})")
            if res["matched_skills"]:
                chips = " ".join([f"<span class='skill-badge-matched'>✓ {s}</span>" for s in res["matched_skills"]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.info("No explicit skills directly matched.")

        with col_missing:
            st.markdown(f"##### ❌ Missing Skills ({len(res['missing_skills'])})")
            if res["missing_skills"]:
                chips = " ".join([f"<span class='skill-badge-missing'>✗ {s}</span>" for s in res["missing_skills"]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.success("Candidate covers 100% of required technical competencies!")

        with col_extra:
            st.markdown(f"##### ➕ Additional Skills ({len(res['additional_skills'])})")
            if res["additional_skills"]:
                chips = " ".join([f"<span class='skill-badge-extra'>+ {s}</span>" for s in res["additional_skills"][:15]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No additional keywords detected.")

        st.markdown("---")

        # Upskilling Recommendations
        st.subheader("💡 Strategic Upskilling & Interview Recommendations")
        recs = res.get("recommendations", [])
        if recs:
            for rec in recs:
                st.warning(f"**{rec['category'].replace('_', ' ').title()} ({rec['impact']} Priority):** {rec['suggestion']}")
        else:
            st.success("No critical skill deficits detected. Candidate strongly aligns with this role.")

        # Candidate Profile Metadata
        with st.expander("🔍 View Extracted Candidate Metadata & Education"):
            c_meta1, c_meta2 = st.columns(2)
            with c_meta1:
                st.markdown(f"**Candidate Name:** {res['candidate_name']}")
                st.markdown(f"**Email:** {res['contact_info'].get('email') or 'Not detected'}")
                st.markdown(f"**Phone:** {res['contact_info'].get('phone') or 'Not detected'}")
                st.markdown(f"**Resume Word Count:** {res['resume_word_count']}")
            with c_meta2:
                st.markdown("**Education Entries:**")
                edus = res.get("education", [])
                if edus:
                    for e in edus:
                        deg_name = e.get('degree', 'Degree')
                        field_name = f" in {e['field']}" if e.get('field') else ""
                        inst_name = f" ({e['institution']})" if e.get('institution') else ""
                        st.markdown(f"- *{deg_name}*{field_name}{inst_name}")
                else:
                    st.caption("No formal degree detected via NER.")

        # Recruiter Notes and Save to Database
        st.markdown("---")
        notes = st.text_input("Recruiter Notes (Optional)", placeholder="e.g., Invited for phone screening on Monday...")
        col_save, col_dl = st.columns([1, 2])

        with col_save:
            if st.button("💾 Save Match to History Database", use_container_width=True):
                rec_id = save_match_result(res, notes=notes)
                st.success(f"Successfully logged evaluation to SQLite database (Record ID: #{rec_id})")

        with col_dl:
            report_json = json.dumps(res, indent=2, default=str)
            st.download_button(
                "📥 Download Match Report (JSON)",
                data=report_json,
                file_name=f"match_{res['candidate_name'].replace(' ', '_').lower()}.json",
                mime="application/json",
                use_container_width=True
            )


# ===========================================================================
# PAGE 2: Batch Candidate Ranking
# ===========================================================================

elif page == "📊 Batch Candidate Ranking":
    st.markdown('<div class="main-header">📊 Batch Candidate Screening & Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ingest a cohort of candidate resumes, batch-encode dense embeddings, and automatically rank candidates against a target job posting.</div>', unsafe_allow_html=True)

    col_b_left, col_b_right = st.columns(2)

    with col_b_left:
        st.subheader("1. Ingest Candidate Cohort")
        batch_source = st.radio("Choose Input Method:", ["Load Demo Cohort (6 Candidates)", "Upload CSV File"])

        batch_candidates_data: List[Dict[str, Any]] = []

        if batch_source == "Load Demo Cohort (6 Candidates)":
            st.info("Loaded 6 diverse test candidates (Senior DS, BI Analyst, Junior Dev, Data Engineer, Product Analyst, Frontend).")
            batch_candidates_data = DEMO_BATCH_CANDIDATES
        else:
            uploaded_csv = st.file_uploader("Upload Candidates CSV", type=["csv"])
            if uploaded_csv is not None:
                try:
                    cand_df = pd.read_csv(uploaded_csv)
                    st.dataframe(cand_df.head(4), use_container_width=True)

                    name_col = st.selectbox("Select Candidate Name Column:", cand_df.columns, index=0)
                    text_col = st.selectbox(
                        "Select Resume Text Column:",
                        cand_df.columns,
                        index=1 if len(cand_df.columns) > 1 else 0
                    )

                    for _, r in cand_df.iterrows():
                        batch_candidates_data.append({
                            "name": str(r[name_col]),
                            "text": str(r[text_col]),
                        })
                    st.success(f"Parsed {len(batch_candidates_data)} candidate records from CSV.")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")

    with col_b_right:
        st.subheader("2. Target Benchmark Job")
        if not jobs_df.empty:
            b_job_idx = st.selectbox(
                "Select target job posting:",
                range(len(jobs_df)),
                format_func=lambda i: f"{jobs_df.iloc[i]['Job Title']} (ID: {jobs_df.iloc[i]['job_id']})"
            )
            b_selected_job = jobs_df.iloc[b_job_idx]
            b_target_title = b_selected_job["Job Title"]
            b_target_desc = b_selected_job["clean_description"]
            st.caption(f"**Required Skills:** `{b_selected_job.get('skills_str', 'N/A')}`")
        else:
            b_target_title = st.text_input("Job Title", value="Senior Data Analyst")
            b_target_desc = st.text_area(
                "Job Description",
                value="Looking for Senior Data Analyst with expertise in Python, SQL, Tableau, Power BI, and Snowflake.",
                height=180
            )

    st.markdown("---")
    top_k_select = st.slider("Select Top K Candidates to Return:", min_value=1, max_value=max(10, len(batch_candidates_data) or 10), value=min(5, len(batch_candidates_data) or 5))

    run_batch = st.button("🚀 Run Batch Vector Screening & Leaderboard", type="primary", use_container_width=True)

    if run_batch:
        if not batch_candidates_data:
            st.error("No candidate data available. Please load the demo cohort or upload a CSV.")
        else:
            with st.spinner(f"Batch embedding {len(batch_candidates_data)} candidates with all-MiniLM-L6-v2..."):
                ranked = rank_candidates_batch(
                    resumes=batch_candidates_data,
                    job_input=b_target_desc,
                    job_title=b_target_title,
                    top_k=top_k_select,
                    weights=(semantic_weight, skill_weight),
                )
                st.session_state["last_ranked_batch"] = ranked
                st.session_state["last_batch_job_title"] = b_target_title

    if "last_ranked_batch" in st.session_state:
        ranked = st.session_state["last_ranked_batch"]
        target_title = st.session_state.get("last_batch_job_title", "Target Role")

        st.markdown(f"### 🏆 Candidate Leaderboard for: **{target_title}**")

        # Convert to display DataFrame
        table_rows = []
        for c in ranked:
            table_rows.append({
                "Rank": f"#{c['rank']}",
                "Candidate Name": c["candidate_name"],
                "Composite Match %": f"{c['composite_score']}%",
                "Semantic %": f"{c['semantic_score']}%",
                "Skill %": f"{c['skill_score']}%",
                "Matched Count": c["matched_count"],
                "Missing Count": c["missing_count"],
                "Match Tier": c["match_tier"],
                "Matched Skills": ", ".join(c["matched_skills"][:5]) + ("..." if len(c["matched_skills"]) > 5 else ""),
            })

        leaderboard_df = pd.DataFrame(table_rows)
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        st.markdown("####")

        # Strategic Fit Scatter Chart (Semantic vs Skill Coverage)
        plot_df = pd.DataFrame([
            {
                "Candidate": c["candidate_name"],
                "Semantic Alignment (%)": c["semantic_score"],
                "Skill Coverage (%)": c["skill_score"],
                "Composite Score (%)": c["composite_score"],
                "Tier": c["match_tier"],
            }
            for c in ranked
        ])

        fig_scatter = px.scatter(
            plot_df,
            x="Semantic Alignment (%)",
            y="Skill Coverage (%)",
            size="Composite Score (%)",
            color="Tier",
            hover_name="Candidate",
            text="Candidate",
            title="Strategic Fit Matrix: Semantic Context vs Technical Skill Coverage",
            color_discrete_map={
                "Exceptional Fit": "#059669",
                "Strong Fit": "#2563EB",
                "Moderate Fit": "#D97706",
                "Low Fit": "#DC2626",
            },
            range_x=[0, 105],
            range_y=[0, 105],
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Persistence & CSV Export Actions
        col_save_b, col_dl_b = st.columns(2)
        with col_save_b:
            if st.button("💾 Save Batch Session to Database", use_container_width=True):
                sess_data = {"job_title": target_title}
                sess_id = save_batch_session(sess_data, ranked)
                st.success(f"Batch session persisted with ID: `{sess_id}`")

        with col_dl_b:
            csv_export = pd.DataFrame(ranked).to_csv(index=False)
            st.download_button(
                "📥 Export Ranked Leaderboard (CSV)",
                data=csv_export,
                file_name=f"ranked_candidates_{target_title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ===========================================================================
# PAGE 3: Match History & Audit Logs
# ===========================================================================

elif page == "📜 Match History & Analytics":
    st.markdown('<div class="main-header">📜 Evaluation Audit Trail & Historical Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review historical candidate screening records stored in SQLite, analyze match distributions, and manage audit logs.</div>', unsafe_allow_html=True)

    db_stats = get_database_stats()

    # Overview KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Single Evaluations", db_stats["total_single_evaluations"])
    with k2:
        st.metric("Total Batch Sessions", db_stats["total_batch_sessions"])
    with k3:
        st.metric("Total Candidates Evaluated", db_stats["total_candidates_evaluated"])
    with k4:
        st.metric("Avg Composite Score", f"{db_stats['average_composite_score']}%")

    st.markdown("---")

    # Filters
    f1, f2 = st.columns([3, 1])
    with f1:
        search_kw = st.text_input("🔍 Search History (Candidate Name, Job Title, or Email)", placeholder="Search...")
    with f2:
        tier_opt = st.selectbox("Filter by Tier", ["All", "Exceptional", "Strong", "Moderate", "Low"])

    resolved_tier = None if tier_opt == "All" else tier_opt.lower()
    history_records = get_match_history(limit=100, search_query=search_kw, tier_filter=resolved_tier)

    if history_records:
        history_df = pd.DataFrame([
            {
                "ID": r["id"],
                "Timestamp": r["timestamp"][:19].replace("T", " "),
                "Candidate": r["candidate_name"],
                "Job Title": r["job_title"],
                "Composite %": r["composite_score"],
                "Semantic %": r["semantic_score"],
                "Skill %": r["skill_score"],
                "Tier": r["match_tier"],
                "Notes": r.get("notes") or "",
            }
            for r in history_records
        ])

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.markdown("####")

        # Score Distribution Histogram
        if len(history_df) >= 2:
            fig_hist = px.histogram(
                history_df,
                x="Composite %",
                color="Tier",
                nbins=10,
                title="Historical Candidate Match Score Distribution",
                color_discrete_map={
                    "Exceptional Fit": "#059669",
                    "Strong Fit": "#2563EB",
                    "Moderate Fit": "#D97706",
                    "Low Fit": "#DC2626",
                }
            )
            fig_hist.update_layout(height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Inspection Expander
        with st.expander("🔎 Inspect Specific Evaluation Record"):
            sel_id = st.selectbox("Select Record ID:", [r["id"] for r in history_records])
            detail = get_match_by_id(sel_id)
            if detail:
                st.json(detail)

        # Actions
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear Entire History", type="secondary"):
                clear_match_history()
                st.warning("History cleared. Reload page to refresh view.")
        with col_export:
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                "📥 Export Full Audit Log (CSV)",
                data=csv_data,
                file_name="evaluation_audit_log.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No evaluation records match your filter criteria or database is currently empty.")


# ===========================================================================
# PAGE 4: System Architecture & Taxonomy Explorer
# ===========================================================================

elif page == "ℹ️ System Architecture & Explorer":
    st.markdown('<div class="main-header">ℹ️ System Architecture & NLP Taxonomy Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into model architecture, device acceleration, token window constraints, and domain skill taxonomies.</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🏛️ System Architecture", "📚 7-Domain Skill Taxonomy", "⚡ Runtime Diagnostics"])

    with t1:
        st.markdown("""
        ### End-to-End Information Retrieval Pipeline

        ```
        ┌────────────────────────────────────────────────────────────────────────┐
        │                        Streamlit Web Interface                         │
        │      (Single Match Dashboard  |  Batch Ranking  |  SQLite History)     │
        └──────────────────┬─────────────────────────────────┬───────────────────┘
                           │ Parse PDF / Clean Text          │ Compute Embeddings
                           ▼                                 ▼
        ┌────────────────────────────────────┐    ┌──────────────────────────────┐
        │          Document Parser           │    │    Sentence-Transformers     │
        │   - Dual-Engine PDF Ingestion      │    │    - all-MiniLM-L6-v2        │
        │   - Unicode / Ligature Fixing      │    │    - 384-dimensional dense   │
        │   - spaCy NER Entity Extraction    │    │    - Cosine similarity calc  │
        └──────────────────┬─────────────────┘    └──────────────┬───────────────┘
                           │ Parsed Metadata                     │ Dense Vectors
                           ▼                                     ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │                     Skill Gap & Ranking Matcher                        │
        │   - 7-Domain Technical Taxonomy Comparison                             │
        │   - Weighted Composite Scoring (60% Semantic + 40% Keyword Coverage)   │
        │   - Actionable Upskilling Recommendations                              │
        └──────────────────┬─────────────────────────────────────────────────────┘
                           │ Persist Evaluations
                           ▼
        ┌────────────────────────────────────┐
        │          SQLite Database           │
        │   - match_history audit records    │
        │   - batch_sessions & candidates    │
        └────────────────────────────────────┘
        ```

        #### Solving the 256-Token Context Window Bottleneck
        - Standard Transformer models (`all-MiniLM-L6-v2`) operate with a **256-token limit** (~180-200 words).
        - Raw job postings average **650 words**, with some exceeding **2,100 words**. Direct truncation would silently discard 60-75% of technical requirements.
        - Our pipeline generates **Dense Profile Chunks** (`Job Title + Extracted Skills + First 120 words of Responsibilities`), ensuring 100% of required skills are vectorized inside the token budget.
        """)

    with t2:
        st.markdown("### Pre-defined 7-Domain Technical Competency Taxonomy")
        st.markdown("Extracted via word-boundary regular expressions and spaCy EntityRuler:")

        for domain_name, skills_list in SKILLS_TAXONOMY.items():
            with st.expander(f"📁 {domain_name.replace('_', ' ').title()} ({len(skills_list)} keywords)"):
                chips = " ".join([f"<span class='skill-badge-extra'>{s}</span>" for s in skills_list])
                st.markdown(chips, unsafe_allow_html=True)

    with t3:
        st.markdown("### Runtime Environment Diagnostics")
        dev = get_optimal_device()

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            st.metric("Compute Acceleration", dev.upper())
        with c_d2:
            st.metric("Embedding Model", DEFAULT_MODEL_NAME)
        with c_d3:
            st.metric("Dense Dimensions", f"{EMBEDDING_DIM}-d")

        st.markdown("#### Hardware Details & Dependencies")
        st.json({
            "optimal_device": dev,
            "embedding_dimension": EMBEDDING_DIM,
            "model_name": DEFAULT_MODEL_NAME,
            "jobs_in_dataset": len(jobs_df) if not jobs_df.empty else 0,
            "sqlite_database_path": str(Path("history.db").resolve()),
        })
