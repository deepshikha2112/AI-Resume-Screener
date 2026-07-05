"""
AI Resume Screening System - Streamlit Application
--------------------------------------------------
A professional offline resume screening dashboard that ranks PDF resumes
against a job description using classical NLP, TF-IDF, and cosine similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from resume_parser import parse_job_description, parse_resume
from similarity import keyword_frequency, rank_resumes
from skill_extractor import compare_skills, extract_skills, generate_interview_questions
from utils import (
    dataframe_to_csv,
    dataframe_to_excel,
    ensure_directories,
    format_list,
    generate_pdf_report,
    get_recommendation,
    save_uploaded_file,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
ASSETS_DIR = BASE_DIR / "assets"


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_css() -> None:
    """Inject a modern dark dashboard style into Streamlit."""

    st.markdown(
        """
        <style>
            :root {
                --primary: #22d3ee;
                --secondary: #a78bfa;
                --success: #34d399;
                --warning: #fbbf24;
                --danger: #fb7185;
                --card: rgba(15, 23, 42, 0.82);
                --card-border: rgba(148, 163, 184, 0.22);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(34, 211, 238, 0.14), transparent 32rem),
                    radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 36rem),
                    linear-gradient(135deg, #020617 0%, #0f172a 52%, #111827 100%);
                color: #e5e7eb;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.96));
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }

            [data-testid="stSidebar"] * { color: #e5e7eb; }

            .main-title {
                font-size: clamp(2.1rem, 5vw, 4rem);
                font-weight: 900;
                line-height: 1.05;
                letter-spacing: -0.05em;
                margin-bottom: 0.25rem;
                background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .subtitle {
                color: #cbd5e1;
                font-size: 1.05rem;
                margin-bottom: 1.4rem;
            }

            .glass-card {
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 1.35rem;
                padding: 1.15rem 1.25rem;
                box-shadow: 0 20px 45px rgba(0, 0, 0, 0.24);
                backdrop-filter: blur(16px);
            }

            .metric-card {
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.8));
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 1.25rem;
                padding: 1rem;
                min-height: 138px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 16px 36px rgba(0, 0, 0, 0.22);
            }

            .metric-label {
                color: #94a3b8;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.4rem;
            }

            .metric-value {
                color: #f8fafc;
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
            }

            .metric-caption { color: #cbd5e1; font-size: 0.88rem; }

            .candidate-card {
                border-left: 5px solid #22d3ee;
                background: linear-gradient(135deg, rgba(8, 47, 73, 0.72), rgba(30, 41, 59, 0.72));
                border-radius: 1.2rem;
                padding: 1.1rem 1.25rem;
                margin: 0.5rem 0 1rem;
            }

            .badge {
                display: inline-block;
                padding: 0.22rem 0.6rem;
                margin: 0.13rem;
                border-radius: 999px;
                font-size: 0.8rem;
                border: 1px solid rgba(255,255,255,0.15);
            }
            .badge-good { background: rgba(52, 211, 153, 0.16); color: #bbf7d0; }
            .badge-bad { background: rgba(251, 113, 133, 0.15); color: #fecdd3; }
            .badge-neutral { background: rgba(34, 211, 238, 0.13); color: #cffafe; }

            div[data-testid="stMetricValue"] { color: #f8fafc; }
            div[data-testid="stMetricLabel"] { color: #cbd5e1; }
            .stDataFrame { border-radius: 1rem; overflow: hidden; }
            .block-container { padding-top: 2rem; padding-bottom: 3rem; }
            hr { border-color: rgba(148, 163, 184, 0.16); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Create Streamlit session state defaults."""

    defaults = {
        "analysis_results": [],
        "job_text": "",
        "job_filename": "",
        "jd_skills": [],
        "errors": [],
        "last_source": "No analysis yet",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def metric_card(icon: str, label: str, value: str, caption: str = "") -> None:
    """Render a reusable dashboard metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(score: float) -> str:
    """Return a semantic color for a match score."""

    if score >= 80:
        return "#34d399"
    if score >= 60:
        return "#fbbf24"
    return "#fb7185"


def badges(items: List[str], css_class: str) -> str:
    """Create HTML badges for skills and keywords."""

    if not items:
        return "<span class='badge badge-neutral'>None detected</span>"
    return "".join(f"<span class='badge {css_class}'>{item}</span>" for item in items[:18])


def build_results_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert analysis result dictionaries into a dashboard-friendly DataFrame."""

    rows = []
    for result in results:
        rows.append(
            {
                "Rank": result["rank"],
                "Candidate": result["name"],
                "File": result["filename"],
                "Match %": round(result["match_score"], 2),
                "Recommendation": result["recommendation"],
                "Email": result["email"],
                "Phone": result["phone"],
                "Matched Skills": format_list(result["matched_skills"]),
                "Missing Skills": format_list(result["missing_skills"]),
                "Education": format_list(result["education"]),
                "Experience": result["experience"],
                "Summary": result["summary"],
            }
        )
    return pd.DataFrame(rows)


def analyze_documents(job_source: Any, resume_sources: List[Any], source_label: str) -> None:
    """Parse, score, rank, and enrich all submitted documents."""

    errors: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        job_payload = parse_job_description(job_source)
    except Exception as exc:
        st.error(f"Unable to parse job description: {exc}")
        return

    if not job_payload.get("text", "").strip():
        st.error("The job description appears to be empty. Please upload a readable PDF, DOCX, or TXT file.")
        return

    candidates: List[Dict[str, Any]] = []
    for resume_source in resume_sources:
        try:
            parsed_resume = parse_resume(resume_source)
            if parsed_resume.get("raw_text", "").strip():
                candidates.append(parsed_resume)
            else:
                errors.append(f"{parsed_resume.get('filename', 'Unknown file')} has no readable text.")
        except Exception as exc:
            filename = getattr(resume_source, "name", str(resume_source))
            errors.append(f"{filename}: {exc}")

    if not candidates:
        st.error("No readable PDF resumes were found. Please upload valid text-based PDF resumes.")
        if errors:
            st.warning("\n".join(errors))
        return

    ranked = rank_resumes(job_payload["text"], candidates)
    jd_skills = sorted(extract_skills(job_payload["text"]))

    for index, item in enumerate(ranked, start=1):
        candidate = item["candidate"]

        candidate_skills = sorted(
            set(candidate.get("skills") or [])
            | set(extract_skills(candidate.get("raw_text") or ""))
        )

        matched_skills, missing_skills = compare_skills(
            candidate.get("raw_text") or "",
            job_payload.get("text") or "",
        )

        recommendation = get_recommendation(item["score"])

        results.append(
            {
                "rank": index,
                "filename": candidate.get("filename", "Unknown"),
                "name": candidate.get("name", "Unknown Candidate"),
                "email": candidate.get("email", "Not found"),
                "phone": candidate.get("phone", "Not found"),
                "education": candidate.get("education", []),
                "skills": candidate_skills,
                "experience": candidate.get("experience", "Not clearly mentioned"),
                "summary": candidate.get("summary", "Summary unavailable."),
                "match_score": round(float(item["score"]), 2),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "recommendation": recommendation,
                "questions": generate_interview_questions(missing_skills),
                "raw_text": candidate.get("raw_text", ""),
            }
        )

    st.session_state.analysis_results = results
    st.session_state.job_text = job_payload["text"]
    st.session_state.job_filename = job_payload.get("filename", "Job Description")
    st.session_state.jd_skills = jd_skills
    st.session_state.errors = errors
    st.session_state.last_source = source_label

    st.success(f"Analysis complete. Ranked {len(results)} resume(s) successfully.")
    if errors:
        st.warning("Some files could not be fully processed:\n" + "\n".join(errors))


def analyze_uploaded_files(job_file: Any, resume_files: List[Any]) -> None:
    """Validate and analyze user-uploaded job and resume files."""

    if job_file is None:
        st.error("Please upload one job description file before analysis.")
        return
    if not resume_files:
        st.error("Please upload at least one PDF resume before analysis.")
        return

    job_suffix = Path(job_file.name).suffix.lower()
    if job_suffix not in {".pdf", ".docx", ".txt"}:
        st.error("Job description must be PDF, DOCX, or TXT.")
        return

    invalid_resumes = [file.name for file in resume_files if Path(file.name).suffix.lower() != ".pdf"]
    if invalid_resumes:
        st.error("Only PDF resumes are supported: " + ", ".join(invalid_resumes))
        return

    saved_job = save_uploaded_file(job_file, UPLOAD_DIR)
    saved_resumes = [save_uploaded_file(file, UPLOAD_DIR) for file in resume_files]
    analyze_documents(saved_job, saved_resumes, "Uploaded files")


def analyze_sample_dataset() -> None:
    """Analyze bundled sample data for quick testing and demos."""

    sample_job = DATA_DIR / "sample_job_description.txt"
    sample_resumes = sorted(DATA_DIR.glob("sample_resume_*.pdf"))

    if not sample_job.exists() or len(sample_resumes) < 3:
        st.error("Sample data is missing. Please check the data folder.")
        return

    analyze_documents(sample_job, sample_resumes, "Bundled sample dataset")


def filtered_results() -> List[Dict[str, Any]]:
    """Apply sidebar search, skill, and score filters to the current analysis."""

    results = st.session_state.analysis_results
    query = st.session_state.get("candidate_search", "").strip().lower()
    skill_query = st.session_state.get("skill_search", "").strip().lower()
    min_score = float(st.session_state.get("min_match", 0))

    filtered = []
    for result in results:
        candidate_blob = " ".join(
            [
                result.get("name", ""),
                result.get("email", ""),
                result.get("filename", ""),
                " ".join(result.get("skills", [])),
            ]
        ).lower()
        skill_blob = " ".join(result.get("skills", [])).lower()
        if query and query not in candidate_blob:
            continue
        if skill_query and skill_query not in skill_blob:
            continue
        if result.get("match_score", 0) < min_score:
            continue
        filtered.append(result)
    return filtered


def render_sidebar() -> str:
    """Render sidebar navigation, uploads, sample loader, and filters."""

    with st.sidebar:
        st.markdown("## 🤖 AI Resume Screener")
        st.caption("Offline NLP • TF-IDF • Cosine Similarity")
        page = st.radio(
            "Navigate",
            ["📊 Dashboard", "🏆 Candidate Analysis", "⚖️ Compare Candidates", "📤 Export Center", "ℹ️ About"],
        )

        st.divider()
        st.markdown("### 📥 Upload Documents")
        job_file = st.file_uploader("Job Description", type=["pdf", "docx", "txt"])
        resume_files = st.file_uploader("Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

        col_a, col_b = st.columns(2)
        with col_a:
            process = st.button("🚀 Analyze", type="primary", use_container_width=True)
        with col_b:
            sample = st.button("🧪 Demo", use_container_width=True)

        if process:
            with st.spinner("Parsing documents and calculating match scores..."):
                analyze_uploaded_files(job_file, resume_files)
        if sample:
            with st.spinner("Loading sample job description and resumes..."):
                analyze_sample_dataset()

        st.divider()
        st.markdown("### 🔎 Search & Filter")
        st.text_input("Search candidate/name/email/skill", key="candidate_search")
        st.text_input("Search by skill", placeholder="python, sql, react...", key="skill_search")
        st.slider("Minimum match percentage", 0, 100, 0, key="min_match")

        st.divider()
        st.caption(f"Last source: {st.session_state.last_source}")

    return page


def render_hero() -> None:
    """Render the landing state before analysis."""

    st.markdown('<div class="main-title">AI Resume Screening System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Rank resumes against a job description with offline classical NLP, skill-gap insights, charts, and downloadable reports.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("📄", "Input", "1 JD + PDFs", "PDF, DOCX, TXT job descriptions")
    with col2:
        metric_card("🧠", "NLP Engine", "TF-IDF", "Tokenization, stopwords, lemmatization")
    with col3:
        metric_card("📊", "Output", "Ranked List", "Charts, skills, recommendations, exports")

    st.info("Upload files from the sidebar or click **Demo** to analyze the bundled sample dataset.")


def render_dashboard(results: List[Dict[str, Any]]) -> None:
    """Render KPI cards, charts, keyword visualization, and ranking table."""

    st.markdown('<div class="main-title">📊 Screening Dashboard</div>', unsafe_allow_html=True)
    if not results:
        render_hero()
        return

    df = build_results_dataframe(results)
    best = results[0]
    highest = max(item["match_score"] for item in results)
    average = sum(item["match_score"] for item in results) / len(results)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("📚", "Resumes Uploaded", str(len(results)), "Readable resumes analyzed")
    with col2:
        metric_card("🏅", "Highest Match", f"{highest:.1f}%", best["name"])
    with col3:
        metric_card("📈", "Average Match", f"{average:.1f}%", "Across filtered candidates")
    with col4:
        metric_card("⭐", "Best Candidate", best["name"], best["recommendation"])

    st.markdown("### 🏆 Top Candidate")
    st.markdown(
        f"""
        <div class="candidate-card">
            <h3>{best['name']} • <span style="color:{score_color(best['match_score'])}">{best['match_score']:.1f}% Match</span></h3>
            <p><strong>Recommendation:</strong> {best['recommendation']}</p>
            <p><strong>Email:</strong> {best['email']} &nbsp; | &nbsp; <strong>Phone:</strong> {best['phone']}</p>
            <p><strong>Summary:</strong> {best['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(best["match_score"] / 100, 1.0), text=f"Resume Score Gauge: {best['match_score']:.1f}%")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_bar = px.bar(
            df,
            x="Candidate",
            y="Match %",
            color="Match %",
            color_continuous_scale=["#fb7185", "#fbbf24", "#34d399"],
            title="Candidate Match Comparison",
            text="Match %",
        )
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        top = best
        pie_values = [len(top["matched_skills"]), len(top["missing_skills"])]
        fig_pie = px.pie(
            names=["Matched Skills", "Missing Skills"],
            values=pie_values,
            title="Top Candidate Skill Match",
            color_discrete_sequence=["#34d399", "#fb7185"],
            hole=0.45,
        )
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    freq_pairs = keyword_frequency(st.session_state.job_text, top_n=15)
    if freq_pairs:
        freq_df = pd.DataFrame(freq_pairs, columns=["Keyword", "Frequency"])
        fig_freq = px.bar(
            freq_df,
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Job Description Keyword Frequency",
            color="Frequency",
            color_continuous_scale="Tealgrn",
        )
        fig_freq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("### 🎨 Resume Ranking Table")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.1f%%"),
            "Summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )


def render_candidate_analysis(results: List[Dict[str, Any]]) -> None:
    """Render detailed candidate cards with matched/missing skills and questions."""

    st.markdown('<div class="main-title">🏆 Candidate Analysis</div>', unsafe_allow_html=True)
    if not results:
        render_hero()
        return

    names = [f"#{item['rank']} - {item['name']} ({item['match_score']:.1f}%)" for item in results]
    selected_label = st.selectbox("Select candidate", names)
    selected = results[names.index(selected_label)]

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>{selected['name']}</h3>
                <p><strong>File:</strong> {selected['filename']}</p>
                <p><strong>Email:</strong> {selected['email']}</p>
                <p><strong>Phone:</strong> {selected['phone']}</p>
                <p><strong>Recommendation:</strong> {selected['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=selected["match_score"],
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": score_color(selected["match_score"])},
                    "steps": [
                        {"range": [0, 60], "color": "rgba(251,113,133,0.28)"},
                        {"range": [60, 80], "color": "rgba(251,191,36,0.28)"},
                        {"range": [80, 100], "color": "rgba(52,211,153,0.28)"},
                    ],
                },
                title={"text": "Resume Score"},
            )
        )
        fig_gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col3:
        st.markdown("#### 🎓 Education")
        st.write(format_list(selected["education"]) or "Not clearly mentioned")
        st.markdown("#### 💼 Experience")
        st.write(selected["experience"])

    st.markdown("### ✅ Matched Skills")
    st.markdown(badges(selected["matched_skills"], "badge-good"), unsafe_allow_html=True)
    st.markdown("### ⚠️ Missing Skills / Skill Gap")
    st.markdown(badges(selected["missing_skills"], "badge-bad"), unsafe_allow_html=True)

    st.markdown("### 📝 Resume Summary")
    st.write(selected["summary"])

    st.markdown("### 🎤 AI-Generated Interview Questions (Rule-Based, Offline)")
    for question in selected["questions"]:
        st.markdown(f"- {question}")

    with st.expander("View extracted skills and raw text preview"):
        st.write(format_list(selected["skills"]))
        st.text(selected["raw_text"][:3500])


def render_candidate_comparison(results: List[Dict[str, Any]]) -> None:
    """Render side-by-side candidate comparison and skill gap charts."""

    st.markdown('<div class="main-title">⚖️ Candidate Comparison</div>', unsafe_allow_html=True)
    if not results:
        render_hero()
        return

    candidate_names = [item["name"] for item in results]
    default_selection = candidate_names[: min(3, len(candidate_names))]
    selected_names = st.multiselect("Choose candidates to compare", candidate_names, default=default_selection)
    selected_results = [item for item in results if item["name"] in selected_names]

    if not selected_results:
        st.warning("Select at least one candidate to compare.")
        return

    comparison_rows = []
    for item in selected_results:
        comparison_rows.append(
            {
                "Candidate": item["name"],
                "Match %": item["match_score"],
                "Matched Skill Count": len(item["matched_skills"]),
                "Missing Skill Count": len(item["missing_skills"]),
                "Detected Skill Count": len(item["skills"]),
                "Recommendation": item["recommendation"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    col1, col2 = st.columns(2)
    with col1:
        fig_scores = px.bar(
            comparison_df,
            x="Candidate",
            y="Match %",
            color="Recommendation",
            title="Selected Candidate Scores",
            color_discrete_map={
                "Highly Recommended": "#34d399",
                "Recommended": "#fbbf24",
                "Needs Improvement": "#fb7185",
            },
        )
        fig_scores.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scores, use_container_width=True)

    with col2:
        skill_df = comparison_df.melt(
            id_vars="Candidate",
            value_vars=["Matched Skill Count", "Missing Skill Count"],
            var_name="Skill Type",
            value_name="Count",
        )
        fig_skills = px.bar(
            skill_df,
            x="Candidate",
            y="Count",
            color="Skill Type",
            barmode="group",
            title="Skill Gap Analysis",
            color_discrete_sequence=["#34d399", "#fb7185"],
        )
        fig_skills.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_skills, use_container_width=True)

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_export_center(results: List[Dict[str, Any]]) -> None:
    """Render CSV, Excel, and PDF report downloads."""

    st.markdown('<div class="main-title">📤 Export Center</div>', unsafe_allow_html=True)
    if not results:
        render_hero()
        return

    df = build_results_dataframe(results)
    top_candidate = results[0]

    st.markdown("Download analysis results for documentation, sharing, or internship project submission.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Download CSV",
            data=dataframe_to_csv(df),
            file_name="resume_screening_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download Excel",
            data=dataframe_to_excel(df),
            file_name="resume_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇️ Download PDF Report",
            data=generate_pdf_report(df, top_candidate, st.session_state.jd_skills),
            file_name="resume_screening_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("### Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_about() -> None:
    """Render project information and usage notes."""

    st.markdown('<div class="main-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <h3>AI Resume Screening System</h3>
            <p>
                This final-year/internship-ready project screens resumes completely offline using classical NLP.
                It extracts candidate details, compares resumes with a job description, highlights matched and missing
                skills, ranks applicants, visualizes insights, and exports reports.
            </p>
            <ul>
                <li><strong>No paid APIs:</strong> No OpenAI, Gemini, Claude, or cloud dependency.</li>
                <li><strong>NLP stack:</strong> NLTK preprocessing, TF-IDF vectors, cosine similarity.</li>
                <li><strong>Exports:</strong> CSV, Excel, and PDF analysis reports.</li>
                <li><strong>Bonus:</strong> keyword frequency, skill gap, comparison dashboard, and interview questions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How recommendations are calculated"):
        st.write(
            "Candidates scoring 80% or above are Highly Recommended, 60% to 79.99% are Recommended, "
            "and below 60% are marked Needs Improvement. The score is based on TF-IDF cosine similarity between "
            "the job description and resume text, enriched by visible skill-gap analysis."
        )


def main() -> None:
    """Application entry point."""

    ensure_directories([DATA_DIR, UPLOAD_DIR, ASSETS_DIR])
    apply_custom_css()
    initialize_state()

    page = render_sidebar()
    results = filtered_results()

    if page == "📊 Dashboard":
        render_dashboard(results)
    elif page == "🏆 Candidate Analysis":
        render_candidate_analysis(results)
    elif page == "⚖️ Compare Candidates":
        render_candidate_comparison(results)
    elif page == "📤 Export Center":
        render_export_center(results)
    else:
        render_about()


if __name__ == "__main__":
    main()