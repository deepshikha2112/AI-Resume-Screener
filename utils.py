"""Utility helpers for AI Resume Screening System."""

from __future__ import annotations
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, List
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", filename.strip())
    return cleaned or "uploaded_file"

def save_uploaded_file(uploaded_file, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(uploaded_file.name)
    destination = upload_dir / safe_name
    counter = 1
    while destination.exists():
        destination = upload_dir / f"{destination.stem}_{counter}{destination.suffix}"
        counter += 1
    destination.write_bytes(uploaded_file.getbuffer())
    return destination

def format_list(items: Iterable[str] | None, max_items: int = 12) -> str:
    if not items:
        return ""
    unique_items = []
    for item in items:
        item = str(item).strip()
        if item and item not in unique_items:
            unique_items.append(item)
    visible = unique_items[:max_items]
    suffix = "" if len(unique_items) <= max_items else f" +{len(unique_items) - max_items} more"
    return ", ".join(visible) + suffix

def get_recommendation(score: float) -> str:
    if score >= 80:
        return "Highly Recommended"
    if score >= 60:
        return "Recommended"
    return "Needs Improvement"

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 25]

def generate_resume_summary(text: str, skills: List[str], education: List[str], experience: str) -> str:
    skill_part = format_list(skills, max_items=8) or "relevant technical skills"
    education_part = format_list(education, max_items=2) or "education details not clearly detected"
    experience_part = experience if experience and experience != "Not clearly mentioned" else "experience details not clearly detected"
    sentences = split_sentences(text)
    intro = sentences[0] if sentences else "Candidate profile extracted from the uploaded resume."
    if len(intro) > 220:
        intro = intro[:217].rsplit(" ", 1)[0] + "..."
    return f"{intro} Key skills include {skill_part}. Education: {education_part}. Experience: {experience_part}."

def extract_years_of_experience(text: str) -> str:
    patterns = [
        r"(\d{1,2}\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+)?experience[^.\n]*)",
        r"(?:experience|work history|employment)[\s:\-]+([^\n]{20,180})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
    return "Not clearly mentioned"

def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screening Results")
        worksheet = writer.sheets["Screening Results"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 14), 48)
    buffer.seek(0)
    return buffer.getvalue()

def generate_pdf_report(df: pd.DataFrame, top_candidate: dict, required_skills: List[str]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DarkTitle", parent=styles["Title"], textColor=colors.HexColor("#0f172a"), fontSize=20, leading=24, spaceAfter=12)
    heading_style = ParagraphStyle("SectionHeading", parent=styles["Heading2"], textColor=colors.HexColor("#164e63"), fontSize=13, leading=16, spaceBefore=10, spaceAfter=8)
    normal_style = ParagraphStyle("NormalWrap", parent=styles["BodyText"], fontSize=9, leading=12)

    story = [
        Paragraph("AI Resume Screening Analysis Report", title_style),
        Paragraph("Offline resume-job matching report generated using NLP preprocessing, TF-IDF vectorization, and cosine similarity.", normal_style),
        Spacer(1, 0.12*inch),
        Paragraph("Top Candidate", heading_style),
        Paragraph(f"<b>{top_candidate.get('name', 'Unknown')}</b> scored <b>{top_candidate.get('match_score', 0):.1f}%</b> and is marked <b>{top_candidate.get('recommendation', 'N/A')}</b>.", normal_style),
        Paragraph(f"Matched skills: {format_list(top_candidate.get('matched_skills', []), 20)}", normal_style),
        Paragraph(f"Missing skills: {format_list(top_candidate.get('missing_skills', []), 20) or 'None detected'}", normal_style),
        Spacer(1, 0.12*inch),
        Paragraph("Required Skills Detected in Job Description", heading_style),
        Paragraph(format_list(required_skills, 40) or "No explicit skills detected.", normal_style),
        Spacer(1, 0.12*inch),
        Paragraph("Candidate Ranking", heading_style),
    ]

    table_columns = ["Rank", "Candidate", "Match %", "Recommendation", "Email"]
    safe_df = df[table_columns].copy()
    table_data = [table_columns] + safe_df.astype(str).values.tolist()
    table = Table(table_data, repeatRows=1, colWidths=[0.45*inch, 1.45*inch, 0.75*inch, 1.45*inch, 2.0*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164e63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.14*inch))
    story.append(Paragraph("Notes", heading_style))
    story.append(Paragraph("This report is a decision-support tool. Final hiring decisions should include human review, interview performance, portfolio quality, and organizational requirements.", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()