"""Resume and job description parsing utilities."""

from __future__ import annotations
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
import pdfplumber
from docx import Document
from skill_extractor import extract_skills   
from utils import extract_years_of_experience, generate_resume_summary

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "doctorate", "b.tech", "m.tech", "b.e", "m.e", "bsc", "msc", "bca", "mca", "mba",
    "computer science", "information technology", "engineering", "university", "college", "institute",
]

SECTION_STOP_WORDS = {"skills", "technical skills", "projects", "education", "certifications", "achievements", "contact", "summary"}

def _source_name(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return getattr(source, "name", "uploaded_file")

def _source_suffix(source: Any) -> str:
    return Path(_source_name(source)).suffix.lower()

def _read_bytes_source(source: Any) -> BytesIO:
    if isinstance(source, (str, Path)):
        return BytesIO(Path(source).read_bytes())
    if hasattr(source, "getvalue"):
        return BytesIO(source.getvalue())
    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        data = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        return BytesIO(data)
    raise ValueError("Unsupported file source.")

def extract_text_from_pdf(source: Any) -> str:
    text_parts: List[str] = []
    buffer = _read_bytes_source(source)
    try:
        with pdfplumber.open(buffer) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise ValueError(f"Could not read PDF text. Ensure it is not scanned/image-only. Details: {exc}") from exc
    return "\n".join(text_parts).strip()

def extract_text_from_docx(source: Any) -> str:
    try:
        document = Document(_read_bytes_source(source))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as exc:
        raise ValueError(f"Could not read DOCX file: {exc}") from exc

def extract_text_from_txt(source: Any) -> str:
    try:
        if isinstance(source, (str, Path)):
            return Path(source).read_text(encoding="utf-8", errors="ignore").strip()
        data = _read_bytes_source(source).getvalue()
        return data.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        raise ValueError(f"Could not read TXT file: {exc}") from exc

def parse_job_description(source: Any) -> Dict[str, Any]:
    suffix = _source_suffix(source)
    if suffix == ".pdf":
        text = extract_text_from_pdf(source)
    elif suffix == ".docx":
        text = extract_text_from_docx(source)
    elif suffix == ".txt":
        text = extract_text_from_txt(source)
    else:
        raise ValueError("Unsupported job description format. Use PDF, DOCX, or TXT.")
    return {
        "filename": _source_name(source),
        "text": clean_extracted_text(text),
        "word_count": len(text.split()),
    }

def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def extract_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text: str) -> str:
    pattern = r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else "Not found"

def extract_name(text: str, filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:10]:
        lowered = line.lower()
        if any(token in lowered for token in ["resume", "curriculum", "email", "phone", "linkedin", "github"]):
            continue
        if "@" in line or re.search(r"\d{3}", line):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'.-]*", line)
        if 2 <= len(words) <= 4 and len(" ".join(words)) <= 50:
            return " ".join(word.capitalize() for word in words)
    fallback = Path(filename).stem.replace("_", " ").replace("-", " ")
    fallback = re.sub(r"sample resume", "", fallback, flags=re.IGNORECASE).strip()
    return fallback.title() or "Unknown Candidate"

def extract_education(text: str) -> List[str]:
    education_lines: List[str] = []
    for line in text.splitlines():
        normalized_line = re.sub(r"\s+", " ", line.strip())
        if not normalized_line or len(normalized_line) > 180:
            continue
        lowered = normalized_line.lower()
        if any(keyword in lowered for keyword in EDUCATION_KEYWORDS):
            if normalized_line not in education_lines:
                education_lines.append(normalized_line)
    return education_lines[:5]

def extract_experience_section(text: str) -> str:
    direct = extract_years_of_experience(text)
    if direct != "Not clearly mentioned":
        return direct
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    capture = False
    captured: List[str] = []
    for line in lines:
        lowered = line.lower().strip(" :-")
        if lowered in {"experience", "work experience", "professional experience", "employment history"}:
            capture = True
            continue
        if capture and lowered in SECTION_STOP_WORDS:
            break
        if capture:
            captured.append(line)
            if len(captured) >= 3:
                break
    if captured:
        return " ".join(captured)[:260]
    return "Not clearly mentioned"

def parse_resume(source: Any) -> Dict[str, Any]:
    if _source_suffix(source) != ".pdf":
        raise ValueError("Only PDF resumes are supported.")
    filename = _source_name(source)
    raw_text = clean_extracted_text(extract_text_from_pdf(source))
    skills = sorted(extract_skills(raw_text))
    education = extract_education(raw_text)
    experience = extract_experience_section(raw_text)
    return {
        "filename": filename,
        "name": extract_name(raw_text, filename),
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "education": education,
        "skills": skills,
        "experience": experience,
        "summary": generate_resume_summary(raw_text, skills, education, experience),
        "raw_text": raw_text,
    }