"""Skill extraction and skill-gap analysis."""
from typing import Iterable, List, Tuple

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "php", "ruby",
    "html", "css", "react", "next.js", "angular", "vue", "node.js", "express", "django", "flask", "streamlit",
    "machine learning", "deep learning", "nlp", "data science", "data analysis", "data visualization",
    "tensorflow", "pytorch", "keras", "scikit-learn", "nltk", "pandas", "numpy", "matplotlib", "plotly",
    "sql", "mysql", "postgresql", "mongodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "ci/cd",
    "object oriented programming", "data structures", "algorithms", "system design",
    "communication", "problem solving", "leadership", "teamwork",
]

SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "nodejs": "node.js", "nextjs": "next.js",
    "reactjs": "react", "postgres": "postgresql", "sklearn": "scikit-learn",
    "natural language processing": "nlp", "google cloud": "gcp", "amazon web services": "aws",
    "oop": "object oriented programming",
}

QUESTION_TEMPLATES = [
    "Can you describe your experience with {skill}?",
    "What projects have you built using {skill}?",
    "How would you rate your proficiency in {skill}, and why?",
    "Describe a challenge you solved using {skill}.",
    "How do you keep your {skill} skills up to date?",
    "Explain a real-world use case where {skill} was important.",
]


def extract_skills(text):
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for alias, canonical in SKILL_ALIASES.items():
        if alias in text_lower:
            found.add(canonical)
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.add(skill)
    return sorted(found)


def compare_skills(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    return matched, missing


def generate_interview_questions(missing_skills, max_questions=6):
    questions = []
    for i, skill in enumerate(list(missing_skills)[:max_questions]):
        questions.append(QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)].format(skill=skill))
    if not questions:
        questions = ["Ask the candidate to explain their strongest project end-to-end."]
    return questions
