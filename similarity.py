"""NLP preprocessing and similarity scoring."""

from __future__ import annotations
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "with", "will", "you", "your", "we", "our", "this", "these", "those", "their", "candidate", "resume", "job", "role", "responsibilities", "requirements",
}

LEMMATIZER = WordNetLemmatizer()

def get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english")) | FALLBACK_STOPWORDS
    except (LookupError, Exception):
        return FALLBACK_STOPWORDS

def tokenize_text(text: str) -> List[str]:
    normalized = text.lower()
    try:
        tokens = nltk.word_tokenize(normalized)
    except (LookupError, Exception):
        tokens = re.findall(r"[a-z][a-z0-9+#.\-]{1,}", normalized)
    return tokens

def lemmatize_token(token: str) -> str:
    try:
        return LEMMATIZER.lemmatize(token)
    except (LookupError, Exception):
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    stop_words = get_stopwords()
    tokens = tokenize_text(text)
    cleaned_tokens: List[str] = []
    for token in tokens:
        token = token.strip("._-/#")
        if len(token) < 2:
            continue
        if token in stop_words:
            continue
        if not re.search(r"[a-z]", token):
            continue
        cleaned_tokens.append(lemmatize_token(token))
    return " ".join(cleaned_tokens)

def calculate_similarity(job_text: str, resume_text: str) -> float:
    processed_job = preprocess_text(job_text)
    processed_resume = preprocess_text(resume_text)
    if not processed_job or not processed_resume:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, min_df=1)
        matrix = vectorizer.fit_transform([processed_job, processed_resume])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(score) * 100, 2)
    except ValueError:
        return 0.0

def rank_resumes(job_text: str, candidates: Iterable[Dict]) -> List[Dict]:
    ranked: List[Dict] = []
    for candidate in candidates:
        score = calculate_similarity(job_text, candidate.get("raw_text", ""))
        ranked.append({"candidate": candidate, "score": score})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked

def keyword_frequency(text: str, top_n: int = 20) -> List[Tuple[str, int]]:
    processed = preprocess_text(text)
    words = [word for word in processed.split() if len(word) > 2 and not word.isdigit()]
    counts = Counter(words)
    return counts.most_common(top_n)
