import re
from functools import lru_cache

import spacy
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import get_settings


DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b",
    re.IGNORECASE,
)
AMOUNT_REGEX = re.compile(
    r"(?:\$|USD\s?|EUR\s?|INR\s?|Rs\.?\s?|GBP\s?|AED\s?|JPY\s?|\u20b9)\s?\d[\d,]*(?:\.\d{1,2})?"
)
ORG_SUFFIX_REGEX = re.compile(
    r"\b[A-Z][A-Za-z0-9&,.\-\s]{1,80}\s(?:Inc|Ltd|LLC|Corp|Corporation|Company|Pvt\.?\sLtd|University|Bank)\b"
)
MONTH_NAME_REGEX = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)
PERSON_LIKE_REGEX = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}$")

TECH_ENTITY_STOPWORDS = {
    "java",
    "spring",
    "spring boot",
    "core java",
    "hibernate",
    "jdbc",
    "redis",
    "mysql",
    "oauth",
    "maven",
    "framework",
    "frameworks",
    "collections",
    "arraylist",
    "linkedlist",
    "hashset",
    "lambda",
    "lambda expressions",
    "streams",
    "restful",
    "web services",
    "multithreading",
    "jsp",
    "basics",
    "concepts",
    "security",
}

DATE_BANNED_SINGLE_TOKENS = {
    "spring",
    "java",
    "hibernate",
    "redis",
    "jdbc",
    "maven",
}

ORG_BANNED_TERMS = TECH_ENTITY_STOPWORDS

ORG_NOISE_WORDS = {
    "problem",
    "description",
    "features",
    "functionality",
    "technical",
    "implementation",
    "available",
    "note",
    "create",
    "response",
    "request",
    "summary",
    "media",
    "type",
    "schema",
    "example",
    "value",
    "code",
    "details",
    "successful",
}

GENERIC_NON_ORG_ACRONYMS = {
    "API",
    "PDF",
    "DOCX",
    "OCR",
    "JSON",
    "HTTP",
    "URL",
}

# Controlled fallback for tech-heavy documents (course outlines, resumes, etc.)
# when strict ORG filtering returns nothing.
TECH_ORG_FALLBACK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bSpring\s+Boot\b", re.IGNORECASE), "Spring Boot"),
    (re.compile(r"\bMySQL\b", re.IGNORECASE), "MySQL"),
    (re.compile(r"\bRedis\b", re.IGNORECASE), "Redis"),
    (re.compile(r"\bMaven\b", re.IGNORECASE), "Maven"),
    (re.compile(r"\bHibernate\b", re.IGNORECASE), "Hibernate"),
    (re.compile(r"\bJDBC\b", re.IGNORECASE), "JDBC"),
    (re.compile(r"\bJSP\b", re.IGNORECASE), "JSP"),
    (re.compile(r"\bOAuth\s*2(?:\.0)?\b", re.IGNORECASE), "OAuth 2.0"),
    (re.compile(r"\bJava\b", re.IGNORECASE), "Java"),
]


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def get_summarizer():
    settings = get_settings()
    if not settings.enable_transformers_summarization:
        return None
    try:
        return pipeline("summarization", model=settings.summarization_model)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_ner_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def summarize_text(text: str) -> str:
    text = _normalize_text(text).strip()
    if not text:
        return "No text could be extracted from the document."

    summarizer = get_summarizer()
    settings = get_settings()

    if summarizer and len(text.split()) > 40:
        try:
            chunks = _chunk_text_for_summary(text, max_words=450)
            partial_summaries: list[str] = []
            for chunk in chunks:
                result = summarizer(
                    chunk,
                    max_length=settings.max_summary_tokens,
                    min_length=30,
                    do_sample=False,
                )
                partial_summaries.append(result[0]["summary_text"].strip())
            return " ".join(partial_summaries).strip()
        except Exception:
            pass

    return _simple_extractive_summary(text)


def extract_entities(text: str) -> dict[str, list[str]]:
    text = _normalize_text(text or "")

    names: set[str] = set()
    dates: set[str] = set(DATE_REGEX.findall(text))
    organizations: set[str] = set(ORG_SUFFIX_REGEX.findall(text))
    amounts: set[str] = set(AMOUNT_REGEX.findall(text))

    ner_model = get_ner_model()
    if ner_model:
        try:
            doc = ner_model(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    names.add(ent.text.strip())
                elif ent.label_ in {"ORG"}:
                    organizations.add(ent.text.strip())
                elif ent.label_ in {"DATE"}:
                    dates.add(ent.text.strip())
                elif ent.label_ in {"MONEY"}:
                    amounts.add(ent.text.strip())
        except Exception:
            pass

    if not names:
        for candidate in re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", text):
            names.add(candidate)

    filtered_names = _filter_names(names)
    filtered_dates = _filter_dates(dates)
    filtered_orgs = _filter_organizations(organizations)
    filtered_amounts = _filter_amounts(amounts)

    if not filtered_orgs:
        filtered_orgs = _fallback_tech_organizations(text)

    return {
        "names": sorted(filtered_names),
        "dates": sorted(filtered_dates),
        "organizations": sorted(filtered_orgs),
        "amounts": sorted(filtered_amounts),
    }


def analyze_sentiment(text: str) -> str:
    analyzer = get_sentiment_analyzer()
    score = analyzer.polarity_scores(text or "")
    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def _chunk_text_for_summary(text: str, max_words: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _simple_extractive_summary(text: str) -> str:
    text = _normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:500]
    summary = " ".join(sentences[:3]).strip()
    return summary[:700] if len(summary) > 700 else summary


def _normalize_text(text: str) -> str:
    # Normalize OCR bullets and noisy separators before NLP.
    normalized = text.replace("\x00", " ")
    normalized = normalized.replace("\u00ab", " ").replace("\u00bb", " ")
    normalized = normalized.replace("•", " ").replace("*", " ")
    normalized = normalized.replace("+", " ").replace("\u201c", " ").replace("\u201d", " ")
    normalized = re.sub(r"[\t\r\f\v]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_stopword_phrase(value: str, stopwords: set[str]) -> bool:
    normalized = value.strip().lower()
    return normalized in stopwords


def _filter_names(names: set[str]) -> set[str]:
    filtered: set[str] = set()
    for name in names:
        candidate = name.strip()
        if not candidate:
            continue
        if _is_stopword_phrase(candidate, TECH_ENTITY_STOPWORDS):
            continue
        if PERSON_LIKE_REGEX.match(candidate):
            filtered.add(candidate)
    return filtered


def _filter_dates(dates: set[str]) -> set[str]:
    filtered: set[str] = set()
    for date in dates:
        candidate = date.strip()
        if not candidate:
            continue

        lower = candidate.lower()
        if lower in DATE_BANNED_SINGLE_TOKENS:
            continue

        has_numeric_date = bool(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", candidate))
        has_day_month_year = bool(
            re.search(
                r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}\b",
                candidate,
                re.IGNORECASE,
            )
        )
        has_month_year = bool(
            re.search(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b",
                candidate,
                re.IGNORECASE,
            )
        )
        is_year_only = bool(re.fullmatch(r"\d{4}", candidate))

        if has_numeric_date or has_day_month_year or has_month_year or is_year_only:
            filtered.add(candidate)
            continue

        # Allow month names only when part of longer valid date fragments.
        if MONTH_NAME_REGEX.search(candidate) and len(candidate.split()) >= 2:
            filtered.add(candidate)

    return filtered


def _filter_organizations(organizations: set[str]) -> set[str]:
    filtered: set[str] = set()
    for org in organizations:
        candidate = org.strip()
        if not candidate:
            continue
        if _is_stopword_phrase(candidate, ORG_BANNED_TERMS):
            continue

        lower = candidate.lower()
        if any(term in lower for term in ORG_BANNED_TERMS):
            continue

        if _is_noisy_organization_candidate(candidate):
            continue

        if len(candidate) < 3:
            continue
        filtered.add(candidate)

    return filtered


def _filter_amounts(amounts: set[str]) -> set[str]:
    return {value.strip() for value in amounts if value and value.strip()}


def _fallback_tech_organizations(text: str) -> set[str]:
    found: set[str] = set()
    for pattern, canonical in TECH_ORG_FALLBACK_PATTERNS:
        if pattern.search(text):
            found.add(canonical)
    return found


def _is_noisy_organization_candidate(candidate: str) -> bool:
    if re.search(r"\d{2,}/\d{2,}", candidate):
        return True
    if re.search(r"[!?]", candidate):
        return True

    tokens = re.findall(r"[A-Za-z0-9.&'-]+", candidate)
    if not tokens:
        return True
    if len(tokens) > 6:
        return True

    lower_tokens = [token.lower() for token in tokens]
    if any(token in ORG_NOISE_WORDS for token in lower_tokens):
        return True

    # Remove generic all-caps technical abbreviations that are not organizations.
    if candidate.isupper() and candidate in GENERIC_NON_ORG_ACRONYMS:
        return True

    # Keep short single-token candidates only for proper nouns/acronyms.
    if len(tokens) == 1:
        token = tokens[0]
        if not (re.fullmatch(r"[A-Z][a-zA-Z0-9]+", token) or re.fullmatch(r"[A-Z]{2,8}", token)):
            return True

    return False
