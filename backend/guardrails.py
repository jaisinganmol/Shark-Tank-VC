from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError


MAX_SHORT_TEXT = 2_000
MAX_LONG_TEXT = 20_000
MAX_TRANSCRIPT = 40_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_QNA_ITEMS = 40
MAX_METRICS_KEYS = 32

_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (all|any|the|previous|prior|above)[^.\n]{0,40}(instruction|prompt|rule|message)s?"),
    re.compile(r"(?i)disregard [^.\n]{0,40}(instruction|prompt|rule|system)"),
    re.compile(r"(?i)forget (everything|all|previous|prior)[^.\n]{0,40}"),
    re.compile(r"(?i)you are now [^.\n]{0,60}"),
    re.compile(r"(?i)act as (?:an? )?(?:different|new)[^.\n]{0,60}"),
    re.compile(r"(?i)system\s*(?:prompt|message)\s*[:=]"),
    re.compile(r"(?i)</?\s*(?:system|assistant|user)\s*>"),
    re.compile(r"(?i)\bBEGIN (?:SYSTEM|ASSISTANT) MESSAGE\b"),
]

_ON_TOPIC_KEYWORDS = (
    "startup", "founder", "product", "customer", "market", "pitch", "investor",
    "revenue", "pricing", "user", "problem", "solution", "team", "mvp", "traction",
    "growth", "sales", "b2b", "b2c", "saas", "ai", "app", "platform", "tool",
    "service", "business", "company", "idea", "build", "launch", "brand",
)


def sanitize_text(value: str, max_length: int = MAX_LONG_TEXT) -> str:
    """Trim, cap, and strip common prompt-injection phrases from free-form user input."""
    if not value:
        return ""
    text = value.replace("\x00", "").strip()
    if len(text) > max_length:
        text = text[:max_length]
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[filtered]", text)
    return text


def sanitize_session(session: dict[str, Any]) -> dict[str, Any]:
    """Sanitize every string field in a session payload and cap list/dict sizes."""
    cleaned: dict[str, Any] = {}
    limits = {
        "startup_idea": MAX_SHORT_TEXT,
        "target_customer": MAX_SHORT_TEXT,
        "problem": MAX_LONG_TEXT,
        "current_solution": MAX_LONG_TEXT,
        "founder_background": MAX_SHORT_TEXT,
        "stage": 64,
        "transcript": MAX_TRANSCRIPT,
        "startup_brief": MAX_LONG_TEXT,
        "deck_text": MAX_LONG_TEXT,
        "context": MAX_LONG_TEXT,
    }
    for key, value in session.items():
        if isinstance(value, str):
            cleaned[key] = sanitize_text(value, limits.get(key, MAX_LONG_TEXT))
        elif isinstance(value, list):
            cleaned[key] = value[:MAX_QNA_ITEMS] if key == "qna" else value
        elif isinstance(value, dict):
            if key == "metrics":
                cleaned[key] = dict(list(value.items())[:MAX_METRICS_KEYS])
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value
    if "qna" in cleaned and isinstance(cleaned["qna"], list):
        trimmed_qna = []
        for item in cleaned["qna"]:
            if not isinstance(item, dict):
                continue
            trimmed_qna.append(
                {
                    "question": sanitize_text(str(item.get("question", "")), MAX_SHORT_TEXT),
                    "answer": sanitize_text(str(item.get("answer", "")), MAX_LONG_TEXT),
                }
            )
        cleaned["qna"] = trimmed_qna
    return cleaned


def is_on_topic(text: str) -> bool:
    """Loose check that content is startup/pitch related. Empty input passes (handled elsewhere)."""
    if not text or len(text) < 20:
        return True
    lower = text.lower()
    return any(word in lower for word in _ON_TOPIC_KEYWORDS)


def truncate_words(text: str, max_words: int) -> str:
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",.;:") + "…"


class PositioningOutput(BaseModel):
    one_liner: str | None = None
    problem_statement: str | None = None
    value_proposition: str | None = None
    business_model: str | None = None
    why_now: str | None = None
    competitive_positioning: str | None = None


def validate_positioning(raw: str) -> dict[str, str]:
    """Parse an LLM positioning response and return only valid string fields. Empty dict on failure."""
    import json

    if not raw:
        return {}
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    try:
        model = PositioningOutput(**parsed)
    except ValidationError:
        return {}
    return {k: sanitize_text(v, 400) for k, v in model.model_dump().items() if isinstance(v, str) and v.strip()}
