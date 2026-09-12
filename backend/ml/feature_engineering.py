from __future__ import annotations

import re
from typing import Any


DEFAULT_FEATURES: dict[str, float] = {
    "problem_clarity_score": 75,
    "market_need_score": 75,
    "founder_market_fit_score": 75,
    "ai_fit_score": 75,
    "business_potential_score": 75,
    "validation_score": 75,
    "confidence": 75,
    "eye_contact": 70,
    "speaking_speed": 140,
    "filler_words": 8,
    "clarity": 75,
    "energy": 75,
    "qna_quality_score": 75,
    "pitch_length_words": 180,
    "has_clear_customer": 1,
    "has_business_model": 0,
    "has_competition_awareness": 0,
    "has_go_to_market": 0,
    "has_revenue_model": 0,
}


SCORE_LABEL_TO_FEATURE = {
    "problem clarity": "problem_clarity_score",
    "market need": "market_need_score",
    "founder-market fit": "founder_market_fit_score",
    "founder market fit": "founder_market_fit_score",
    "ai fit": "ai_fit_score",
    "business potential": "business_potential_score",
}


KEYWORDS = {
    "has_business_model": [
        "subscription",
        "pricing",
        "revenue",
        "business model",
        "customers",
        "b2b",
        "saas",
    ],
    "has_go_to_market": [
        "sales",
        "marketing",
        "partnership",
        "distribution",
        "go-to-market",
        "gtm",
    ],
    "has_revenue_model": [
        "revenue",
        "paid",
        "subscription",
        "pricing",
        "monetization",
    ],
    "has_competition_awareness": [
        "competitor",
        "competition",
        "alternative",
        "market",
        "differentiation",
    ],
    "has_clear_customer": [
        "customer",
        "users",
        "founders",
        "students",
        "businesses",
        "operators",
        "teams",
    ],
}


def get_feature_columns() -> list[str]:
    """Return the exact ordered feature list used by training and inference."""
    return [
        "problem_clarity_score",
        "market_need_score",
        "founder_market_fit_score",
        "ai_fit_score",
        "business_potential_score",
        "validation_score",
        "confidence",
        "eye_contact",
        "speaking_speed",
        "filler_words",
        "clarity",
        "energy",
        "qna_quality_score",
        "pitch_length_words",
        "has_clear_customer",
        "has_business_model",
        "has_competition_awareness",
        "has_go_to_market",
        "has_revenue_model",
    ]


def build_feature_vector(session_state: dict[str, Any]) -> dict[str, float]:
    """Build a robust feature vector from the current backend session or report state."""
    state = session_state or {}
    session = _dict_value(state.get("session"))
    metrics = {
        **_dict_value(session.get("metrics")),
        **_dict_value(state.get("metrics")),
    }
    validation = _first_dict(state, "validate", "validation", "deck")
    simulation = _dict_value(state.get("simulate"))
    delivery = {
        **_dict_value(simulation.get("delivery")),
        **_dict_value(state.get("delivery")),
        **metrics,
    }

    features = dict(DEFAULT_FEATURES)
    _apply_validation_features(features, validation)
    _apply_delivery_features(features, delivery)

    qna = state.get("qna") or session.get("qna") or []
    transcript = str(state.get("transcript") or session.get("transcript") or "")
    all_text = _collect_text(state)

    if transcript.strip():
        features["pitch_length_words"] = _word_count(transcript)
    elif all_text.strip():
        features["pitch_length_words"] = _word_count(all_text)

    if qna:
        features["qna_quality_score"] = _score_qna(qna)

    for feature_name, keywords in KEYWORDS.items():
        if _contains_keyword(all_text, keywords):
            features[feature_name] = 1

    return {
        column: _clip_feature(column, features.get(column, DEFAULT_FEATURES[column]))
        for column in get_feature_columns()
    }


def _apply_validation_features(features: dict[str, float], validation: dict[str, Any]) -> None:
    score = _number(validation.get("score"))
    if score is not None:
        features["validation_score"] = score

    for item in validation.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        feature_name = SCORE_LABEL_TO_FEATURE.get(label)
        value = _number(item.get("score"))
        if feature_name and value is not None:
            features[feature_name] = value * 10 if value <= 10 else value


def _apply_delivery_features(features: dict[str, float], delivery: dict[str, Any]) -> None:
    mapping = {
        "confidence": ["confidence"],
        "eye_contact": ["eye_contact", "eyeContact"],
        "speaking_speed": ["speaking_speed", "words_per_minute", "wpm", "pace"],
        "filler_words": ["filler_words", "fillerWords"],
        "clarity": ["clarity"],
        "energy": ["energy"],
    }
    for feature_name, keys in mapping.items():
        value = _first_number(delivery, keys)
        if value is not None:
            features[feature_name] = value


def _score_qna(qna: Any) -> float:
    if not isinstance(qna, list) or not qna:
        return DEFAULT_FEATURES["qna_quality_score"]

    answers = []
    for item in qna:
        if isinstance(item, dict):
            answers.append(str(item.get("answer", "")))
        else:
            answers.append(str(item))
    avg_words = sum(_word_count(answer) for answer in answers) / max(len(answers), 1)
    keyword_bonus = sum(
        1
        for answer in answers
        if _contains_keyword(answer, ["validated", "customers", "revenue", "market", "pilot", "pricing", "growth"])
    )
    return min(96, 58 + min(22, len(answers) * 5) + min(12, avg_words / 4) + min(8, keyword_bonus * 2))


def _collect_text(value: Any) -> str:
    pieces: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            pieces.append(item)
        elif isinstance(item, dict):
            for key, nested in item.items():
                if key in {"market_evidence", "meta"}:
                    continue
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return " ".join(pieces)


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])", lower) for keyword in keywords)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def _first_dict(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_number(values: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        number = _number(values.get(key))
        if number is not None:
            return number
    return None


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_feature(column: str, value: Any) -> float:
    number = _number(value)
    if number is None:
        number = DEFAULT_FEATURES[column]

    if column in {
        "has_clear_customer",
        "has_business_model",
        "has_competition_awareness",
        "has_go_to_market",
        "has_revenue_model",
    }:
        return 1 if number >= 0.5 else 0
    if column == "speaking_speed":
        return min(220, max(60, number))
    if column == "filler_words":
        return min(80, max(0, number))
    if column == "pitch_length_words":
        return min(1000, max(0, number))
    return min(100, max(0, number))

