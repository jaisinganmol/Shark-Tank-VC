from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import joblib
    import pandas as pd
except Exception:  # pragma: no cover - backend should still boot without ML deps
    joblib = None
    pd = None

from .feature_engineering import build_feature_vector, get_feature_columns


BACKEND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "investor_readiness_model.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


class ReadinessModelService:
    def __init__(
        self,
        model_path: Path | None = None,
        feature_columns_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or MODEL_PATH
        self.feature_columns_path = feature_columns_path or FEATURE_COLUMNS_PATH
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.load_error: str | None = None

    def predict_from_state(self, session_state: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self._ensure_loaded():
                return self.fallback_prediction(session_state)
            features = build_feature_vector(session_state)
            frame = pd.DataFrame([[features[column] for column in self.feature_columns]], columns=self.feature_columns)
            raw_score = float(self.model.predict(frame)[0])
            score = int(round(min(100, max(0, raw_score))))
            return self._prediction_payload(score, "ml_model", self._top_signals(features))
        except Exception as exc:
            self.load_error = str(exc)
            return self.fallback_prediction(session_state)

    def fallback_prediction(self, session_state: dict[str, Any]) -> dict[str, Any]:
        features = build_feature_vector(session_state)
        score = (
            features["validation_score"] * 0.36
            + features["clarity"] * 0.18
            + features["confidence"] * 0.16
            + features["qna_quality_score"] * 0.12
            + features["problem_clarity_score"] * 0.08
            + features["market_need_score"] * 0.06
            + features["founder_market_fit_score"] * 0.04
        )
        score += features["has_business_model"] * 2.5
        score += features["has_revenue_model"] * 2.5
        score += features["has_go_to_market"] * 2
        score -= min(features["filler_words"] * 0.35, 10)
        score -= max(abs(features["speaking_speed"] - 140) - 28, 0) * 0.18
        score -= max(150 - features["pitch_length_words"], 0) * 0.06
        score = int(round(min(100, max(0, score))))
        return self._prediction_payload(score, "fallback_heuristic", self._top_signals(features))

    def _ensure_loaded(self) -> bool:
        if self.model is not None and self.feature_columns:
            return True
        if joblib is None or pd is None:
            self.load_error = "ML dependencies are unavailable."
            return False
        if not self.model_path.exists() or not self.feature_columns_path.exists():
            self.load_error = "Model artifacts are missing."
            return False

        self.model = joblib.load(self.model_path)
        self.feature_columns = json.loads(self.feature_columns_path.read_text(encoding="utf-8"))
        expected = get_feature_columns()
        if self.feature_columns != expected:
            self.load_error = "Feature column artifact does not match current feature engineering."
            return False
        self.load_error = None
        return True

    def _prediction_payload(self, score: int, score_source: str, top_signals: list[str]) -> dict[str, Any]:
        return {
            "readiness_score": score,
            "readiness_probability": round(score / 100, 2),
            "risk_level": self._risk_level(score),
            "model_version": "synthetic_v1",
            "score_source": score_source,
            "top_signals": top_signals,
        }

    def _top_signals(self, features: dict[str, float]) -> list[str]:
        signals: list[str] = []
        if features["founder_market_fit_score"] >= 80:
            signals.append("Strong founder-market fit")
        if features["problem_clarity_score"] >= 80:
            signals.append("Clear problem statement")
        if features["confidence"] >= 78 and features["clarity"] >= 78:
            signals.append("Strong pitch delivery")
        if features["validation_score"] >= 80:
            signals.append("Strong validation signal")
        if features["has_business_model"] and features["has_revenue_model"]:
            signals.append("Business and revenue model are present")
        if features["market_need_score"] >= 80:
            signals.append("Strong market need")
        if not signals:
            signals = [
                "Validation score is usable but needs proof",
                "Pitch delivery is directionally clear",
                "Business model needs more evidence",
            ]
        return signals[:3]

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 80:
            return "Low"
        if score >= 65:
            return "Medium"
        return "High"


def fallback_prediction(session_state: dict[str, Any]) -> dict[str, Any]:
    return ReadinessModelService().fallback_prediction(session_state)

