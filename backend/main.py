from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agents import PitchBridgeGraph
from .guardrails import (
    MAX_LONG_TEXT,
    MAX_QNA_ITEMS,
    MAX_SHORT_TEXT,
    MAX_TRANSCRIPT,
    MAX_UPLOAD_BYTES,
    is_on_topic,
    sanitize_session,
    sanitize_text,
)


app = FastAPI(title="InvestorBridge AI", version="0.2.0")
graph = PitchBridgeGraph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    stage: str = Field(default="problem", max_length=64)
    context: str = Field(default="", max_length=MAX_LONG_TEXT)


class FinalReportRequest(BaseModel):
    deck_text: str = Field(default="", max_length=MAX_LONG_TEXT)
    startup_brief: str = Field(default="", max_length=MAX_LONG_TEXT)
    transcript: str = Field(default="", max_length=MAX_TRANSCRIPT)
    metrics: dict[str, Any] = Field(default_factory=dict)
    qna: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_QNA_ITEMS)


class SessionRunRequest(BaseModel):
    startup_idea: str = Field(default="", max_length=MAX_SHORT_TEXT)
    target_customer: str = Field(default="", max_length=MAX_SHORT_TEXT)
    problem: str = Field(default="", max_length=MAX_LONG_TEXT)
    current_solution: str = Field(default="", max_length=MAX_LONG_TEXT)
    founder_background: str = Field(default="", max_length=MAX_SHORT_TEXT)
    stage: str = Field(default="idea", max_length=64)
    transcript: str = Field(default="", max_length=MAX_TRANSCRIPT)
    metrics: dict[str, Any] = Field(default_factory=dict)
    qna: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_QNA_ITEMS)


class MLPredictRequest(BaseModel):
    session_state: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "InvestorBridge AI"}


@app.post("/api/session/run")
def run_session(request: SessionRunRequest) -> dict[str, Any]:
    return graph.run_session(sanitize_session(request.model_dump()))

@app.post("/api/analyze-deck")
async def analyze_deck(
    startup_brief: str = Form(""),
    deck_text: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    uploaded_text = ""
    file_meta: dict[str, Any] | None = None
    if file:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes")
        file_meta = {"filename": file.filename, "content_type": file.content_type, "bytes": len(raw)}
        if (file.content_type or "").startswith("text/") or (file.filename or "").lower().endswith((".txt", ".md")):
            uploaded_text = raw.decode("utf-8", errors="ignore")

    startup_brief = sanitize_text(startup_brief, MAX_LONG_TEXT)
    deck_text = sanitize_text(deck_text, MAX_LONG_TEXT)
    uploaded_text = sanitize_text(uploaded_text, MAX_LONG_TEXT)

    text = "\n".join(part for part in [startup_brief, deck_text, uploaded_text] if part.strip())
    if not text.strip():
        text = "InvestorBridge AI helps immigrant technical founders become investor-ready through live pitch simulation, deck analysis, delivery coaching, and investor Q&A."
    elif not is_on_topic(text):
        raise HTTPException(status_code=422, detail="input does not appear to be startup or pitch content")

    result = graph.analyze_deck(text)
    result["file"] = file_meta
    return result


@app.post("/api/investor-question")
def investor_question(request: QuestionRequest) -> dict[str, str]:
    return graph.investor_question(
        sanitize_text(request.stage, 64),
        sanitize_text(request.context, MAX_LONG_TEXT),
    )


@app.post("/api/ml/predict-readiness")
def predict_readiness(request: MLPredictRequest) -> dict[str, Any]:
    return {"ml_prediction": graph.predict_readiness(request.session_state)}


@app.post("/api/final-report")
def final_report(request: FinalReportRequest) -> dict[str, Any]:
    return graph.final_report(sanitize_session(request.model_dump()))
