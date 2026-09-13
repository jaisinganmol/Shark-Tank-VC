# Shark-Tank VC

Shark-Tank VC is an AI pitch room for immigrant technical founders. It helps founders validate their startup idea, sharpen positioning, build an investor-ready narrative, simulate a live investor pitch, and improve through AI feedback.

## Run locally

Backend:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Demo Flow

1. Validate the startup idea and founder-market fit.
2. Sharpen ICP, problem statement, wedge, why now, and business model.
3. Build an investor-ready deck narrative with slide-by-slide notes.
4. Simulate a live pitch with screen share, camera, microphone, transcript, and investor Q&A.
5. Generate the final Investor Readiness Report and improvement roadmap.

## Agentic Layer

The backend uses a LangGraph `StateGraph` orchestration layer:

```text
local_market_rag
  -> validate_agent
  -> sharpen_agent
  -> build_agent
  -> simulate_agent
  -> improve_agent
```

Tools:

- `local_market_rag`: retrieves curated market evidence for the founder's idea.
- `baseten_deepseek_llm`: rewrites investor Q&A answers through Baseten Model APIs using `deepseek-ai/DeepSeek-V3.1`.
- deterministic fallback: keeps the demo reliable when no API key is configured.

## ML Investor Readiness Pipeline

InvestorBridge AI includes a simple ML lifecycle for predicting an Investor Readiness Score.
Synthetic data is used because this is a hackathon prototype and real historical investor outcome data is not available.

Pipeline steps:

- Generate synthetic founder pitch/session data in a notebook.
- Engineer a stable feature vector from validation, delivery, Q&A, and pitch text.
- Train a `GradientBoostingRegressor` to predict investor readiness.
- Evaluate with MAE, RMSE, and R2.
- Save model artifacts under `backend/artifacts/`.
- Load the model in the FastAPI backend.
- Add `ml_prediction` and `final_score` to the final readiness flow.

Training notebook:

Open `backend/ml/investor_readiness_training_pipeline.ipynb` and run all cells.
That notebook is the single end-to-end ML training pipeline for this prototype.
It includes synthetic data generation, EDA visualizations, preprocessing, feature engineering,
model comparison, best-model selection, hyperparameter tuning, final evaluation, artifact saving,
and a backend prediction smoke test.

Generated artifacts:

- `backend/artifacts/synthetic_pitch_data.csv`
- `backend/artifacts/investor_readiness_model.joblib`
- `backend/artifacts/feature_columns.json`
- `backend/artifacts/model_metrics.json`
