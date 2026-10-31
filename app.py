"""FastAPI backend for the NovelMate web application."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
INFO_FILE = MODEL_DIR / "model_info.joblib"

app = FastAPI(title="NovelMate API", version="1.0.0",
              description="Predict and compare novel ratings using trained machine-learning models.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class BookInput(BaseModel):
    original_publication_year: float = Field(..., ge=-4000, le=2100)
    books_count: float = Field(..., ge=0)
    ratings_count: float = Field(..., ge=0)
    work_ratings_count: float = Field(..., ge=0)
    work_text_reviews_count: float = Field(..., ge=0)
    language_code: str = Field(default="unknown", max_length=30)

class PredictionInput(BookInput):
    algorithm: str

def load_registry():
    if not INFO_FILE.exists():
        raise HTTPException(status_code=503, detail="Models are not trained yet. Run NovelMate_Colab.ipynb first.")
    try:
        return joblib.load(INFO_FILE)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read model registry: {exc}")

def get_model(task, key, info):
    if key not in info["models"].get(task, {}):
        raise HTTPException(status_code=404, detail=f"Unknown model '{key}' for {task}.")
    path = MODEL_DIR / f"{task}_{key}.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model file is missing: {path.name}. Retrain the models.")
    return joblib.load(path)

def input_frame(book, info):
    values = book.model_dump()
    return pd.DataFrame([{feature: values[feature] for feature in info["features"]}])

def predict_one(task, key, row, info):
    model = get_model(task, key, info)
    if task == "regression":
        value = float(model.predict(row)[0])
        return {"algorithm": key, "name": info["models"][task][key],
                "predicted_average_rating": round(min(5.0, max(0.0, value)), 2)}
    label = str(model.predict(row)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = {str(label): round(float(prob), 4)
                         for label, prob in zip(model.classes_, model.predict_proba(row)[0])}
    return {"algorithm": key, "name": info["models"][task][key],
            "category": label, "range": info["class_ranges"].get(label, ""),
            "probabilities": probabilities}

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "service": "NovelMate API"}

@app.get("/api/info")
def api_info():
    return load_registry()

@app.post("/api/predict/regression")
def predict_regression(request: PredictionInput):
    info = load_registry()
    return predict_one("regression", request.algorithm, input_frame(request, info), info)

@app.post("/api/predict/classification")
def predict_classification(request: PredictionInput):
    info = load_registry()
    return predict_one("classification", request.algorithm, input_frame(request, info), info)

@app.post("/api/compare/regression")
def compare_regression(request: BookInput):
    info = load_registry()
    row = input_frame(request, info)
    return {"results": [predict_one("regression", key, row, info)
                        for key in info["models"]["regression"]]}

@app.post("/api/compare/classification")
def compare_classification(request: BookInput):
    info = load_registry()
    row = input_frame(request, info)
    return {"results": [predict_one("classification", key, row, info)
                        for key in info["models"]["classification"]]}
