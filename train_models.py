"""Train all NovelMate regression and classification models."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "books.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

YEAR = "original_publication_year"
COUNTS = ["books_count", "ratings_count", "work_ratings_count", "work_text_reviews_count"]
LANG = "language_code"
TARGET = "average_rating"
FEATURES = [YEAR] + COUNTS + [LANG]
LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42

def make_preprocessor():
    count_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ])
    year_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    language_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("counts", count_pipe, COUNTS),
        ("year", year_pipe, [YEAR]),
        ("language", language_pipe, [LANG])
    ])

def make_pipeline(estimator):
    return Pipeline([("preprocessing", make_preprocessor()), ("model", estimator)])

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}. Put books.csv inside data/.")
    books = pd.read_csv(DATA_FILE)
    missing = [c for c in FEATURES + [TARGET] if c not in books.columns]
    if missing:
        raise ValueError(f"books.csv is missing required columns: {missing}")

    data = books[FEATURES + [TARGET]].copy()
    for col in [YEAR] + COUNTS + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].between(0, 5)].copy()
    data[COUNTS] = data[COUNTS].where(data[COUNTS] >= 0)
    data[LANG] = data[LANG].fillna("unknown").astype(str)
    if len(data) < 30:
        raise ValueError("Not enough valid rows to train the models.")

    # Three rating bands based on dataset tertiles.
    q1, q2 = data[TARGET].quantile([1/3, 2/3]).tolist()
    data["rating_group"] = pd.cut(
        data[TARGET], [-np.inf, q1, q2, np.inf],
        labels=LABELS, right=False, duplicates="drop"
    ).astype(str)
    X = data[FEATURES]
    y_reg = data[TARGET]
    y_clf = data["rating_group"]
    X_train, X_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    regression_models = {
        "linear_regression": ("Linear Regression", LinearRegression()),
        "decision_tree": ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE)),
        "random_forest": ("Random Forest Regressor", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)),
        "svr": ("Support Vector Regression", SVR(C=10, epsilon=0.1))
    }
    classification_models = {
        "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=1500)),
        "decision_tree_gini": ("Decision Tree (Gini)", DecisionTreeClassifier(criterion="gini", max_depth=8, random_state=RANDOM_STATE)),
        "decision_tree_entropy": ("Decision Tree (Entropy)", DecisionTreeClassifier(criterion="entropy", max_depth=8, random_state=RANDOM_STATE)),
        "random_forest": ("Random Forest Classifier", RandomForestClassifier(n_estimators=150, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)),
        "svm": ("Support Vector Machine", SVC(C=2, probability=True, random_state=RANDOM_STATE)),
        "naive_bayes": ("Gaussian Naive Bayes", GaussianNB())
    }

    info = {
        "project": "NovelMate", "dataset_rows": int(len(data)),
        "features": FEATURES, "languages": sorted(data[LANG].astype(str).unique().tolist()),
        "class_labels": LABELS,
        "class_ranges": {"Low": f"below {q1:.2f}", "Medium": f"{q1:.2f} to below {q2:.2f}", "High": f"{q2:.2f} and above"},
        "models": {"regression": {}, "classification": {}},
        "metrics": {"regression": [], "classification": []}
    }

    print(f"Cleaned dataset: {len(data):,} books")
    print("Training regression models...")
    for key, (name, estimator) in regression_models.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, yr_train)
        pred = pipe.predict(X_test)
        metrics = {
            "r2": float(r2_score(yr_test, pred)),
            "mae": float(mean_absolute_error(yr_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yr_test, pred)))
        }
        joblib.dump(pipe, MODEL_DIR / f"regression_{key}.joblib", compress=3)
        info["models"]["regression"][key] = name
        info["metrics"]["regression"].append({"algorithm": key, "name": name, **metrics})
        print(f"  {name}: R2={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")

    print("Training classification models...")
    for key, (name, estimator) in classification_models.items():
        pipe = make_pipeline(estimator)
        pipe.fit(Xc_train, yc_train)
        pred = pipe.predict(Xc_test)
        metrics = {
            "accuracy": float(accuracy_score(yc_test, pred)),
            "f1": float(f1_score(yc_test, pred, average="weighted"))
        }
        joblib.dump(pipe, MODEL_DIR / f"classification_{key}.joblib", compress=3)
        info["models"]["classification"][key] = name
        info["metrics"]["classification"].append({"algorithm": key, "name": name, **metrics})
        print(f"  {name}: accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

    data.to_csv(BASE_DIR / "data" / "books_clean.csv", index=False)
    joblib.dump(info, MODEL_DIR / "model_info.joblib", compress=3)
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nTraining complete. Model files are saved in the models/ folder.")
    print("Next step: run the app with `uvicorn app:app --reload`.")

if __name__ == "__main__":
    main()
