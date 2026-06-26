from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data_prep import load_hw7_dataset

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

BEST_PARAMS = {
    "objective": "binary",
    "n_estimators": 752,
    "learning_rate": 0.01871833986018486,
    "num_leaves": 96,
    "max_depth": 14,
    "min_child_samples": 244,
    "subsample": 0.9912205687305236,
    "colsample_bytree": 0.5339101338813105,
    "reg_lambda": 0.9220816073823441,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def split_xy(df: pd.DataFrame):
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    return X, y


def build_metadata(X: pd.DataFrame) -> dict:
    fill: dict = {}
    categoricals = []
    category_levels: dict = {}
    for col in X.columns:
        if str(X[col].dtype) == "category" or X[col].dtype == object:
            categoricals.append(col)
            series = X[col].astype("category")
            category_levels[col] = series.cat.categories.tolist()
            fill[col] = series.mode(dropna=True).iloc[0] if series.notna().any() else ""
        else:
            fill[col] = float(X[col].median()) if X[col].notna().any() else 0.0
    return {
        "columns": list(X.columns),
        "categorical_columns": categoricals,
        "category_levels": category_levels,
        "fill_values": fill,
        "model_version": "hw7-improved-v1",
    }


def main() -> None:
    print("Loading HW7 dataset...")
    _, df = load_hw7_dataset(ROOT / "data")
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    X, y = split_xy(df)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print("Training LightGBM...")
    model = lgb.LGBMClassifier(**BEST_PARAMS)
    model.fit(X_tr, y_tr)

    raw_va = model.predict_proba(X_va)[:, 1]
    auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_va, raw_va)
    print(f"Validation AUC (sample): {auc:.4f}")

    print("Fitting isotonic calibrator...")
    raw_tr = model.predict_proba(X_tr)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_tr, y_tr)

    metadata = build_metadata(X)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")
    joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")
    with open(ARTIFACTS / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved to {ARTIFACTS}: model.joblib, calibrator.joblib, feature_metadata.json")


if __name__ == "__main__":
    main()
