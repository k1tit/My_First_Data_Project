from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shared.config import ARTIFACTS_DIR
from shared.risk import apply_business_rules


class ScoringEngine:
    def __init__(self, artifacts_dir: str | Path | None = None):
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self.model = None
        self.calibrator = None
        self.metadata: dict = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        meta_path = self.artifacts_dir / "feature_metadata.json"
        model_path = self.artifacts_dir / "model.joblib"
        cal_path = self.artifacts_dir / "calibrator.joblib"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Артефакты не найдены в {self.artifacts_dir}. "
                "Запустите: python task8/scripts/export_artifacts.py"
            )
        with open(meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.model = joblib.load(model_path)
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None
        self._loaded = True

    def predict(self, payload: dict) -> tuple[float, float, str]:
        self.load()
        row = self._build_row(payload)
        raw = float(self.model.predict_proba(row)[0, 1])
        calibrated = float(self.calibrator.predict([raw])[0]) if self.calibrator else raw
        level = apply_business_rules(payload, raw)
        return raw, calibrated, level

    def _build_row(self, payload: dict) -> pd.DataFrame:
        columns = self.metadata["columns"]
        fill = self.metadata.get("fill_values", {})
        categoricals = set(self.metadata.get("categorical_columns", []))
        category_levels = self.metadata.get("category_levels", {})

        data = {
            col: self._resolve_value(col, payload, fill)
            for col in columns
        }
        df = pd.DataFrame([data], columns=columns)

        for col in categoricals:
            if col not in df.columns:
                continue
            value = df.at[0, col]
            if pd.isna(value):
                value = fill.get(col, "")
            levels = category_levels.get(col)
            if levels:
                df[col] = pd.Categorical([value], categories=levels)
            else:
                df[col] = pd.Categorical([value])

        for col in df.columns:
            if col in categoricals:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[columns]

    @staticmethod
    def _resolve_value(col: str, payload: dict, fill: dict):
        if col in payload and payload[col] is not None:
            return payload[col]
        if col in fill:
            return fill[col]
        return np.nan
