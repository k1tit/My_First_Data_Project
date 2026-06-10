"""Загрузка и очистка Home Credit для ноутбуков."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "application_train.csv"


def clean_application(
    df: pd.DataFrame,
    housing_miss_threshold: float = 0.65,
    days_employed_anomaly: int = 365243,
) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=["SK_ID_CURR"], keep="first")
    out.loc[out["DAYS_EMPLOYED"] == days_employed_anomaly, "DAYS_EMPLOYED"] = np.nan

    miss_rate = out.isna().mean()
    drop_cols = miss_rate[miss_rate >= housing_miss_threshold].index
    drop_cols = [c for c in drop_cols if c not in ("TARGET", "SK_ID_CURR")]
    out = out.drop(columns=drop_cols)

    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if col in out.columns:
            out[f"{col}_MISSING"] = out[col].isna().astype("int8")

    return out


def build_bureau_features(path: Path | str) -> pd.DataFrame:
    bureau = pd.read_csv(
        path,
        usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM"],
    )
    active = bureau["CREDIT_ACTIVE"].eq("Active")
    return (
        bureau.assign(_active=active.astype("int8"))
        .groupby("SK_ID_CURR", as_index=False)
        .agg(
            bureau_n_credits=("SK_ID_BUREAU", "count"),
            bureau_active_n=("_active", "sum"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_amt_sum_mean=("AMT_CREDIT_SUM", "mean"),
        )
    )


def build_installments_features(path: Path | str) -> pd.DataFrame:
    inst = pd.read_csv(path, usecols=["SK_ID_CURR", "AMT_INSTALMENT", "AMT_PAYMENT"])
    inst["pay_diff"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    return (
        inst.assign(_underpay=(inst["pay_diff"] < 0).astype("int8"))
        .groupby("SK_ID_CURR", as_index=False)
        .agg(
            inst_n_payments=("AMT_PAYMENT", "count"),
            inst_amt_mean=("AMT_PAYMENT", "mean"),
            inst_amt_max=("AMT_PAYMENT", "max"),
            inst_underpay_share=("_underpay", "mean"),
        )
    )


def load_train_datasets(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    train_path = data_dir / "application_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Нет файла {train_path}. Скачайте данные: python download.py")

    app = clean_application(pd.read_csv(train_path))
    app["SK_ID_CURR"] = app["SK_ID_CURR"].astype("int64")

    df_base = app.copy()
    df_full = app.merge(
        build_bureau_features(data_dir / "bureau.csv"),
        on="SK_ID_CURR",
        how="left",
    )
    df_full = df_full.merge(
        build_installments_features(data_dir / "installments_payments.csv"),
        on="SK_ID_CURR",
        how="left",
    )
    return df_base, df_full
