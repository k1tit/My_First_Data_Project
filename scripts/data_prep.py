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
    med = inst.groupby("SK_ID_CURR")["AMT_PAYMENT"].transform("median")
    inst["_large_pay"] = (inst["AMT_PAYMENT"] > 2 * med).astype("int8")
    g = inst.groupby("SK_ID_CURR")
    cv = g["AMT_PAYMENT"].apply(lambda s: float(s.std() / s.mean()) if s.mean() > 0 else 0.0)
    return (
        inst.assign(_underpay=(inst["pay_diff"] < 0).astype("int8"))
        .groupby("SK_ID_CURR", as_index=False)
        .agg(
            inst_n_payments=("AMT_PAYMENT", "count"),
            inst_amt_mean=("AMT_PAYMENT", "mean"),
            inst_amt_max=("AMT_PAYMENT", "max"),
            inst_underpay_share=("_underpay", "mean"),
            inst_large_pay_share=("_large_pay", "mean"),
            inst_pay_diff_abs_mean=("pay_diff", lambda s: float(np.abs(s).mean())),
        )
        .merge(cv.rename("inst_amt_cv"), on="SK_ID_CURR", how="left")
    )


def build_previous_application_features(path: Path | str) -> pd.DataFrame:
    prev = pd.read_csv(
        path,
        usecols=["SK_ID_CURR", "SK_ID_PREV", "AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_STATUS"],
    )
    refused = prev["NAME_CONTRACT_STATUS"].eq("Refused").astype("int8")
    return (
        prev.assign(_refused=refused)
        .groupby("SK_ID_CURR", as_index=False)
        .agg(
            prev_n_apps=("SK_ID_PREV", "count"),
            prev_amt_app_mean=("AMT_APPLICATION", "mean"),
            prev_amt_credit_mean=("AMT_CREDIT", "mean"),
            prev_refused_share=("_refused", "mean"),
        )
    )


def build_credit_card_features(path: Path | str) -> pd.DataFrame:
    cc = pd.read_csv(
        path,
        usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD", "SK_DPD_DEF"],
    )
    return cc.groupby("SK_ID_CURR", as_index=False).agg(
        cc_balance_mean=("AMT_BALANCE", "mean"),
        cc_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        cc_dpd_max=("SK_DPD", "max"),
        cc_dpd_def_max=("SK_DPD_DEF", "max"),
    )


def build_pos_cash_features(path: Path | str) -> pd.DataFrame:
    pos = pd.read_csv(path, usecols=["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT_FUTURE"])
    return pos.groupby("SK_ID_CURR", as_index=False).agg(
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_def_max=("SK_DPD_DEF", "max"),
        pos_inst_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
    )


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ratio_credit_income"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ratio_annuity_income"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(out.columns):
        out["ratio_employed_birth"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"].replace(0, np.nan)
    if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(out.columns):
        ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        out["ext_source_mean"] = ext
    return out


def merge_related_features(app: pd.DataFrame, data_dir: Path | str, extended: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir)
    out = app.copy()
    merges = [
        build_bureau_features(data_dir / "bureau.csv"),
        build_installments_features(data_dir / "installments_payments.csv"),
    ]
    if extended:
        merges.extend(
            [
                build_previous_application_features(data_dir / "previous_application.csv"),
                build_credit_card_features(data_dir / "credit_card_balance.csv"),
                build_pos_cash_features(data_dir / "POS_CASH_balance.csv"),
            ]
        )
    for feat in merges:
        out = out.merge(feat, on="SK_ID_CURR", how="left")
    return add_ratio_features(out)


def feature_pipeline_audit(df_raw: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    related_cols = [c for c in df_features.columns if c not in df_raw.columns and c not in ("TARGET", "SK_ID_CURR")]
    return pd.DataFrame(
        {
            "этап": ["сырой application", "после pipeline"],
            "строк": [len(df_raw), len(df_features)],
            "столбцов": [df_raw.shape[1], df_features.shape[1]],
            "новых признаков": [0, len(related_cols)],
        }
    )


def load_train_datasets(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    train_path = data_dir / "application_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Нет файла {train_path}. Скачайте данные: python download.py")

    app = clean_application(pd.read_csv(train_path))
    app["SK_ID_CURR"] = app["SK_ID_CURR"].astype("int64")

    df_base = app.copy()
    df_full = merge_related_features(app, data_dir, extended=False)
    return df_base, df_full


def load_hw7_dataset(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline (HW5-6) и расширенный датасет для HW7."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    train_path = data_dir / "application_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Нет файла {train_path}. Скачайте данные: python download.py")

    app_raw = pd.read_csv(train_path)
    app = clean_application(app_raw)
    app["SK_ID_CURR"] = app["SK_ID_CURR"].astype("int64")

    df_baseline = merge_related_features(app, data_dir, extended=False)
    df_improved = merge_related_features(app, data_dir, extended=True)
    return df_baseline, df_improved
