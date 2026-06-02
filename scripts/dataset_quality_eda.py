"""
HW4 Data Understanding: EDA + визуализации Home Credit Default Risk.
Графики сохраняются в reports/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TRAIN_PATH = DATA_DIR / "application_train.csv"

OPTIONAL_TABLES = {
    "bureau": DATA_DIR / "bureau.csv",
    "installments_payments": DATA_DIR / "installments_payments.csv",
    "credit_card_balance": DATA_DIR / "credit_card_balance.csv",
}


def configure_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8")
                except (OSError, ValueError):
                    pass


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_train() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    return df


def save_fig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  сохранено: {path.relative_to(ROOT)}")


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["TARGET"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["0 — без дефолта", "1 — дефолт"], counts.values, color=["#4c78a8", "#e45756"])
    ax.set_title("Распределение TARGET")
    ax.set_ylabel("Количество заявок")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}\n({pct(val / len(df))})",
                ha="center", va="bottom", fontsize=9)
    save_fig("01_target_distribution.png")


def plot_missing_top(df: pd.DataFrame, top_n: int = 15) -> None:
    miss = df.isna().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=miss.values * 100, y=miss.index, ax=ax, color="#72b7b2")
    ax.set_xlabel("Доля пропусков, %")
    ax.set_title(f"Топ-{top_n} признаков по доле пропусков")
    save_fig("02_missing_top15.png")


def plot_corr_with_target(df: pd.DataFrame, top_n: int = 12) -> None:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)["TARGET"].drop("TARGET", errors="ignore")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e45756" if v < 0 else "#4c78a8" for v in corr.values]
    sns.barplot(x=corr.values, y=corr.index, ax=ax, palette=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Корреляция с TARGET")
    ax.set_title("Топ признаков по корреляции с TARGET")
    save_fig("03_corr_with_target.png")


def plot_ext_source_by_target(df: pd.DataFrame) -> None:
    cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if not cols:
        return
    melted = df[cols + ["TARGET"]].melt(id_vars="TARGET", var_name="feature", value_name="value")
    melted["TARGET"] = melted["TARGET"].map({0: "без дефолта", 1: "дефолт"})
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=melted, x="feature", y="value", hue="TARGET", ax=ax)
    ax.set_title("EXT_SOURCE по классам TARGET")
    ax.set_ylabel("значение")
    save_fig("04_ext_source_by_target.png")


def plot_income_by_target(df: pd.DataFrame) -> None:
    if "AMT_INCOME_TOTAL" not in df.columns:
        return
    sub = df[["AMT_INCOME_TOTAL", "TARGET"]].copy()
    sub = sub[sub["AMT_INCOME_TOTAL"] < sub["AMT_INCOME_TOTAL"].quantile(0.99)]
    sub["TARGET"] = sub["TARGET"].map({0: "без дефолта", 1: "дефолт"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=sub, x="AMT_INCOME_TOTAL", hue="TARGET", bins=40, kde=True, ax=ax)
    ax.set_title("AMT_INCOME_TOTAL (без топ-1% выбросов)")
    save_fig("05_income_by_target.png")


def plot_days_employed_anomaly(df: pd.DataFrame) -> None:
    if "DAYS_EMPLOYED" not in df.columns:
        return
    is_anom = df["DAYS_EMPLOYED"] == 365243
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["обычные значения", "365243 (код пропуска)"], [(~is_anom).sum(), is_anom.sum()],
           color=["#4c78a8", "#f58518"])
    ax.set_title("Аномалия DAYS_EMPLOYED")
    ax.set_ylabel("число строк")
    save_fig("06_days_employed_anomaly.png")


def plot_table_coverage() -> None:
    if not TRAIN_PATH.exists():
        return
    train_ids = set(pd.read_csv(TRAIN_PATH, usecols=["SK_ID_CURR"])["SK_ID_CURR"])
    n_train = len(train_ids)
    rows = []
    for name, path in OPTIONAL_TABLES.items():
        if not path.exists():
            rows.append((name, 0.0))
            continue
        seen: set[int] = set()
        for chunk in pd.read_csv(path, usecols=["SK_ID_CURR"], chunksize=200_000):
            seen.update(chunk["SK_ID_CURR"].dropna().astype(int).tolist())
        rows.append((name, 100.0 * len(train_ids & seen) / n_train))
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=list(values), y=list(labels), ax=ax, color="#54a24b")
    ax.set_xlabel("Покрытие train, %")
    ax.set_title("Пересечение связанных таблиц с application_train")
    save_fig("07_related_tables_coverage.png")


def print_text_eda(df: pd.DataFrame) -> dict:
    section("1. Источник и состав")
    print("Источник: Kaggle Home Credit Default Risk")
    print("TARGET: 1 — дефолт, 0 — без дефолта")
    for name, path in {"application_train": TRAIN_PATH, **OPTIONAL_TABLES}.items():
        print(f"  {name}: {'OK' if path.exists() else 'нет'}")

    section("2. Базовое качество")
    n_rows, n_cols = df.shape
    n_dup = int(df["SK_ID_CURR"].duplicated().sum())
    print(f"Размер: {n_rows:,} x {n_cols}")
    print(f"Дубликаты SK_ID_CURR: {n_dup}")

    top_miss = df.isna().mean().sort_values(ascending=False).head(10)
    print("Топ пропусков:")
    for col, rate in top_miss.items():
        print(f"  {col}: {pct(rate)}")

    if "DAYS_EMPLOYED" in df.columns:
        anom = int((df["DAYS_EMPLOYED"] == 365243).sum())
        print(f"DAYS_EMPLOYED == 365243: {anom:,} ({pct(anom / n_rows)})")

    section("3. Разметка TARGET")
    counts = df["TARGET"].value_counts().sort_index()
    for label, cnt in counts.items():
        print(f"  TARGET={label}: {cnt:,} ({pct(cnt / len(df))})")

    if "EXT_SOURCE_3" in df.columns:
        med_def = df.loc[df["TARGET"] == 1, "EXT_SOURCE_3"].median()
        med_ok = df.loc[df["TARGET"] == 0, "EXT_SOURCE_3"].median()
        print(f"  EXT_SOURCE_3 median: дефолт={med_def:.3f}, без дефолта={med_ok:.3f}")

    section("4. EDA для моделирования")
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)["TARGET"].drop("TARGET", errors="ignore")
    corr = corr.abs().sort_values(ascending=False).head(10)
    print("Топ |corr| с TARGET:")
    for name, val in corr.items():
        print(f"  {name}: {val:.4f}")

    section("5. Валидация")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(df, df["TARGET"]))
    sizes = [len(t) for _, t in folds]
    rates = [df.iloc[t]["TARGET"].mean() for _, t in folds]
    print("Размеры тестовых фолдов:", sizes)
    print("Доля TARGET=1:", [round(r, 4) for r in rates])

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_dup_id": n_dup,
        "default_rate": float(df["TARGET"].mean()),
        "class_counts": counts.to_dict(),
        "top_missing": {k: float(v) for k, v in top_miss.items()},
    }


def main() -> None:
    configure_console()
    sns.set_theme(style="whitegrid")
    print("Загрузка application_train...")
    df = load_train()

    section("Визуализации")
    plot_target_distribution(df)
    plot_missing_top(df)
    plot_corr_with_target(df)
    plot_ext_source_by_target(df)
    plot_income_by_target(df)
    plot_days_employed_anomaly(df)
    plot_table_coverage()

    summary = print_text_eda(df)
    REPORT_DIR.mkdir(exist_ok=True)
    summary_path = REPORT_DIR / "dataset_quality_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nСводка: {summary_path.relative_to(ROOT)}")
    section("Готово")
    print("Графики: reports/figures/*.png")


if __name__ == "__main__":
    main()
