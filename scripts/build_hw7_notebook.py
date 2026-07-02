"""Сборка notebooks/hw7_improved_model.ipynb."""
import json
from pathlib import Path

cells = []


def md(text: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})


def code(text: str):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [text],
        }
    )


md("""# ДЗ 7. Улучшение модели скоринга (Home Credit)

**Продукт:** AI-система оценки финансового риска и поведенческого профиля клиента.

**Данные:** [Яндекс.Диск — data.rar](https://disk.yandex.ru/client/disk?idApp=client&dialog=slider&idDialog=%2Fdisk%2Fdata.rar) — распаковать в `data/`.

| Раздел | Содержание |
|--------|------------|
| 1. Pipeline FE | очистка → join таблиц → ratio/поведенческие признаки |
| 2. Улучшенная модель | LightGBM + Optuna на расширенных признаках vs baseline |
| 3. Постобработка | калибровка вероятностей + бизнес-порог отсечения |
| 4. Анализ качества | ROC/PR, calibration, сегменты, важность признаков |
""")

code("""from pathlib import Path
import sys
import warnings
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
import lightgbm as lgb
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
try:
    from tqdm import TqdmWarning
    warnings.filterwarnings("ignore", category=TqdmWarning)
except ImportError:
    pass
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep import feature_pipeline_audit, load_hw7_dataset

DATA_DIR = ROOT / "data"
RANDOM_STATE = 42
N_SPLITS = 5
OPTUNA_TRIALS = 20
""")

md("""---
# 1. Пайплайн предобработки и feature engineering

**Схема pipeline:**

```
application_train.csv
    → clean_application (дедуп, аномалии, пропуски)
    → merge bureau / installments / previous_app / card / POS
    → add_ratio_features (доход, кредит, EXT_SOURCE)
    → df_improved
```

**Новые поведенческие признаки (installments):** доля недоплат, крупных платежей (>2× median), CV платежей.

**Baseline для сравнения:** тот же pipeline, но без previous_app / card / POS и без расширенных installments.
""")

code("""app_raw = pd.read_csv(DATA_DIR / "application_train.csv")
df_baseline, df_improved = load_hw7_dataset(DATA_DIR)

audit = feature_pipeline_audit(app_raw, df_improved)
display(audit)

new_cols = [c for c in df_improved.columns if c not in df_baseline.columns]
print(f"Добавлено признаков vs baseline HW5-6: {len(new_cols)}")
print("Примеры:", new_cols[:12])

engineered = [
    "inst_large_pay_share", "inst_amt_cv", "inst_pay_diff_abs_mean",
    "prev_refused_share", "cc_dpd_max", "pos_dpd_max",
    "ratio_credit_income", "ext_source_mean",
]
display(df_improved[engineered].describe().T.round(4))
""")

md("""---
# 2. Улучшенная архитектура модели

| Вариант | Признаки | Тюнинг |
|---------|----------|--------|
| Baseline LGBM (HW5-6) | анкета + bureau + installments | Optuna, 15 trials |
| Improved LGBM (HW7) | + previous_app + card + POS + ratio + поведение | Optuna, 20 trials |

**Валидация:** Stratified K-Fold, k=5, одинаковые сплиты для сравнения.
""")

code("""def split_xy(df):
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    return X, y

def ci95(values):
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    h = stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1)
    return mean, mean - h, mean + h

def lgbm_cv_auc(params, X_tr, y_tr, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in skf.split(X_tr, y_tr):
        m = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X_tr.iloc[tr], y_tr.iloc[tr], eval_set=[(X_tr.iloc[va], y_tr.iloc[va])],
              eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(roc_auc_score(y_tr.iloc[va], m.predict_proba(X_tr.iloc[va])[:, 1]))
    return float(np.mean(scores))

def tune_lgbm(X_tr, y_tr, n_trials=OPTUNA_TRIALS):
    def objective(trial):
        params = {
            "objective": "binary",
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 96),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 250),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "class_weight": "balanced",
        }
        return lgbm_cv_auc(params, X_tr, y_tr)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def eval_lgbm_oof(X, y, params):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    aucs, prs = [], []
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**params, objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        prs.append(average_precision_score(y.iloc[va], p))
    am, al, ah = ci95(aucs)
    pm, pl, ph = ci95(prs)
    return {
        "oof": oof,
        "auc_mean": am,
        "auc_ci": (al, ah),
        "pr_mean": pm,
        "pr_ci": (pl, ph),
        "auc_folds": aucs,
        "pr_folds": prs,
    }

X_base, y = split_xy(df_baseline)
X_imp, y_imp = split_xy(df_improved)
assert y.equals(y_imp)

tune_idx = y.sample(frac=0.8, random_state=RANDOM_STATE, replace=False).index
base_params = tune_lgbm(X_base.loc[tune_idx], y.loc[tune_idx], n_trials=15)
imp_params = tune_lgbm(X_imp.loc[tune_idx], y.loc[tune_idx], n_trials=OPTUNA_TRIALS)

base_res = eval_lgbm_oof(X_base, y, base_params)
imp_res = eval_lgbm_oof(X_imp, y, imp_params)

model_cmp = pd.DataFrame([
    {"Модель": "Baseline LGBM (HW5-6)", "AUC": base_res["auc_mean"], "PR-AUC": base_res["pr_mean"], "признаков": X_base.shape[1]},
    {"Модель": "Improved LGBM (HW7)", "AUC": imp_res["auc_mean"], "PR-AUC": imp_res["pr_mean"], "признаков": X_imp.shape[1]},
]).round(4)
display(model_cmp)
print("Лучшие гиперпараметры (improved):", imp_params)
""")

md("""---
# 3. Постобработка предсказаний

1. **Калибровка** — Isotonic Regression поверх OOF-скоров модели. Для **оценки** калибратор обучается fold-wise: на каждом фолде fit только на остальных фолдах, predict — на текущем (без in-sample fit/predict на тех же объектах).
2. **Порог отсечения** — отказываем клиентам из верхних 20% по риску (бизнес-правило).
3. **Бизнес-метрики** — bad rate среди одобренных и доля дефолтов в топ-10% риска.
4. **Метрики калибровки** — Brier и log loss для сырого и OOS-калиброванного скора (AUC от isotonic почти не меняется).
""")

code("""y_arr = y.to_numpy()
raw_oof = imp_res["oof"]


def calibrate_oof_out_of_sample(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    cal = np.zeros(len(y_true))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(scores, y_true):
        iso_fold = IsotonicRegression(out_of_bounds="clip")
        iso_fold.fit(scores[tr], y_true[tr])
        cal[va] = iso_fold.predict(scores[va])
    return cal


cal_oof = calibrate_oof_out_of_sample(raw_oof, y_arr)

reject_share = 0.20
threshold_raw = float(np.quantile(raw_oof, 1 - reject_share))
threshold = float(np.quantile(cal_oof, 1 - reject_share))
approved_raw = raw_oof < threshold_raw
approved = cal_oof < threshold


def bad_rate(mask):
    return float(y_arr[mask].mean()) if mask.sum() else np.nan


def topk_default_share(scores, k=0.10):
    cut = np.quantile(scores, 1 - k)
    top = scores >= cut
    return float(y_arr[top].mean())


calib_metrics = pd.DataFrame([
    {
        "скор": "сырой OOF",
        "Brier": brier_score_loss(y_arr, raw_oof),
        "log_loss": log_loss(y_arr, raw_oof),
    },
    {
        "скор": "калибр. OOF (fold-wise)",
        "Brier": brier_score_loss(y_arr, cal_oof),
        "log_loss": log_loss(y_arr, np.clip(cal_oof, 1e-6, 1 - 1e-6)),
    },
])
display(calib_metrics.round(4))

post_df = pd.DataFrame([
    {
        "вариант": "сырой скор",
        "bad_rate_одобренных": bad_rate(approved_raw),
        "дефолт_в_топ10%": topk_default_share(raw_oof, 0.10),
    },
    {
        "вариант": "калиброванный + порог top-20%",
        "bad_rate_одобренных": bad_rate(approved),
        "дефолт_в_топ10%": topk_default_share(cal_oof, 0.10),
    },
])
display(post_df.round(4))
print(f"Порог отсечения (калибр. OOF, top-{int(reject_share*100)}% риска): {threshold:.4f}")
""")

md("""---
# 4. Подробный анализ качества модели

**Метрики:** AUC-ROC, PR-AUC, Precision/Recall/F1 при пороге, calibration, сегменты, важность признаков.
""")

code("""def fmt_ci(mean, ci):
    return f"{mean:.4f} [{ci[0]:.4f}; {ci[1]:.4f}]"

delta_auc = np.array(imp_res["auc_folds"]) - np.array(base_res["auc_folds"])
d_mean, d_lo, d_hi = ci95(delta_auc.tolist())

summary = pd.DataFrame([
    {
        "Модель": "Baseline LGBM",
        "AUC-ROC": fmt_ci(base_res["auc_mean"], base_res["auc_ci"]),
        "PR-AUC": fmt_ci(base_res["pr_mean"], base_res["pr_ci"]),
    },
    {
        "Модель": "Improved LGBM",
        "AUC-ROC": fmt_ci(imp_res["auc_mean"], imp_res["auc_ci"]),
        "PR-AUC": fmt_ci(imp_res["pr_mean"], imp_res["pr_ci"]),
    },
])
display(summary)
print(f"Прирост AUC (improved − baseline): {d_mean:.4f} [{d_lo:.4f}; {d_hi:.4f}]")

fpr_b, tpr_b, _ = roc_curve(y_arr, base_res["oof"])
fpr_i, tpr_i, _ = roc_curve(y_arr, imp_res["oof"])
prec_b, rec_b, _ = precision_recall_curve(y_arr, base_res["oof"])
prec_i, rec_i, _ = precision_recall_curve(y_arr, imp_res["oof"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(fpr_b, tpr_b, label=f"baseline AUC={base_res['auc_mean']:.3f}")
axes[0].plot(fpr_i, tpr_i, label=f"improved AUC={imp_res['auc_mean']:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_title("ROC-кривая")
axes[0].legend()

axes[1].plot(rec_b, prec_b, label=f"baseline PR={base_res['pr_mean']:.3f}")
axes[1].plot(rec_i, prec_i, label=f"improved PR={imp_res['pr_mean']:.3f}")
axes[1].set_title("PR-кривая")
axes[1].legend()
plt.tight_layout()
plt.show()

pt_raw, pp_raw = calibration_curve(y_arr, raw_oof, n_bins=10, strategy="quantile")
pt, pp = calibration_curve(y_arr, cal_oof, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(pp_raw, pt_raw, marker="s", label="сырой OOF")
ax.plot(pp, pt, marker="o", label="калибр. OOF (fold-wise)")
ax.plot([0, 1], [0, 1], "k--", label="идеал")
ax.set_xlabel("предсказанная вероятность")
ax.set_ylabel("доля дефолтов")
ax.set_title("Калибровка вероятностей (OOS)")
ax.legend()
plt.show()

y_pred = (cal_oof >= threshold).astype(int)
cm = confusion_matrix(y_arr, y_pred)
cm_df = pd.DataFrame(cm, index=["факт 0", "факт 1"], columns=["pred 0", "pred 1"])
display(cm_df)
print(
    f"Precision={precision_score(y_arr, y_pred, zero_division=0):.4f}, "
    f"Recall={recall_score(y_arr, y_pred, zero_division=0):.4f}, "
    f"F1={f1_score(y_arr, y_pred, zero_division=0):.4f}"
)

for k in (0.05, 0.10, 0.20):
    print(f"Доля дефолтов в топ-{int(k*100)}% (improved): {topk_default_share(cal_oof, k):.4f}")
""")

code("""seg = pd.DataFrame({
    "y": y_arr,
    "score": imp_res["oof"],
    "has_bureau": (df_improved["bureau_n_credits"].fillna(0) > 0).to_numpy(),
    "has_inst": (df_improved["inst_n_payments"].fillna(0) > 0).to_numpy(),
})

rows = []
for name, mask in [
    ("все клиенты", np.ones(len(seg), dtype=bool)),
    ("есть bureau", seg["has_bureau"]),
    ("есть installments", seg["has_inst"]),
    ("нет bureau", ~seg["has_bureau"]),
]:
    sub = seg.loc[mask]
    if sub["y"].nunique() < 2 or len(sub) < 100:
        continue
    rows.append({
        "сегмент": name,
        "n": len(sub),
        "AUC": roc_auc_score(sub["y"], sub["score"]),
        "bad_rate": sub["y"].mean(),
    })
display(pd.DataFrame(rows).round(4))

final_model = lgb.LGBMClassifier(**imp_params, objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
final_model.fit(X_imp, y)
imp = pd.Series(final_model.feature_importances_, index=X_imp.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#4c78a8")
ax.set_title("Топ-15 признаков (improved LGBM)")
plt.tight_layout()
plt.show()
""")

md("""---
## Выводы

- **Pipeline FE:** расширенный join и ratio/поведение увеличивают число признаков относительно baseline HW5-6 (audit §1, столбец «признаков» в model_cmp, §2).
- **Модель:** Improved LGBM vs baseline на Stratified K-Fold (k=5); AUC, PR-AUC и 95% CI прироста — в model_cmp (§2) и summary (§4).
- **Постобработка:** isotonic оценивается fold-wise OOS без in-sample утечки (§3). Brier, log loss, bad rate и top-10% — в calib_metrics и post_df; на этих метриках калибровка не даёт заметного улучшения относительно сырого OOF (значения совпадают или Δ≈0).
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "hw7_improved_model.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"written: {out}")
