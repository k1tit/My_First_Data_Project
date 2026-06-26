from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.joblib").exists(),
    reason="Run python task8/scripts/export_artifacts.py first",
)


@pytest.fixture
def engine():
    from shared.scoring_engine import ScoringEngine

    return ScoringEngine(artifacts_dir=ARTIFACTS)


def test_predict_returns_valid_probability(engine):
    payload = {
        "AMT_INCOME_TOTAL": 150_000,
        "AMT_CREDIT": 300_000,
        "AMT_ANNUITY": 15_000,
        "DAYS_BIRTH": -12_000,
        "EXT_SOURCE_2": 0.6,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
    }
    raw, calibrated, level = engine.predict(payload)

    assert 0.0 <= raw <= 1.0
    assert 0.0 <= calibrated <= 1.0
    assert level in {"low", "medium", "high"}


def test_pensioner_profile_not_zero_risk(engine):
    payload = {
        "AMT_INCOME_TOTAL": 150_000,
        "AMT_CREDIT": 300_000,
        "AMT_ANNUITY": 15_000,
        "DAYS_BIRTH": -round(88 * 365.25),
        "EXT_SOURCE_1": 0.2,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Pensioner",
        "age_years": 88,
        "is_pensioner": True,
        "employed_years_ui": 0,
        "employed_months_ui": 0,
    }
    raw, calibrated, level = engine.predict(payload)
    assert raw > 0.05
    assert level in {"medium", "high", "low"}


def test_predict_with_null_optional_fields(engine):
    payload = {
        "AMT_INCOME_TOTAL": 150_000,
        "AMT_CREDIT": 300_000,
        "AMT_ANNUITY": 15_000,
        "DAYS_BIRTH": -12_000,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "AMT_GOODS_PRICE": None,
        "CNT_FAM_MEMBERS": None,
        "bureau_n_credits": None,
        "inst_n_payments": None,
    }
    raw, calibrated, level = engine.predict(payload)
    assert 0.0 <= raw <= 1.0
    assert level in {"low", "medium", "high"}


def test_high_risk_when_payment_exceeds_income(engine):
    payload = {
        "AMT_INCOME_TOTAL": 40_000,
        "AMT_CREDIT": 500_000,
        "AMT_ANNUITY": 50_000,
        "DAYS_BIRTH": -12_000,
        "age_years": 40,
        "is_pensioner": False,
        "employed_years_ui": 10,
        "employed_months_ui": 0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
    }
    _, _, level = engine.predict(payload)
    assert level == "high"
