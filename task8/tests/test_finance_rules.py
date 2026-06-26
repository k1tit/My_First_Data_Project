from shared.finance import monthly_annuity
from shared.risk import apply_business_rules, risk_level


def test_monthly_annuity_zero_rate():
    pay = monthly_annuity(120_000, 0, 2)
    assert pay == 5_000


def test_monthly_annuity_with_rate():
    pay = monthly_annuity(300_000, 18, 5)
    assert 7_000 < pay < 8_000


def test_risk_payment_exceeds_income():
    payload = {
        "AMT_INCOME_TOTAL": 50_000,
        "AMT_ANNUITY": 60_000,
        "age_years": 40,
        "is_pensioner": False,
        "employed_years_ui": 5,
        "employed_months_ui": 0,
    }
    assert apply_business_rules(payload, 0.05) == "high"


def test_risk_age50_no_experience():
    payload = {
        "AMT_INCOME_TOTAL": 150_000,
        "AMT_ANNUITY": 15_000,
        "age_years": 55,
        "is_pensioner": False,
        "employed_years_ui": 0,
        "employed_months_ui": 0,
    }
    assert apply_business_rules(payload, 0.05) == "high"


def test_risk_pensioner_skips_experience_rule():
    payload = {
        "AMT_INCOME_TOTAL": 150_000,
        "AMT_ANNUITY": 15_000,
        "age_years": 70,
        "is_pensioner": True,
        "employed_years_ui": 0,
        "employed_months_ui": 0,
    }
    assert apply_business_rules(payload, 0.05) == "low"


def test_risk_model_only_when_no_rules():
    assert apply_business_rules(
        {
            "AMT_INCOME_TOTAL": 150_000,
            "AMT_ANNUITY": 15_000,
            "age_years": 33,
            "is_pensioner": False,
            "employed_years_ui": 5,
            "employed_months_ui": 0,
        },
        0.20,
    ) == risk_level(0.20)
