def risk_level(probability: float) -> str:
    if probability < 0.10:
        return "low"
    if probability < 0.25:
        return "medium"
    return "high"


def _age_years(payload: dict) -> float | None:
    if payload.get("age_years") is not None:
        return float(payload["age_years"])
    days_birth = payload.get("DAYS_BIRTH")
    if days_birth is not None:
        return -float(days_birth) / 365.25
    return None


def apply_business_rules(payload: dict, model_probability: float) -> str:
    income = float(payload.get("AMT_INCOME_TOTAL") or 0)
    annuity = float(payload.get("AMT_ANNUITY") or 0)
    if income > 0 and annuity > income:
        return "high"

    is_pensioner = bool(payload.get("is_pensioner"))
    if not is_pensioner:
        age = _age_years(payload)
        years = float(payload.get("employed_years_ui") or 0)
        months = float(payload.get("employed_months_ui") or 0)
        if age is not None and age >= 50 and years <= 0 and months <= 0:
            return "high"

    return risk_level(model_probability)
