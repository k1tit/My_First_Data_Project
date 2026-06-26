def monthly_annuity(principal: float, annual_rate_pct: float, term_years: float) -> float:
    if principal <= 0 or term_years <= 0:
        raise ValueError("Сумма кредита и срок должны быть больше нуля")
    months = max(1, round(term_years * 12))
    if annual_rate_pct <= 0:
        return principal / months
    rate = annual_rate_pct / 100 / 12
    factor = (1 + rate) ** months
    return principal * rate * factor / (factor - 1)
