from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ApplicationInput(BaseModel):

    AMT_INCOME_TOTAL: float = Field(..., gt=0, le=1_000_000_000, description="Доход")
    AMT_CREDIT: float = Field(..., gt=0, le=1_000_000_000, description="Сумма кредита")
    AMT_ANNUITY: float = Field(..., gt=0, le=1_000_000_000, description="Ежемесячный платёж")
    DAYS_BIRTH: float = Field(..., le=-6570, description="Дней от рождения (отриц., ≥18 лет)")
    DAYS_EMPLOYED: float | None = Field(None, description="Дней на работе")
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)
    NAME_CONTRACT_TYPE: str = "Cash loans"
    NAME_INCOME_TYPE: str = "Working"
    AMT_GOODS_PRICE: float | None = None
    CNT_FAM_MEMBERS: float | None = None
    bureau_n_credits: float | None = None
    inst_n_payments: float | None = None
    age_years: float | None = None
    is_pensioner: bool = False
    employed_years_ui: float | None = None
    employed_months_ui: float | None = None
    loan_term_years: float | None = None
    interest_rate_annual: float | None = None


class ScoreCreateResponse(BaseModel):
    request_id: int
    status: Literal["pending"] = "pending"
    message: str = "Заявка принята в очередь на скоринг"


class ScoreResultOut(BaseModel):
    request_id: int
    status: str
    probability: float | None = None
    probability_calibrated: float | None = None
    risk_level: str | None = None
    model_version: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
