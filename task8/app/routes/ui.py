from pathlib import Path

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from shared.db import get_session
from shared.finance import monthly_annuity
from shared.schemas import ApplicationInput
from app.services.scoring_service import create_scoring_request, enqueue_scoring, list_recent

from pydantic import ValidationError

router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

MAX_AMOUNT = 1_000_000_000

ERROR_MESSAGES = {
    "form": "Не удалось принять заявку — проверьте обязательные поля и отправьте снова.",
    "limit_income": f"Превышен лимит: доход не может быть больше {MAX_AMOUNT:,} ₽.".replace(",", " "),
    "limit_credit": f"Превышен лимит: сумма кредита не может быть больше {MAX_AMOUNT:,} ₽.".replace(",", " "),
    "limit_loan": "Некорректные параметры кредита: проверьте сумму, срок и ставку.",
    "limit_age": "Возраст заёмщика должен быть от 18 до 100 лет.",
    "validation": "Проверьте введённые значения — одно или несколько полей выходят за допустимые пределы.",
}


def _days_birth(age_years: float) -> float:
    return -round(age_years * 365.25)


def _days_employed(years: float | None, months: float | None) -> float | None:
    y = years or 0
    m = months or 0
    if y <= 0 and m <= 0:
        return None
    return -round(y * 365.25 + m * 30.4375)


def _optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _form_bool(value: str | None) -> bool:
    return value in ("on", "true", "True", "1", "yes")


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    session: Session = Depends(get_session),
    error: str | None = Query(None),
):
    history = list_recent(session, limit=10)
    error_message = ERROR_MESSAGES.get(error) if error else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {"history": history, "error_message": error_message},
    )


@router.post("/score")
async def submit_form(
    request: Request,
    AMT_INCOME_TOTAL: float = Form(...),
    AMT_CREDIT: float = Form(...),
    loan_term_years: float = Form(...),
    interest_rate_annual: float = Form(...),
    age_years: float = Form(...),
    employed_years: float | None = Form(None),
    employed_months: float | None = Form(None),
    is_pensioner: str | None = Form(None),
    EXT_SOURCE_1: str | None = Form(None),
    EXT_SOURCE_2: str | None = Form(None),
    EXT_SOURCE_3: str | None = Form(None),
    NAME_CONTRACT_TYPE: str = Form("Cash loans"),
    NAME_INCOME_TYPE: str = Form("Working"),
    session: Session = Depends(get_session),
):
    if AMT_INCOME_TOTAL > MAX_AMOUNT:
        return RedirectResponse(url="/?error=limit_income", status_code=303)
    if AMT_CREDIT > MAX_AMOUNT:
        return RedirectResponse(url="/?error=limit_credit", status_code=303)
    if age_years < 18 or age_years > 100:
        return RedirectResponse(url="/?error=limit_age", status_code=303)

    pensioner = _form_bool(is_pensioner)
    try:
        annuity = monthly_annuity(AMT_CREDIT, interest_rate_annual, loan_term_years)
    except ValueError:
        return RedirectResponse(url="/?error=limit_loan", status_code=303)

    try:
        data = ApplicationInput(
            AMT_INCOME_TOTAL=AMT_INCOME_TOTAL,
            AMT_CREDIT=AMT_CREDIT,
            AMT_ANNUITY=round(annuity, 2),
            DAYS_BIRTH=_days_birth(age_years),
            DAYS_EMPLOYED=None if pensioner else _days_employed(employed_years, employed_months),
            EXT_SOURCE_1=_optional_float(EXT_SOURCE_1),
            EXT_SOURCE_2=_optional_float(EXT_SOURCE_2),
            EXT_SOURCE_3=_optional_float(EXT_SOURCE_3),
            NAME_CONTRACT_TYPE=NAME_CONTRACT_TYPE,
            NAME_INCOME_TYPE="Pensioner" if pensioner else NAME_INCOME_TYPE,
            age_years=age_years,
            is_pensioner=pensioner,
            employed_years_ui=0 if pensioner else (employed_years or 0),
            employed_months_ui=0 if pensioner else (employed_months or 0),
            loan_term_years=loan_term_years,
            interest_rate_annual=interest_rate_annual,
        )
    except ValidationError:
        return RedirectResponse(url="/?error=validation", status_code=303)

    req = create_scoring_request(data, session)
    await enqueue_scoring(req.id)
    return RedirectResponse(url=f"/result/{req.id}", status_code=303)


@router.get("/result/{request_id}", response_class=HTMLResponse)
def result_page(request_id: int, request: Request, session: Session = Depends(get_session)):
    from app.services.scoring_service import get_scoring_result

    result = get_scoring_result(request_id, session)
    return templates.TemplateResponse(
        request,
        "result.html",
        {"result": result, "request_id": request_id},
    )
