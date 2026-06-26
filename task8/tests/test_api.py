from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from shared.db import Base, get_session
from shared.models import ScoringRequest, ScoringResult
from shared.risk import risk_level


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session, monkeypatch):
    from app.main import app

    def override_get_session():
        yield db_session

    async def noop_enqueue(_request_id: int):
        return None

    app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr("app.routes.score.enqueue_scoring", noop_enqueue)
    monkeypatch.setattr("app.routes.ui.enqueue_scoring", noop_enqueue)

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_risk_levels():
    assert risk_level(0.05) == "low"
    assert risk_level(0.15) == "medium"
    assert risk_level(0.30) == "high"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_submit_score_api(client, db_session):
    payload = {
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 300000,
        "AMT_ANNUITY": 15000,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_2": 0.6,
    }
    r = client.post("/api/v1/score", json=payload)
    assert r.status_code == 202
    data = r.json()
    assert "request_id" in data
    assert db_session.get(ScoringRequest, data["request_id"]) is not None


def test_get_score_not_found(client):
    r = client.get("/api/v1/score/9999")
    assert r.status_code == 404


def test_ui_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Оценка кредитного риска" in r.text


def test_ui_limit_income_shows_error(client):
    r = client.post(
        "/score",
        data={
            "AMT_INCOME_TOTAL": "2000000000",
            "AMT_CREDIT": "300000",
            "loan_term_years": "5",
            "interest_rate_annual": "18",
            "age_years": "33",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/?error=limit_income"


def test_ui_large_credit_calculates_payment(client, db_session):
    r = client.post(
        "/score",
        data={
            "AMT_INCOME_TOTAL": "50000000",
            "AMT_CREDIT": "200000000",
            "loan_term_years": "5",
            "interest_rate_annual": "18",
            "age_years": "33",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/result/")
    r = client.post(
        "/score",
        data={
            "AMT_INCOME_TOTAL": "150000",
            "AMT_CREDIT": "300000",
            "loan_term_years": "5",
            "interest_rate_annual": "18",
            "age_years": "33",
            "employed_years": "5",
            "employed_months": "0",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/result/")
    req_id = int(r.headers["location"].split("/")[-1])
    req = db_session.get(ScoringRequest, req_id)
    assert req is not None
    import json
    payload = json.loads(req.payload_json)
    assert payload["AMT_ANNUITY"] > 0
    assert payload.get("loan_term_years") == 5


def test_completed_result(client, db_session):
    req = ScoringRequest(status="completed", payload_json="{}")
    db_session.add(req)
    db_session.flush()
    db_session.add(
        ScoringResult(
            request_id=req.id,
            probability=0.2,
            probability_calibrated=0.18,
            risk_level="medium",
        )
    )
    db_session.commit()
    r = client.get(f"/api/v1/score/{req.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["risk_level"] == "medium"
