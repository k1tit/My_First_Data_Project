from sqlalchemy.orm import Session

from shared.messaging import publish_scoring_job
from shared.models import ScoringRequest, ScoringResult
from shared.schemas import ApplicationInput, ScoreResultOut


def create_scoring_request(data: ApplicationInput, session: Session) -> ScoringRequest:
    req = ScoringRequest(
        status="pending",
        payload_json=data.model_dump_json(exclude_none=True),
    )
    session.add(req)
    session.commit()
    session.refresh(req)
    return req


async def enqueue_scoring(request_id: int) -> None:
    await publish_scoring_job(request_id)


def get_scoring_result(request_id: int, session: Session) -> ScoreResultOut | None:
    req = session.get(ScoringRequest, request_id)
    if not req:
        return None

    out = ScoreResultOut(
        request_id=req.id,
        status=req.status,
        model_version=req.model_version,
        error_message=req.error_message,
        created_at=req.created_at,
    )
    if req.result:
        out.probability = req.result.probability
        out.probability_calibrated = req.result.probability_calibrated
        out.risk_level = req.result.risk_level
    return out


def list_recent(session: Session, limit: int = 20) -> list[ScoreResultOut]:
    rows = (
        session.query(ScoringRequest)
        .order_by(ScoringRequest.id.desc())
        .limit(limit)
        .all()
    )
    out: list[ScoreResultOut] = []
    for row in rows:
        item = get_scoring_result(row.id, session)
        if item:
            out.append(item)
    return out
