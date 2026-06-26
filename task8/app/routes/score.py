from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from shared.db import get_session
from shared.schemas import ApplicationInput, ScoreCreateResponse, ScoreResultOut
from app.services.scoring_service import create_scoring_request, enqueue_scoring, get_scoring_result

router = APIRouter(prefix="/api/v1/score", tags=["score"])


@router.post("", response_model=ScoreCreateResponse, status_code=202)
async def submit_score(
    data: ApplicationInput,
    session: Session = Depends(get_session),
):
    req = create_scoring_request(data, session)
    await enqueue_scoring(req.id)
    return ScoreCreateResponse(request_id=req.id)


@router.get("/{request_id}", response_model=ScoreResultOut)
def fetch_score(request_id: int, session: Session = Depends(get_session)):
    result = get_scoring_result(request_id, session)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found")
    return result
