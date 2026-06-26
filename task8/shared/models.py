from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from shared.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ScoringRequest(Base):
    __tablename__ = "scoring_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(32), default="pending", index=True)
    payload_json = Column(Text, nullable=False)
    model_version = Column(String(64), default="hw7-improved-v1")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    result = relationship(
        "ScoringResult",
        back_populates="request",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ScoringResult(Base):
    __tablename__ = "scoring_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(
        Integer,
        ForeignKey("scoring_requests.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    probability = Column(Float, nullable=False)
    probability_calibrated = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    request = relationship("ScoringRequest", back_populates="result")
