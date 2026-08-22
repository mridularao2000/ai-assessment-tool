import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._utils import utcnow


class LateSubmissionToken(Base):
    """A single late-submission credential, identified by its own UUID.

    Single-tenant system — no owning user column. The monthly cron job
    issues new rows so the pool of unused tokens (used_at is None) reaches
    MAX_BALANCE (see LateTokenService). Spending a token sets used_at and
    used_by_assessment_id, letting a submission be accepted after an
    assessment's due_date has passed.
    """

    __tablename__ = "late_submission_tokens"
    __table_args__ = (
        Index("ix_late_submission_tokens_used_at", "used_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    issued_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    used_by_assessment_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("assessments.id"), nullable=True, default=None
    )
