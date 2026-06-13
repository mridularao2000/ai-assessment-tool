import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.assessment import Assessment

# Defined here for reference; authoritative copy lives in RescheduleService.
# Application code maps these to approved=True/False.
#
#   Approved:  interview | medical | emergency | work_escalation
#   Denied:    procrastination | lack_of_preparation | missed_schedule


class RescheduleRequest(Base):
    __tablename__ = "reschedule_requests"
    __table_args__ = (
        Index("ix_reschedule_requests_assessment_id", "assessment_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    assessment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessments.id"), nullable=False
    )
    reason_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Claude provides only these two fields — Claude never decides approval.
    classification_category: Mapped[Optional[str]] = mapped_column(
        String(50), default=None
    )
    category_reasoning: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Set by RescheduleService based on whether classification_category
    # is in APPROVED_CATEGORIES. Null until classification completes.
    approved: Mapped[Optional[bool]] = mapped_column(Boolean, default=None)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    assessment: Mapped["Assessment"] = relationship(
        "Assessment", back_populates="reschedule_requests"
    )
