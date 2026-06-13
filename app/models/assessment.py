import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.curriculum import Curriculum
    from app.models.prompt_template import PromptTemplate
    from app.models.reschedule_request import RescheduleRequest
    from app.models.submission import Submission


class AssessmentStatus(str, enum.Enum):
    scheduled = "scheduled"   # created, email job queued, not yet sent
    active = "active"         # assessment email sent, awaiting submission
    submitted = "submitted"   # submission received, grading pending
    completed = "completed"   # graded
    expired = "expired"       # due_date passed with no submission


class Assessment(Base):
    __tablename__ = "assessments"
    __table_args__ = (
        Index("ix_assessments_curriculum_id", "curriculum_id"),
        Index("ix_assessments_status", "status"),
        Index("ix_assessments_scheduled_at", "scheduled_at"),
        Index("ix_assessments_due_date", "due_date"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    curriculum_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("curricula.id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Populated by AssessmentService after Claude generation.
    # Null while status == scheduled and generation is pending.
    assessment_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Never exposed to the user — grading reference only.
    rubric: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Claude-determined from curriculum complexity. Included in assessment email.
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    generation_prompt_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("prompt_templates.id"), nullable=True, default=None
    )
    # ── Scheduling ────────────────────────────────────────────────────────────
    # scheduled_at is always within [target_completion_date + min_days,
    #                                 target_completion_date + max_days].
    # Enforced by AssessmentService._calculate_scheduled_at() (primary) and
    # SchedulerService.schedule_assessment_jobs() (secondary guard).
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    reminder_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    due_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[AssessmentStatus] = mapped_column(
        Enum(AssessmentStatus), nullable=False, default=AssessmentStatus.scheduled
    )
    # HMAC-signed token embedded in the assessment email link.
    # Acts as the sole credential for submission and reschedule endpoints.
    submission_token: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    # APScheduler job ID map: {"send_reminder": id, "send_assessment": id, "expire": id}
    # Written by SchedulerService; used for targeted cancellation on reschedule.
    scheduled_job_ids: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    curriculum: Mapped["Curriculum"] = relationship(
        "Curriculum", back_populates="assessments"
    )
    generation_prompt: Mapped[Optional["PromptTemplate"]] = relationship(
        "PromptTemplate",
        foreign_keys="[Assessment.generation_prompt_id]",
    )
    submission: Mapped[Optional["Submission"]] = relationship(
        "Submission",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    reschedule_requests: Mapped[list["RescheduleRequest"]] = relationship(
        "RescheduleRequest",
        back_populates="assessment",
        cascade="all, delete-orphan",
    )
