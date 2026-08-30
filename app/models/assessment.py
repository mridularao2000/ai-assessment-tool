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
    late_submitted = "late_submitted"  # submitted after due_date using a late token
    completed = "completed"   # graded
    expired = "expired"       # due_date passed with no submission
    # Generation exhausted its tool-call budget without producing valid
    # output (LLMToolBudgetExceededError — see AnthropicLLMAdapter._retry).
    # Distinct from a generic failure specifically so recheck_stuck_
    # assessments_job's sweep (which only matches status == scheduled)
    # never auto-retries it — a human must resolve the cause and use
    # POST /resend. No DB migration needed: status is a plain VARCHAR
    # column with no CHECK constraint (confirmed against the live schema).
    needs_manual_diagnosis = "needs_manual_diagnosis"


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
    # For curriculum-upload Midterm-type entries, these two stay null and
    # part1_text/part2_text are used instead (see below) — never both.
    assessment_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Never exposed to the user — grading reference only.
    rubric: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Midterm-type only (curriculum-upload) — null for standalone and
    # Assessment-type entries. Populated by AssessmentService.
    # generate_midterm_content() at send-time, same as assessment_text.
    part1_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    part1_rubric: Mapped[Optional[str]] = mapped_column(Text, default=None)
    part2_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    part2_rubric: Mapped[Optional[str]] = mapped_column(Text, default=None)
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
    # TOCTOU guard for send_assessment_job: claimed atomically (an
    # UPDATE...WHERE send_job_claimed_at IS NULL) before any generation/send
    # work, so two concurrent executions for the same assessment_id (e.g. a
    # manual retry racing a still-in-flight scheduled firing — APScheduler's
    # own max_instances=1 only stops the SAME job_id from double-running
    # through the scheduler itself, not an out-of-band re-invocation) can't
    # both generate content and both send a duplicate email. Released back
    # to None on failure so a genuine retry after a crash isn't permanently
    # locked out; left set on success (nothing legitimately re-invokes this
    # job for an already-sent assessment).
    send_job_claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )

    @property
    def content_generated(self) -> bool:
        """True once the actual LLM-generated content exists on this row —
        assessment_text (standalone/Assessment-type) or part1_text
        (Midterm-type). Distinct from `status`: an `expired` row can be
        either a genuine generate-then-never-submitted case, or one created
        directly in `expired` with content still null (see
        CurriculumUploadService._create_retroactive_expired_assessment,
        whose generation is deferred to first access) — `status` alone
        can't tell those apart, this can."""
        return self.assessment_text is not None or self.part1_text is not None

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
