import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.prompt_template import PromptTemplate
    from app.models.submission import Submission


class Grade(Base):
    __tablename__ = "grades"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # UNIQUE: one grade per submission.
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.id"), nullable=False, unique=True
    )
    mastery_score: Mapped[float] = mapped_column(Float, nullable=False)
    # Claude-identified weak areas from this attempt, e.g. ["Promises", "Async/Await"].
    # Passed to retest_generation prompt when mastery_score < mastery_threshold.
    weak_areas: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    overall_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    # Midterm-type only — raw points per part. Null for standalone and
    # Assessment-type entries (they use mastery_score directly).
    part1_score: Mapped[Optional[float]] = mapped_column(Float, default=None)
    part2_score: Mapped[Optional[float]] = mapped_column(Float, default=None)
    # GPA inputs. Null for standalone rows (they don't participate in GPA).
    # For Assessment-type entries: score_earned = mastery_score/100*max_marks.
    # For Midterm-type entries: score_earned = part1_score+part2_score, and
    # mastery_score is back-derived as score_earned/max_marks*100 so the
    # existing mastery_threshold pass/fail comparison works unchanged for
    # all three cases without a branch in the retake-cap logic.
    score_earned: Mapped[Optional[float]] = mapped_column(Float, default=None)
    max_marks: Mapped[Optional[float]] = mapped_column(Float, default=None)
    grading_prompt_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("prompt_templates.id"), nullable=True, default=None
    )
    graded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    # ── Relationships ─────────────────────────────────────────────────────────
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="grade"
    )
    grading_prompt: Mapped[Optional["PromptTemplate"]] = relationship(
        "PromptTemplate",
        foreign_keys="[Grade.grading_prompt_id]",
    )
