import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.assessment import Assessment
    from app.models.grade import Grade


class SubmissionType(str, enum.Enum):
    github_url = "github_url"
    text = "text"
    file = "file"


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # UNIQUE: one submission per assessment.
    assessment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessments.id"), nullable=False, unique=True
    )
    submission_type: Mapped[SubmissionType] = mapped_column(
        Enum(SubmissionType), nullable=False
    )
    # Exactly one of the three content fields is non-null, matching submission_type.
    # Mutual exclusivity is enforced by the submission schema validator.
    github_url: Mapped[Optional[str]] = mapped_column(Text, default=None)
    text_content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    file_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )
    # Full POST body stored for audit. Not used in grading logic.
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSON, default=None)

    # ── Relationships ─────────────────────────────────────────────────────────
    assessment: Mapped["Assessment"] = relationship(
        "Assessment", back_populates="submission"
    )
    grade: Mapped[Optional["Grade"]] = relationship(
        "Grade",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan",
    )
