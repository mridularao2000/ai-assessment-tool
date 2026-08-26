import enum
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.assessment import Assessment
    from app.models.curriculum_upload import CurriculumUpload
    from app.models.midterm_detail import MidtermDetail
    from app.models.resource import Resource


class CurriculumEntryType(str, enum.Enum):
    """None (the column default) = standalone curriculum, exactly today's flow.

    Set only for rows created by CurriculumUploadService — distinguishes the
    two genuinely different exam shapes without duplicating the table.
    """

    assessment = "assessment"
    midterm = "midterm"


class CurriculumStatus(str, enum.Enum):
    pending = "pending"
    analyzing = "analyzing"
    ready = "ready"
    complete = "complete"


class Curriculum(Base):
    __tablename__ = "curricula"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    target_completion_date: Mapped[date] = mapped_column(Date, nullable=False)
    extracted_content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    mastery_achieved: Mapped[Optional[bool]] = mapped_column(Boolean, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    status: Mapped[CurriculumStatus] = mapped_column(
        Enum(CurriculumStatus), nullable=False, default=CurriculumStatus.pending
    )
    # Nice-to-have fields: reserved for future hyperfocus / priority features.
    # No application logic is implemented against these yet.
    priority: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    is_active_focus: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )

    # ── Curriculum-upload extension fields ─────────────────────────────────────
    # All default to None/False for every row created by the standalone flow —
    # no behavior change there. Only set by CurriculumUploadService.
    entry_type: Mapped[Optional[CurriculumEntryType]] = mapped_column(
        Enum(CurriculumEntryType), nullable=True, default=None
    )
    upload_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("curriculum_uploads.id"), nullable=True, default=None
    )
    # Verbatim `chapter` field from the upload file — DISPLAY ONLY. The real
    # Midterm Part-1 cumulative pool is always computed fresh from
    # target_completion_date comparisons, never parsed from this string.
    chapter_label: Mapped[Optional[str]] = mapped_column(Text, default=None)
    max_marks: Mapped[Optional[float]] = mapped_column(Float, default=None)
    # True only while a midterm awaits pending_completion resources past its
    # target_completion_date. See CurriculumUploadService / the daily recheck job.
    resources_hold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_hold_reminder_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    # ── Relationships ─────────────────────────────────────────────────────────
    resources: Mapped[list["Resource"]] = relationship(
        "Resource",
        back_populates="curriculum",
        cascade="all, delete-orphan",
    )
    assessments: Mapped[list["Assessment"]] = relationship(
        "Assessment",
        back_populates="curriculum",
        cascade="all, delete-orphan",
        order_by="Assessment.attempt_number",
    )
    upload: Mapped[Optional["CurriculumUpload"]] = relationship(
        "CurriculumUpload", back_populates="entries"
    )
    midterm_detail: Mapped[Optional["MidtermDetail"]] = relationship(
        "MidtermDetail",
        back_populates="curriculum",
        uselist=False,
        cascade="all, delete-orphan",
    )
