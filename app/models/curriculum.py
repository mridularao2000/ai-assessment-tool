import enum
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.assessment import Assessment
    from app.models.resource import Resource


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
