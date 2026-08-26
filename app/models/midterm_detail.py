import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.curriculum import Curriculum


class MidtermDetail(Base):
    """Midterm-only fields, kept off Curriculum so standalone and
    Assessment-type rows aren't polluted with columns that are always null
    for them.

    known_now / pending_completion are handled as genuinely different states
    (per the upload spec): known_now is usable immediately; pending_completion
    doesn't exist yet at upload time and gets restructured into named slots
    that a later update fills in.
    """

    __tablename__ = "midterm_details"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    curriculum_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("curricula.id"), nullable=False, unique=True
    )
    known_now: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # {slot_slug: original_label_text} — derived from pending_completion's
    # placeholder labels at ingestion time, e.g. {"repo_url": "repo URL"}.
    pending_completion_labels: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # {slot_slug: value|None} — filled in via PATCH .../pending-resources.
    pending_completion_slots: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    probe_focus: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Verbatim seed prose (e.g. PM System's fallback explanation) — audit/
    # display only. The known_now fallback is always derived generically
    # (zero qualifying Assessments), never by branching on this string.
    special_case: Mapped[Optional[str]] = mapped_column(Text, default=None)
    part1_max_marks: Mapped[float] = mapped_column(Float, nullable=False)
    part2_max_marks: Mapped[float] = mapped_column(Float, nullable=False)

    curriculum: Mapped["Curriculum"] = relationship(
        "Curriculum", back_populates="midterm_detail"
    )
