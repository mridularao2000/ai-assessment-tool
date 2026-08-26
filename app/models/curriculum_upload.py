import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.curriculum import Curriculum


class CurriculumUpload(Base):
    """One uploaded syllabus file, grouping its CurriculumEntry rows.

    Nothing in the standalone flow references this table — it exists purely
    to scope a batch of Curriculum rows (entry_type set) to the file they
    came from, for the syllabus email and future GPA/transcript queries.
    """

    __tablename__ = "curriculum_uploads"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_filename: Mapped[str] = mapped_column(Text, nullable=False)
    # Verbatim top-level metadata string from the file, e.g. explaining the
    # known_now/pending_completion split. Audit/display only.
    note_on_pending_resources: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # JSON list of {"chapter": ..., "note": ...} — chapters with no standalone
    # Assessment, verbatim from the file.
    chapters_with_no_standalone_assessment: Mapped[Optional[list]] = mapped_column(
        JSON, default=None
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    syllabus_email_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    # Frozen "Course Material" section for the transcript email — captured
    # once, the first time send_syllabus_email() succeeds, and reused
    # unchanged on every subsequent transcript send rather than
    # recomputed (so it doesn't drift as pending_completion_slots get
    # filled in after upload). See syllabus_builder.serialize_syllabus_content.
    course_material_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    course_material_captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    entries: Mapped[list["Curriculum"]] = relationship(
        "Curriculum", back_populates="upload"
    )
