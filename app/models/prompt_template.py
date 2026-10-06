import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._utils import utcnow

# Valid slug values — enforced at the service layer, not at the DB level.
#   assessment_generation   — first-attempt assessment creation (MCQ + Coding Project format, v2.0+)
#   retest_generation       — subsequent-attempt assessment creation (receives weak_areas)
#   midterm_generation      — two-part Midterm exam creation (curriculum-upload only)
#   grading                 — submission grading (legacy free-form format only)
#   mcq_grading             — Section 1 (5 MCQs) grading for the MCQ + Coding Project format
#   project_grading         — Section 2 (coding project) grading for the MCQ + Coding Project format
#   reschedule_classification — excuse classification (Claude outputs category only)
VALID_SLUGS = frozenset(
    {
        "assessment_generation",
        "retest_generation",
        "midterm_generation",
        "grading",
        "mcq_grading",
        "project_grading",
        "reschedule_classification",
        "curriculum_analysis",   # optional enrichment in CurriculumService.create()
    }
)


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (
        # Each (slug, version) pair must be unique.
        UniqueConstraint("slug", "version", name="uq_prompt_slug_version"),
        # Supports the common service-layer query:
        #   WHERE slug = ? AND is_active = true LIMIT 1
        Index("ix_prompt_templates_slug_active", "slug", "is_active"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # At most one row per slug should have is_active=True.
    # This invariant is enforced in the service layer, not with a partial
    # unique index (SQLAlchemy has no portable partial-index API for SQLite).
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow
    )
