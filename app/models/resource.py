import enum
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.curriculum import Curriculum


class ResourceType(str, enum.Enum):
    url = "url"
    pdf = "pdf"
    note = "note"
    github_repo = "github_repo"
    markdown = "markdown"


class Resource(Base):
    __tablename__ = "resources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    curriculum_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("curricula.id"),
        nullable=False,
        index=True,
    )
    type: Mapped[ResourceType] = mapped_column(Enum(ResourceType), nullable=False)
    # Populated by the appropriate ingestor (pdf_parser, url_scraper, etc.)
    # after the resource is submitted. Null until ingestion completes.
    raw_content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Relationships ─────────────────────────────────────────────────────────
    curriculum: Mapped["Curriculum"] = relationship(
        "Curriculum", back_populates="resources"
    )
