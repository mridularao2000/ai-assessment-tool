from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy.orm import Session

from app.interfaces.llm import LLMInterface

if TYPE_CHECKING:
    from app.models.curriculum import Curriculum


class CurriculumService:
    """Handles curriculum creation, resource ingestion, and mastery state.

    Depends on:
      db  — SQLAlchemy session for all persistence
      llm — LLMInterface for curriculum analysis only
    """

    def __init__(self, db: Session, llm: LLMInterface) -> None:
        self.db = db
        self.llm = llm

    # ── Public methods ─────────────────────────────────────────────────────────

    def create(
        self,
        topic: str,
        target_completion_date: date,
        links: list[str],
        github_repos: list[str],
        notes: Optional[str],
        uploaded_files: list[tuple[str, bytes]],
    ) -> Curriculum:
        """Ingest all supplied resources and persist a new Curriculum.

        Steps:
          1. Create a Curriculum row with status=pending.
          2. For each resource, dispatch to the appropriate ingestor:
               - links       → url_scraper.fetch(url)
               - github_repos → github_ingestor.ingest(repo_url)
               - notes       → stored inline as raw_content
               - uploaded files → pdf_parser.extract(bytes) or
                                  markdown_ingestor.extract(bytes)
               Each produces a Resource row with raw_content set.
          3. Concatenate all raw_content values into Curriculum.extracted_content.
          4. Call llm.analyze_curriculum() with topic and extracted_content
             to obtain summary, key_topics, and complexity_level.
          5. Set Curriculum.status = ready.
          6. Commit and return the Curriculum.

        Returns:
            The persisted Curriculum with resources eagerly loaded.

        Raises:
            IngestionError: if any URL, repo, or file cannot be fetched or parsed.
            LLMValidationError: if curriculum analysis fails after all retries.
        """
        raise NotImplementedError

    def get(self, curriculum_id: str) -> Curriculum:
        """Fetch a Curriculum by ID with resources and assessments loaded.

        Raises:
            NotFoundError: if no Curriculum exists with the given ID.
        """
        raise NotImplementedError

    def mark_mastery(self, curriculum_id: str) -> None:
        """Record that the student has achieved mastery on this curriculum.

        Sets:
          Curriculum.mastery_achieved = True
          Curriculum.completed_at     = utcnow()
          Curriculum.status           = complete

        Called by the grade_submission_job after GradingService.grade()
        returns a score >= settings.mastery_threshold.

        Raises:
            NotFoundError: if no Curriculum exists with the given ID.
        """
        raise NotImplementedError
