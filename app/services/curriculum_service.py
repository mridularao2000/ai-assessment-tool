from __future__ import annotations

# ── Orchestration note ────────────────────────────────────────────────────────
#
# CurriculumService.create() is the SINGLE orchestration point for the entire
# curriculum-creation pipeline:
#
#   1. Ingest resources (URLs, GitHub repos, files, notes)
#   2. Enrich extracted_content via LLM analysis (optional)
#   3. Delegate to AssessmentService.create_for_curriculum()
#        → generates assessment text via LLM
#        → commits curriculum + resources + assessment in ONE transaction
#          (a curriculum that fails assessment generation is never persisted)
#   4. Delegate to SchedulerService.schedule_assessment_jobs()
#        → creates APScheduler jobs (reminder, send, expire)
#        → persists job IDs to the Assessment row
#
# The route at POST /api/v1/curriculum/ calls this method and does nothing else.
# Assessment creation and scheduling MUST NOT be duplicated there.
#
# ─────────────────────────────────────────────────────────────────────────────

import logging
import uuid
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.exceptions import IngestionError, InvalidStateError, NotFoundError
from app.interfaces.llm import CurriculumAnalysisRequest, LLMError, LLMInterface
from app.models._utils import utcnow
from app.models.curriculum import Curriculum, CurriculumStatus
from app.models.prompt_template import PromptTemplate
from app.models.resource import Resource, ResourceType
from app.services.assessment_service import AssessmentService
from app.services.scheduler_service import SchedulerService

logger = logging.getLogger(__name__)


class CurriculumService:
    """Ingests resources, generates assessments, and manages mastery state.

    Depends on:
      db        — SQLAlchemy session for all persistence
      llm       — LLMInterface for curriculum analysis and assessment generation
      scheduler — SchedulerService for APScheduler job registration
    """

    def __init__(
        self,
        db: Session,
        llm: LLMInterface,
        scheduler: SchedulerService,
    ) -> None:
        self.db = db
        self.llm = llm
        self._scheduler = scheduler

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
        """Full pipeline: ingest → enrich → create assessment → schedule jobs.

        Transaction safety:
          No DB commit is issued for the curriculum itself.
          AssessmentService.create_for_curriculum() issues the single commit
          that atomically writes curriculum + resources + assessment together.
          If LLM assessment generation fails, nothing is persisted.

        Returns:
            The persisted and refreshed Curriculum.

        Raises:
            InvalidStateError: topic is blank, or target_completion_date is past.
            IngestionError: any URL, repo, or file cannot be fetched/parsed.
            LLMValidationError: assessment generation fails after all retries.
            SchedulerNotRunningError: APScheduler not yet started.
        """
        # ── 1. Validate ────────────────────────────────────────────────────────
        topic = topic.strip()
        if not topic:
            raise InvalidStateError("topic must not be empty.")
        if target_completion_date < date.today():
            raise InvalidStateError(
                "target_completion_date must be today or a future date."
            )

        # Generate the ID in Python so it is available immediately on the ORM
        # object. Curriculum.id uses a SQLAlchemy column default (evaluated at
        # flush time), so without an explicit id the value stays None until
        # flush — which never fires automatically because autoflush=False.
        curriculum = Curriculum(
            id=str(uuid.uuid4()),
            topic=topic,
            target_completion_date=target_completion_date,
            status=CurriculumStatus.pending,
        )
        self.db.add(curriculum)
        if curriculum.id is None:
            raise RuntimeError("Curriculum ID not generated")

        # ── 2. Ingest resources ────────────────────────────────────────────────
        content_parts: list[str] = []

        if notes and notes.strip():
            self.db.add(Resource(
                curriculum_id=curriculum.id,
                type=ResourceType.note,
                source_ref="notes",
                raw_content=notes,
            ))
            content_parts.append(notes)

        for url in links:
            text = self._fetch_url(url)
            self.db.add(Resource(
                curriculum_id=curriculum.id,
                type=ResourceType.url,
                source_ref=url,
                raw_content=text,
            ))
            content_parts.append(text)

        for repo_url in github_repos:
            text = self._fetch_github_repo(repo_url)
            self.db.add(Resource(
                curriculum_id=curriculum.id,
                type=ResourceType.github_repo,
                source_ref=repo_url,
                raw_content=text,
            ))
            content_parts.append(text)

        for filename, file_bytes in uploaded_files:
            text, resource_type = self._process_file(filename, file_bytes)
            self.db.add(Resource(
                curriculum_id=curriculum.id,
                type=resource_type,
                source_ref=filename,
                raw_content=text,
            ))
            content_parts.append(text)

        curriculum.extracted_content = "\n\n".join(content_parts)

        # ── 3. Optional LLM enrichment ─────────────────────────────────────────
        # Appends structured analysis to extracted_content so that downstream
        # assessment generation has richer context. Non-fatal: skipped if the
        # curriculum_analysis template has not been seeded or the call fails.
        curriculum.status = CurriculumStatus.analyzing
        try:
            template = self._fetch_prompt("curriculum_analysis")
            analysis = self.llm.analyze_curriculum(
                CurriculumAnalysisRequest(
                    topic=topic,
                    curriculum_content=curriculum.extracted_content,
                    prompt_template_body=template.body,
                )
            )
            curriculum.extracted_content += (
                f"\n\n=== CURRICULUM ANALYSIS ===\n"
                f"Summary: {analysis.summary}\n"
                f"Key Topics: {', '.join(analysis.key_topics)}\n"
                f"Complexity: {analysis.complexity_level}\n"
                f"Estimated Study Hours: {analysis.estimated_study_hours}"
            )
        except (NotFoundError, LLMError):
            logger.info(
                "Curriculum analysis enrichment skipped for topic %r "
                "(template missing or LLM error)",
                topic,
            )

        curriculum.status = CurriculumStatus.ready

        # ── 4. Build assessment (pure factory — no DB side effects) ───────────────
        # create_for_curriculum() returns an Assessment ORM object with all
        # fields set but NOT yet added to the session or committed.
        assessment = AssessmentService(self.db, self.llm).create_for_curriculum(
            curriculum
        )
        self.db.add(assessment)

        # Capture scheduling fields now — db.commit() below expires all
        # attributes on session-tracked objects, so reading them afterwards
        # would trigger lazy SELECTs that are unnecessary.
        assessment_id = assessment.id
        scheduled_at = assessment.scheduled_at
        reminder_at = assessment.reminder_at
        due_date = assessment.due_date

        # ── 5. Single commit: curriculum + resources + assessment ──────────────
        # This is the ONLY db.commit() in the creation pipeline.
        # If assessment generation (step 4) raised, nothing is committed.
        self.db.commit()
        self.db.refresh(curriculum)

        # ── 6. Schedule APScheduler jobs ───────────────────────────────────────
        # Called after commit so SchedulerService.schedule_assessment_jobs()
        # can find the assessment row via db.get() when it persists job IDs.
        self._scheduler.schedule_assessment_jobs(
            assessment_id=assessment_id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
        )

        return curriculum

    def get(self, curriculum_id: str) -> Curriculum:
        """Fetch a Curriculum by ID with resources and assessments loaded.

        Raises:
            NotFoundError: if no Curriculum exists with the given ID.
        """
        curriculum = (
            self.db.query(Curriculum)
            .options(
                joinedload(Curriculum.resources),
                joinedload(Curriculum.assessments),
            )
            .filter(Curriculum.id == curriculum_id)
            .first()
        )
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")
        return curriculum

    def mark_mastery(self, curriculum_id: str) -> None:
        """Record mastery achievement on this curriculum.

        Sets mastery_achieved=True, completed_at=now, status=complete.

        Raises:
            NotFoundError: if no Curriculum exists with the given ID.
        """
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")
        curriculum.mastery_achieved = True
        curriculum.completed_at = utcnow()
        curriculum.status = CurriculumStatus.complete
        self.db.commit()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_prompt(self, slug: str) -> PromptTemplate:
        template = (
            self.db.query(PromptTemplate)
            .filter(PromptTemplate.slug == slug, PromptTemplate.is_active.is_(True))
            .first()
        )
        if template is None:
            raise NotFoundError(f"No active {slug!r} prompt template found.")
        return template

    def _fetch_url(self, url: str) -> str:
        try:
            import httpx
            from bs4 import BeautifulSoup

            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception as exc:
            raise IngestionError(f"Failed to fetch URL {url!r}: {exc}") from exc

    def _fetch_github_repo(self, repo_url: str) -> str:
        try:
            from app.ingestors.github_ingestor import fetch_repo_content
            return fetch_repo_content(repo_url)
        except ImportError:
            # Module not yet implemented — fall back to plain URL fetch.
            return self._fetch_url(repo_url)
        except Exception as exc:
            raise IngestionError(
                f"Failed to fetch GitHub repo {repo_url!r}: {exc}"
            ) from exc

    def _process_file(
        self, filename: str, file_bytes: bytes
    ) -> tuple[str, ResourceType]:
        if filename.lower().endswith(".pdf"):
            return self._extract_pdf(filename, file_bytes), ResourceType.pdf
        try:
            return file_bytes.decode("utf-8", errors="replace"), ResourceType.markdown
        except Exception as exc:
            raise IngestionError(
                f"Failed to decode file {filename!r}: {exc}"
            ) from exc

    def _extract_pdf(self, filename: str, file_bytes: bytes) -> str:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n\n".join(page.get_text() for page in doc)
        except Exception as exc:
            raise IngestionError(
                f"Failed to extract PDF {filename!r}: {exc}"
            ) from exc
