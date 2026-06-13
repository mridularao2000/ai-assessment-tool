"""Shared test infrastructure for the AI Assessment System integration suite.

Provides:
  - In-memory SQLite engine (StaticPool) shared across all sessions in a test
  - FakeLLM / FakeLLMBelowThreshold / FakeLLMDenied — deterministic LLM stubs
  - FakeScheduler — in-memory scheduler that records every call
  - client fixture — FastAPI TestClient with all external boundaries replaced
  - DB seed helpers (plain functions, importable in test modules)
"""

import unittest.mock
import uuid
from datetime import date, datetime, timedelta
from typing import Annotated, Generator

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.dependencies import (
    get_assessment_service,
    get_grading_service,
    get_reschedule_service,
    get_scheduler_service,
)
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    GradingRequest,
    GradingResult,
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RetestGenerationRequest,
)
from app.interfaces.scheduler import AssessmentJobIds
from app.main import app
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumStatus
from app.models.grade import Grade
from app.models.prompt_template import PromptTemplate
from app.models.submission import Submission, SubmissionType
from app.services.assessment_service import AssessmentService
from app.services.grading_service import GradingService
from app.services.reschedule_service import RescheduleService
from app.services.scheduler_service import SchedulerService
from app.utils.token_auth import generate_submission_token


# ── In-memory test database ───────────────────────────────────────────────────
# StaticPool ensures all sessions reuse the same in-memory connection, so
# data committed by one session is immediately visible to another.

TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def _tables():
    """Create all tables before each test; drop after."""
    Base.metadata.create_all(test_engine)
    yield
    Base.metadata.drop_all(test_engine)


@pytest.fixture(autouse=True)
def _no_apscheduler():
    """Prevent the real APScheduler from starting during tests.

    The module-level _scheduler_adapter in dependencies.py would otherwise
    try to start BackgroundScheduler with the production SQLite job store.
    """
    with (
        unittest.mock.patch("app.adapters.apscheduler_adapter.APSchedulerAdapter.start"),
        unittest.mock.patch("app.adapters.apscheduler_adapter.APSchedulerAdapter.shutdown"),
    ):
        yield


@pytest.fixture
def db(_tables) -> Generator[Session, None, None]:
    """Test session for seeding data and asserting DB state."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── LLM fakes ─────────────────────────────────────────────────────────────────


class FakeLLM:
    """Deterministic LLM stub: all calls return valid, predictable results.

    Used as the default LLM in the client fixture.
    Grading always returns mastery_score=90.0 (above the default 85.0 threshold).
    Reschedule always classifies as 'medical' (approved).
    """

    def analyze_curriculum(self, req: CurriculumAnalysisRequest) -> CurriculumAnalysisResult:
        return CurriculumAnalysisResult(
            summary="Test curriculum summary.",
            key_topics=["async", "await", "event loop"],
            complexity_level="intermediate",
            estimated_study_hours=10.0,
        )

    def generate_assessment(self, req: AssessmentGenerationRequest) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text="Explain the Python event loop and async/await.",
            rubric="Award full marks for: event loop mechanics, coroutine definition, await semantics.",
            duration_minutes=60,
        )

    def generate_retest(self, req: RetestGenerationRequest) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text="Retest: focus on weak areas identified previously.",
            rubric="Retest rubric — full marks for correcting weak areas.",
            duration_minutes=45,
        )

    def grade_submission(self, req: GradingRequest) -> GradingResult:
        return GradingResult(
            mastery_score=90.0,
            weak_areas=[],
            overall_feedback="Excellent understanding demonstrated.",
        )

    def classify_reschedule_request(
        self, req: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        return RescheduleClassificationResult(
            category="medical",
            reasoning="User cited a confirmed medical appointment.",
        )


class FakeLLMBelowThreshold(FakeLLM):
    """Grades at 70.0 — below the default mastery threshold (85.0)."""

    def grade_submission(self, req: GradingRequest) -> GradingResult:
        return GradingResult(
            mastery_score=70.0,
            weak_areas=["event loop internals", "coroutine lifecycle"],
            overall_feedback="Needs improvement on core concurrency concepts.",
        )


class FakeLLMDenied(FakeLLM):
    """Classifies all reschedule requests as 'procrastination' (denied)."""

    def classify_reschedule_request(
        self, req: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        return RescheduleClassificationResult(
            category="procrastination",
            reasoning="No legitimate reason for reschedule was provided.",
        )


# ── Fake scheduler ────────────────────────────────────────────────────────────


class FakeScheduler:
    """In-memory scheduler stub. Records every call; never starts a real thread."""

    def __init__(self):
        self.schedule_assessment_jobs_calls: list[dict] = []
        self.schedule_grade_job_calls: list[str] = []
        self.cancel_jobs_calls: list[AssessmentJobIds] = []
        self.reschedule_calls: list[dict] = []

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def schedule_assessment_jobs(
        self,
        assessment_id: str,
        scheduled_at: datetime,
        reminder_at: datetime,
        due_date: datetime,
    ) -> AssessmentJobIds:
        self.schedule_assessment_jobs_calls.append(
            dict(assessment_id=assessment_id, scheduled_at=scheduled_at)
        )
        return AssessmentJobIds(
            send_reminder=f"send_reminder_{assessment_id}",
            send_assessment=f"assessment_{assessment_id}",
            expire=f"expire_{assessment_id}",
        )

    def schedule_grade_job(self, submission_id: str) -> str:
        self.schedule_grade_job_calls.append(submission_id)
        return f"grade_{submission_id}"

    def cancel_jobs_for_assessment(self, job_ids: AssessmentJobIds) -> None:
        self.cancel_jobs_calls.append(job_ids)

    def reschedule_assessment(
        self,
        assessment_id: str,
        new_scheduled_at: datetime,
        new_reminder_at: datetime,
        new_due_date: datetime,
        existing_job_ids: AssessmentJobIds,
    ) -> AssessmentJobIds:
        self.reschedule_calls.append(
            dict(assessment_id=assessment_id, new_scheduled_at=new_scheduled_at)
        )
        return AssessmentJobIds(
            send_reminder=f"send_reminder_{assessment_id}_v2",
            send_assessment=f"assessment_{assessment_id}_v2",
            expire=f"expire_{assessment_id}_v2",
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def _make_client(fake_scheduler_instance, fake_llm_instance):
    """Build a TestClient with all external boundaries replaced.

    - DB → in-memory SQLite via TestSessionLocal
    - Scheduler → FakeScheduler (records calls, no threads)
    - LLM → provided fake (deterministic responses)
    """

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    def override_get_scheduler_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> SchedulerService:
        return SchedulerService(db, fake_scheduler_instance)

    def override_get_assessment_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> AssessmentService:
        return AssessmentService(db, fake_llm_instance)

    def override_get_grading_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> GradingService:
        return GradingService(db, fake_llm_instance)

    def override_get_reschedule_service(
        db: Annotated[Session, Depends(get_db)],
        svc: Annotated[SchedulerService, Depends(get_scheduler_service)],
    ) -> RescheduleService:
        return RescheduleService(db, fake_llm_instance, svc)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_scheduler_service] = override_get_scheduler_service
    app.dependency_overrides[get_assessment_service] = override_get_assessment_service
    app.dependency_overrides[get_grading_service] = override_get_grading_service
    app.dependency_overrides[get_reschedule_service] = override_get_reschedule_service

    return TestClient(app)


@pytest.fixture
def client(fake_scheduler, fake_llm) -> Generator[TestClient, None, None]:
    """TestClient with FakeLLM (grade=90.0, reschedule=medical/approved)."""
    with _make_client(fake_scheduler, fake_llm) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def denied_client(fake_scheduler) -> Generator[TestClient, None, None]:
    """TestClient with FakeLLMDenied (reschedule classified as procrastination)."""
    with _make_client(fake_scheduler, FakeLLMDenied()) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def below_threshold_client(fake_scheduler) -> Generator[TestClient, None, None]:
    """TestClient with FakeLLMBelowThreshold (grade=70.0, below mastery_threshold)."""
    with _make_client(fake_scheduler, FakeLLMBelowThreshold()) as c:
        yield c
    app.dependency_overrides.clear()


# ── DB seed helpers (importable plain functions) ───────────────────────────────


def seed_prompt_templates(db: Session) -> None:
    """Insert one active PromptTemplate for each required slug."""
    for slug in (
        "assessment_generation",
        "retest_generation",
        "grading",
        "reschedule_classification",
    ):
        db.add(
            PromptTemplate(
                id=str(uuid.uuid4()),
                slug=slug,
                version="1.0",
                body=f"System prompt body for {slug}.",
                is_active=True,
            )
        )
    db.commit()


def make_curriculum(
    db: Session,
    *,
    topic: str = "Python async programming",
    target_completion_date: date | None = None,
    status: CurriculumStatus = CurriculumStatus.ready,
    extracted_content: str = "Comprehensive notes on Python async/await and the event loop.",
) -> Curriculum:
    curriculum = Curriculum(
        id=str(uuid.uuid4()),
        topic=topic,
        target_completion_date=target_completion_date or date(2026, 8, 1),
        extracted_content=extracted_content,
        status=status,
    )
    db.add(curriculum)
    db.commit()
    db.refresh(curriculum)
    return curriculum


def make_assessment(
    db: Session,
    curriculum: Curriculum,
    *,
    status: AssessmentStatus = AssessmentStatus.active,
    due_offset_days: int = 7,
) -> tuple[Assessment, str]:
    """Return (assessment, token). assessment.scheduled_job_ids is pre-populated."""
    assessment_id = str(uuid.uuid4())
    token = generate_submission_token(assessment_id)
    now = datetime.utcnow()
    assessment = Assessment(
        id=assessment_id,
        curriculum_id=curriculum.id,
        attempt_number=1,
        assessment_text="Explain the Python event loop in detail.",
        rubric="Full marks for: event loop, coroutines, await semantics.",
        duration_minutes=60,
        scheduled_at=now + timedelta(days=2),
        reminder_at=now + timedelta(hours=24),
        due_date=now + timedelta(days=due_offset_days),
        status=status,
        submission_token=token,
        scheduled_job_ids={
            "send_reminder": f"send_reminder_{assessment_id}",
            "send_assessment": f"assessment_{assessment_id}",
            "expire": f"expire_{assessment_id}",
        },
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment, token


def make_submission(
    db: Session,
    assessment: Assessment,
    *,
    text_content: str = "Async/await enables concurrent I/O without OS threads.",
) -> Submission:
    """Create a text submission and mark the assessment as submitted."""
    submission = Submission(
        id=str(uuid.uuid4()),
        assessment_id=assessment.id,
        submission_type=SubmissionType.text,
        text_content=text_content,
    )
    assessment.status = AssessmentStatus.submitted
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def make_grade(
    db: Session,
    submission: Submission,
    *,
    mastery_score: float = 90.0,
    weak_areas: list | None = None,
    overall_feedback: str = "Well done.",
) -> Grade:
    """Create a grade and mark the assessment as completed."""
    grade = Grade(
        id=str(uuid.uuid4()),
        submission_id=submission.id,
        mastery_score=mastery_score,
        weak_areas=weak_areas if weak_areas is not None else [],
        overall_feedback=overall_feedback,
    )
    db.query(Assessment).filter(
        Assessment.id == submission.assessment_id
    ).update({"status": AssessmentStatus.completed})
    db.add(grade)
    db.commit()
    db.refresh(grade)
    return grade
