"""Shared test infrastructure for the AI Assessment System integration suite.

Provides:
  - In-memory SQLite engine (StaticPool) shared across all sessions in a test
  - FakeLLM / FakeLLMBelowThreshold / FakeLLMDenied — deterministic LLM stubs
  - FakeScheduler — in-memory scheduler that records every call
  - client fixture — FastAPI TestClient with all external boundaries replaced
  - DB seed helpers (plain functions, importable in test modules)
"""

import os

# Must run before ANY `app.*` import in this file (or anything it pulls
# in transitively, e.g. app.main -> app.database -> app.config). Settings
# is cached via @lru_cache on first call, and app.database.engine is a
# process-global singleton bound to database_url at import time — FastAPI's
# TestClient triggers app.main.lifespan on the first request even without
# a `with` block, and without this, that engine defaults to
# sqlite:///./assessment.db (the real tracked dev DB), so create_all()/
# seed_prompt_templates() would silently run against it instead of the
# isolated test_engine below. See app/config.py's run_schema_bootstrap.
os.environ["RUN_SCHEMA_BOOTSTRAP"] = "false"

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
    get_curriculum_service,
    get_curriculum_upload_service,
    get_email_service,
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
    MidtermGenerationRequest,
    MidtermGenerationResult,
    MidtermGradingRequest,
    MidtermGradingResult,
    LLMValidationError,
    MidtermRetestGenerationRequest,
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
from app.services.curriculum_service import CurriculumService
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.email_service import EmailService
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


class NoopEmailAdapter:
    """Safe no-op EmailInterface — does nothing, never touches the network."""

    def send_assessment_email(self, data) -> None:
        pass

    def send_reminder_email(self, data) -> None:
        pass

    def send_results_email(self, data) -> None:
        pass

    def send_syllabus_email(self, data) -> None:
        pass

    def send_transcript_email(self, data) -> None:
        pass

    def send_midterm_hold_reminder_email(self, data) -> None:
        pass


class RecordingEmailAdapter(NoopEmailAdapter):
    """Records every call instead of doing nothing, so tests can assert on
    exactly what was sent (recipients, content) without touching the network."""

    def __init__(self):
        self.assessment_calls = []
        self.reminder_calls = []
        self.results_calls = []
        self.syllabus_calls = []
        self.transcript_calls = []
        self.hold_reminder_calls = []

    def send_assessment_email(self, data) -> None:
        self.assessment_calls.append(data)

    def send_reminder_email(self, data) -> None:
        self.reminder_calls.append(data)

    def send_results_email(self, data) -> None:
        self.results_calls.append(data)

    def send_syllabus_email(self, data) -> None:
        self.syllabus_calls.append(data)

    def send_transcript_email(self, data) -> None:
        self.transcript_calls.append(data)

    def send_midterm_hold_reminder_email(self, data) -> None:
        self.hold_reminder_calls.append(data)


@pytest.fixture(autouse=True)
def _no_real_email_in_jobs(monkeypatch):
    """Prevent any test from ever sending a real email via Resend.

    app/jobs/{grade_submission_job,send_assessment_job,send_reminder_job,
    recheck_pending_midterms_job}.py each import `_email` as a module-level
    singleton directly from app.dependencies (`from app.dependencies import
    _email`), bypassing FastAPI's dependency-override system entirely. The
    `client` fixture's app.dependency_overrides never touches this — it
    only intercepts Depends()-injected services at the route layer.

    A test that invokes one of those job functions directly (rather than
    through an HTTP call whose scheduling is faked) would otherwise hit the
    REAL email adapter (GmailSMTPEmailAdapter as of the Resend->Gmail SMTP
    migration; previously ResendEmailAdapter) with REAL credentials from
    .env. This bit us once already (a retake-cap test that called
    grade_submission_job() directly mailed the real registered inbox), so
    this is a blanket default for every test going forward — individual
    tests may still monkeypatch a specific job's `_email` further (e.g. to
    assert on calls), which layers cleanly on top since monkeypatch undoes
    in LIFO order.
    """
    noop = NoopEmailAdapter()
    for module in (
        "app.jobs.grade_submission_job",
        "app.jobs.send_assessment_job",
        "app.jobs.send_reminder_job",
        "app.jobs.recheck_pending_midterms_job",
    ):
        monkeypatch.setattr(f"{module}._email", noop)


class _UnconfiguredLLMAdapter:
    """Autouse default for _llm in job modules that call the real LLM
    directly (grade_submission_job, send_assessment_job). Raises
    immediately and loudly — rather than a silent no-op — so a test that
    forgets to monkeypatch _llm to FakeLLM/FakeLLMBelowThreshold fails fast
    with a clear message, instead of either producing nonsense downstream
    or (the real risk) hitting the real, billed Anthropic API using the
    real key from .env.
    """

    def _boom(self, *args, **kwargs):
        raise RuntimeError(
            "Test invoked a job's real _llm without patching it to a Fake* "
            "LLM first — see _no_real_llm_in_jobs in conftest.py."
        )

    analyze_curriculum = _boom
    generate_assessment = _boom
    generate_retest = _boom
    generate_midterm = _boom
    grade_submission = _boom
    classify_reschedule_request = _boom


@pytest.fixture(autouse=True)
def _no_real_llm_in_jobs(monkeypatch):
    """Prevent any test from ever calling the real, billed Anthropic API.

    Same rationale as _no_real_email_in_jobs above, for `_llm` — jobs that
    import it as a module-level singleton bypass FastAPI's
    dependency-override system entirely. Unlike the email case this
    defaults to a loud failure, not a silent no-op, since a test relying on
    generated content needs an explicit Fake* LLM to produce something
    meaningful anyway.
    """
    unconfigured = _UnconfiguredLLMAdapter()
    for module in ("app.jobs.grade_submission_job", "app.jobs.send_assessment_job"):
        monkeypatch.setattr(f"{module}._llm", unconfigured)


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

    def generate_midterm(self, req: MidtermGenerationRequest) -> MidtermGenerationResult:
        return MidtermGenerationResult(
            part1_text="Part 1: small coding questions on cumulative material.",
            part1_rubric="Part 1 rubric — full marks for correct implementations.",
            part2_text="Part 2: defend your project's design decisions.",
            part2_rubric="Part 2 rubric — full marks for well-justified decisions.",
            duration_minutes=120,
        )

    def grade_submission(self, req: GradingRequest) -> GradingResult:
        return GradingResult(
            mastery_score=90.0,
            weak_areas=[],
            overall_feedback="Excellent understanding demonstrated.",
        )

    def grade_midterm_submission(self, req: MidtermGradingRequest) -> MidtermGradingResult:
        return MidtermGradingResult(
            part1_score=req.part1_max_marks * 0.9,
            part2_score=req.part2_max_marks * 0.9,
            weak_areas=[],
            overall_feedback="Strong grasp of cumulative material and clear project defense.",
        )

    def generate_midterm_retest(self, req: MidtermRetestGenerationRequest) -> MidtermGenerationResult:
        return MidtermGenerationResult(
            part1_text="Part 1 retest: focus on weak areas identified previously.",
            part1_rubric="Part 1 retest rubric — full marks for correcting weak areas.",
            part2_text="Part 2 retest: defend your project's design decisions again.",
            part2_rubric="Part 2 retest rubric — full marks for well-justified decisions.",
            duration_minutes=120,
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

    def grade_midterm_submission(self, req: MidtermGradingRequest) -> MidtermGradingResult:
        return MidtermGradingResult(
            part1_score=req.part1_max_marks * 0.7,
            part2_score=req.part2_max_marks * 0.7,
            weak_areas=["state management internals", "project architecture rationale"],
            overall_feedback="Needs improvement on both cumulative concepts and project defense.",
        )


class FakeLLMRetestFails(FakeLLMBelowThreshold):
    """Grades below threshold (like FakeLLMBelowThreshold), but generate_retest
    raises LLMValidationError — reproduces the real failure mode where the
    model exhausts its token budget on tool calls before producing retest
    content, to prove grade_submission_job's retest step is non-fatal."""

    def generate_retest(self, req: RetestGenerationRequest) -> AssessmentGenerationResult:
        raise LLMValidationError("No text block in response content: [...]")

    def generate_midterm_retest(self, req: MidtermRetestGenerationRequest) -> MidtermGenerationResult:
        raise LLMValidationError("No text block in response content: [...]")


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

    def override_get_email_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> EmailService:
        # Every route this client can reach must be safe from real Resend
        # calls, same reasoning as the LLM/scheduler fakes above — a route
        # added later that starts depending on get_email_service must not
        # silently start hitting the network just because this override
        # list wasn't updated for it.
        return EmailService(db, RecordingEmailAdapter())

    def override_get_curriculum_service(
        db: Annotated[Session, Depends(get_db)],
        svc: Annotated[SchedulerService, Depends(get_scheduler_service)],
    ) -> CurriculumService:
        # Was missing from this override list entirely — POST /curriculum/
        # was only ever exercised by constructing CurriculumService(fake)
        # directly at the service layer, never through this client fixture,
        # which is why the gap went unnoticed. The first route-level test
        # for it (or any future one) would otherwise silently hit the real
        # Anthropic API via app.dependencies._llm.
        return CurriculumService(db, fake_llm_instance, svc)

    def override_get_curriculum_upload_service(
        db: Annotated[Session, Depends(get_db)],
        svc: Annotated[SchedulerService, Depends(get_scheduler_service)],
    ) -> CurriculumUploadService:
        # Same gap as above, for POST /curriculum-uploads/ — real _email
        # (Resend/Gmail SMTP) instead of a fake, undetected because nothing
        # exercised this route through the client fixture yet either.
        return CurriculumUploadService(db, RecordingEmailAdapter(), svc)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_scheduler_service] = override_get_scheduler_service
    app.dependency_overrides[get_assessment_service] = override_get_assessment_service
    app.dependency_overrides[get_grading_service] = override_get_grading_service
    app.dependency_overrides[get_reschedule_service] = override_get_reschedule_service
    app.dependency_overrides[get_email_service] = override_get_email_service
    app.dependency_overrides[get_curriculum_service] = override_get_curriculum_service
    app.dependency_overrides[get_curriculum_upload_service] = override_get_curriculum_upload_service

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
        "curriculum_analysis",
        "retest_generation",
        "midterm_generation",
        "midterm_retest_generation",
        "grading",
        "midterm_grading",
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
    entry_type=None,
) -> Curriculum:
    curriculum = Curriculum(
        id=str(uuid.uuid4()),
        topic=topic,
        target_completion_date=target_completion_date or date(2026, 8, 1),
        extracted_content=extracted_content,
        status=status,
        entry_type=entry_type,
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
    attempt_number: int = 1,
    part1_text: str | None = None,
    part1_rubric: str | None = None,
    part2_text: str | None = None,
    part2_rubric: str | None = None,
) -> tuple[Assessment, str]:
    """Return (assessment, token). assessment.scheduled_job_ids is pre-populated."""
    assessment_id = str(uuid.uuid4())
    token = generate_submission_token(assessment_id)
    now = datetime.utcnow()
    assessment = Assessment(
        id=assessment_id,
        curriculum_id=curriculum.id,
        attempt_number=attempt_number,
        assessment_text="Explain the Python event loop in detail.",
        rubric="Full marks for: event loop, coroutines, await semantics.",
        part1_text=part1_text,
        part1_rubric=part1_rubric,
        part2_text=part2_text,
        part2_rubric=part2_rubric,
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
    part1_text_content: str | None = None,
) -> Submission:
    """Create a text submission and mark the assessment as submitted."""
    submission = Submission(
        id=str(uuid.uuid4()),
        assessment_id=assessment.id,
        submission_type=SubmissionType.text,
        text_content=text_content,
        part1_text_content=part1_text_content,
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
    part1_score: float | None = None,
    part2_score: float | None = None,
    score_earned: float | None = None,
    max_marks: float | None = None,
) -> Grade:
    """Create a grade and mark the assessment as completed."""
    grade = Grade(
        id=str(uuid.uuid4()),
        submission_id=submission.id,
        mastery_score=mastery_score,
        weak_areas=weak_areas if weak_areas is not None else [],
        overall_feedback=overall_feedback,
        part1_score=part1_score,
        part2_score=part2_score,
        score_earned=score_earned,
        max_marks=max_marks,
    )
    db.query(Assessment).filter(
        Assessment.id == submission.assessment_id
    ).update({"status": AssessmentStatus.completed})
    db.add(grade)
    db.commit()
    db.refresh(grade)
    return grade
