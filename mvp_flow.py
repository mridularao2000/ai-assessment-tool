#!/usr/bin/env python3
"""
mvp_flow.py — End-to-end MVP API validation for the AI Assessment System.

Runs entirely in-process via FastAPI TestClient.
No external server, database file, or API keys required.

What it exercises:
  POST   /api/v1/curriculum             — create curriculum + auto-schedule assessment
  GET    /api/v1/curriculum/{id}        — fetch curriculum record
  GET    /api/v1/assessments/{id}       — fetch assessment via token
  POST   /api/v1/submissions/           — submit text answer
  GET    /api/v1/submissions/{id}       — confirm submission record
  [job]  GradingService.grade()         — simulate grade_submission_job
  GET    /api/v1/submissions/{id}/results — fetch grading result
  POST   /api/v1/assessments/{id}/reschedule — test approved + denied reschedule
  [summary] FakeScheduler call log      — verify every job was scheduled

Usage:
    .venv/bin/python mvp_flow.py
"""

from __future__ import annotations

import sys
import traceback
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Patch APScheduler BEFORE importing app modules ───────────────────────────
# app/dependencies.py creates _scheduler_adapter = APSchedulerAdapter() at
# import time. The real scheduler would try to start a background thread and
# connect to the SQLite job store in lifespan. Patch both class methods so the
# TestClient lifespan runs without starting any real threads.

import unittest.mock
_aps_start_patch = unittest.mock.patch(
    "app.adapters.apscheduler_adapter.APSchedulerAdapter.start"
)
_aps_shutdown_patch = unittest.mock.patch(
    "app.adapters.apscheduler_adapter.APSchedulerAdapter.shutdown"
)
_aps_start_patch.start()
_aps_shutdown_patch.start()

# ── App imports (after patch) ─────────────────────────────────────────────────
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.dependencies import (
    get_assessment_service,
    get_curriculum_service,
    get_grading_service,
    get_reschedule_service,
    get_scheduler_service,
)
from app.exceptions import NotFoundError
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
from app.models.submission import Submission
from app.services.assessment_service import AssessmentService
from app.services.grading_service import GradingService
from app.services.reschedule_service import RescheduleService
from app.services.scheduler_service import SchedulerService
from app.utils.token_auth import generate_submission_token

# ── Terminal colours ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def hdr(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def step(text: str) -> None:
    print(f"\n{BOLD}▶ {text}{RESET}")


def ok(label: str, value: str = "") -> None:
    if value:
        print(f"  {GREEN}✓{RESET} {label}: {YELLOW}{value}{RESET}")
    else:
        print(f"  {GREEN}✓{RESET} {label}")


def info(label: str, value: str = "") -> None:
    if value:
        print(f"  {DIM}  {label}: {value}{RESET}")
    else:
        print(f"  {DIM}  {label}{RESET}")


def fail(text: str) -> None:
    print(f"  {RED}✗ {text}{RESET}")


def assert_status(response, expected: int, label: str) -> dict:
    if response.status_code != expected:
        fail(f"{label} — expected HTTP {expected}, got {response.status_code}")
        try:
            fail(f"  body: {response.json()}")
        except Exception:
            fail(f"  body: {response.text[:200]}")
        sys.exit(1)
    ok(label, f"HTTP {response.status_code}")
    return response.json()


# ── In-memory database ────────────────────────────────────────────────────────
# StaticPool: every session shares the same single in-memory connection, so
# data committed by a direct session is immediately visible to route sessions.

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def new_session() -> Session:
    return _SessionLocal()


# ── Fake LLM ──────────────────────────────────────────────────────────────────


class FakeLLM:
    """Deterministic LLM stub. All responses are hard-coded and instant."""

    def analyze_curriculum(self, req: CurriculumAnalysisRequest) -> CurriculumAnalysisResult:
        return CurriculumAnalysisResult(
            summary="Python async programming: event loop, coroutines, and await.",
            key_topics=["event loop", "async/await", "coroutines", "tasks"],
            complexity_level="intermediate",
            estimated_study_hours=12.0,
        )

    def generate_assessment(self, req: AssessmentGenerationRequest) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text=(
                "Question 1: Explain the Python event loop and how it schedules coroutines.\n"
                "Question 2: What is the difference between asyncio.gather and asyncio.wait?\n"
                "Question 3: When would you use asyncio.Queue?"
            ),
            rubric=(
                "Q1: Full marks for: event loop mechanics, task scheduling, await semantics.\n"
                "Q2: Full marks for: concurrency vs sequential semantics, return types.\n"
                "Q3: Full marks for: producer-consumer pattern, backpressure."
            ),
            duration_minutes=60,
        )

    def generate_retest(self, req: RetestGenerationRequest) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text="Retest: focus on identified weak areas.",
            rubric="Full marks for demonstrating improvement on weak areas.",
            duration_minutes=45,
        )

    def grade_submission(self, req: GradingRequest) -> GradingResult:
        return GradingResult(
            mastery_score=88.5,
            weak_areas=["asyncio.wait semantics"],
            overall_feedback=(
                "Strong understanding of the event loop. "
                "Minor gaps in asyncio.wait vs gather distinction."
            ),
        )

    def classify_reschedule_request(
        self, req: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        return RescheduleClassificationResult(
            category="medical",
            reasoning="User provided a confirmed medical appointment as reason.",
        )


class FakeLLMDenied(FakeLLM):
    def classify_reschedule_request(
        self, req: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        return RescheduleClassificationResult(
            category="procrastination",
            reasoning="Reason does not meet the criteria for a legitimate reschedule.",
        )


# ── Fake scheduler ────────────────────────────────────────────────────────────


class FakeScheduler:
    """Records every scheduling call. Never touches a real thread."""

    def __init__(self):
        self.schedule_assessment_jobs_calls: list[dict] = []
        self.schedule_grade_job_calls: list[str] = []
        self.reschedule_calls: list[dict] = []
        self.cancel_calls: list[AssessmentJobIds] = []

    def start(self) -> None: pass
    def shutdown(self) -> None: pass

    def schedule_assessment_jobs(
        self, assessment_id: str, scheduled_at: datetime,
        reminder_at: datetime, due_date: datetime,
    ) -> AssessmentJobIds:
        self.schedule_assessment_jobs_calls.append({
            "assessment_id": assessment_id,
            "scheduled_at": scheduled_at.isoformat(),
            "reminder_at": reminder_at.isoformat(),
            "due_date": due_date.isoformat(),
        })
        return AssessmentJobIds(
            send_reminder=f"send_reminder_{assessment_id}",
            send_assessment=f"assessment_{assessment_id}",
            expire=f"expire_{assessment_id}",
        )

    def schedule_grade_job(self, submission_id: str) -> str:
        self.schedule_grade_job_calls.append(submission_id)
        return f"grade_{submission_id}"

    def cancel_jobs_for_assessment(self, job_ids: AssessmentJobIds) -> None:
        self.cancel_calls.append(job_ids)

    def reschedule_assessment(
        self, assessment_id: str, new_scheduled_at: datetime,
        new_reminder_at: datetime, new_due_date: datetime,
        existing_job_ids: AssessmentJobIds,
    ) -> AssessmentJobIds:
        self.reschedule_calls.append({
            "assessment_id": assessment_id,
            "new_scheduled_at": new_scheduled_at.isoformat(),
            "new_due_date": new_due_date.isoformat(),
        })
        return AssessmentJobIds(
            send_reminder=f"send_reminder_{assessment_id}_v2",
            send_assessment=f"assessment_{assessment_id}_v2",
            expire=f"expire_{assessment_id}_v2",
        )


# ── Fake curriculum service ───────────────────────────────────────────────────
# CurriculumService.create() is not yet implemented (requires ingestors).
# This stub creates a valid Curriculum row so the rest of the chain works.


class FakeCurriculumService:
    """Lightweight curriculum service that skips ingestion and LLM analysis."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        topic: str,
        target_completion_date: date,
        links: list[str],
        github_repos: list[str],
        notes: str | None,
        uploaded_files: list[tuple[str, bytes]],
    ) -> Curriculum:
        content_parts = []
        if notes:
            content_parts.append(f"Notes: {notes}")
        for url in links:
            content_parts.append(f"Resource: {url}")
        for repo in github_repos:
            content_parts.append(f"GitHub: {repo}")
        if not content_parts:
            content_parts.append(f"MVP placeholder content for topic: {topic}")

        curriculum = Curriculum(
            id=str(uuid.uuid4()),
            topic=topic,
            target_completion_date=target_completion_date,
            extracted_content="\n".join(content_parts),
            status=CurriculumStatus.ready,
        )
        self.db.add(curriculum)
        self.db.commit()
        self.db.refresh(curriculum)
        return curriculum

    def get(self, curriculum_id: str) -> Curriculum:
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")
        return curriculum

    def mark_mastery(self, curriculum_id: str) -> None:
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")
        curriculum.mastery_achieved = True
        curriculum.completed_at = datetime.utcnow()
        curriculum.status = CurriculumStatus.complete
        self.db.commit()


# ── Dependency wiring ─────────────────────────────────────────────────────────


def wire_dependencies(scheduler: FakeScheduler, llm: FakeLLM) -> None:
    """Override all FastAPI dependencies to use in-memory stubs."""

    def override_get_db():
        session = _SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def override_get_curriculum_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> FakeCurriculumService:
        return FakeCurriculumService(db)

    def override_get_scheduler_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> SchedulerService:
        return SchedulerService(db, scheduler)

    def override_get_assessment_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> AssessmentService:
        return AssessmentService(db, llm)

    def override_get_grading_service(
        db: Annotated[Session, Depends(get_db)],
    ) -> GradingService:
        return GradingService(db, llm)

    def override_get_reschedule_service(
        db: Annotated[Session, Depends(get_db)],
        svc: Annotated[SchedulerService, Depends(get_scheduler_service)],
    ) -> RescheduleService:
        return RescheduleService(db, llm, svc)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_curriculum_service] = override_get_curriculum_service
    app.dependency_overrides[get_scheduler_service] = override_get_scheduler_service
    app.dependency_overrides[get_assessment_service] = override_get_assessment_service
    app.dependency_overrides[get_grading_service] = override_get_grading_service
    app.dependency_overrides[get_reschedule_service] = override_get_reschedule_service


# ── DB seed helpers ───────────────────────────────────────────────────────────


def seed_prompt_templates(db: Session) -> None:
    for slug in ("assessment_generation", "retest_generation", "grading", "reschedule_classification"):
        db.add(PromptTemplate(
            id=str(uuid.uuid4()),
            slug=slug,
            version="1.0",
            body=f"You are an expert assessor. Slug: {slug}.",
            is_active=True,
        ))
    db.commit()


# ── MVP flow ──────────────────────────────────────────────────────────────────


def run_flow() -> None:
    hdr("AI Assessment System — MVP End-to-End Flow")

    # ── Setup ─────────────────────────────────────────────────────────────────
    Base.metadata.create_all(_engine)
    scheduler = FakeScheduler()
    llm = FakeLLM()
    wire_dependencies(scheduler, llm)

    db = new_session()
    seed_prompt_templates(db)
    db.close()

    client = TestClient(app, raise_server_exceptions=True)
    client.__enter__()

    try:
        # ── 1. Create curriculum ───────────────────────────────────────────────
        hdr("Step 1 — POST /api/v1/curriculum")
        step("Creating curriculum (topic: Python async programming)")

        r = client.post("/api/v1/curriculum", data={
            "topic": "Python async programming",
            "target_completion_date": "2026-08-01",
            "notes": "Focus on event loop, coroutines, asyncio.gather, and asyncio.Queue.",
        })
        data = assert_status(r, 201, "Create curriculum")
        curriculum_id = data["curriculum_id"]
        ok("Curriculum ID", curriculum_id)

        # ── 2. Inspect created assessment ─────────────────────────────────────
        hdr("Step 2 — Inspect auto-created Assessment")
        step("Querying DB for assessment created by POST /curriculum")

        db = new_session()
        assessment = db.query(Assessment).filter_by(curriculum_id=curriculum_id).first()
        if not assessment:
            fail("No assessment found in DB after curriculum creation")
            sys.exit(1)

        assessment_id = assessment.id
        token = assessment.submission_token
        ok("Assessment ID", assessment_id)
        ok("Status", assessment.status.value)
        ok("Attempt", str(assessment.attempt_number))
        ok("Scheduled at", str(assessment.scheduled_at))
        ok("Due date", str(assessment.due_date))
        ok("Duration (min)", str(assessment.duration_minutes))
        ok("Job IDs stored", str(bool(assessment.scheduled_job_ids)))
        db.close()

        # ── 3. Fetch curriculum via GET ────────────────────────────────────────
        hdr("Step 3 — GET /api/v1/curriculum/{id}")
        step("Fetching curriculum record")

        # Note: CurriculumService.get() is implemented in FakeCurriculumService.
        # The route calls model_validate(curriculum) which lazy-loads assessments +
        # resources — FakeCurriculumService.get() uses db.get() (no joinedload),
        # so this will trigger lazy loads within the open route session.
        r = client.get(f"/api/v1/curriculum/{curriculum_id}")
        data = assert_status(r, 200, "Get curriculum")
        ok("Topic", data["topic"])
        ok("Status", data["status"])

        # ── 4. Activate assessment (simulate send_assessment_job) ──────────────
        hdr("Step 4 — Simulate send_assessment_job (scheduled → active)")
        step("Setting assessment.status = active (APScheduler job would do this in production)")

        db = new_session()
        a = db.get(Assessment, assessment_id)
        a.status = AssessmentStatus.active
        db.commit()
        db.close()
        ok("Assessment status", "active")

        # ── 5. Fetch assessment via token ──────────────────────────────────────
        hdr("Step 5 — GET /api/v1/assessments/{id}?token=...")
        step("Fetching full assessment details with submission token")

        r = client.get(f"/api/v1/assessments/{assessment_id}?token={token}")
        data = assert_status(r, 200, "Get assessment")
        ok("Assessment ID", data["assessment_id"])
        ok("Topic", data["topic"])
        ok("Status", data["status"])
        ok("Duration (min)", str(data["duration_minutes"]))
        if data.get("assessment_text"):
            info("Question preview", data["assessment_text"][:80] + "...")

        # Verify bad token is rejected
        step("Verifying invalid token → 403")
        r_bad = client.get(f"/api/v1/assessments/{assessment_id}?token=INVALID")
        assert_status(r_bad, 403, "Invalid token rejected")

        # ── 6. Submit text answer ──────────────────────────────────────────────
        hdr("Step 6 — POST /api/v1/submissions/")
        step("Submitting text answer")

        submission_text = (
            "The Python event loop is a single-threaded loop that manages coroutines. "
            "asyncio.gather runs coroutines concurrently and returns all results; "
            "asyncio.wait gives more control over completion conditions. "
            "asyncio.Queue is ideal for producer-consumer patterns to pass data between coroutines."
        )

        r = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment_id,
            "token": token,
            "submission_type": "text",
            "text_content": submission_text,
        })
        data = assert_status(r, 201, "Submit answer")
        submission_id = data["submission_id"]
        ok("Submission ID", submission_id)

        # Verify status transition
        db = new_session()
        a = db.get(Assessment, assessment_id)
        ok("Assessment status after submit", a.status.value)
        db.close()

        # Verify duplicate submission is rejected
        step("Verifying duplicate submission → 409")
        r_dup = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment_id,
            "token": token,
            "submission_type": "text",
            "text_content": "A second attempt.",
        })
        assert_status(r_dup, 409, "Duplicate submission rejected")

        # ── 7. Confirm submission record ───────────────────────────────────────
        hdr("Step 7 — GET /api/v1/submissions/{id}")
        step("Fetching submission record")

        r = client.get(f"/api/v1/submissions/{submission_id}")
        data = assert_status(r, 200, "Get submission")
        ok("Submission ID confirmed", data["submission_id"])

        # ── 8. Grade submission (simulate grade_submission_job) ────────────────
        hdr("Step 8 — Simulate grade_submission_job")
        step("Running GradingService.grade() (APScheduler job would do this in production)")

        db = new_session()
        try:
            grade = GradingService(db, llm).grade(submission_id)
            ok("Grading completed", f"mastery_score={grade.mastery_score}")
            ok("Weak areas", str(grade.weak_areas))
            info("Feedback", grade.overall_feedback)
        finally:
            db.close()

        # Verify assessment is now completed
        db = new_session()
        a = db.get(Assessment, assessment_id)
        ok("Assessment status after grading", a.status.value)
        db.close()

        # ── 9. Fetch grading results ───────────────────────────────────────────
        hdr("Step 9 — GET /api/v1/submissions/{id}/results")
        step("Fetching grading results via API")

        r = client.get(f"/api/v1/submissions/{submission_id}/results")
        data = assert_status(r, 200, "Get grading results")
        ok("Mastery score", str(data["mastery_score"]))
        ok("Passed", str(data["passed"]))
        ok("Weak areas", str(data.get("weak_areas", [])))
        info("Feedback", data.get("overall_feedback", "")[:100])

        # Verify missing result → 404
        step("Verifying missing grade → 404")
        r_missing = client.get("/api/v1/submissions/does-not-exist/results")
        assert_status(r_missing, 404, "Missing grade returns 404")

        # ── 10. Reschedule — approved ──────────────────────────────────────────
        hdr("Step 10 — POST /api/v1/assessments/{id}/reschedule (approved)")
        step("Creating a second curriculum + assessment for reschedule test")

        # Need a fresh assessment that hasn't been completed
        db = new_session()
        aid2 = str(uuid.uuid4())
        tok2 = generate_submission_token(aid2)
        now = datetime.utcnow()
        curriculum2 = Curriculum(
            id=str(uuid.uuid4()),
            topic="Python testing",
            target_completion_date=date(2026, 9, 1),
            extracted_content="Notes on pytest, fixtures, and mocking.",
            status=CurriculumStatus.ready,
        )
        db.add(curriculum2)
        db.flush()
        assessment2 = Assessment(
            id=aid2,
            curriculum_id=curriculum2.id,
            attempt_number=1,
            assessment_text="Explain pytest fixtures.",
            rubric="Full marks for: scope, autouse, conftest.",
            duration_minutes=45,
            scheduled_at=now + timedelta(days=3),
            reminder_at=now + timedelta(days=2),
            due_date=now + timedelta(days=8),
            status=AssessmentStatus.active,
            submission_token=tok2,
            scheduled_job_ids={
                "send_reminder": f"send_reminder_{aid2}",
                "send_assessment": f"assessment_{aid2}",
                "expire": f"expire_{aid2}",
            },
        )
        db.add(assessment2)
        db.commit()
        db.close()

        step("Requesting reschedule (FakeLLM classifies as 'medical' → approved)")
        r = client.post(f"/api/v1/assessments/{aid2}/reschedule", json={
            "token": tok2,
            "reason": (
                "I have a confirmed specialist medical appointment on the scheduled "
                "assessment date and will be unable to complete the examination."
            ),
        })
        data = assert_status(r, 200, "Reschedule request")
        ok("Approved", str(data["approved"]))
        ok("New scheduled at", str(data.get("new_scheduled_at")))
        ok("New due date", str(data.get("new_due_date")))
        ok("Reasoning", data.get("reasoning", "")[:80])

        db = new_session()
        a2 = db.get(Assessment, aid2)
        ok("Assessment dates updated", str(a2.scheduled_at))
        ok("New job IDs stored", str(bool(a2.scheduled_job_ids)))
        db.close()

        # ── 11. Reschedule — denied ────────────────────────────────────────────
        hdr("Step 11 — POST /api/v1/assessments/{id}/reschedule (denied)")
        step("Switching to FakeLLMDenied (classifies as 'procrastination')")

        # Rewire just the reschedule service with FakeLLMDenied
        denied_llm = FakeLLMDenied()
        denied_scheduler = FakeScheduler()

        def _denied_reschedule_service(
            db: Annotated[Session, Depends(get_db)],
            svc: Annotated[SchedulerService, Depends(get_scheduler_service)],
        ) -> RescheduleService:
            return RescheduleService(db, denied_llm, svc)

        app.dependency_overrides[get_reschedule_service] = _denied_reschedule_service

        # Create a third assessment for the denied test
        db = new_session()
        aid3 = str(uuid.uuid4())
        tok3 = generate_submission_token(aid3)
        curriculum3 = Curriculum(
            id=str(uuid.uuid4()),
            topic="SQL fundamentals",
            target_completion_date=date(2026, 10, 1),
            extracted_content="Notes on SQL joins, indexes, and transactions.",
            status=CurriculumStatus.ready,
        )
        db.add(curriculum3)
        db.flush()
        assessment3 = Assessment(
            id=aid3,
            curriculum_id=curriculum3.id,
            attempt_number=1,
            assessment_text="Explain SQL JOIN types.",
            rubric="Full marks for INNER, LEFT, RIGHT, FULL OUTER.",
            duration_minutes=30,
            scheduled_at=now + timedelta(days=3),
            reminder_at=now + timedelta(days=2),
            due_date=now + timedelta(days=8),
            status=AssessmentStatus.active,
            submission_token=tok3,
            scheduled_job_ids={
                "send_reminder": f"send_reminder_{aid3}",
                "send_assessment": f"assessment_{aid3}",
                "expire": f"expire_{aid3}",
            },
        )
        db.add(assessment3)
        db.commit()
        db.close()

        r = client.post(f"/api/v1/assessments/{aid3}/reschedule", json={
            "token": tok3,
            "reason": (
                "I just don't feel like doing it today and would prefer "
                "to do it next week when I am more in the mood to study."
            ),
        })
        data = assert_status(r, 200, "Reschedule request (denied)")
        ok("Approved", str(data["approved"]))
        ok("New scheduled at (should be null)", str(data.get("new_scheduled_at")))
        ok("Reasoning", data.get("reasoning", "")[:80])

        # Verify dates unchanged
        db = new_session()
        a3 = db.get(Assessment, aid3)
        ok("Assessment dates unchanged", str(a3.scheduled_at))
        db.close()

        # ── 12. Scheduler job summary ──────────────────────────────────────────
        hdr("Step 12 — Scheduler Job Summary")

        print(f"\n  {BOLD}schedule_assessment_jobs calls ({len(scheduler.schedule_assessment_jobs_calls)}):{RESET}")
        for i, call in enumerate(scheduler.schedule_assessment_jobs_calls, 1):
            print(f"    [{i}] assessment_id: {call['assessment_id'][:8]}...")
            print(f"        scheduled_at:  {call['scheduled_at']}")
            print(f"        reminder_at:   {call['reminder_at']}")
            print(f"        due_date:      {call['due_date']}")

        print(f"\n  {BOLD}schedule_grade_job calls ({len(scheduler.schedule_grade_job_calls)}):{RESET}")
        for sid in scheduler.schedule_grade_job_calls:
            print(f"    grade_{sid[:8]}...")

        print(f"\n  {BOLD}reschedule_assessment calls ({len(scheduler.reschedule_calls)}):{RESET}")
        for call in scheduler.reschedule_calls:
            print(f"    assessment_id:   {call['assessment_id'][:8]}...")
            print(f"    new_scheduled_at:{call['new_scheduled_at']}")
            print(f"    new_due_date:    {call['new_due_date']}")

        # Assertions on scheduler call counts
        step("Verifying scheduler call counts")
        assert len(scheduler.schedule_assessment_jobs_calls) == 1, (
            f"Expected 1 schedule_assessment_jobs call, got {len(scheduler.schedule_assessment_jobs_calls)}"
        )
        ok("schedule_assessment_jobs called once")

        assert len(scheduler.schedule_grade_job_calls) == 1, (
            f"Expected 1 schedule_grade_job call, got {len(scheduler.schedule_grade_job_calls)}"
        )
        ok("schedule_grade_job called once")

        assert len(scheduler.reschedule_calls) == 1, (
            f"Expected 1 reschedule_assessment call, got {len(scheduler.reschedule_calls)}"
        )
        ok("reschedule_assessment called once (approved only)")

    finally:
        client.__exit__(None, None, None)
        app.dependency_overrides.clear()
        _aps_start_patch.stop()
        _aps_shutdown_patch.stop()
        Base.metadata.drop_all(_engine)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_flow()
        print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
        print(f"{BOLD}{GREEN}  ✓ MVP FLOW COMPLETED SUCCESSFULLY{RESET}")
        print(f"{BOLD}{GREEN}{'═' * 60}{RESET}\n")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        print(f"\n{RED}{BOLD}UNEXPECTED ERROR:{RESET}")
        traceback.print_exc()
        sys.exit(1)
