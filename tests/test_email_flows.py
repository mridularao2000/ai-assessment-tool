"""Section 5 — Email flows / recipients.

Covers:
  - Recipient configuration for flows (b) reminder, (c) exam delivery,
    (d) grading result — standalone unaffected, entries use the new
    configured recipient(s), falling back to user_email when unset
  - results_recipient_emails config parsing (comma-separated, 2 addresses)
  - Flow (b)'s reminder re-anchored to due_date (the deadline) for entries,
    with a configurable offset — standalone's own reminder (anchored to
    scheduled_at, hardcoded 1 day) stays untouched

Flow (a) syllabus / hold-reminder recipients were already covered in
Section 1's tests. Flow (e) transcript doesn't exist yet (Sections 6-7).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.email_service import EmailService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeScheduler,
    NoopEmailAdapter,
    RecordingEmailAdapter,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "curriculum_seed.json"


class _FixedDate(date):
    @classmethod
    def today(cls):
        return date(2026, 8, 26)


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setattr("app.services.curriculum_upload_service.date", _FixedDate)


@pytest.fixture
def seed_raw() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _scheduler(db) -> SchedulerService:
    return SchedulerService(db, FakeScheduler())


class TestResultsRecipientEmailsConfig:

    def test_comma_separated_parses_to_two_addresses(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("RESULTS_RECIPIENT_EMAILS_RAW", "a@example.com, b@example.com")
        settings = get_settings()

        assert settings.results_recipient_emails == ["a@example.com", "b@example.com"]
        get_settings.cache_clear()

    def test_unset_falls_back_to_user_email(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("USER_EMAIL", "owner@example.com")
        monkeypatch.delenv("RESULTS_RECIPIENT_EMAILS_RAW", raising=False)
        settings = get_settings()

        assert settings.results_recipient_emails == ["owner@example.com"]
        get_settings.cache_clear()


class TestRecipientRoutingByEntryType:
    """EmailService branches recipients on curriculum.entry_type, not on
    submission/grading outcome — standalone must be byte-identical to
    today regardless of what's configured for entries."""

    def test_standalone_assessment_email_uses_user_email_regardless_of_entry_config(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("USER_EMAIL", "owner@example.com")
        monkeypatch.setenv("EXAM_RECIPIENT_EMAIL", "exam-only@example.com")

        curriculum = make_curriculum(db)  # entry_type=None
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_assessment_email(assessment.id)

        assert email.assessment_calls[0].recipient_emails == ["owner@example.com"]
        get_settings.cache_clear()

    def test_entry_assessment_email_uses_exam_recipient(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("USER_EMAIL", "owner@example.com")
        monkeypatch.setenv("EXAM_RECIPIENT_EMAIL", "exam-only@example.com")

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_assessment_email(assessment.id)

        assert email.assessment_calls[0].recipient_emails == ["exam-only@example.com"]
        get_settings.cache_clear()

    def test_entry_assessment_email_falls_back_to_user_email_when_unset(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("USER_EMAIL", "owner@example.com")
        monkeypatch.delenv("EXAM_RECIPIENT_EMAIL", raising=False)

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_assessment_email(assessment.id)

        assert email.assessment_calls[0].recipient_emails == ["owner@example.com"]
        get_settings.cache_clear()

    def test_entry_reminder_email_uses_exam_recipient(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("EXAM_RECIPIENT_EMAIL", "exam-only@example.com")

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_reminder_email(assessment.id)

        assert email.reminder_calls[0].recipient_emails == ["exam-only@example.com"]
        get_settings.cache_clear()

    def test_standalone_results_email_uses_user_email(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("USER_EMAIL", "owner@example.com")
        monkeypatch.setenv("RESULTS_RECIPIENT_EMAILS_RAW", "a@example.com,b@example.com")

        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=90.0)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_results_email(submission.id)

        assert email.results_calls[0].recipient_emails == ["owner@example.com"]
        get_settings.cache_clear()

    def test_entry_results_email_uses_two_configured_recipients(self, db, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("RESULTS_RECIPIENT_EMAILS_RAW", "a@example.com,b@example.com")

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=90.0)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_results_email(submission.id)

        assert email.results_calls[0].recipient_emails == ["a@example.com", "b@example.com"]
        get_settings.cache_clear()


class TestEntryReminderAnchoredToDeadline:
    """Flow (b): for entries, reminder_at counts back from due_date (the
    deadline) with a configurable offset — not from scheduled_at (the send
    date) like standalone's own, untouched reminder."""

    def test_entry_reminder_defaults_to_24h_before_due_date(self, db, seed_raw):
        from tests.conftest import seed_prompt_templates

        seed_prompt_templates(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        redux = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("State Management — Redux%"))
            .first()
        )
        assessment = redux.assessments[0]
        assert assessment.reminder_at == assessment.due_date - timedelta(hours=24)

    def test_entry_reminder_offset_is_configurable(self, db, seed_raw, monkeypatch):
        from app.config import get_settings
        from tests.conftest import seed_prompt_templates

        get_settings.cache_clear()
        monkeypatch.setenv("ENTRY_REMINDER_HOURS_BEFORE_DEADLINE", "6")

        seed_prompt_templates(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        redux = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("State Management — Redux%"))
            .first()
        )
        assessment = redux.assessments[0]
        assert assessment.reminder_at == assessment.due_date - timedelta(hours=6)
        get_settings.cache_clear()

    def test_standalone_reminder_still_anchored_to_scheduled_at(self, db, monkeypatch):
        """Regression: standalone's own date math (AssessmentService.
        _build_dates, exercised via create_for_curriculum) must stay
        exactly as it was before Section 5 — 1 day before scheduled_at,
        not touching due_date at all. This calls the real service method,
        not just a test fixture's own hardcoded value."""
        from app.config import get_settings
        from app.services.assessment_service import AssessmentService
        from tests.conftest import FakeLLM, seed_prompt_templates

        get_settings.cache_clear()
        monkeypatch.setenv("ENTRY_REMINDER_HOURS_BEFORE_DEADLINE", "6")  # must have zero effect

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)  # entry_type=None, status=ready by default
        assessment = AssessmentService(db, FakeLLM()).create_for_curriculum(curriculum)

        assert assessment.reminder_at == assessment.scheduled_at - timedelta(days=1)
        get_settings.cache_clear()


class TestReminderCopyMatchesActualTiming:
    """Found during audit: the reminder email's copy ("scheduled for
    tomorrow... will arrive in a separate email") was written for
    standalone's pre-SEND timing and stayed unchanged when flow (b) was
    re-anchored to pre-DEADLINE for entries — meaning the email would tell
    a student their exam "will arrive" when it had already been sent days
    earlier. EmailService now passes is_pre_deadline through so the
    adapter can render correct copy for each case."""

    def test_is_pre_deadline_flag_set_for_entries(self, db):
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_reminder_email(assessment.id)

        assert email.reminder_calls[0].is_pre_deadline is True

    def test_is_pre_deadline_flag_unset_for_standalone(self, db):
        curriculum = make_curriculum(db)  # entry_type=None
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        email = RecordingEmailAdapter()

        EmailService(db, email).send_reminder_email(assessment.id)

        assert email.reminder_calls[0].is_pre_deadline is False

    def test_adapter_copy_says_arrives_tomorrow_when_not_pre_deadline(self, monkeypatch):
        from app.adapters.resend_email import ResendEmailAdapter
        from app.interfaces.email import ReminderEmailData
        from datetime import datetime

        monkeypatch.setenv("RESEND_API_KEY", "test-key")
        from app.config import get_settings
        get_settings.cache_clear()
        adapter = ResendEmailAdapter()
        sent = {}
        monkeypatch.setattr(
            "resend.Emails.send", lambda params: sent.update(params)
        )

        adapter.send_reminder_email(ReminderEmailData(
            recipient_emails=["a@example.com"], topic="Async/Await",
            scheduled_at=datetime(2026, 9, 1), expire_date=datetime(2026, 9, 3),
            key_topics=[], is_pre_deadline=False,
        ))

        assert "scheduled for tomorrow" in sent["html"]
        assert "will arrive in a separate email" in sent["html"]
        assert "Tomorrow" in sent["subject"]
        get_settings.cache_clear()

    def test_adapter_copy_says_deadline_approaching_when_pre_deadline(self, monkeypatch):
        from app.adapters.resend_email import ResendEmailAdapter
        from app.interfaces.email import ReminderEmailData
        from datetime import datetime

        monkeypatch.setenv("RESEND_API_KEY", "test-key")
        from app.config import get_settings
        get_settings.cache_clear()
        adapter = ResendEmailAdapter()
        sent = {}
        monkeypatch.setattr(
            "resend.Emails.send", lambda params: sent.update(params)
        )

        adapter.send_reminder_email(ReminderEmailData(
            recipient_emails=["a@example.com"], topic="Async/Await",
            scheduled_at=datetime(2026, 9, 1), expire_date=datetime(2026, 9, 3),
            key_topics=[], is_pre_deadline=True,
        ))

        assert "deadline is approaching" in sent["html"]
        assert "scheduled for tomorrow" not in sent["html"]
        assert "will arrive in a separate email" not in sent["html"]
        assert "deadline approaching" in sent["subject"]
        get_settings.cache_clear()


class TestRetestRespectsEntryTypeAndResources:
    """Found during audit: create_retest() predates curriculum-upload
    entirely and never branched on entry_type — a Midterm's retest would
    have silently built single-part content instead of two-part, and an
    Assessment-type entry's retest never got resources/tools passed
    (violating "web_search/web_fetch on every exam-generation call"), nor
    the entry-specific reminder timing from Section 5.
    """

    def test_assessment_type_retest_passes_resources_and_uses_entry_dates(self, db):
        from app.services.assessment_service import AssessmentService
        from tests.conftest import FakeLLMBelowThreshold, make_grade, make_submission, seed_prompt_templates

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        from app.models.resource import Resource, ResourceType
        db.add(Resource(curriculum_id=curriculum.id, type=ResourceType.note,
                         source_ref="react.dev (hooks)", raw_content=None))
        db.commit()

        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        grade = make_grade(db, submission, mastery_score=70.0, weak_areas=["hooks"])

        class SpyLLM(FakeLLMBelowThreshold):
            def __init__(self):
                self.retest_requests = []
            def generate_retest(self, req):
                self.retest_requests.append(req)
                return super().generate_retest(req)

        spy = SpyLLM()
        retest = AssessmentService(db, spy).create_retest(curriculum.id, previous_grade_id=grade.id)

        assert spy.retest_requests[0].resources == ["react.dev (hooks)"]
        # Entry-specific date math: reminder_at counts back from due_date,
        # not from scheduled_at like standalone's create_retest used to.
        assert retest.reminder_at == retest.due_date - timedelta(hours=24)
        assert retest.attempt_number == 2

    def test_standalone_retest_passes_no_resources(self, db):
        """Regression: standalone must never suddenly start passing
        resources/enabling tools — that would be a real behavior change."""
        from app.services.assessment_service import AssessmentService
        from tests.conftest import FakeLLMBelowThreshold, make_grade, make_submission, seed_prompt_templates

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)  # entry_type=None
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        grade = make_grade(db, submission, mastery_score=70.0, weak_areas=["event loop"])

        class SpyLLM(FakeLLMBelowThreshold):
            def __init__(self):
                self.retest_requests = []
            def generate_retest(self, req):
                self.retest_requests.append(req)
                return super().generate_retest(req)

        spy = SpyLLM()
        retest = AssessmentService(db, spy).create_retest(curriculum.id, previous_grade_id=grade.id)

        assert spy.retest_requests[0].resources is None
        assert retest.reminder_at == retest.scheduled_at - timedelta(days=1)

    def test_midterm_retest_regenerates_two_part_content(self, db):
        """Section 6: a failing Midterm gets the same 2-attempt cap as
        Assessments/standalone, but a full two-part regeneration (not a
        single-part retest) — see AssessmentService._create_midterm_retest."""
        from app.models.midterm_detail import MidtermDetail
        from app.services.assessment_service import AssessmentService
        from tests.conftest import FakeLLM, make_grade, make_submission, seed_prompt_templates

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["design doc"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")
        grade = make_grade(
            db, submission, mastery_score=70.0, weak_areas=["design"],
            part1_score=21.0, part2_score=49.0, score_earned=70.0, max_marks=100.0,
        )

        retest = AssessmentService(db, FakeLLM()).create_retest(
            curriculum.id, previous_grade_id=grade.id
        )

        assert retest.attempt_number == 2
        assert retest.part1_text is not None
        assert retest.part2_text is not None
        assert retest.assessment_text is None  # single-part field stays unused
        assert retest.status == AssessmentStatus.scheduled


class TestMidtermGrading:
    """Section 6: grading a Midterm scores Part 1 and Part 2 independently
    against their own rubric/max_marks and combines them into score_earned
    (the GPA input) — see GradingService._grade_midterm."""

    def test_grading_a_midterm_submission_computes_part_scores(self, db):
        from app.models.midterm_detail import MidtermDetail
        from app.services.grading_service import GradingService
        from tests.conftest import FakeLLM, make_submission, seed_prompt_templates

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.max_marks = 100.0
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["design doc"],
            pending_completion_labels={},
            pending_completion_slots={"repo_url": "https://github.com/example/repo"},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")

        grade = GradingService(db, FakeLLM()).grade(submission.id)

        assert grade.part1_score == pytest.approx(30.0 * 0.9)
        assert grade.part2_score == pytest.approx(70.0 * 0.9)
        assert grade.score_earned == pytest.approx(grade.part1_score + grade.part2_score)
        assert grade.max_marks == 100.0
        assert grade.mastery_score == pytest.approx(grade.score_earned / 100.0 * 100.0)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.completed


class TestFlowDCoversLateAndRetakeAttempts:
    """Spec names flow (d) as firing "on grading (including late/retake
    attempts)" explicitly — verifying both, end-to-end through
    grade_submission_job, for an entry (not just standalone)."""

    def _patch_job(self, monkeypatch, llm, fake_scheduler):
        from tests.conftest import TestSessionLocal
        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", llm)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: fake_scheduler
        )

    def test_retake_attempt_grading_sends_to_two_configured_recipients(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLMBelowThreshold, FakeScheduler, make_submission, seed_prompt_templates

        get_settings_mod = __import__("app.config", fromlist=["get_settings"])
        get_settings_mod.get_settings.cache_clear()
        monkeypatch.setenv("RESULTS_RECIPIENT_EMAILS_RAW", "a@example.com,b@example.com")

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)  # attempt_number=1

        email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.grade_submission_job._email", email)
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), FakeScheduler())

        grade_submission_job(submission.id)  # fails -> schedules attempt 2 (a retest)

        assert len(email.results_calls) == 1
        assert email.results_calls[0].recipient_emails == ["a@example.com", "b@example.com"]
        assert email.results_calls[0].attempt_number == 1

        db.expire_all()
        retest = (
            db.query(Assessment)
            .filter(Assessment.curriculum_id == curriculum.id, Assessment.attempt_number == 2)
            .first()
        )
        assert retest is not None  # the retake attempt really was created

        get_settings_mod.get_settings.cache_clear()

    def test_late_submitted_entry_grading_sends_to_two_configured_recipients(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job
        from app.models.late_submission_token import LateSubmissionToken
        from app.models.submission import Submission, SubmissionType
        from tests.conftest import FakeLLM, FakeScheduler, seed_prompt_templates

        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("RESULTS_RECIPIENT_EMAILS_RAW", "a@example.com,b@example.com")

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        from app.services.submission_service import SubmissionService
        from app.services.scheduler_service import SchedulerService
        from app.services.late_token_service import LateTokenService
        submission_svc = SubmissionService(
            db, SchedulerService(db, FakeScheduler()), LateTokenService(db)
        )
        submission = submission_svc.create(
            assessment_id=assessment.id, token=token,
            submission_type=SubmissionType.text, text_content="Late answer.",
        )
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.late_submitted

        email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.grade_submission_job._email", email)
        self._patch_job(monkeypatch, FakeLLM(), FakeScheduler())

        grade_submission_job(submission.id)

        assert len(email.results_calls) == 1
        assert email.results_calls[0].recipient_emails == ["a@example.com", "b@example.com"]

        get_settings.cache_clear()
