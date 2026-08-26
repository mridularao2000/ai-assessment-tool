"""Section 2 (exam scheduling) + Section 3 (resource fetching).

Covers:
  - Scheduling deferred-generation Assessment rows for ready entries,
    skipping already-past and held ones
  - A cleared hold schedules from today, not the original completion_date
  - Content generation happens at send_assessment_job time, not eagerly
  - web_search/web_fetch are actually passed as enabled tools on the real
    request payload (not just defined elsewhere) — checked against the
    mocked anthropic SDK call, not just against our own DTOs
  - The four named non-fetchable resources get general-knowledge guidance;
    everything else gets search-then-fetch guidance
  - Midterm Part 1 pool assembly is threaded correctly into generation
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.adapters.nonfetchable_resources import build_resource_guidance
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.midterm_detail import MidtermDetail
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeLLM,
    FakeScheduler,
    NoopEmailAdapter,
    TestSessionLocal,
    make_curriculum,
    seed_prompt_templates,
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


def _scheduler(db) -> tuple[SchedulerService, FakeScheduler]:
    fake = FakeScheduler()
    return SchedulerService(db, fake), fake


class SpyLLM(FakeLLM):
    """Records every generate_assessment/generate_midterm request it receives."""

    def __init__(self):
        self.assessment_requests = []
        self.midterm_requests = []

    def generate_assessment(self, req):
        self.assessment_requests.append(req)
        return super().generate_assessment(req)

    def generate_midterm(self, req):
        self.midterm_requests.append(req)
        return super().generate_midterm(req)


class TestScheduleReadyEntries:

    def test_ready_entries_get_null_content_assessment_rows(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        # 18 entries total: PM System held (no Assessment), 14 scheduled
        # normally with empty content, 3 already-past-at-upload entries
        # retroactively created straight to expired (see
        # TestRetroactiveExpiredAssessments below).
        scheduled = (
            db.query(Assessment)
            .join(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Assessment.status == AssessmentStatus.scheduled)
            .all()
        )
        assert len(scheduled) == 14
        assert all(a.assessment_text is None and a.part1_text is None for a in scheduled)

    def test_pm_system_gets_no_assessment_row_while_held(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        assert pm_system.assessments == []

    def test_jobs_scheduled_for_every_ready_entry(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, fake = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        service.ingest(seed_raw, "curriculum_seed.json")

        assert len(fake.schedule_assessment_jobs_calls) == 14

    def test_scheduling_is_idempotent(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, fake = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        service.ingest(seed_raw, "curriculum_seed.json")

        # Calling again must not duplicate Assessment rows or jobs.
        service.schedule_ready_entries()
        assert len(fake.schedule_assessment_jobs_calls) == 14
        db.expire_all()
        assert db.query(Assessment).count() == 17  # 14 scheduled + 3 retroactive


class TestRetroactiveExpiredAssessments:
    """Entries whose window had already closed before upload (JS Internals,
    React Rendering, State Mgmt-Context in the real seed) get an Assessment
    row created directly in `expired` status — no jobs, no email — purely
    so a late-submission token has something real to submit against.
    Without this, "Missed — Late-Eligible" would be a label with no
    assessment_id/token behind it.
    """

    def test_already_past_entries_get_expired_assessment_with_no_jobs(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, fake = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        for topic_prefix in ("JS Internals", "React Rendering", "State Management — Context"):
            entry = (
                db.query(Curriculum)
                .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like(f"{topic_prefix}%"))
                .first()
            )
            assert len(entry.assessments) == 1, f"{topic_prefix} should have a retroactive Assessment"
            assessment = entry.assessments[0]
            assert assessment.status == AssessmentStatus.expired
            assert assessment.assessment_text is None
            assert assessment.scheduled_job_ids is None

        # None of the 3 retroactive ones registered jobs — only the 14
        # normally-scheduled entries did.
        assert len(fake.schedule_assessment_jobs_calls) == 14

    def test_lazy_generation_on_first_view_then_late_submittable(self, client, db, seed_raw):
        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        js_internals = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("JS Internals%"))
            .first()
        )
        assessment = js_internals.assessments[0]
        token = assessment.submission_token

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={token}")
        assert response.status_code == 200
        assert response.json()["assessment_text"] is not None

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.assessment_text is not None
        assert refreshed.status == AssessmentStatus.expired  # viewing doesn't activate it

        from app.models.late_submission_token import LateSubmissionToken
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        submit_response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "Late answer for JS Internals.",
        })
        assert submit_response.status_code == 201
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.late_submitted

    def test_normally_scheduled_entry_not_generated_early_via_get(self, client, db, seed_raw):
        """A future entry's content must wait for send_assessment_job — the
        GET endpoint must not let it be peeked/generated before its actual
        send time just because someone has the link early."""
        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        redux = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("State Management — Redux%"))
            .first()
        )
        assessment = redux.assessments[0]
        assert assessment.status == AssessmentStatus.scheduled

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={assessment.submission_token}")
        assert response.status_code == 200

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.assessment_text is None
        assert refreshed.status == AssessmentStatus.scheduled

    def test_lazy_generation_on_retroactive_midterm_uses_midterm_branch(self, client, db):
        """The GET route's lazy-generation branch dispatches on is_midterm —
        confirm the midterm arm (generate_midterm_content, populating
        part1_text/part2_text) actually fires for a retroactive expired
        Midterm, not just the assessment-type arm exercised by the seed's
        3 retroactive entries (the seed never produces a retroactive
        Midterm — PM System, the only past-due one, goes to resources_hold
        instead)."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(
            db,
            topic="Retroactive Midterm Project",
            target_completion_date=date(2026, 1, 1),
            entry_type=CurriculumEntryType.midterm,
        )
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["Design doc"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()

        svc, _ = _scheduler(db)
        upload_service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        assessment = upload_service._create_retroactive_expired_assessment(curriculum)
        assert assessment.part1_text is None
        assert assessment.assessment_text is None

        response = client.get(
            f"/api/v1/assessments/{assessment.id}?token={assessment.submission_token}"
        )
        assert response.status_code == 200
        assert response.json()["is_midterm"] is True
        assert response.json()["assessment_text"] is not None  # part1_text surfaced here

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.part1_text is not None
        assert refreshed.part2_text is not None
        assert refreshed.assessment_text is None  # single-part field stays unused


class TestHoldClearingSchedulesFromToday:

    def test_cleared_hold_window_is_relative_to_today_not_original_date(self, db, seed_raw):
        seed_prompt_templates(db)
        svc, fake = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)
        service.fill_pending_resources(pm_system.id, {s: f"value-{s}" for s in slugs})

        db.expire_all()
        refreshed = db.get(Curriculum, pm_system.id)
        assert refreshed.resources_hold is False
        assert len(refreshed.assessments) == 1
        assessment = refreshed.assessments[0]
        # window = today(2026-08-26)+1..+3, NOT 2026-08-14(original)+1..+3
        assert date(2026, 8, 27) <= assessment.scheduled_at.date() <= date(2026, 8, 29)


class TestSendAssessmentJobGeneratesContentAtSendTime:

    def test_assessment_type_entry_generates_content_when_sent(self, db, monkeypatch, seed_raw):
        from app.jobs.send_assessment_job import send_assessment_job

        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        entry = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("State Management — Redux%"))
            .first()
        )
        assessment = entry.assessments[0]
        assert assessment.assessment_text is None  # not generated yet

        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", NoopEmailAdapter())

        send_assessment_job(assessment.id)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.assessment_text is not None
        assert refreshed.status == AssessmentStatus.active

    def test_midterm_type_entry_generates_two_part_content_when_sent(self, db, monkeypatch, seed_raw):
        from app.jobs.send_assessment_job import send_assessment_job

        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        capstone = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("VS Code Extension Phase 4%"))
            .first()
        )
        assessment = capstone.assessments[0]
        assert assessment.part1_text is None

        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", NoopEmailAdapter())

        send_assessment_job(assessment.id)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.part1_text is not None
        assert refreshed.part2_text is not None
        assert refreshed.assessment_text is None  # single-part field stays unused
        assert refreshed.status == AssessmentStatus.active


class TestMidtermContentAssembly:

    def test_pm_system_uses_known_now_fallback_in_generation_request(self, db, monkeypatch, seed_raw):
        from app.jobs.send_assessment_job import send_assessment_job

        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)
        service.fill_pending_resources(pm_system.id, {s: f"value-{s}" for s in slugs})

        db.expire_all()
        refreshed = db.get(Curriculum, pm_system.id)
        assessment = refreshed.assessments[0]

        spy = SpyLLM()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", spy)
        monkeypatch.setattr("app.jobs.send_assessment_job._email", NoopEmailAdapter())

        send_assessment_job(assessment.id)

        assert len(spy.midterm_requests) == 1
        req = spy.midterm_requests[0]
        assert "WAI-ARIA APG" in req.cumulative_pool_content
        assert "axe DevTools" in req.cumulative_pool_content
        # own_resources includes known_now + the values just filled in
        assert any("value-" in r for r in req.own_resources)

    def test_chat_app_part1_pool_is_9_assessments_at_generation_time(self, db, monkeypatch, seed_raw):
        from app.jobs.send_assessment_job import send_assessment_job

        seed_prompt_templates(db)
        svc, _ = _scheduler(db)
        service = CurriculumUploadService(db, NoopEmailAdapter(), svc)
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        chat_app = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("Chat App%"))
            .first()
        )
        assessment = chat_app.assessments[0]

        spy = SpyLLM()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", spy)
        monkeypatch.setattr("app.jobs.send_assessment_job._email", NoopEmailAdapter())

        send_assessment_job(assessment.id)

        req = spy.midterm_requests[0]
        # 9 Assessments, not the seed's stale "7" label — one bullet per entry
        assert req.cumulative_pool_content.count("Resources:") == 9
        assert "Frontend Architecture" in req.cumulative_pool_content
        assert "Browser Internals" in req.cumulative_pool_content


class TestBuildResourceGuidance:

    def test_empty_resources_returns_empty_string(self):
        assert build_resource_guidance([]) == ""

    @pytest.mark.parametrize("resource", [
        '"System Design Interview" by Alex Xu, Vol 1 (scalability, reliability patterns, databases at scale, distributed systems)',
        "ByteByteGo (caching/CDNs, message queues)",
        "Chrome DevTools Memory panel (hands-on)",
        "bigfrontend.dev/react (memoization landscape)",
    ])
    def test_named_non_fetchable_resources_get_general_knowledge_note(self, resource):
        guidance = build_resource_guidance([resource])
        assert "Do NOT attempt to fetch" in guidance
        assert "general knowledge" in guidance

    def test_generic_resource_gets_search_then_fetch_instruction(self):
        guidance = build_resource_guidance(["react.dev (rendering, keys, reconciliation, JSX docs)"])
        assert "web_search" in guidance
        assert "web_fetch" in guidance
        assert "Do NOT attempt to fetch" not in guidance

    def test_mixed_list_handles_each_resource_independently(self):
        guidance = build_resource_guidance([
            "ByteByteGo (caching/CDNs, message queues)",
            "react.dev (rendering, keys, reconciliation, JSX docs)",
        ])
        assert "Do NOT attempt to fetch" in guidance  # ByteByteGo
        assert "web_search" in guidance  # react.dev


class TestToolsActuallyPassedOnRealRequest:
    """Checkpoint-2's explicit question: are web_search/web_fetch actually
    enabled on the exam-generation API call — checked against the mocked
    anthropic SDK request payload itself, not just our own DTOs."""

    def _adapter(self, monkeypatch):
        from app.adapters.anthropic_llm import AnthropicLLMAdapter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from app.config import get_settings
        get_settings.cache_clear()
        adapter = AnthropicLLMAdapter()
        get_settings.cache_clear()
        return adapter

    def _mock_response(self, payload: dict):
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text=json.dumps(payload))]
        return msg

    def test_standalone_call_has_no_tools_in_payload(self, monkeypatch):
        from app.interfaces.llm import AssessmentGenerationRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "assessment_text": "x", "rubric": "y", "duration_minutes": 60,
        }))
        adapter._client.messages.create = mock_create

        adapter.generate_assessment(AssessmentGenerationRequest(
            topic="Async", curriculum_content="notes", prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
        ))

        assert "tools" not in mock_create.call_args.kwargs

    def test_entry_call_has_both_tools_in_payload(self, monkeypatch):
        from app.adapters.anthropic_llm import WEB_FETCH_TOOL, WEB_SEARCH_TOOL
        from app.interfaces.llm import AssessmentGenerationRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "assessment_text": "x", "rubric": "y", "duration_minutes": 60,
        }))
        adapter._client.messages.create = mock_create

        adapter.generate_assessment(AssessmentGenerationRequest(
            topic="React Rendering",
            curriculum_content="",
            prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
            resources=["react.dev (rendering, keys, reconciliation, JSX docs)"],
        ))

        sent_tools = mock_create.call_args.kwargs["tools"]
        assert WEB_SEARCH_TOOL in sent_tools
        assert WEB_FETCH_TOOL in sent_tools

    def test_midterm_call_has_tools_when_own_resources_present(self, monkeypatch):
        from app.adapters.anthropic_llm import WEB_FETCH_TOOL, WEB_SEARCH_TOOL
        from app.interfaces.llm import MidtermGenerationRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "part1_text": "a", "part1_rubric": "b",
            "part2_text": "c", "part2_rubric": "d",
            "duration_minutes": 120,
        }))
        adapter._client.messages.create = mock_create

        adapter.generate_midterm(MidtermGenerationRequest(
            topic="PM System",
            cumulative_pool_content="",
            own_resources=["project README"],
            probe_focus="Defend the design.",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
            prompt_template_body=(
                "{topic}{cumulative_pool_content}{own_resources_list}"
                "{resource_guidance}{probe_focus}{part1_max_marks}{part2_max_marks}"
            ),
        ))

        sent_tools = mock_create.call_args.kwargs["tools"]
        assert WEB_SEARCH_TOOL in sent_tools
        assert WEB_FETCH_TOOL in sent_tools

    def test_midterm_grading_call_has_no_tools_when_resources_empty(self, monkeypatch):
        """grade_midterm_submission() with no resources — mirrors the
        standalone-call-has-no-tools case above, checked directly against
        this function rather than inferred from generate_midterm()."""
        from app.interfaces.llm import MidtermGradingRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "part1_score": 27.0, "part2_score": 63.0,
            "weak_areas": [], "overall_feedback": "Good.",
        }))
        adapter._client.messages.create = mock_create

        adapter.grade_midterm_submission(MidtermGradingRequest(
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
            part1_max_marks=30.0, part2_max_marks=70.0,
            part1_submission_content="answer", part2_submission_content="submission",
            prompt_template_body=(
                "{part1_text}{part1_rubric}{part2_text}{part2_rubric}"
                "{part1_max_marks}{part2_max_marks}"
                "{part1_submission_content}{part2_submission_content}{resource_guidance}"
            ),
        ))

        assert "tools" not in mock_create.call_args.kwargs

    def test_midterm_grading_call_has_both_tools_when_resources_present(self, monkeypatch):
        """Direct proof that grade_midterm_submission() enables web_search/
        web_fetch on the real Anthropic SDK call — not inferred from
        generate_midterm()/TestToolsActuallyPassedOnRealRequest's other
        cases, which test a different function. Three prior bugs in this
        codebase were exactly this pattern: assumed to follow an existing
        pattern, and silently didn't — this asserts it directly against the
        mocked SDK request payload, same as every other case in this class."""
        from app.adapters.anthropic_llm import WEB_FETCH_TOOL, WEB_SEARCH_TOOL
        from app.interfaces.llm import MidtermGradingRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "part1_score": 27.0, "part2_score": 63.0,
            "weak_areas": [], "overall_feedback": "Good.",
        }))
        adapter._client.messages.create = mock_create

        adapter.grade_midterm_submission(MidtermGradingRequest(
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
            part1_max_marks=30.0, part2_max_marks=70.0,
            part1_submission_content="answer", part2_submission_content="submission",
            prompt_template_body=(
                "{part1_text}{part1_rubric}{part2_text}{part2_rubric}"
                "{part1_max_marks}{part2_max_marks}"
                "{part1_submission_content}{part2_submission_content}{resource_guidance}"
            ),
            resources=["project README"],
        ))

        assert "tools" in mock_create.call_args.kwargs
        sent_tools = mock_create.call_args.kwargs["tools"]
        assert WEB_SEARCH_TOOL in sent_tools
        assert WEB_FETCH_TOOL in sent_tools

    def test_resource_guidance_text_reaches_the_actual_prompt(self, monkeypatch):
        from app.interfaces.llm import AssessmentGenerationRequest

        adapter = self._adapter(monkeypatch)
        mock_create = MagicMock(return_value=self._mock_response({
            "assessment_text": "x", "rubric": "y", "duration_minutes": 60,
        }))
        adapter._client.messages.create = mock_create

        adapter.generate_assessment(AssessmentGenerationRequest(
            topic="System Design",
            curriculum_content="",
            prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
            resources=['"System Design Interview" by Alex Xu, Vol 1'],
        ))

        sent_prompt = mock_create.call_args.kwargs["messages"][0]["content"]
        assert "Do NOT attempt to fetch" in sent_prompt
        assert "general knowledge" in sent_prompt
