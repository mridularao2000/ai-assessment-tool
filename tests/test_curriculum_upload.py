"""Section 1 — Curriculum upload + syllabus email.

Covers:
  - Parsing/validation against the actual curriculum_seed.json fixture
    (not a synthetic stand-in)
  - Classification: PM System -> resources_hold, everything else -> normal
  - Part-1 cumulative pool assembly, including the generic known_now
    fallback (PM System) and the date-driven pool for Chat App (9
    Assessments / chapters 1,2,3,4,5,6,11 — NOT the seed's own stale
    "7 Assessments / chapters 1,3,4,5,6" annotation)
  - Syllabus email content: chapter order, verbatim resources, Chapter 9
    heading with no standalone Assessment, PM System known_now shown
  - Pending-resources PATCH endpoint and the daily recheck job
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.exceptions import CurriculumUploadValidationError, InvalidStateError, NotFoundError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.late_submission_token import LateSubmissionToken
from app.models.midterm_detail import MidtermDetail
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.late_token_service import LateTokenService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeScheduler,
    NoopEmailAdapter,
    RecordingEmailAdapter,
    make_assessment,
    make_curriculum,
)


def _scheduler(db) -> SchedulerService:
    """Fresh SchedulerService + FakeScheduler pair for one test call."""
    return SchedulerService(db, FakeScheduler())

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "curriculum_seed.json"


class _FixedDate(date):
    """Pins date.today() to 2026-08-26 so classification is deterministic
    regardless of when the test actually runs."""

    @classmethod
    def today(cls):
        return date(2026, 8, 26)


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setattr("app.services.curriculum_upload_service.date", _FixedDate)


@pytest.fixture
def seed_raw() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class TestParsing:

    def test_rejects_missing_topics(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError):
            service.ingest({}, "bad.json")

    def test_rejects_unknown_type(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError, match="type"):
            service.ingest(
                {"topics": [{"topic": "X", "type": "quiz", "chapter": "Ch 1",
                             "resources": [], "completion_date": "2026-08-15", "max_marks": 50}]},
                "bad.json",
            )

    def test_rejects_missing_required_field(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError, match="max_marks"):
            service.ingest(
                {"topics": [{"topic": "X", "type": "assessment", "chapter": "Ch 1",
                             "resources": ["a"], "completion_date": "2026-08-15"}]},
                "bad.json",
            )

    def test_rejects_assessment_resources_as_object(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError, match="resources"):
            service.ingest(
                {"topics": [{"topic": "X", "type": "assessment", "chapter": "Ch 1",
                             "resources": {"known_now": []}, "completion_date": "2026-08-15",
                             "max_marks": 50}]},
                "bad.json",
            )

    def test_rejects_midterm_resources_as_flat_list(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError, match="resources"):
            service.ingest(
                {"topics": [{"topic": "X", "type": "midterm", "chapter": "Ch 1",
                             "resources": ["a", "b"], "completion_date": "2026-08-15",
                             "max_marks": 100}]},
                "bad.json",
            )

    def test_bad_upload_persists_nothing(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        with pytest.raises(CurriculumUploadValidationError):
            service.ingest(
                {"topics": [{"topic": "Good", "type": "assessment", "chapter": "Ch 1",
                             "resources": ["a"], "completion_date": "2026-08-15", "max_marks": 50},
                            {"topic": "Bad", "type": "assessment", "chapter": "Ch 2",
                             "resources": ["b"], "completion_date": "not-a-date", "max_marks": 50}]},
                "bad.json",
            )
        assert db.query(Curriculum).count() == 0
        assert db.query(CurriculumUpload).count() == 0


class TestRealSeedIngestion:

    def test_creates_18_entries_13_assessments_5_midterms(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        entries = db.query(Curriculum).filter(Curriculum.upload_id == upload.id).all()
        assert len(entries) == 18
        assert sum(1 for e in entries if e.entry_type == CurriculumEntryType.assessment) == 13
        assert sum(1 for e in entries if e.entry_type == CurriculumEntryType.midterm) == 5

    def test_pm_system_enters_resources_hold(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        assert pm_system is not None
        assert pm_system.resources_hold is True
        assert pm_system.target_completion_date == date(2026, 8, 14)

    def test_future_midterms_are_not_held(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        midterms = (
            db.query(Curriculum)
            .filter(
                Curriculum.upload_id == upload.id,
                Curriculum.entry_type == CurriculumEntryType.midterm,
                ~Curriculum.topic.like("PM System%"),
            )
            .all()
        )
        assert len(midterms) == 4
        assert all(m.resources_hold is False for m in midterms)

    def test_assessment_resources_stored_verbatim(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        js_internals = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("JS Internals%"))
            .first()
        )
        resource_texts = {r.source_ref for r in js_internals.resources}
        assert "javascript.info (review — closures, prototypes, promises/async)" in resource_texts
        assert "Chrome DevTools Memory panel (hands-on)" in resource_texts
        assert all(r.raw_content is None for r in js_internals.resources)

    def test_midterm_pending_resources_restructured_into_slots(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        detail = pm_system.midterm_detail
        assert set(detail.pending_completion_labels.values()) == {
            "project README", "repo URL", "accessibility audit results",
        }
        assert all(v is None for v in detail.pending_completion_slots.values())
        assert detail.known_now == [
            "react.dev (component/rendering resources, for Part 1)",
            "WAI-ARIA APG",
            "axe DevTools + Lighthouse",
            "MDN (semantic HTML, ARIA)",
        ]
        # Default split: Part 1 (assignment) = 30%, Part 2 (project) = 70%.
        assert detail.part1_max_marks == 30.0
        assert detail.part2_max_marks == 70.0

    def test_syllabus_email_sent_and_upload_marked(self, db, seed_raw):
        email = RecordingEmailAdapter()
        service = CurriculumUploadService(db, email, _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        assert len(email.syllabus_calls) == 1
        db.expire_all()
        refreshed = db.get(CurriculumUpload, upload.id)
        assert refreshed.syllabus_email_sent_at is not None

    def test_hold_reminder_sent_immediately_for_pm_system(self, db, seed_raw):
        email = RecordingEmailAdapter()
        service = CurriculumUploadService(db, email, _scheduler(db))
        service.ingest(seed_raw, "curriculum_seed.json")

        assert len(email.hold_reminder_calls) == 1
        assert email.hold_reminder_calls[0].topic.startswith("PM System")
        assert set(email.hold_reminder_calls[0].missing_labels) == {
            "project README", "repo URL", "accessibility audit results",
        }


class TestPart1PoolAssembly:
    """Checkpoint-1's explicit question: does PM System correctly use
    known_now instead of an empty cumulative pool, and is the pool always
    computed from dates rather than trusting a stale chapter-label count?
    """

    def test_pm_system_falls_back_to_known_now(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        qualifying, used_fallback = service.assemble_part1_pool(pm_system)
        assert qualifying == []
        assert used_fallback is True

    def test_chat_app_pool_is_9_not_7_and_ignores_stale_label(self, db, seed_raw):
        """The seed's own chapter label says "7 Assessments, Chapters
        1,3,4,5,6" for Chat App (completion 2026-09-13) — but Frontend
        Architecture (Ch 11, 2026-09-11) and Browser Internals (Ch 2,
        2026-09-12) both complete before that date too. The real,
        date-driven pool must include them regardless of what the label
        claims.
        """
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        chat_app = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("Chat App%"))
            .first()
        )
        qualifying, used_fallback = service.assemble_part1_pool(chat_app)
        assert used_fallback is False
        assert len(qualifying) == 9
        topics = {e.topic for e in qualifying}
        assert "Frontend Architecture & Patterns" in topics
        assert "Browser Internals — Rendering Path, Reflow/Repaint" in topics

    def test_capstone_pool_is_all_13_assessments(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        capstone = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("VS Code Extension Phase 4%"))
            .first()
        )
        qualifying, used_fallback = service.assemble_part1_pool(capstone)
        assert used_fallback is False
        assert len(qualifying) == 13


class TestSyllabusContent:

    def test_chapters_appear_in_numeric_order_including_chapter_9(self, db, seed_raw):
        from app.services.syllabus_builder import build_syllabus

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        full_upload = service.get_upload(upload.id)
        content = build_syllabus(full_upload, list(full_upload.entries))

        chapter_numbers = []
        for section in content.chapters:
            import re
            m = re.match(r"Chapter (\d+)", section.chapter_label)
            assert m is not None, f"unexpected chapter label: {section.chapter_label!r}"
            chapter_numbers.append(int(m.group(1)))
        assert chapter_numbers == sorted(chapter_numbers)
        assert 9 in chapter_numbers

        ch9 = next(s for s in content.chapters if s.chapter_label.startswith("Chapter 9"))
        assert ch9.assessments == []
        assert "PM System" in ch9.no_standalone_note

        ch1 = next(s for s in content.chapters if s.chapter_label.startswith("Chapter 1:"))
        js_internals = next(a for a in ch1.assessments if a.topic.startswith("JS Internals"))
        real_entry = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("JS Internals%"))
            .first()
        )
        assert js_internals.id == real_entry.id  # the real Curriculum.id, not a display label

    def test_resources_appear_verbatim_not_paraphrased(self, db, seed_raw):
        from app.services.syllabus_builder import build_syllabus

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        full_upload = service.get_upload(upload.id)
        content = build_syllabus(full_upload, list(full_upload.entries))

        ch1 = next(s for s in content.chapters if s.chapter_label.startswith("Chapter 1:"))
        all_resources = [r for a in ch1.assessments for r in a.resources]
        assert "javascript.info (review — closures, prototypes, promises/async)" in all_resources
        assert "Chrome DevTools Memory panel (hands-on)" in all_resources

    def test_pm_system_shown_with_known_now_and_hold_flag(self, db, seed_raw):
        from app.services.syllabus_builder import build_syllabus

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        full_upload = service.get_upload(upload.id)
        content = build_syllabus(full_upload, list(full_upload.entries))

        pm_row = next(m for m in content.midterms if m.topic.startswith("PM System"))
        assert pm_row.resources_hold is True
        assert "WAI-ARIA APG" in pm_row.known_now
        assert pm_row.pending_status[0][1] is False  # nothing filled yet

        real_pm = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        assert pm_row.id == real_pm.id


class TestSyllabusEmailRendering:
    """The rendered syllabus email must actually show each entry's real
    Curriculum.id (and the upload's own id) — the whole point being that
    it's a genuinely useful reference later (Update Entry, View Entries,
    late-send, etc.), not just present in the underlying dataclass with
    nothing surfacing it to the person reading the email."""

    def test_entry_ids_and_upload_id_appear_in_rendered_html(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        js_internals = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("JS Internals%"))
            .first()
        )
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )

        from app.adapters.resend_email import ResendEmailAdapter
        from app.services.email_service import EmailService

        sent = {}

        class _CapturingResendAdapter(ResendEmailAdapter):
            def __init__(self):
                self._from = "Test <test@example.com>"

            def _send(self, to, subject, body_html):
                sent["html"] = body_html

        EmailService(db, _CapturingResendAdapter()).send_syllabus_email(upload.id)

        html = sent["html"]
        assert upload.id in html
        assert js_internals.id in html
        assert pm_system.id in html


class TestPendingResourcesPatch:

    def test_fill_all_slots_clears_hold(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")
        # A held midterm's completion_date has already passed by
        # construction, so clearing it now is a late recovery — same
        # token-gated monthly window as a late assessment submission (see
        # CurriculumUploadService.check_and_clear_hold).
        LateTokenService(db).grant_monthly(upload.id)

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)

        updated = service.fill_pending_resources(
            pm_system.id, {slugs[0]: "https://github.com/x/pm-system"}
        )
        assert updated.resources_hold is True  # still 2 slots empty

        updated = service.fill_pending_resources(
            pm_system.id,
            {slugs[1]: "https://github.com/x/pm-system#readme", slugs[2]: "audit passed"},
        )
        assert updated.resources_hold is False

    def test_fill_all_slots_without_a_token_stays_held(self, db, seed_raw):
        """Same fill, but this upload's pool is exhausted — the hold must
        NOT clear, since resources_hold is only set once completion_date
        has already passed (a held midterm is, by construction, always a
        late recovery). ingest() now auto-grants 2 tokens to a fresh
        upload's pool (see CurriculumUploadService.ingest), so to exercise
        the zero-balance case we drain the pool it was just given, standing
        in for both tokens already having been spent elsewhere this month."""
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")
        db.query(LateSubmissionToken).filter(
            LateSubmissionToken.curriculum_upload_id == upload.id
        ).delete()
        db.commit()

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)

        updated = service.fill_pending_resources(
            pm_system.id,
            {slugs[0]: "a", slugs[1]: "b", slugs[2]: "c"},
        )
        assert updated.resources_hold is True
        assert updated.assessments == []  # window never opened

    def test_unknown_slot_raises(self, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        with pytest.raises(InvalidStateError):
            service.fill_pending_resources(pm_system.id, {"not_a_real_slot": "x"})

    def test_missing_curriculum_raises_not_found(self, db):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        with pytest.raises(NotFoundError):
            service.fill_pending_resources("does-not-exist", {"x": "y"})

    def test_blank_value_is_rejected_not_silently_dropped(self, db, seed_raw):
        """Regression: a blank-form submission (every field "") used to
        silently no-op — 200-style success with nothing actually written,
        resources_hold and every slot left untouched. It must now be
        rejected so the caller (and the UI) can tell the difference
        between "succeeded" and "nothing happened"."""
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")
        LateTokenService(db).grant_monthly(upload.id)

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)

        with pytest.raises(InvalidStateError):
            service.fill_pending_resources(pm_system.id, {slugs[0]: "", slugs[1]: "   "})

        db.expire_all()
        refreshed = db.get(Curriculum, pm_system.id)
        assert refreshed.resources_hold is True
        assert all(v is None for v in refreshed.midterm_detail.pending_completion_slots.values())

    def test_patch_endpoint_clears_hold(self, client, db, seed_raw):
        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")
        LateTokenService(db).grant_monthly(upload.id)

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)
        values = {slug: f"value-for-{slug}" for slug in slugs}

        response = client.patch(
            f"/api/v1/curriculum-uploads/entries/{pm_system.id}/pending-resources",
            json={"values": values},
        )

        assert response.status_code == 200
        assert response.json()["resources_hold"] is False


class TestDailyRecheckJob:

    def test_recheck_clears_hold_when_slots_complete(self, db, monkeypatch, seed_raw):
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
        from tests.conftest import TestSessionLocal

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")
        LateTokenService(db).grant_monthly(upload.id)

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        slugs = list(pm_system.midterm_detail.pending_completion_slots)
        detail = pm_system.midterm_detail
        detail.pending_completion_slots = {slug: f"filled-{slug}" for slug in slugs}
        db.commit()

        recording_email = RecordingEmailAdapter()
        fake_scheduler = FakeScheduler()
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording_email)
        monkeypatch.setattr(
            "app.jobs.recheck_pending_midterms_job.get_scheduler_adapter", lambda: fake_scheduler
        )

        recheck_pending_midterms_job()

        db.expire_all()
        refreshed = db.get(Curriculum, pm_system.id)
        assert refreshed.resources_hold is False
        assert recording_email.hold_reminder_calls == []  # cleared, no reminder needed
        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1

    def test_recheck_resends_reminder_after_throttle_interval(self, db, monkeypatch, seed_raw):
        from datetime import timedelta
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
        from app.models._utils import utcnow
        from tests.conftest import TestSessionLocal

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        pm_system.last_hold_reminder_at = utcnow() - timedelta(days=8)
        db.commit()

        recording_email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording_email)

        recheck_pending_midterms_job()

        assert len(recording_email.hold_reminder_calls) == 1

    def test_recheck_does_not_resend_within_throttle_window(self, db, monkeypatch, seed_raw):
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
        from app.models._utils import utcnow
        from tests.conftest import TestSessionLocal

        service = CurriculumUploadService(db, RecordingEmailAdapter(), _scheduler(db))
        upload = service.ingest(seed_raw, "curriculum_seed.json")

        db.expire_all()
        pm_system = (
            db.query(Curriculum)
            .filter(Curriculum.upload_id == upload.id, Curriculum.topic.like("PM System%"))
            .first()
        )
        pm_system.last_hold_reminder_at = utcnow()  # just sent, at ingestion
        db.commit()

        recording_email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording_email)

        recheck_pending_midterms_job()

        assert recording_email.hold_reminder_calls == []


class TestBulkReschedule:
    """CurriculumUploadService.bulk_reschedule_entries — shifts due_date/
    scheduled_date forward for missed ('expired') entries without
    regenerating already-generated content, so the existing send pipeline
    resends it as-is once the new scheduled_at arrives."""

    def test_shifts_dates_resets_status_and_keeps_content(self, db):
        curriculum = make_curriculum(db, entry_type="assessment", target_completion_date=date(2026, 8, 15))
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-5)
        original_text = assessment.assessment_text
        original_due_date = assessment.due_date
        assert original_text is not None
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        [updated] = service.bulk_reschedule_entries([curriculum.id], shift_days=10)

        assert updated.target_completion_date == date(2026, 8, 25)
        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.scheduled
        assert refreshed.assessment_text == original_text  # untouched, not regenerated
        assert refreshed.due_date > original_due_date
        assert refreshed.send_job_claimed_at is None

    def test_refuses_an_entry_not_yet_due(self, db):
        curriculum = make_curriculum(db, entry_type="assessment")
        make_assessment(db, curriculum, status=AssessmentStatus.scheduled, due_offset_days=7)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        with pytest.raises(InvalidStateError, match="not eligible"):
            service.bulk_reschedule_entries([curriculum.id], shift_days=10)

    def test_refuses_an_already_graded_entry(self, db):
        curriculum = make_curriculum(db, entry_type="assessment")
        make_assessment(db, curriculum, status=AssessmentStatus.completed, due_offset_days=-5)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        with pytest.raises(InvalidStateError, match="not eligible"):
            service.bulk_reschedule_entries([curriculum.id], shift_days=10)

    def test_refuses_a_nonpositive_shift(self, db):
        curriculum = make_curriculum(db, entry_type="assessment")
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-5)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        with pytest.raises(InvalidStateError, match="shift_days"):
            service.bulk_reschedule_entries([curriculum.id], shift_days=0)

    def test_atomic_one_invalid_id_writes_nothing(self, db):
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-5)
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        with pytest.raises(NotFoundError):
            service.bulk_reschedule_entries([curriculum.id, "does-not-exist"], shift_days=10)

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.expired

    def test_bulk_shifts_multiple_entries_in_one_call(self, db):
        curricula = [
            make_curriculum(db, entry_type="assessment", topic=f"Topic {i}")
            for i in range(3)
        ]
        assessments = [
            make_assessment(db, c, status=AssessmentStatus.expired, due_offset_days=-5)[0]
            for c in curricula
        ]
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        updated = service.bulk_reschedule_entries([c.id for c in curricula], shift_days=5)

        assert len(updated) == 3
        db.expire_all()
        for a in assessments:
            assert db.get(Assessment, a.id).status == AssessmentStatus.scheduled

    def test_http_route_bulk_reschedules(self, client, db):
        curriculum = make_curriculum(db, entry_type="assessment")
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-5)

        response = client.post(
            "/api/v1/curriculum-uploads/entries/bulk-reschedule",
            json={"curriculum_ids": [curriculum.id], "shift_days": 7},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["updated"]) == 1
        assert body["updated"][0]["status"] == "Not Yet Due"
