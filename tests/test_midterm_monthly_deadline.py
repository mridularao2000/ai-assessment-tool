"""Part 3: the same token-gated monthly deadline that already applies to
missed Assessments now applies to a held Midterm's pending_completion.

A held midterm's completion_date has, by construction, already passed
(resources_hold is only ever set once completion_date <= today — see
CurriculumUploadService._create_entry) — so clearing it is always a late
recovery, gated exactly like SubmissionService.create() gates a late
assessment submission:
  - completion_date's calendar month == today's month: clearing spends one
    late-submission token from the entry's own pool. No token -> stays
    held.
  - a LATER calendar month: permanently unscoreable, no token can help —
    reusing transcript_service.MISSED_NO_SCORE verbatim, the same terminal
    state a permanently-missed assessment gets, not a distinct label. GPA
    and the daily recheck job both pick this up for free, since they
    already key off that shared constant rather than entry_type.

Retroactive-risk check (done BEFORE writing this code, per instruction):
ran the real curriculum_seed.json through CurriculumUploadService.ingest()
against a live server on today's actual date (2026-08-29) — the only
currently-held midterm is PM System (completion_date 2026-08-14), and
August is still the current month. Zero entries have a fully-elapsed
grace month as of today, so nothing needed grandfathering; the new rule
applies directly, per the "still within its due-date month" carve-out.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.gpa_service import compute_gpa
from app.services.late_token_service import LateTokenService
from app.services.transcript_service import HELD, MISSED_NO_SCORE, compute_transcript, display_status
from tests.conftest import FakeScheduler, RecordingEmailAdapter, make_curriculum
from app.services.scheduler_service import SchedulerService


def _make_upload(db, source_filename="test.json") -> CurriculumUpload:
    upload = CurriculumUpload(source_filename=source_filename)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _held_midterm(db, upload, *, completion_date, slots_filled=True, max_marks=100.0):
    curriculum = make_curriculum(
        db, entry_type=CurriculumEntryType.midterm, target_completion_date=completion_date,
    )
    curriculum.upload_id = upload.id
    curriculum.max_marks = max_marks
    curriculum.chapter_label = "Midterm 1"
    curriculum.resources_hold = True
    db.add(curriculum)
    slot_value = "https://github.com/x/y" if slots_filled else None
    db.add(MidtermDetail(
        curriculum_id=curriculum.id,
        known_now=["general design principles"],
        pending_completion_labels={"repo_url": "repo URL"},
        pending_completion_slots={"repo_url": slot_value},
        part1_max_marks=30.0,
        part2_max_marks=70.0,
    ))
    db.commit()
    db.refresh(curriculum)
    return curriculum


def _service(db):
    return CurriculumUploadService(db, RecordingEmailAdapter(), SchedulerService(db, FakeScheduler()))


class TestCheckAndClearHoldTokenGate:

    def test_same_month_with_token_clears_and_spends_it(self, db):
        upload = _make_upload(db)
        curriculum = _held_midterm(db, upload, completion_date=date.today() - timedelta(days=5))
        LateTokenService(db).grant_monthly(upload.id)
        balance_before = LateTokenService(db).get_balance(upload.id)

        cleared = _service(db).check_and_clear_hold(curriculum)

        assert cleared is True
        assert curriculum.resources_hold is False
        assert len(curriculum.assessments) == 1
        assert LateTokenService(db).get_balance(upload.id) == balance_before - 1
        # The token is tied to the assessment it unlocked, same audit
        # trail a late assessment submission gets.
        spent = LateTokenService(db).list_unused_tokens(upload.id)
        assert curriculum.assessments[0].id not in spent  # sanity: it's not "unused"

    def test_same_month_without_a_token_stays_held(self, db):
        upload = _make_upload(db)
        curriculum = _held_midterm(db, upload, completion_date=date.today() - timedelta(days=5))
        # No grant_monthly() call — pool starts at 0.

        cleared = _service(db).check_and_clear_hold(curriculum)

        assert cleared is False
        assert curriculum.resources_hold is True
        assert curriculum.assessments == []

    def test_month_already_passed_is_permanently_blocked_even_with_a_token(self, db):
        upload = _make_upload(db)
        last_month = date.today().replace(day=1) - timedelta(days=1)
        curriculum = _held_midterm(db, upload, completion_date=last_month)
        LateTokenService(db).grant_monthly(upload.id)
        balance_before = LateTokenService(db).get_balance(upload.id)

        cleared = _service(db).check_and_clear_hold(curriculum)

        assert cleared is False
        assert curriculum.resources_hold is True
        assert curriculum.assessments == []
        # No token was spent trying — the block is unconditional, not a
        # failed attempt that happened to also cost nothing.
        assert LateTokenService(db).get_balance(upload.id) == balance_before

    def test_slots_still_incomplete_is_unaffected_by_the_new_gate(self, db):
        """The pre-existing "not every field is filled in yet" case must
        keep behaving exactly as before — the new gate only applies once
        slots are actually complete."""
        upload = _make_upload(db)
        curriculum = _held_midterm(
            db, upload, completion_date=date.today() - timedelta(days=5), slots_filled=False,
        )
        LateTokenService(db).grant_monthly(upload.id)  # even with a token available

        cleared = _service(db).check_and_clear_hold(curriculum)

        assert cleared is False
        assert curriculum.resources_hold is True


class TestDisplayStatusAndDownstreamConsumers:

    def test_held_within_grace_month_is_still_awaiting_resources(self, db):
        upload = _make_upload(db)
        curriculum = _held_midterm(db, upload, completion_date=date.today() - timedelta(days=5))
        assert display_status(db, curriculum) == HELD

    def test_held_past_grace_month_is_missed_no_score(self, db):
        upload = _make_upload(db)
        last_month = date.today().replace(day=1) - timedelta(days=1)
        curriculum = _held_midterm(db, upload, completion_date=last_month)
        assert display_status(db, curriculum) == MISSED_NO_SCORE

    def test_held_with_completion_date_pushed_into_a_future_month_is_still_held(self, db):
        """Regression: target_completion_date can be moved into the future
        out-of-band (e.g. a manual reschedule while resources are still
        pending) — the month-mismatch check must not treat a FUTURE month
        the same as a past one. A held entry due next month is still
        recoverable, not permanently missed, just because it isn't due
        THIS month — that was the actual 2026-08-31 incident (PM System
        moved from Aug 14 to Sep 5, still resources_hold=True, and the
        transcript wrongly showed "Missed — No Score")."""
        upload = _make_upload(db)
        next_month = (date.today().replace(day=28) + timedelta(days=10)).replace(day=5)
        curriculum = _held_midterm(db, upload, completion_date=next_month)
        assert display_status(db, curriculum) == HELD

    def test_transcript_shows_permanently_missed_midterm_as_missed(self, db):
        upload = _make_upload(db)
        last_month = date.today().replace(day=1) - timedelta(days=1)
        _held_midterm(db, upload, completion_date=last_month)

        content = compute_transcript(db, upload.id)

        assert content.resolved_count == 1
        assert content.chapter_groups[0].rows[0].status_label == "MISSED"
        assert content.chapter_groups[0].rows[0].points is None

    def test_gpa_counts_permanently_missed_midterm_as_zero_in_denominator(self, db):
        """No GPA-side code change was needed for this — compute_gpa
        already keys off display_status()==MISSED_NO_SCORE generically,
        not by entry_type, for any curriculum with no Assessment row yet."""
        upload = _make_upload(db)
        last_month = date.today().replace(day=1) - timedelta(days=1)
        _held_midterm(db, upload, completion_date=last_month, max_marks=100.0)

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == 0.0
        assert summary.total_max == 100.0
        assert summary.gpa == 0.0
        assert summary.missed_count == 1

    def test_still_within_month_is_excluded_from_gpa_denominator(self, db):
        """Not yet a final outcome — a token could still clear it before
        the month ends, mirroring Missed-Late-Eligible for assessments."""
        upload = _make_upload(db)
        curriculum = _held_midterm(db, upload, completion_date=date.today() - timedelta(days=5))
        curriculum.max_marks = 100.0
        db.commit()

        summary = compute_gpa(db, upload.id)

        assert summary.total_max == 0.0
        assert summary.missed_count == 0


class TestDailyRecheckStopsRemindingOncePermanentlyMissed:

    def test_no_reminder_sent_for_a_permanently_missed_entry(self, db, monkeypatch):
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        last_month = date.today().replace(day=1) - timedelta(days=1)
        curriculum = _held_midterm(db, upload, completion_date=last_month)
        # Due for a reminder by every pre-existing throttle rule.
        curriculum.last_hold_reminder_at = None
        db.commit()

        recording_email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording_email)
        monkeypatch.setattr(
            "app.jobs.recheck_pending_midterms_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        recheck_pending_midterms_job()

        assert recording_email.hold_reminder_calls == []
        db.expire_all()
        refreshed = db.get(type(curriculum), curriculum.id)
        assert refreshed.resources_hold is True  # left as-is, not force-cleared

    def test_reminder_still_sent_for_an_entry_still_within_its_month(self, db, monkeypatch):
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        curriculum = _held_midterm(
            db, upload, completion_date=date.today() - timedelta(days=5), slots_filled=False,
        )
        curriculum.last_hold_reminder_at = None
        db.commit()

        recording_email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording_email)
        monkeypatch.setattr(
            "app.jobs.recheck_pending_midterms_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        recheck_pending_midterms_job()

        assert len(recording_email.hold_reminder_calls) == 1
