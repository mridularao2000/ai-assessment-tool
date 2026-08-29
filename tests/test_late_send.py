"""AssessmentService.trigger_late_send — the UI-facing way to enter the
generation/send path for a 'Missed — Late-Eligible' assessment, instead of
that only being reachable through a raw manual API call.

Covers: not-found cases, every InvalidStateError gate (not expired, expired
in a previous calendar month, zero token balance), content generation only
happening when content is actually missing, and that a token is never spent
by this call itself — only by an actual submission afterward.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fastapi import HTTPException

from app.exceptions import InvalidStateError, NotFoundError
from app.interfaces.email import EmailDeliveryError
from app.models.assessment import AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.midterm_detail import MidtermDetail
from app.services.assessment_service import AssessmentService
from app.services.email_service import EmailService
from app.services.late_token_service import LateTokenService
from tests.conftest import (
    FakeLLM,
    NoopEmailAdapter,
    RecordingEmailAdapter,
    make_assessment,
    make_curriculum,
    seed_prompt_templates,
)


def _svc(db):
    return AssessmentService(db, FakeLLM())


class TestNotFound:
    def test_unknown_curriculum_id_raises_not_found(self, db):
        with pytest.raises(NotFoundError):
            _svc(db).trigger_late_send("nope", EmailService(db, RecordingEmailAdapter()))

    def test_curriculum_with_no_assessment_yet_raises_not_found(self, db):
        curriculum = make_curriculum(db)
        with pytest.raises(NotFoundError):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))


class TestStateGates:
    def test_active_assessment_is_refused(self, db):
        curriculum = make_curriculum(db)
        make_assessment(db, curriculum, status=AssessmentStatus.active, due_offset_days=2)
        with pytest.raises(InvalidStateError, match="not expired"):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))

    def test_completed_assessment_is_refused(self, db):
        curriculum = make_curriculum(db)
        make_assessment(db, curriculum, status=AssessmentStatus.completed, due_offset_days=-2)
        with pytest.raises(InvalidStateError, match="not expired"):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))

    def test_expired_in_a_previous_calendar_month_is_refused(self, db):
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        # Force due_date into last month regardless of what day of the
        # month the test happens to run on.
        assessment.due_date = datetime.utcnow().replace(day=1) - timedelta(days=1)
        db.commit()
        LateTokenService(db).grant_monthly(None)

        with pytest.raises(InvalidStateError, match="previous calendar month"):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))

    def test_expired_this_month_with_zero_token_balance_is_refused(self, db):
        curriculum = make_curriculum(db)
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        # No grant_monthly call — standalone pool starts at 0.

        with pytest.raises(InvalidStateError, match="No late-submission tokens"):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))


class TestSuccessPaths:
    def test_content_already_present_is_not_regenerated_and_email_is_sent(self, db):
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        original_text = assessment.assessment_text
        assert original_text is not None
        LateTokenService(db).grant_monthly(None)
        recording = RecordingEmailAdapter()

        result = _svc(db).trigger_late_send(curriculum.id, EmailService(db, recording))

        assert result.assessment_text == original_text  # untouched
        assert len(recording.assessment_calls) == 1
        # Status stays expired — SubmissionService's late-path branch keys
        # on exactly this, so trigger_late_send must never flip it.
        assert result.status == AssessmentStatus.expired

    def test_missing_content_is_generated_then_sent(self, db):
        """The retroactive case: an entry whose window closed before it was
        ever scheduled has no content yet — see
        CurriculumUploadService._create_retroactive_expired_assessment."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        assessment.assessment_text = None
        assessment.rubric = None
        db.commit()
        LateTokenService(db).grant_monthly(None)
        recording = RecordingEmailAdapter()

        result = _svc(db).trigger_late_send(curriculum.id, EmailService(db, recording))

        assert result.assessment_text is not None  # FakeLLM populated it
        assert len(recording.assessment_calls) == 1

    def test_missing_midterm_content_is_generated_via_midterm_path(self, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["general design principles"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2,
        )
        # make_assessment sets assessment_text but leaves part1/part2 null
        # by default — exactly the "midterm, not yet generated" shape.
        assert assessment.part1_text is None
        LateTokenService(db).grant_monthly(None)
        recording = RecordingEmailAdapter()

        result = _svc(db).trigger_late_send(curriculum.id, EmailService(db, recording))

        assert result.part1_text is not None
        assert result.part2_text is not None
        assert len(recording.assessment_calls) == 1

    def test_token_is_not_spent_by_the_trigger_itself(self, db):
        curriculum = make_curriculum(db)
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        LateTokenService(db).grant_monthly(None)
        balance_before = LateTokenService(db).get_balance(None)

        _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))

        assert LateTokenService(db).get_balance(None) == balance_before

    def test_upload_scoped_entry_checks_its_own_token_pool(self, db):
        """A curriculum-upload entry's late-eligibility must be gated by
        its own upload's token pool, not the standalone pool — mirrors
        SubmissionService.create()'s existing per-pool scoping."""
        from tests.conftest import TestSessionLocal
        from app.models.curriculum_upload import CurriculumUpload

        upload = CurriculumUpload(source_filename="x.json")
        db.add(upload)
        db.commit()
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id = upload.id
        db.commit()
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)

        # Standalone pool has tokens, but this entry's own upload pool does not.
        LateTokenService(db).grant_monthly(None)
        with pytest.raises(InvalidStateError, match="No late-submission tokens"):
            _svc(db).trigger_late_send(curriculum.id, EmailService(db, RecordingEmailAdapter()))

        LateTokenService(db).grant_monthly(upload.id)
        recording = RecordingEmailAdapter()
        _svc(db).trigger_late_send(curriculum.id, EmailService(db, recording))
        assert len(recording.assessment_calls) == 1


class _BrokenEmailAdapter(NoopEmailAdapter):
    """A configured-but-failing provider (e.g. Gmail SMTP auth rejected)."""

    def send_assessment_email(self, data) -> None:
        raise EmailDeliveryError("Gmail SMTP failed: (535, b'BadCredentials')")


class _UnconfiguredEmailAdapter(NoopEmailAdapter):
    """Mirrors StubEmailAdapter (app.dependencies): no provider configured
    at all, so every send raises loudly rather than silently no-op'ing."""

    def send_assessment_email(self, data) -> None:
        raise NotImplementedError(
            "StubEmailAdapter: set GMAIL_ADDRESS/GMAIL_APP_PASSWORD to enable email."
        )


class TestRouteErrorMapping:
    """The /late-send route (app/api/v1/curriculum_upload.py) must never
    leak a bare, undiagnosable 500 for a failure that already has a clear
    cause — a real production incident: GMAIL_ADDRESS/GMAIL_APP_PASSWORD
    not yet set in Render's dashboard (env vars declared in render.yaml
    with sync:false are NOT populated by a git push — they need manual
    entry) left the live server on StubEmailAdapter, whose NotImplementedError
    propagated as an opaque 500 with no indication of the real cause."""

    def _entry(self, db):
        curriculum = make_curriculum(db)
        make_assessment(db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2)
        LateTokenService(db).grant_monthly(None)
        return curriculum

    def test_email_delivery_error_maps_to_502(self, db):
        from app.api.v1.curriculum_upload import trigger_late_send as route

        curriculum = self._entry(db)
        with pytest.raises(HTTPException) as exc_info:
            route(curriculum.id, _svc(db), EmailService(db, _BrokenEmailAdapter()))
        assert exc_info.value.status_code == 502
        assert "Gmail SMTP failed" in exc_info.value.detail

    def test_unconfigured_email_provider_maps_to_503(self, db):
        from app.api.v1.curriculum_upload import trigger_late_send as route

        curriculum = self._entry(db)
        with pytest.raises(HTTPException) as exc_info:
            route(curriculum.id, _svc(db), EmailService(db, _UnconfiguredEmailAdapter()))
        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail
