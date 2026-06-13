"""D + E: Reschedule request tests and scheduler interaction verification.

Covers:
  - Approved reschedule (medical): updates scheduled_at, due_date, reminder_at
  - Approved reschedule calls scheduler.reschedule_assessment()
  - Denied reschedule (procrastination): approved=False, no date changes
  - Invalid token → 403
  - Missing assessment → 404
  - Completed / expired assessment → 409
  - RescheduleRequest is persisted to DB for both approved and denied
  - New job IDs are stored on the Assessment after approved reschedule
"""

import pytest
from datetime import datetime

from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.reschedule_request import RescheduleRequest
from app.services.reschedule_service import RescheduleService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeLLM,
    FakeLLMDenied,
    FakeScheduler,
    make_assessment,
    make_curriculum,
    seed_prompt_templates,
)


VALID_REASON = (
    "I have a confirmed medical appointment on the scheduled assessment day "
    "and cannot attend the examination as planned."
)


class TestApprovedReschedule:

    def test_approved_reschedule_returns_approved_true(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.status_code == 200
        assert response.json()["approved"] is True

    def test_approved_reschedule_returns_new_dates(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        data = response.json()
        assert data["new_scheduled_at"] is not None
        assert data["new_due_date"] is not None

    def test_approved_reschedule_updates_assessment_scheduled_at(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        original_scheduled_at = assessment.scheduled_at

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.scheduled_at != original_scheduled_at

    def test_approved_reschedule_updates_assessment_due_date(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        original_due_date = assessment.due_date

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.due_date != original_due_date

    def test_approved_reschedule_updates_reminder_at(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        original_reminder_at = assessment.reminder_at

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.reminder_at != original_reminder_at

    def test_approved_reschedule_stores_new_job_ids(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.scheduled_job_ids is not None
        assert refreshed.scheduled_job_ids.get("send_reminder", "").endswith("_v2")

    def test_approved_reschedule_calls_scheduler_reschedule(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert len(fake_scheduler.reschedule_calls) == 1
        assert fake_scheduler.reschedule_calls[0]["assessment_id"] == assessment.id

    def test_approved_reschedule_persists_request_record(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        db.expire_all()
        record = db.query(RescheduleRequest).filter_by(assessment_id=assessment.id).first()
        assert record is not None
        assert record.approved is True
        assert record.classification_category == "medical"
        assert record.reason_text == VALID_REASON

    def test_approved_reschedule_reasoning_returned(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.json()["reasoning"] != ""

    def test_approved_reschedule_on_scheduled_assessment(self, client, db):
        """Reschedule is allowed on 'scheduled' status too (not yet active)."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.status_code == 200
        assert response.json()["approved"] is True


class TestDeniedReschedule:

    def test_denied_reschedule_returns_approved_false_via_service(self, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        svc = RescheduleService(db, FakeLLMDenied(), SchedulerService(db, fake_scheduler))
        result, updated = svc.request_reschedule(
            assessment_id=assessment.id,
            token=token,
            reason=VALID_REASON,
        )

        assert result.approved is False
        assert updated is None

    def test_denied_reschedule_no_date_change(self, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        original_scheduled_at = assessment.scheduled_at
        original_due_date = assessment.due_date

        svc = RescheduleService(db, FakeLLMDenied(), SchedulerService(db, fake_scheduler))
        svc.request_reschedule(assessment_id=assessment.id, token=token, reason=VALID_REASON)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.scheduled_at == original_scheduled_at
        assert refreshed.due_date == original_due_date

    def test_denied_reschedule_does_not_call_scheduler(self, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        svc = RescheduleService(db, FakeLLMDenied(), SchedulerService(db, fake_scheduler))
        svc.request_reschedule(assessment_id=assessment.id, token=token, reason=VALID_REASON)

        assert len(fake_scheduler.reschedule_calls) == 0

    def test_denied_reschedule_persists_request_record(self, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        svc = RescheduleService(db, FakeLLMDenied(), SchedulerService(db, fake_scheduler))
        svc.request_reschedule(assessment_id=assessment.id, token=token, reason=VALID_REASON)

        db.expire_all()
        record = db.query(RescheduleRequest).filter_by(assessment_id=assessment.id).first()
        assert record is not None
        assert record.approved is False
        assert record.classification_category == "procrastination"

    def test_denied_reschedule_via_api(self, denied_client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = denied_client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["approved"] is False
        assert data["new_scheduled_at"] is None
        assert data["new_due_date"] is None


class TestRescheduleValidation:

    def test_invalid_token_returns_403(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": "INVALID_TOKEN", "reason": VALID_REASON},
        )

        assert response.status_code == 403

    def test_missing_assessment_returns_404(self, client, db):
        response = client.post(
            "/api/v1/assessments/does-not-exist/reschedule",
            json={"token": "anything", "reason": VALID_REASON},
        )

        assert response.status_code == 404

    def test_completed_assessment_returns_409(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.completed)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.status_code == 409

    def test_expired_assessment_returns_409(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.expired)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert response.status_code == 409

    def test_reason_too_short_returns_422(self, client, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": "Too short"},  # < 20 chars
        )

        assert response.status_code == 422

    def test_missing_prompt_template_raises_not_found(self, db, fake_scheduler):
        """If 'reschedule_classification' PromptTemplate is absent, NotFoundError is raised."""
        # No prompt templates seeded
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        svc = RescheduleService(db, FakeLLM(), SchedulerService(db, fake_scheduler))
        with pytest.raises(NotFoundError, match="reschedule_classification"):
            svc.request_reschedule(
                assessment_id=assessment.id,
                token=token,
                reason=VALID_REASON,
            )

    def test_reschedule_requires_existing_job_ids(self, db, fake_scheduler):
        """Approved reschedule requires assessment.scheduled_job_ids to cancel old jobs."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        # Create assessment WITHOUT job IDs
        from app.models.assessment import Assessment
        from app.utils.token_auth import generate_submission_token
        import uuid
        from datetime import timedelta

        aid = str(uuid.uuid4())
        token = generate_submission_token(aid)
        now = __import__("datetime").datetime.utcnow()
        assessment = Assessment(
            id=aid,
            curriculum_id=curriculum.id,
            attempt_number=1,
            assessment_text="text",
            rubric="rubric",
            duration_minutes=60,
            scheduled_at=now + timedelta(days=2),
            reminder_at=now + timedelta(hours=24),
            due_date=now + timedelta(days=7),
            status=AssessmentStatus.active,
            submission_token=token,
            scheduled_job_ids=None,  # <-- no job IDs
        )
        db.add(assessment)
        db.commit()

        svc = RescheduleService(db, FakeLLM(), SchedulerService(db, fake_scheduler))
        with pytest.raises(InvalidStateError):
            svc.request_reschedule(
                assessment_id=aid,
                token=token,
                reason=VALID_REASON,
            )


class TestSchedulerInteraction:

    def test_schedule_assessment_jobs_called_on_create(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        client.post(f"/api/v1/assessments/{curriculum.id}")

        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1

    def test_schedule_grade_job_called_on_submission(self, client, db, fake_scheduler):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "My answer about the event loop.",
        })

        assert len(fake_scheduler.schedule_grade_job_calls) == 1

    def test_reschedule_assessment_called_on_approved(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post(
            f"/api/v1/assessments/{assessment.id}/reschedule",
            json={"token": token, "reason": VALID_REASON},
        )

        assert len(fake_scheduler.reschedule_calls) == 1

    def test_reschedule_not_called_on_denied(self, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        svc = RescheduleService(db, FakeLLMDenied(), SchedulerService(db, fake_scheduler))
        svc.request_reschedule(assessment_id=assessment.id, token=token, reason=VALID_REASON)

        assert len(fake_scheduler.reschedule_calls) == 0
