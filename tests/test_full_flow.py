"""A + G: Full happy-path integration flow and token auth tests.

Happy-path flow under test:
  1. Seed curriculum (status=ready) and prompt templates directly in DB.
  2. POST /api/v1/assessments/{curriculum_id} → creates Assessment via FakeLLM,
     schedules jobs via FakeScheduler (status=scheduled).
  3. Manually set status=active (simulates send_assessment_job firing).
  4. GET /api/v1/assessments/{id}?token=... → returns assessment details.
  5. POST /api/v1/submissions/ → text submission (status→submitted).
  6. Call GradingService.grade() directly (simulates grade_submission_job).
  7. GET /api/v1/submissions/{id}/results → grade result with passed=True.
"""

from app.models.assessment import Assessment, AssessmentStatus
from app.models.grade import Grade
from app.models.submission import Submission
from app.services.grading_service import GradingService
from app.utils.token_auth import generate_submission_token, verify_submission_token
from tests.conftest import (
    FakeLLM,
    TestSessionLocal,
    make_assessment,
    make_curriculum,
    seed_prompt_templates,
)


# ── G. Token auth ─────────────────────────────────────────────────────────────


class TestTokenAuth:

    def test_token_is_deterministic(self):
        assessment_id = "fixed-id-for-token-test"
        assert generate_submission_token(assessment_id) == generate_submission_token(assessment_id)

    def test_different_assessment_ids_produce_different_tokens(self):
        assert generate_submission_token("id-alpha") != generate_submission_token("id-beta")

    def test_verify_returns_true_for_correct_token(self):
        aid = "verify-correct-test"
        token = generate_submission_token(aid)
        assert verify_submission_token(aid, token) is True

    def test_verify_returns_false_for_wrong_token(self):
        assert verify_submission_token("some-id", "notavalidtoken") is False

    def test_verify_returns_false_for_tampered_token(self):
        aid = "tamper-test-id"
        token = generate_submission_token(aid)
        tampered = token[:-4] + ("XXXX" if not token.endswith("XXXX") else "YYYY")
        assert verify_submission_token(aid, tampered) is False

    def test_verify_returns_false_for_wrong_assessment_id(self):
        token = generate_submission_token("real-id")
        assert verify_submission_token("other-id", token) is False

    def test_token_is_url_safe_base64(self):
        token = generate_submission_token("url-safe-test")
        # URL-safe base64 uses only A-Z a-z 0-9 - _
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in token)


# ── A. Full happy-path flow ───────────────────────────────────────────────────


class TestFullHappyPathFlow:

    def test_create_assessment_via_api(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        response = client.post(f"/api/v1/assessments/{curriculum.id}")

        assert response.status_code == 201
        data = response.json()
        assert data["attempt_number"] == 1
        assert data["status"] == "scheduled"
        assert data["duration_minutes"] == 60  # from FakeLLM

    def test_create_assessment_schedules_jobs(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        response = client.post(f"/api/v1/assessments/{curriculum.id}")
        assert response.status_code == 201
        assessment_id = response.json()["id"]

        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1
        call = fake_scheduler.schedule_assessment_jobs_calls[0]
        assert call["assessment_id"] == assessment_id

    def test_create_assessment_persists_job_ids(self, client, db, fake_scheduler):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        response = client.post(f"/api/v1/assessments/{curriculum.id}")
        assessment_id = response.json()["id"]

        db.expire_all()
        assessment = db.get(Assessment, assessment_id)
        assert assessment is not None
        assert assessment.scheduled_job_ids is not None
        assert "send_reminder" in assessment.scheduled_job_ids
        assert "send_assessment" in assessment.scheduled_job_ids
        assert "expire" in assessment.scheduled_job_ids

    def test_get_assessment_returns_details(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={token}")

        assert response.status_code == 200
        data = response.json()
        assert data["assessment_id"] == assessment.id
        assert data["topic"] == curriculum.topic
        assert data["status"] == "active"
        assert data["assessment_text"] is not None
        assert data["duration_minutes"] == 60

    def test_submit_text_assignment(self, client, db, fake_scheduler):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "The event loop runs coroutines until completion.",
        })

        assert response.status_code == 201
        submission_id = response.json()["submission_id"]
        assert submission_id

    def test_submission_transitions_assessment_to_submitted(self, client, db, fake_scheduler):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "Concurrency is cooperative via await.",
        })

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.submitted

    def test_submission_triggers_grade_job_scheduling(self, client, db, fake_scheduler):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "Coroutines are paused at await points.",
        })
        submission_id = response.json()["submission_id"]

        assert len(fake_scheduler.schedule_grade_job_calls) == 1
        assert fake_scheduler.schedule_grade_job_calls[0] == submission_id

    def test_grade_submission_via_service(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.submitted)
        from tests.conftest import make_submission
        submission = make_submission(db, assessment)

        grade_session = TestSessionLocal()
        try:
            grade = GradingService(grade_session, fake_llm).grade(submission.id)
        finally:
            grade_session.close()

        assert grade.mastery_score == 90.0
        assert grade.overall_feedback == "Excellent understanding demonstrated."

    def test_grade_marks_assessment_completed(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        from tests.conftest import make_submission
        submission = make_submission(db, assessment)

        grade_session = TestSessionLocal()
        try:
            GradingService(grade_session, fake_llm).grade(submission.id)
        finally:
            grade_session.close()

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.completed

    def test_get_grade_results_via_api(self, client, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        from tests.conftest import make_submission
        submission = make_submission(db, assessment)

        grade_session = TestSessionLocal()
        try:
            GradingService(grade_session, fake_llm).grade(submission.id)
        finally:
            grade_session.close()

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        assert response.status_code == 200
        data = response.json()
        assert data["mastery_score"] == 90.0
        assert data["passed"] is True
        assert data["overall_feedback"] == "Excellent understanding demonstrated."

    def test_full_end_to_end_with_all_status_transitions(
        self, client, db, fake_scheduler, fake_llm
    ):
        # Setup
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)

        # Step 1: Create assessment (scheduled)
        r = client.post(f"/api/v1/assessments/{curriculum.id}")
        assert r.status_code == 201
        assessment_id = r.json()["id"]

        db.expire_all()
        assert db.get(Assessment, assessment_id).status == AssessmentStatus.scheduled

        # Step 2: Activate (simulates send_assessment_job)
        db.expire_all()
        a = db.get(Assessment, assessment_id)
        a.status = AssessmentStatus.active
        db.commit()

        # Step 3: Fetch assessment via token
        db.expire_all()
        a = db.get(Assessment, assessment_id)
        token = a.submission_token

        r = client.get(f"/api/v1/assessments/{assessment_id}?token={token}")
        assert r.status_code == 200
        assert r.json()["status"] == "active"

        # Step 4: Submit
        r = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment_id,
            "token": token,
            "submission_type": "text",
            "text_content": "The event loop schedules coroutines cooperatively.",
        })
        assert r.status_code == 201
        submission_id = r.json()["submission_id"]

        db.expire_all()
        assert db.get(Assessment, assessment_id).status == AssessmentStatus.submitted

        # Step 5: Grade
        grade_session = TestSessionLocal()
        try:
            GradingService(grade_session, fake_llm).grade(submission_id)
        finally:
            grade_session.close()

        db.expire_all()
        assert db.get(Assessment, assessment_id).status == AssessmentStatus.completed

        # Step 6: Fetch results
        r = client.get(f"/api/v1/submissions/{submission_id}/results")
        assert r.status_code == 200
        result = r.json()
        assert result["mastery_score"] == 90.0
        assert result["passed"] is True

        # F. Verify DB state
        db.expire_all()
        assert db.query(Submission).count() == 1
        assert db.query(Grade).count() == 1
        assert db.query(Assessment).filter_by(status=AssessmentStatus.completed).count() == 1

        # E. Verify scheduler calls
        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1
        assert len(fake_scheduler.schedule_grade_job_calls) == 1

    def test_submission_persisted_before_scheduler_called(self, client, db, fake_scheduler):
        """Submission row exists in DB before schedule_grade_job is invoked.

        Even if the scheduler later fails, the submission is already committed.
        """
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "Coroutines yield control at await expressions.",
        })
        assert response.status_code == 201
        submission_id = response.json()["submission_id"]

        # Submission exists in DB
        db.expire_all()
        submission = db.get(Submission, submission_id)
        assert submission is not None
        assert submission.assessment_id == assessment.id
        assert submission.text_content is not None
