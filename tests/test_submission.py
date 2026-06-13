"""B + H: Submission validation tests and submission edge cases.

Covers:
  - Invalid token → 403
  - Assessment not found → 404
  - Assessment not active (scheduled / submitted / expired / completed) → 409
  - Duplicate submission → 409
  - Text submission happy path
  - GitHub URL submission happy path
  - File upload happy path
  - GET submission → 200 / 404
"""

import io

import pytest

from app.models.assessment import Assessment, AssessmentStatus
from app.models.submission import Submission, SubmissionType
from tests.conftest import (
    FakeScheduler,
    make_assessment,
    make_curriculum,
    make_submission,
)


VALID_TEXT = "Async/await allows cooperative multitasking without OS threads."


class TestSubmissionValidation:

    def test_invalid_token_returns_403(self, client, db):
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": "WRONG_TOKEN_VALUE",
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 403

    def test_missing_assessment_returns_404(self, client, db):
        response = client.post("/api/v1/submissions/", data={
            "assessment_id": "does-not-exist-id",
            "token": "some-token",
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 404

    def test_scheduled_assessment_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409

    def test_submitted_assessment_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)
        make_submission(db, assessment)  # moves status → submitted

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409

    def test_completed_assessment_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.completed)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409

    def test_expired_assessment_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.expired)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409

    def test_duplicate_submission_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        # First submission succeeds
        r1 = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })
        assert r1.status_code == 201

        # Status is now 'submitted' — second attempt must fail
        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": "A second attempt.",
        })
        assert response.status_code == 409


class TestTextSubmission:

    def test_text_submission_returns_201_and_submission_id(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 201
        data = response.json()
        assert "submission_id" in data
        assert data["submission_id"]

    def test_text_submission_persists_content(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        db.expire_all()
        submission = db.query(Submission).filter_by(assessment_id=assessment.id).first()
        assert submission is not None
        assert submission.submission_type == SubmissionType.text
        assert submission.text_content == VALID_TEXT
        assert submission.github_url is None
        assert submission.file_path is None

    def test_text_submission_transitions_status_to_submitted(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.submitted


class TestGithubUrlSubmission:

    def test_github_url_submission_accepted(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/project",
        })

        assert response.status_code == 201

    def test_github_url_submission_persists_url(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/project",
        })

        db.expire_all()
        submission = db.query(Submission).filter_by(assessment_id=assessment.id).first()
        assert submission is not None
        assert submission.submission_type == SubmissionType.github_url
        assert submission.github_url == "https://github.com/example/project"
        assert submission.text_content is None


class TestFileSubmission:

    def test_file_submission_persists_file_path(self, client, db, tmp_path, monkeypatch):
        """File content is written to uploads_dir and the relative path stored."""
        from app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
        try:
            curriculum = make_curriculum(db)
            assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

            file_bytes = b"My Python code submission content."
            response = client.post(
                "/api/v1/submissions/",
                data={
                    "assessment_id": assessment.id,
                    "token": token,
                    "submission_type": "file",
                },
                files={"file": ("solution.py", io.BytesIO(file_bytes), "text/plain")},
            )

            assert response.status_code == 201

            db.expire_all()
            submission = db.query(Submission).filter_by(assessment_id=assessment.id).first()
            assert submission is not None
            assert submission.submission_type == SubmissionType.file
            assert submission.file_path is not None
            assert "solution.py" in submission.file_path

            # Verify the file was actually written
            written = (tmp_path / submission.file_path).read_bytes()
            assert written == file_bytes
        finally:
            get_settings.cache_clear()


class TestGetSubmission:

    def test_get_existing_submission_returns_200(self, client, db):
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        response = client.get(f"/api/v1/submissions/{submission.id}")

        assert response.status_code == 200
        assert response.json()["submission_id"] == submission.id

    def test_get_missing_submission_returns_404(self, client, db):
        response = client.get("/api/v1/submissions/does-not-exist")

        assert response.status_code == 404

    def test_submission_schedule_grade_job_called(self, client, db, fake_scheduler):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        r = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })
        submission_id = r.json()["submission_id"]

        assert submission_id in fake_scheduler.schedule_grade_job_calls
