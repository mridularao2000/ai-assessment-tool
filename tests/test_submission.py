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
import uuid

import pytest

from app.models._utils import utcnow
from app.models.assessment import Assessment, AssessmentStatus
from app.models.late_submission_token import LateSubmissionToken
from app.models.submission import Submission, SubmissionType
from app.services.late_token_service import LateTokenService
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


class TestLateSubmissionTokens:

    def test_expired_with_token_returns_201(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 201

    def test_expired_with_token_spends_one_token_and_marks_late(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.late_submitted

        remaining = db.query(LateSubmissionToken).all()
        unused = [t for t in remaining if t.used_at is None]
        used = [t for t in remaining if t.used_at is not None]
        assert len(unused) == 1
        assert len(used) == 1
        assert used[0].used_by_assessment_id == assessment.id

    def test_expired_without_token_still_returns_409(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409

    def test_expired_with_zero_balance_returns_409_and_does_not_spend(self, client, db):
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        used_token = LateSubmissionToken(used_at=utcnow(), used_by_assessment_id=None)
        db.add(used_token)
        db.commit()

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409
        db.expire_all()
        assert db.get(LateSubmissionToken, used_token.id).used_by_assessment_id is None

    def test_expired_in_a_previous_calendar_month_rejected_even_with_tokens(self, client, db):
        """An assessment that expired last month is no longer late-eligible,
        regardless of token balance — the spec's "within that same calendar
        month" bound, which nothing previously enforced."""
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-40
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
        })

        assert response.status_code == 409
        db.expire_all()
        # No token was spent — rejected before the balance check ever mattered.
        assert all(t.used_at is None for t in db.query(LateSubmissionToken).all())

    def test_entry_based_assessment_type_late_submission_with_token(self, client, db):
        """Confirms the generic late-submission mechanism (built for
        standalone) also works for a curriculum-upload Assessment-type
        entry, since both share the same Assessment/Submission tables —
        the "extend, don't duplicate" architecture's whole point."""
        from app.models.curriculum import CurriculumEntryType

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/late-project",
        })

        assert response.status_code == 201
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.late_submitted

    def test_entry_based_midterm_late_submission_with_token_and_part1(self, client, db):
        """Late submission composes correctly with the Part 1 + Part 2
        two-field requirement for Midterms."""
        from app.models.curriculum import CurriculumEntryType

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-1
        )
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/late-project",
            "part1_text_content": "Late Part 1 answer.",
        })

        assert response.status_code == 201
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.late_submitted
        submission = db.query(Submission).filter_by(assessment_id=assessment.id).first()
        assert submission.part1_text_content == "Late Part 1 answer."
        assert submission.github_url == "https://github.com/example/late-project"

    def test_balance_endpoint_returns_current_balance_and_uuids(self, client, db):
        db.add(LateSubmissionToken())
        db.commit()

        response = client.get("/api/v1/late-tokens/")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 1
        assert len(data["tokens"]) == 1
        # Confirm it's a real UUID, not just a placeholder count.
        uuid.UUID(data["tokens"][0])

    def test_balance_endpoint_returns_zero_when_no_tokens_exist(self, client, db):
        response = client.get("/api/v1/late-tokens/")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 0
        assert data["tokens"] == []

    def test_grant_endpoint_tops_up_from_zero(self, client, db):
        response = client.post("/api/v1/late-tokens/grant")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 2
        assert len(data["tokens"]) == 2

    def test_grant_endpoint_is_idempotent_within_a_cycle(self, client, db):
        client.post("/api/v1/late-tokens/grant")
        response = client.post("/api/v1/late-tokens/grant")

        assert response.status_code == 200
        assert response.json()["balance"] == 2
        assert db.query(LateSubmissionToken).count() == 2


class TestLateTokenService:

    def test_grant_monthly_from_zero_issues_two_uuid_tokens(self, db):
        service = LateTokenService(db)

        balance = service.grant_monthly()

        assert balance == 2
        tokens = db.query(LateSubmissionToken).all()
        assert len(tokens) == 2
        for t in tokens:
            uuid.UUID(t.id)  # each token is a real UUID

    def test_grant_monthly_tops_up_to_two_when_already_at_one(self, db):
        db.add(LateSubmissionToken())
        db.commit()
        service = LateTokenService(db)

        balance = service.grant_monthly()

        assert balance == 2
        assert db.query(LateSubmissionToken).count() == 2

    def test_grant_monthly_issues_nothing_when_already_at_cap(self, db):
        db.add_all([LateSubmissionToken(), LateSubmissionToken()])
        db.commit()
        service = LateTokenService(db)

        balance = service.grant_monthly()

        assert balance == 2
        assert db.query(LateSubmissionToken).count() == 2

    def test_spend_with_zero_balance_raises(self, db):
        from app.exceptions import InvalidStateError

        service = LateTokenService(db)

        with pytest.raises(InvalidStateError):
            service.spend("some-assessment-id")

    def test_spend_marks_oldest_unused_token(self, db):
        older = LateSubmissionToken()
        db.add(older)
        db.commit()
        service = LateTokenService(db)

        spent_id = service.spend("assessment-123")
        db.commit()

        assert spent_id == older.id
        db.expire_all()
        refreshed = db.get(LateSubmissionToken, older.id)
        assert refreshed.used_at is not None
        assert refreshed.used_by_assessment_id == "assessment-123"


class TestMidtermTwoPartSubmission:
    """A Midterm submission needs Part 1 (assignment answer) alongside the
    normal Part 2 (project) fields — both in one request. See
    SubmissionService.create()'s part1_text_content handling.
    """

    def test_assessment_detail_reports_is_midterm_true(self, client, db):
        from app.models.curriculum import CurriculumEntryType

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={token}")

        assert response.status_code == 200
        assert response.json()["is_midterm"] is True

    def test_assessment_detail_reports_is_midterm_false_for_standalone(self, client, db):
        curriculum = make_curriculum(db)  # entry_type=None
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={token}")

        assert response.status_code == 200
        assert response.json()["is_midterm"] is False

    def test_midterm_submission_without_part1_rejected(self, client, db):
        from app.models.curriculum import CurriculumEntryType

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/project",
        })

        assert response.status_code == 409

    def test_midterm_submission_with_part1_and_project_succeeds(self, client, db):
        from app.models.curriculum import CurriculumEntryType

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/project",
            "part1_text_content": "Part 1 answer covering the cumulative assignment questions.",
        })

        assert response.status_code == 201
        db.expire_all()
        submission = db.query(Submission).filter_by(assessment_id=assessment.id).first()
        assert submission.github_url == "https://github.com/example/project"
        assert submission.part1_text_content == (
            "Part 1 answer covering the cumulative assignment questions."
        )

    def test_non_midterm_submission_rejects_part1_text_content(self, client, db):
        curriculum = make_curriculum(db)  # entry_type=None
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "text",
            "text_content": VALID_TEXT,
            "part1_text_content": "This should not be accepted here.",
        })

        assert response.status_code == 409
