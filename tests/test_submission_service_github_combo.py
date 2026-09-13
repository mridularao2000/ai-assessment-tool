"""Regression coverage for Fix 3 at the service layer: SubmissionService
must persist text_content alongside github_url when both are submitted
together, not silently drop it (the schema validator alone isn't enough —
the service used to null out text_content for anything but
submission_type == text)."""
from __future__ import annotations

from app.models.submission import Submission, SubmissionType
from app.services.late_token_service import LateTokenService
from app.services.submission_service import SubmissionService
from tests.conftest import FakeScheduler, make_assessment, make_curriculum, seed_prompt_templates


class TestSubmissionServiceGithubUrlCombo:

    def test_text_content_persisted_alongside_github_url(self, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum)

        svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission = svc.create(
            assessment_id=assessment.id,
            token=token,
            submission_type=SubmissionType.github_url,
            github_url="https://github.com/octocat/hello-world",
            text_content="My implementation is in src/foo.py.",
        )

        db.expire_all()
        persisted = db.get(Submission, submission.id)
        assert persisted.submission_type == SubmissionType.github_url
        assert persisted.github_url == "https://github.com/octocat/hello-world"
        assert persisted.text_content == "My implementation is in src/foo.py."

    def test_github_url_alone_still_leaves_text_content_none(self, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum)

        svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission = svc.create(
            assessment_id=assessment.id,
            token=token,
            submission_type=SubmissionType.github_url,
            github_url="https://github.com/octocat/hello-world",
        )

        db.expire_all()
        persisted = db.get(Submission, submission.id)
        assert persisted.text_content is None

    def test_text_submission_type_unaffected(self, db):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, token = make_assessment(db, curriculum)

        svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission = svc.create(
            assessment_id=assessment.id,
            token=token,
            submission_type=SubmissionType.text,
            text_content="Plain text answer.",
        )

        db.expire_all()
        persisted = db.get(Submission, submission.id)
        assert persisted.text_content == "Plain text answer."
        assert persisted.github_url is None
