"""Unit tests for app.schemas.submission.SubmissionCreate's content-field
mutual-exclusivity validation.

Fix 3: text_content + github_url together is now the one deliberate
exception (a written explanation submitted alongside the repo URL those
files get fetched from) — every other combination stays mutually exclusive.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.submission import SubmissionType
from app.schemas.submission import SubmissionCreate


def _base(**overrides):
    fields = dict(
        assessment_id="assessment-1",
        token="token-1",
        submission_type=SubmissionType.text,
        github_url=None,
        text_content="An answer.",
    )
    fields.update(overrides)
    return fields


class TestGithubUrlCombinations:

    def test_github_url_alone_is_accepted(self):
        s = SubmissionCreate(**_base(
            submission_type=SubmissionType.github_url,
            github_url="https://github.com/octocat/hello-world",
            text_content=None,
        ))
        assert s.github_url == "https://github.com/octocat/hello-world"
        assert s.text_content is None

    def test_github_url_with_text_content_is_now_accepted(self):
        s = SubmissionCreate(**_base(
            submission_type=SubmissionType.github_url,
            github_url="https://github.com/octocat/hello-world",
            text_content="My implementation is in src/foo.py.",
        ))
        assert s.github_url == "https://github.com/octocat/hello-world"
        assert s.text_content == "My implementation is in src/foo.py."

    def test_github_url_missing_github_url_still_rejected(self):
        with pytest.raises(ValidationError, match="github_url is required"):
            SubmissionCreate(**_base(
                submission_type=SubmissionType.github_url,
                github_url=None,
                text_content=None,
            ))


class TestTextStillExclusive:

    def test_text_alone_is_accepted(self):
        s = SubmissionCreate(**_base(submission_type=SubmissionType.text, text_content="An answer."))
        assert s.text_content == "An answer."

    def test_text_missing_text_content_still_rejected(self):
        with pytest.raises(ValidationError, match="text_content is required"):
            SubmissionCreate(**_base(submission_type=SubmissionType.text, text_content=None))

    def test_text_with_github_url_still_rejected(self):
        """The combined exception only runs one way: github_url MAY carry
        text_content, but submission_type == text must still not carry a
        github_url — the combo is only meaningful/allowed under github_url."""
        with pytest.raises(ValidationError, match="github_url must not be set"):
            SubmissionCreate(**_base(
                submission_type=SubmissionType.text,
                text_content="An answer.",
                github_url="https://github.com/octocat/hello-world",
            ))


class TestFileStillExclusive:

    def test_file_with_github_url_still_rejected(self):
        with pytest.raises(ValidationError, match="github_url must not be set"):
            SubmissionCreate(**_base(
                submission_type=SubmissionType.file,
                text_content=None,
                github_url="https://github.com/octocat/hello-world",
            ))

    def test_file_with_text_content_still_rejected(self):
        with pytest.raises(ValidationError, match="text_content must not be set"):
            SubmissionCreate(**_base(
                submission_type=SubmissionType.file,
                text_content="An answer.",
                github_url=None,
            ))

    def test_file_with_neither_is_accepted(self):
        s = SubmissionCreate(**_base(submission_type=SubmissionType.file, text_content=None, github_url=None))
        assert s.text_content is None
        assert s.github_url is None
