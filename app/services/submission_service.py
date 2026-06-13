from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.interfaces.scheduler import SchedulerInterface
from app.models.assessment import Assessment, AssessmentStatus
from app.models.submission import Submission, SubmissionType
from app.utils import token_auth


class SubmissionService:
    """Accepts and persists submissions for active assessments.

    Depends on:
      db                — SQLAlchemy session for all persistence
      scheduler_service — SchedulerInterface to trigger grade job

    Does NOT grade, email, or call the LLM.
    The grade job is scheduled after the Submission row is committed so that
    a scheduler failure does not roll back the submission record.
    """

    def __init__(self, db: Session, scheduler_service: SchedulerInterface) -> None:
        self.db = db
        self.scheduler_service = scheduler_service

    # ── Public methods ─────────────────────────────────────────────────────────

    def create(
        self,
        assessment_id: str,
        token: str,
        submission_type: SubmissionType,
        github_url: Optional[str] = None,
        text_content: Optional[str] = None,
        uploaded_file: Optional[tuple[str, bytes]] = None,
    ) -> Submission:
        """Validate, persist, and schedule grading for a new submission.

        Steps:
          1. Load Assessment by assessment_id.
             Raise NotFoundError if missing.
          2. Verify token via token_auth.verify_submission_token.
             Raise InvalidTokenError if verification fails.
          3. Verify Assessment.status == active.
             Raise InvalidStateError otherwise.
          4. Resolve file_path for submission_type == file:
               - Resolve settings.uploads_dir, create directory if absent.
               - Write bytes under <uploads_dir>/<uuid>_<original_filename>.
          5. Create and flush the Submission row.
          6. Transition Assessment.status → submitted.
          7. Commit.
          8. Call scheduler_service.schedule_grade_job(submission.id).
             Scheduler failure is NOT rolled back — submission is already committed.
          9. Return the Submission.

        Raises:
            NotFoundError: if assessment_id does not exist.
            InvalidTokenError: if the token does not match.
            InvalidStateError: if Assessment.status is not active, or a
                               submission already exists for this assessment.
        """
        assessment = self.db.get(Assessment, assessment_id)
        if assessment is None:
            raise NotFoundError(f"Assessment {assessment_id!r} not found.")

        if not token_auth.verify_submission_token(assessment_id, token):
            raise InvalidTokenError(f"Invalid token for assessment {assessment_id!r}.")

        if assessment.status != AssessmentStatus.active:
            raise InvalidStateError(
                f"Assessment {assessment_id!r} is not active "
                f"(current status: {assessment.status.value!r})."
            )

        if assessment.submission is not None:
            raise InvalidStateError(
                f"Assessment {assessment_id!r} already has a submission."
            )

        file_path: Optional[str] = None
        if submission_type == SubmissionType.file:
            file_path = self._save_file(uploaded_file)

        submission = Submission(
            assessment_id=assessment_id,
            submission_type=submission_type,
            github_url=github_url if submission_type == SubmissionType.github_url else None,
            text_content=text_content if submission_type == SubmissionType.text else None,
            file_path=file_path,
        )
        self.db.add(submission)
        self.db.flush()  # populate submission.id before status transition

        assessment.status = AssessmentStatus.submitted
        self.db.commit()
        self.db.refresh(submission)

        self.scheduler_service.schedule_grade_job(submission.id)

        return submission

    def get(self, submission_id: str) -> Submission:
        """Return a Submission by id.

        Raises:
            NotFoundError: if submission_id does not exist.
        """
        submission = self.db.get(Submission, submission_id)
        if submission is None:
            raise NotFoundError(f"Submission {submission_id!r} not found.")
        return submission

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _save_file(self, uploaded_file: Optional[tuple[str, bytes]]) -> str:
        """Write uploaded bytes to disk and return the relative file path.

        The path is stored relative to settings.uploads_dir so that the
        uploads root can be relocated without invalidating existing records.

        Raises:
            InvalidStateError: if uploaded_file is None.
        """
        if uploaded_file is None:
            raise InvalidStateError(
                "submission_type is 'file' but no file was uploaded."
            )

        original_filename, file_bytes = uploaded_file
        settings = get_settings()
        uploads_root = Path(settings.uploads_dir)
        uploads_root.mkdir(parents=True, exist_ok=True)

        stored_name = f"{uuid.uuid4()}_{original_filename}"
        dest = uploads_root / stored_name
        dest.write_bytes(file_bytes)

        return stored_name
