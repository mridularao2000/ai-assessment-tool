from __future__ import annotations

from sqlalchemy.orm import Session

from app.interfaces.email import (
    AssessmentEmailData,
    EmailInterface,
    ReminderEmailData,
    ResultsEmailData,
)


class StubEmailAdapter:
    """Development stub — raises NotImplementedError for all email sends.
    Replace with ResendEmailAdapter when the email adapter is implemented.
    """

    def send_assessment_email(self, data: AssessmentEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: wire ResendEmailAdapter.")

    def send_reminder_email(self, data: ReminderEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: wire ResendEmailAdapter.")

    def send_results_email(self, data: ResultsEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: wire ResendEmailAdapter.")


class EmailService:
    """Sends transactional emails for assessment lifecycle events.

    Depends on:
      db    — SQLAlchemy session for loading Assessment/Submission/Grade data
      email — EmailInterface implementation (e.g. ResendEmailAdapter)
    """

    def __init__(self, db: Session, email: EmailInterface) -> None:
        self.db = db
        self.email = email

    def send_assessment_email(self, assessment_id: str) -> None:
        """Load assessment data and send the assessment delivery email.

        Raises:
            NotFoundError: if assessment_id does not exist.
            EmailDeliveryError: if the email provider fails.
        """
        raise NotImplementedError

    def send_reminder_email(self, assessment_id: str) -> None:
        """Load assessment data and send the 24-hour reminder email.

        Raises:
            NotFoundError: if assessment_id does not exist.
            EmailDeliveryError: if the email provider fails.
        """
        raise NotImplementedError

    def send_results_email(self, submission_id: str) -> None:
        """Load grade data and send the results email.

        Raises:
            NotFoundError: if submission_id or its grade does not exist.
            EmailDeliveryError: if the email provider fails.
        """
        raise NotImplementedError
