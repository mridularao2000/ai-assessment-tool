from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.submission import SubmissionType


class SubmissionCreate(BaseModel):
    """Request body for POST /api/v1/submissions.

    Exactly one content field must be provided, matching submission_type:
      - github_url  → github_url must be set; text_content must be absent
      - text        → text_content must be set; github_url must be absent
      - file        → neither github_url nor text_content should be set;
                      the route validates that an UploadFile was received

    submission_token is provided via the `token` field so it can be verified
    before the submission is accepted.
    """

    assessment_id: str = Field(..., min_length=1)
    token: str = Field(..., min_length=1)
    submission_type: SubmissionType
    github_url: Optional[str] = None
    text_content: Optional[str] = None
    # `file` is not modelled here — it is received as UploadFile by the route.

    @field_validator("github_url")
    @classmethod
    def validate_github_url_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("github_url must be a valid HTTP or HTTPS URL")
        return v

    @model_validator(mode="after")
    def validate_exactly_one_content(self) -> "SubmissionCreate":
        t = self.submission_type

        if t == SubmissionType.github_url:
            if not self.github_url:
                raise ValueError(
                    "github_url is required when submission_type is 'github_url'"
                )
            if self.text_content:
                raise ValueError(
                    "text_content must not be set when submission_type is 'github_url'"
                )

        elif t == SubmissionType.text:
            if not self.text_content:
                raise ValueError(
                    "text_content is required when submission_type is 'text'"
                )
            if self.github_url:
                raise ValueError(
                    "github_url must not be set when submission_type is 'text'"
                )

        elif t == SubmissionType.file:
            if self.github_url:
                raise ValueError(
                    "github_url must not be set when submission_type is 'file'"
                )
            if self.text_content:
                raise ValueError(
                    "text_content must not be set when submission_type is 'file'"
                )
            # Presence of the actual file is validated by the route after
            # checking the UploadFile parameter.

        return self


class SubmissionResponse(BaseModel):
    submission_id: str
