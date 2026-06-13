from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.curriculum import CurriculumStatus
from app.schemas.assessment import AssessmentSummary
from app.schemas.resource import ResourceResponse


class CurriculumCreate(BaseModel):
    """Request body for POST /api/v1/curriculum.

    Because this endpoint accepts multipart/form-data with file uploads,
    the route assembles this schema after collecting all form fields and
    counting UploadFile items:

        schema = CurriculumCreate(
            topic=topic,
            target_completion_date=target_completion_date,
            links=parsed_links,
            github_repos=parsed_repos,
            notes=notes,
            has_file_uploads=len(files) > 0,
        )

    `has_file_uploads` is injected by the route and excluded from
    serialisation — it exists only to allow the validator to satisfy
    the "at least one resource" requirement when files are present.
    """

    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        strip_whitespace=True,
    )
    target_completion_date: date
    links: list[str] = Field(
        default_factory=list,
        description="HTTP/HTTPS URLs to learning resources",
    )
    github_repos: list[str] = Field(
        default_factory=list,
        description="GitHub repository URLs",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-form text notes to include as a resource",
    )
    # Set True by the route when UploadFile items are present.
    # Excluded from schema serialisation.
    has_file_uploads: bool = Field(default=False, exclude=True)

    @field_validator("links", "github_repos", mode="before")
    @classmethod
    def validate_urls(cls, v: list) -> list:
        for url in v:
            if not isinstance(url, str):
                raise ValueError(f"Expected a string URL, got {type(url).__name__}")
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"'{url}' is not a valid HTTP or HTTPS URL")
        return v

    @model_validator(mode="after")
    def require_at_least_one_resource(self) -> "CurriculumCreate":
        has_links = bool(self.links)
        has_repos = bool(self.github_repos)
        has_notes = bool(self.notes and self.notes.strip())
        has_files = self.has_file_uploads
        if not (has_links or has_repos or has_notes or has_files):
            raise ValueError(
                "At least one resource must be supplied: "
                "a URL, GitHub repo, note, or uploaded file."
            )
        return self


class CurriculumCreateResponse(BaseModel):
    curriculum_id: str


class CurriculumResponse(BaseModel):
    """Full curriculum view, including nested assessment summaries and resources."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    topic: str
    target_completion_date: date
    status: CurriculumStatus
    mastery_achieved: Optional[bool] = None
    completed_at: Optional[datetime] = None
    priority: Optional[int] = None
    is_active_focus: bool
    created_at: datetime
    assessments: list[AssessmentSummary] = []
    resources: list[ResourceResponse] = []
