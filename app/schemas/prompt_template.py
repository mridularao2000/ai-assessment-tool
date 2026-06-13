from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.prompt_template import VALID_SLUGS

_VALID_SLUGS_STR = ", ".join(sorted(VALID_SLUGS))


class PromptTemplateCreate(BaseModel):
    """Request body for creating a new prompt template version."""

    slug: str = Field(
        ...,
        description=f"Must be one of: {_VALID_SLUGS_STR}",
    )
    version: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^v\d+$",
        description="Version string in the form 'v1', 'v2', etc.",
    )
    body: str = Field(
        ...,
        min_length=10,
        description="Prompt template body. Use {placeholder} syntax for variable slots.",
    )
    is_active: bool = Field(
        default=False,
        description=(
            "Whether this version is the active one for its slug. "
            "At most one version per slug should be active. "
            "Enforced at the service layer."
        ),
    )

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        if v not in VALID_SLUGS:
            raise ValueError(
                f"slug must be one of: {_VALID_SLUGS_STR}. Got '{v}'."
            )
        return v


class PromptTemplateUpdate(BaseModel):
    """Partial update — only is_active can be changed after creation."""

    is_active: bool


class PromptTemplateResponse(BaseModel):
    """Full prompt template read view."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    version: str
    body: str
    is_active: bool
    created_at: datetime
