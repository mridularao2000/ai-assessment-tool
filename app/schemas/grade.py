from typing import Optional

from pydantic import BaseModel, Field


class GradeResponse(BaseModel):
    """Grading result returned to the user after submission is evaluated.

    Fields intentionally excluded:
      - grading_prompt_id — internal audit field
      - submission_id     — not needed by the caller

    `passed` is not computed here — the route/service supplies it explicitly
    as: mastery_score >= settings.mastery_threshold.  This keeps the schema
    free of settings coupling.
    """

    mastery_score: float = Field(..., ge=0.0, le=100.0)
    overall_feedback: str
    weak_areas: Optional[list[str]] = None
    passed: bool
    # Populated only for the new MCQ + Coding Project format (see
    # AssessmentGenerationResult) — None for a legacy-format or Midterm
    # grade, even though Midterm grades also happen to populate
    # Grade.part1_score/part2_score internally; this field is specifically
    # the MCQ/project split, not a generic alias for those columns.
    mcq_score: Optional[float] = Field(None, ge=0.0, le=30.0)
    project_score: Optional[float] = Field(None, ge=0.0, le=70.0)
