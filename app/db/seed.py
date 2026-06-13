"""Idempotent prompt-template seeder.

Run directly to seed a local database:
    python -m app.db.seed

Called automatically from app startup (lifespan) to ensure production
environments have all required templates on first boot.
"""
from __future__ import annotations

import logging
from typing import Final

from sqlalchemy.orm import Session

from app.models.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)

# ── Template definitions ──────────────────────────────────────────────────────
#
# Each body uses Python str.format()-style substitution.
# Variables available at render time are shown in the leading comment.
# Literal JSON braces in the body MUST be doubled: {{ and }}.
#
# ─────────────────────────────────────────────────────────────────────────────

# Variables: {topic}, {curriculum_content}
_ASSESSMENT_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform.

A student has completed a self-directed learning curriculum and needs to be \
assessed on their progress.

Topic: {topic}

Curriculum Materials (what the student studied):
{curriculum_content}

IMPORTANT — choose the right assessment FORMAT based on the curriculum:

1. CODING / IMPLEMENTATION-FOCUSED curriculum
   Indicators: building components, writing code, implementing features, \
creating apps, practicing a framework or language, working with APIs or \
data structures.
   → Generate a TIMED CODING PROBLEM. State the problem clearly with \
requirements, constraints, and example inputs/outputs (if applicable). \
Frame it as: "You have X minutes to implement the following…". \
Do NOT ask written explanation questions — the code is the answer.
   → Set duration_minutes to 60–120 depending on complexity.

2. CONCEPTUAL / THEORETICAL curriculum
   Indicators: reading documentation, studying concepts, understanding \
architecture, comparing approaches, reading articles or books.
   → Use written questions: explain, compare, describe trade-offs, \
give examples. No coding required.
   → Set duration_minutes to 45–90.

3. MIXED curriculum (coding practice + theory)
   → Combine: 1–2 short written questions AND 1 coding exercise.
   → Set duration_minutes to 90–120.

Infer the format entirely from the curriculum materials above. \
Do not default to written questions for coding-focused content.

Respond with a single JSON object containing exactly these fields:
{{
  "assessment_text": "The full assessment presented to the student. Use markdown formatting. For coding assessments: state the problem, requirements, constraints, and examples clearly. For written assessments: number each question. For mixed: separate the sections with headings.",
  "rubric": "A detailed marking rubric. For coding: describe what a correct implementation looks like, edge cases that must be handled, and what partial credit covers. For written: per-question expected answers with full/partial/no marks.",
  "duration_minutes": 90
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {curriculum_content}
_CURRICULUM_ANALYSIS = """\
You are an expert curriculum analyst for a software engineering learning platform.

Analyse the following learning materials and provide a structured summary.

Topic: {topic}

Curriculum Materials:
{curriculum_content}

Respond with a single JSON object containing exactly these fields:
{{
  "summary": "A concise 2–3 sentence summary of what this curriculum covers and what a student will be able to do after completing it.",
  "key_topics": ["topic1", "topic2", "topic3"],
  "complexity_level": "intermediate",
  "estimated_study_hours": 10.0
}}

Rules:
- complexity_level must be exactly one of: "beginner", "intermediate", "advanced"
- key_topics should list 4–8 specific technical concepts covered by the materials
- estimated_study_hours should reflect realistic self-study time (e.g. 5.0, 12.5, 20.0)
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {curriculum_content}, {previous_mastery_score},
#            {weak_areas}, {attempt_number}
_RETEST_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform.

A student is retaking an assessment. Create a targeted retest that focuses on \
their identified areas of weakness.

Topic: {topic}

Curriculum Materials:
{curriculum_content}

Previous Attempt Results:
- Attempt number: {attempt_number}
- Previous mastery score: {previous_mastery_score}%
- Identified weak areas: {weak_areas}

IMPORTANT — match the retest FORMAT to the curriculum type (same rules as the original assessment):
- CODING / IMPLEMENTATION-FOCUSED → timed coding problem targeting the weak areas.
- CONCEPTUAL / THEORETICAL → written questions on the weak areas.
- MIXED → 1 written question + 1 coding exercise covering the weak areas.

Design a retest that:
- Focuses primarily (70 %+) on the student's identified weak areas
- Includes some questions on stronger areas to confirm retained knowledge
- Uses different questions and scenarios from previous attempts
- Is calibrated to let an improved student demonstrate that improvement

Respond with a single JSON object containing exactly these fields:
{{
  "assessment_text": "The full retest presented to the student. Use markdown formatting. For coding retests: state the problem and requirements clearly. For written retests: number each question.",
  "rubric": "A detailed marking rubric. For coding: what a correct implementation covers, edge cases, partial credit. For written: per-question expected answers with full/partial/no marks.",
  "duration_minutes": 90
}}

Set duration_minutes based on the complexity of the weak areas (60–120 minutes).
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {assessment_text}, {rubric}, {curriculum_content},
#            {submission_content}
_GRADING = """\
You are an expert technical assessor for a software engineering learning platform.

Grade the following student submission against the assessment and rubric provided.

Assessment:
{assessment_text}

Grading Rubric:
{rubric}

Curriculum Reference (for context):
{curriculum_content}

Student Submission:
{submission_content}

Evaluate the submission carefully and provide an objective grade.

Respond with a single JSON object containing exactly these fields:
{{
  "mastery_score": 75.0,
  "weak_areas": ["specific concept 1", "specific concept 2"],
  "overall_feedback": "Detailed, constructive feedback for the student."
}}

Rules:
- mastery_score must be a number between 0.0 and 100.0
- weak_areas lists 0–5 specific topics where the student showed gaps \
(use an empty list [] if they demonstrated strong mastery throughout)
- overall_feedback should be 2–4 sentences: acknowledge strengths, name \
specific gaps, and give one actionable improvement suggestion
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {reason}
_RESCHEDULE_CLASSIFICATION = """\
You are an assessment coordinator evaluating a student's request to reschedule \
their technical assessment.

Student's reason for rescheduling:
{reason}

Classify the reason into exactly one of the following categories:
- interview          — student has a job/internship interview
- medical            — illness, medical appointment, or health emergency
- emergency          — family emergency, bereavement, or other acute personal crisis
- work_escalation    — urgent work deadline, on-call incident, or business-critical task
- procrastination    — student is not ready, wants more preparation time without a specific reason
- lack_of_preparation — student explicitly states they have not studied enough
- missed_schedule    — student simply forgot or missed the scheduled time

Respond with a single JSON object containing exactly these fields:
{{
  "category": "medical",
  "reasoning": "One sentence explaining why this reason maps to the chosen category."
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# ── Public constants ──────────────────────────────────────────────────────────

# Maps slug → (version, body). Version is bumped when the prompt changes
# in a way that meaningfully affects LLM behaviour.
SEED_TEMPLATES: Final[dict[str, tuple[str, str]]] = {
    "assessment_generation":     ("1.1", _ASSESSMENT_GENERATION),
    "curriculum_analysis":       ("1.0", _CURRICULUM_ANALYSIS),
    "retest_generation":         ("1.1", _RETEST_GENERATION),
    "grading":                   ("1.0", _GRADING),
    "reschedule_classification": ("1.0", _RESCHEDULE_CLASSIFICATION),
}

# The subset whose absence will block core user-facing functionality.
REQUIRED_SLUGS: Final[frozenset[str]] = frozenset({
    "assessment_generation",
    "curriculum_analysis",
    "retest_generation",
    "grading",
    "reschedule_classification",
})


# ── Seeder ────────────────────────────────────────────────────────────────────

def seed_prompt_templates(db: Session) -> list[str]:
    """Idempotently insert or update prompt templates.

    - Missing templates are inserted.
    - Existing templates whose version differs from SEED_TEMPLATES are updated
      in-place (body + version). This lets a version bump propagate to existing
      DBs by re-running `python -m app.db.seed`.
    - Templates already at the current version are left untouched.

    Returns the list of slugs that were inserted or updated.
    """
    changed: list[str] = []
    for slug, (version, body) in SEED_TEMPLATES.items():
        exists = (
            db.query(PromptTemplate)
            .filter(PromptTemplate.slug == slug, PromptTemplate.is_active.is_(True))
            .first()
        )
        if exists is None:
            db.add(PromptTemplate(slug=slug, version=version, body=body, is_active=True))
            changed.append(slug)
        elif exists.version != version:
            exists.body = body
            exists.version = version
            changed.append(slug)

    if changed:
        db.commit()
        logger.info("Seeded/updated %d prompt template(s): %s", len(changed), changed)
    return changed


def check_missing_templates(db: Session) -> list[str]:
    """Return the list of REQUIRED_SLUGS that have no active template row."""
    return [
        slug for slug in sorted(REQUIRED_SLUGS)
        if not db.query(PromptTemplate)
        .filter(PromptTemplate.slug == slug, PromptTemplate.is_active.is_(True))
        .first()
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        inserted = seed_prompt_templates(db)
        missing = check_missing_templates(db)
        if inserted:
            print(f"Seeded {len(inserted)} template(s): {inserted}")
        else:
            print("All required templates already present — nothing to do.")
        if missing:
            print(f"WARNING: still missing after seed: {missing}")
    finally:
        db.close()
