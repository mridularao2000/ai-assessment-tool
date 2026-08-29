"""GET /health/prompts must report "degraded" for ANY missing required
prompt template — including midterm_generation/midterm_grading, the two
templates every Midterm (100 marks each, including the capstone) needs
for generation/grading to succeed at all. Before this fix, REQUIRED_SLUGS
omitted both, so health reported "ok" even with them missing/deactivated.
"""
from __future__ import annotations

from app.db.seed import REQUIRED_SLUGS
from app.models.prompt_template import PromptTemplate
from tests.conftest import seed_prompt_templates


def test_missing_midterm_templates_are_now_required(db):
    assert "midterm_generation" in REQUIRED_SLUGS
    assert "midterm_grading" in REQUIRED_SLUGS


def test_ok_once_every_required_template_is_seeded(client, db):
    seed_prompt_templates(db)

    response = client.get("/health/prompts")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["missing"] == []
    assert "midterm_generation" in body["present"]
    assert "midterm_grading" in body["present"]


def test_degraded_when_midterm_generation_is_missing(client, db):
    seed_prompt_templates(db)
    db.query(PromptTemplate).filter(
        PromptTemplate.slug == "midterm_generation"
    ).delete()
    db.commit()

    response = client.get("/health/prompts")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "midterm_generation" in body["missing"]


def test_degraded_when_midterm_grading_is_deactivated(client, db):
    """Deactivated (is_active=False), not just deleted — the seeder's own
    'active row' check is what check_missing_templates() keys off."""
    seed_prompt_templates(db)
    template = (
        db.query(PromptTemplate)
        .filter(PromptTemplate.slug == "midterm_grading")
        .first()
    )
    template.is_active = False
    db.commit()

    response = client.get("/health/prompts")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "midterm_grading" in body["missing"]
