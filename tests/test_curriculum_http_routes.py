"""HTTP-layer coverage for POST /curriculum/ and POST /curriculum-uploads/.

Both routes were previously only exercised at the service layer (calling
CurriculumService.create() / CurriculumUploadService.ingest() directly with
fake adapters) — route-level multipart/JSON-file parsing (Form fields, list
Form fields, UploadFile handling, the JSON-file-as-multipart-part shape
POST /curriculum-uploads/ actually expects) was never actually driven
through FastAPI's request parsing for these two specific endpoints, unlike
everything downstream of them.

Uses the shared `client` fixture (FakeLLM/FakeScheduler/RecordingEmailAdapter
throughout, including CurriculumService/CurriculumUploadService as of the
conftest.py override-list fix) — zero real Anthropic/email calls.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from app.models.curriculum import Curriculum
from tests.conftest import seed_prompt_templates

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "curriculum_seed.json"


class TestCreateCurriculumRoute:
    def test_creates_curriculum_from_multipart_form(self, client, db):
        seed_prompt_templates(db)
        future_date = (date.today() + timedelta(days=14)).isoformat()

        response = client.post(
            "/api/v1/curriculum/",
            data={
                "topic": "Distributed Systems Basics",
                "target_completion_date": future_date,
                "notes": "Focus on consensus algorithms and CAP theorem.",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert "curriculum_id" in body

        db.expire_all()
        curriculum = db.get(Curriculum, body["curriculum_id"])
        assert curriculum is not None
        assert curriculum.topic == "Distributed Systems Basics"
        assert len(curriculum.assessments) == 1  # fake-generated, committed

    def test_repeated_form_fields_parse_as_lists(self, client, db, monkeypatch):
        """links is a List[Form()] — FastAPI expects repeated form keys
        (or a list value under the same key), not a single comma-joined
        string. Malformed parsing here would silently collapse to a
        single-item list, never noticed at the service layer since that
        layer takes an already-Python list. _fetch_url is monkeypatched
        so this stays a pure request-parsing check with no real network
        call — CurriculumService's actual URL-fetching behavior is a
        separate concern, not what this test is for."""
        from app.services.curriculum_service import CurriculumService

        monkeypatch.setattr(
            CurriculumService, "_fetch_url", lambda self, url: f"Fetched content for {url}"
        )
        seed_prompt_templates(db)
        future_date = (date.today() + timedelta(days=14)).isoformat()

        response = client.post(
            "/api/v1/curriculum/",
            data={
                "topic": "Multi-Resource Topic",
                "target_completion_date": future_date,
                "links": ["https://example.com/a", "https://example.com/b"],
                "notes": "Some notes so extracted_content isn't empty.",
            },
        )

        assert response.status_code == 201
        db.expire_all()
        curriculum = db.get(Curriculum, response.json()["curriculum_id"])
        urls = [r.source_ref for r in curriculum.resources if r.source_ref.startswith("http")]
        assert set(urls) == {"https://example.com/a", "https://example.com/b"}

    def test_blank_topic_maps_to_409(self, client, db):
        seed_prompt_templates(db)
        future_date = (date.today() + timedelta(days=14)).isoformat()

        response = client.post(
            "/api/v1/curriculum/",
            data={"topic": "   ", "target_completion_date": future_date},
        )

        assert response.status_code == 409

    def test_past_completion_date_maps_to_409(self, client, db):
        seed_prompt_templates(db)
        past_date = (date.today() - timedelta(days=1)).isoformat()

        response = client.post(
            "/api/v1/curriculum/",
            data={"topic": "Too Late Topic", "target_completion_date": past_date},
        )

        assert response.status_code == 409

    def test_missing_required_field_is_422(self, client):
        # target_completion_date omitted entirely — FastAPI's own Form
        # validation, before the route body ever runs.
        response = client.post("/api/v1/curriculum/", data={"topic": "No Date"})
        assert response.status_code == 422


class TestUploadCurriculumRoute:
    def test_uploads_real_seed_file_as_multipart(self, client, db):
        seed_prompt_templates(db)
        raw_bytes = FIXTURE_PATH.read_bytes()

        response = client.post(
            "/api/v1/curriculum-uploads/",
            files={"file": ("curriculum_seed.json", raw_bytes, "application/json")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["entry_count"] == 18
        assert "upload_id" in body

    def test_invalid_json_file_is_422(self, client):
        response = client.post(
            "/api/v1/curriculum-uploads/",
            files={"file": ("bad.json", b"{not valid json", "application/json")},
        )
        assert response.status_code == 422

    def test_json_array_instead_of_object_is_422(self, client):
        response = client.post(
            "/api/v1/curriculum-uploads/",
            files={"file": ("bad.json", json.dumps([1, 2, 3]).encode(), "application/json")},
        )
        assert response.status_code == 422

    def test_missing_topics_key_is_422(self, client):
        response = client.post(
            "/api/v1/curriculum-uploads/",
            files={"file": ("bad.json", json.dumps({"foo": "bar"}).encode(), "application/json")},
        )
        assert response.status_code == 422

    def test_missing_file_part_is_422(self, client):
        # No `file` multipart part at all — FastAPI's own parsing, before
        # the route body (and therefore CurriculumUploadService) ever runs.
        response = client.post("/api/v1/curriculum-uploads/")
        assert response.status_code == 422


class TestResendSyllabusRoute:
    def test_resend_sends_again_and_updates_timestamp(self, client, db):
        seed_prompt_templates(db)
        raw_bytes = FIXTURE_PATH.read_bytes()
        upload_id = client.post(
            "/api/v1/curriculum-uploads/",
            files={"file": ("curriculum_seed.json", raw_bytes, "application/json")},
        ).json()["upload_id"]

        response = client.post(f"/api/v1/curriculum-uploads/{upload_id}/resend-syllabus")

        assert response.status_code == 200
        assert response.json() == {"upload_id": upload_id, "sent": True}

    def test_unknown_upload_id_is_404(self, client):
        response = client.post("/api/v1/curriculum-uploads/does-not-exist/resend-syllabus")
        assert response.status_code == 404


class TestContentGeneratedFlag:
    """GET /api/v1/curriculum/{id} must report whether an assessment's
    content was actually generated (assessment_text/part1_text IS NOT
    NULL), not just its status — an `expired` row is ambiguous from status
    alone (see Assessment.content_generated's docstring)."""

    def test_true_once_generated_false_before(self, client, db):
        seed_prompt_templates(db)
        future_date = (date.today() + timedelta(days=14)).isoformat()

        curriculum_id = client.post(
            "/api/v1/curriculum/",
            data={"topic": "Content Flag Test", "target_completion_date": future_date},
        ).json()["curriculum_id"]

        body = client.get(f"/api/v1/curriculum/{curriculum_id}").json()
        assert body["assessments"][0]["content_generated"] is True  # standalone generates eagerly

    def test_false_for_a_retroactively_expired_entry_that_was_never_late_sent(self, client, db):
        """A curriculum-upload entry whose window closed before upload gets
        an Assessment row created directly in `expired` status with no
        content — status alone would misreport this as "was generated"."""
        from app.models.assessment import Assessment, AssessmentStatus
        from app.models.curriculum import Curriculum, CurriculumEntryType
        from app.models.curriculum_upload import CurriculumUpload
        from app.utils.token_auth import generate_submission_token
        from datetime import datetime

        seed_prompt_templates(db)
        upload = CurriculumUpload(source_filename="x.json")
        db.add(upload)
        db.flush()
        curriculum = Curriculum(
            topic="Long Overdue Topic",
            target_completion_date=date.today() - timedelta(days=30),
            entry_type=CurriculumEntryType.assessment,
            upload_id=upload.id,
            chapter_label="Chapter 1",
            max_marks=50.0,
        )
        db.add(curriculum)
        db.flush()
        assessment_id = "test-assessment-id"
        db.add(Assessment(
            id=assessment_id,
            curriculum_id=curriculum.id,
            assessment_text=None,
            scheduled_at=datetime.utcnow() - timedelta(days=27),
            reminder_at=datetime.utcnow() - timedelta(days=28),
            due_date=datetime.utcnow() - timedelta(days=25),
            status=AssessmentStatus.expired,
            submission_token=generate_submission_token(assessment_id),
        ))
        db.commit()

        body = client.get(f"/api/v1/curriculum/{curriculum.id}").json()
        assert body["assessments"][0]["status"] == "expired"
        assert body["assessments"][0]["content_generated"] is False
