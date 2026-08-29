from sqlalchemy import create_engine, inspect, text

from app.db.schema_sync import sync_schema

_OLD_CURRICULA = """
CREATE TABLE curricula (
    id VARCHAR(36) PRIMARY KEY,
    topic TEXT NOT NULL,
    target_completion_date DATE NOT NULL,
    extracted_content TEXT,
    mastery_achieved BOOLEAN,
    completed_at DATETIME,
    status VARCHAR NOT NULL,
    priority INTEGER,
    is_active_focus BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL
)
"""

_OLD_ASSESSMENTS = """
CREATE TABLE assessments (
    id VARCHAR(36) PRIMARY KEY,
    curriculum_id VARCHAR(36) NOT NULL,
    attempt_number INTEGER NOT NULL,
    assessment_text TEXT,
    rubric TEXT,
    duration_minutes INTEGER,
    generation_prompt_id VARCHAR(36),
    scheduled_at DATETIME NOT NULL,
    reminder_at DATETIME NOT NULL,
    due_date DATETIME NOT NULL,
    status VARCHAR NOT NULL,
    submission_token VARCHAR(255) NOT NULL UNIQUE,
    scheduled_job_ids JSON,
    created_at DATETIME NOT NULL
)
"""

_OLD_GRADES = """
CREATE TABLE grades (
    id VARCHAR(36) PRIMARY KEY,
    submission_id VARCHAR(36) NOT NULL UNIQUE,
    mastery_score FLOAT NOT NULL,
    weak_areas JSON,
    overall_feedback TEXT NOT NULL,
    grading_prompt_id VARCHAR(36),
    graded_at DATETIME NOT NULL
)
"""

_OLD_SUBMISSIONS = """
CREATE TABLE submissions (
    id VARCHAR(36) PRIMARY KEY,
    assessment_id VARCHAR(36) NOT NULL UNIQUE,
    submission_type VARCHAR NOT NULL,
    github_url TEXT,
    text_content TEXT,
    file_path TEXT,
    submitted_at DATETIME NOT NULL,
    raw_payload JSON
)
"""

_OLD_LATE_SUBMISSION_TOKENS = """
CREATE TABLE late_submission_tokens (
    id VARCHAR(36) PRIMARY KEY,
    issued_at DATETIME NOT NULL,
    used_at DATETIME,
    used_by_assessment_id VARCHAR(36)
)
"""

_OLD_CURRICULUM_UPLOADS = """
CREATE TABLE curriculum_uploads (
    id VARCHAR(36) PRIMARY KEY,
    source_filename TEXT NOT NULL,
    uploaded_at DATETIME NOT NULL
)
"""


def _make_old_schema_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(_OLD_CURRICULA))
        conn.execute(text(_OLD_ASSESSMENTS))
        conn.execute(text(_OLD_GRADES))
        conn.execute(text(_OLD_SUBMISSIONS))
        conn.execute(text(_OLD_LATE_SUBMISSION_TOKENS))
        conn.execute(text(_OLD_CURRICULUM_UPLOADS))
    return engine


class TestSyncSchema:
    def test_reproduces_missing_column_error_before_fix(self):
        # Confirms the exact failure mode reported in production: inserting
        # a row using a column the pre-existing table doesn't have yet.
        engine = _make_old_schema_engine()
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO curricula (id, topic, target_completion_date, "
                        "status, is_active_focus, created_at, entry_type) "
                        "VALUES ('x', 't', '2026-01-01', 'pending', 0, '2026-01-01', 'midterm')"
                    )
                )
            assert False, "expected OperationalError for missing column"
        except Exception as exc:
            assert "has no column named" in str(exc)

    def test_adds_all_missing_columns_to_preexisting_tables(self):
        engine = _make_old_schema_engine()
        applied = sync_schema(engine)

        assert set(applied) == {
            "curricula.entry_type",
            "curricula.upload_id",
            "curricula.chapter_label",
            "curricula.max_marks",
            "curricula.resources_hold",
            "curricula.last_hold_reminder_at",
            "assessments.part1_text",
            "assessments.part1_rubric",
            "assessments.part2_text",
            "assessments.part2_rubric",
            "assessments.send_job_claimed_at",
            "grades.part1_score",
            "grades.part2_score",
            "grades.score_earned",
            "grades.max_marks",
            "submissions.part1_text_content",
            "late_submission_tokens.curriculum_upload_id",
            "curriculum_uploads.closed_at",
            "curriculum_uploads.last_secondary_transcript_sent_at",
        }

        inspector = inspect(engine)
        curricula_cols = {c["name"] for c in inspector.get_columns("curricula")}
        assert "entry_type" in curricula_cols
        assert "resources_hold" in curricula_cols

    def test_post_fix_insert_with_new_columns_succeeds(self):
        # Same insert as the reproduction test above, now succeeds after sync.
        engine = _make_old_schema_engine()
        sync_schema(engine)

        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO curricula (id, topic, target_completion_date, "
                    "status, is_active_focus, created_at, entry_type, upload_id, "
                    "chapter_label, max_marks) "
                    "VALUES ('x', 't', '2026-01-01', 'pending', 0, '2026-01-01', "
                    "'midterm', NULL, 'Ch 1', 50.0)"
                )
            )
            row = conn.execute(
                text("SELECT entry_type, resources_hold, max_marks FROM curricula WHERE id='x'")
            ).fetchone()
        assert row[0] == "midterm"
        assert row[1] == 0  # default False
        assert row[2] == 50.0

    def test_idempotent_on_second_run(self):
        engine = _make_old_schema_engine()
        first = sync_schema(engine)
        second = sync_schema(engine)
        assert first
        assert second == []

    def test_noop_on_fresh_schema_created_by_create_all(self):
        from app.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        assert sync_schema(engine) == []

    def test_skips_tables_that_do_not_exist_yet(self):
        # A table absent entirely (e.g. never deployed at all) is left for
        # create_all() to create fresh with every column already present.
        engine = create_engine("sqlite:///:memory:")
        assert sync_schema(engine) == []
