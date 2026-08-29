"""Additive schema reconciliation, shared by app startup and the one-off
migrations/align_assessment_db_schema.py CLI script.

Base.metadata.create_all() only creates tables that don't exist yet — it
never alters an existing table to add a new column. On Render the SQLite
file lives on a persistent disk and survives every redeploy, so any column
added to a model after its table already exists in production needs an
explicit ALTER TABLE here, or every write through the ORM raises "no such
column" as an uncaught 500. This is invisible in the test suite, since
every test starts from a blank DB that create_all() creates complete, and
was previously only fixed for the local dev DB via the CLI script above —
never applied to Render's disk, which is why it kept failing there.

PENDING_COLUMNS is additive-only history: once a column ships here, it
stays forever, even if the model changes again later. It is not a mirror
of the current model definitions.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

PENDING_COLUMNS: dict[str, list[str]] = {
    "assessments": [
        "ALTER TABLE assessments ADD COLUMN part1_text TEXT",
        "ALTER TABLE assessments ADD COLUMN part1_rubric TEXT",
        "ALTER TABLE assessments ADD COLUMN part2_text TEXT",
        "ALTER TABLE assessments ADD COLUMN part2_rubric TEXT",
        "ALTER TABLE assessments ADD COLUMN send_job_claimed_at DATETIME",
    ],
    "curricula": [
        "ALTER TABLE curricula ADD COLUMN entry_type VARCHAR(10)",
        "ALTER TABLE curricula ADD COLUMN upload_id VARCHAR(36)",
        "ALTER TABLE curricula ADD COLUMN chapter_label TEXT",
        "ALTER TABLE curricula ADD COLUMN max_marks FLOAT",
        "ALTER TABLE curricula ADD COLUMN resources_hold BOOLEAN NOT NULL DEFAULT 0",
        "ALTER TABLE curricula ADD COLUMN last_hold_reminder_at DATETIME",
    ],
    "grades": [
        "ALTER TABLE grades ADD COLUMN part1_score FLOAT",
        "ALTER TABLE grades ADD COLUMN part2_score FLOAT",
        "ALTER TABLE grades ADD COLUMN score_earned FLOAT",
        "ALTER TABLE grades ADD COLUMN max_marks FLOAT",
    ],
    "submissions": [
        "ALTER TABLE submissions ADD COLUMN part1_text_content TEXT",
    ],
    "late_submission_tokens": [
        "ALTER TABLE late_submission_tokens ADD COLUMN curriculum_upload_id VARCHAR(36)",
    ],
    "curriculum_uploads": [
        "ALTER TABLE curriculum_uploads ADD COLUMN closed_at DATETIME",
    ],
}


def sync_schema(engine: Engine) -> list[str]:
    """Add any column in PENDING_COLUMNS missing from its (already
    existing) table. Returns the "table.column" additions actually
    applied — empty on a fully up-to-date DB, e.g. a fresh install that
    create_all() just created from scratch. Tables that don't exist at all
    yet are skipped — create_all() creates those complete, columns
    included."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    applied: list[str] = []
    with engine.begin() as conn:
        for table, statements in PENDING_COLUMNS.items():
            if table not in existing_tables:
                continue
            existing_columns = {c["name"] for c in inspector.get_columns(table)}
            for stmt in statements:
                col_name = stmt.split("ADD COLUMN")[1].strip().split(" ")[0]
                if col_name in existing_columns:
                    continue
                conn.execute(text(stmt))
                applied.append(f"{table}.{col_name}")
    return applied
