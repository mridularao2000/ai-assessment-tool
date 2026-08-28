"""One-off schema-alignment migration for the tracked dev DB (assessment.db).

Why this exists: the app only ever calls Base.metadata.create_all() (see
app/main.py's lifespan) — that creates missing TABLES but never ALTERs an
existing table to add new columns. Every model change since the
curriculum-upload feature was added (Section 3 onward) has silently drifted
away from assessment.db's on-disk schema as a result. This script brings
that one file back in line with the current models, in place, without
touching data (every added column is nullable-or-defaulted, so existing
rows are never rewritten in an unsafe way).

This project has no Alembic (migrations/versions/ is an empty, untracked
scaffold — never wired up). This script is intentionally a plain, explicit,
one-off fix rather than introducing a migration framework for a single-file
SQLite dev DB.

Usage:
    .venv/bin/python migrations/align_assessment_db_schema.py [path-to-db]

Defaults to ./assessment.db (the app's configured DATABASE_URL) if no path
is given. Idempotent — safe to re-run; already-present columns/tables are
skipped.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# ── New tables (created via SQLAlchemy so they exactly match the models —
# indexes, constraints, column order, everything) ──────────────────────────
NEW_TABLES = ["curriculum_uploads", "midterm_details"]

# ── Columns missing from existing tables, hand-written to mirror the
# models exactly (app/models/assessment.py, curriculum.py, grade.py,
# submission.py) ────────────────────────────────────────────────────────────
ADD_COLUMNS: dict[str, list[str]] = {
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


def _existing_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _existing_tables(cur: sqlite3.Cursor) -> set[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def migrate(db_path: str) -> None:
    print(f"Migrating {db_path}")

    # New tables first, via SQLAlchemy so they're byte-for-byte what the
    # models declare (this needs the real engine/metadata, not raw sqlite3).
    from sqlalchemy import create_engine

    from app.database import Base
    import app.models  # noqa: F401 - populates Base.metadata

    engine = create_engine(f"sqlite:///{db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        live_tables = _existing_tables(cur)

        to_create = [Base.metadata.tables[t] for t in NEW_TABLES if t not in live_tables]
        if to_create:
            Base.metadata.create_all(bind=engine, tables=to_create)
            for t in to_create:
                print(f"  created table: {t.name}")
        else:
            print("  no missing tables")

        for table, statements in ADD_COLUMNS.items():
            existing_cols = _existing_columns(cur, table)
            for stmt in statements:
                # e.g. "ALTER TABLE assessments ADD COLUMN part1_text TEXT" -> "part1_text"
                col_name = stmt.split("ADD COLUMN")[1].strip().split(" ")[0]
                if col_name in existing_cols:
                    continue
                cur.execute(stmt)
                print(f"  added column: {table}.{col_name}")
        conn.commit()
    finally:
        conn.close()
        engine.dispose()

    print("Done.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "assessment.db")
    migrate(db_path)
