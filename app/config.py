from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./assessment.db"
    # Gates the startup schema-bootstrap step (create_all + prompt-template
    # seed) in app.main.lifespan — this app has no separate migration tool,
    # so that step IS the real, intended way its schema/seed data gets
    # created. It must stay explicit rather than implicit: FastAPI's
    # TestClient triggers lifespan on the first request even without a
    # `with` block, and app.database.engine is a process-global singleton
    # bound to database_url at import time — with no guard, ANY test that
    # exercises a route silently ran create_all()/seed against the real
    # tracked assessment.db, not the test session's isolated engine. Tests
    # force this to False (see tests/conftest.py) before app.database is
    # ever imported; real app startup leaves it True.
    run_schema_bootstrap: bool = True

    # ── User ──────────────────────────────────────────────────────────────────
    user_email: str = ""
    submission_token_secret: str = ""

    # ── Assessment scheduling ─────────────────────────────────────────────────
    assessment_window_min_days: int = Field(default=1, ge=1)
    assessment_window_max_days: int = Field(default=3, ge=1)
    reminder_hours_before: int = Field(default=24, ge=1)
    assessment_due_days: int = Field(default=5, ge=1)

    # ── File storage ──────────────────────────────────────────────────────────
    uploads_dir: str = "uploads"

    # ── Mastery ───────────────────────────────────────────────────────────────
    mastery_threshold: float = Field(default=85.0, ge=0.0, le=100.0)

    # ── Email (Gmail SMTP) ────────────────────────────────────────────────────
    # Live send path — see app.dependencies._build_email(). App Password from
    # Google Account → Security → 2-Step Verification → App Passwords, not the
    # account's regular login password. The sending account doesn't need to be
    # a real inbox — a dedicated free Gmail account works.
    gmail_address: str = ""
    gmail_app_password: str = ""

    # ── Email (Resend) ────────────────────────────────────────────────────────
    # Migrated away from as the live send path (see gmail_address above) —
    # kept configurable, unused by default, as a rollback path only.
    resend_api_key: str = ""
    resend_from_email: str = ""
    resend_from_name: str = "AI Assessment System"
    app_base_url: str = "http://localhost:8000"
    # All three below apply ONLY to curriculum-upload entries — standalone
    # keeps sending to user_email exactly as it does today. Each falls back
    # to user_email when unset.
    # Flows (a) syllabus + PM-System-style hold reminders.
    syllabus_recipient_email: str = ""
    # Flows (b) pre-deadline reminder + (c) the exam itself — 1 recipient.
    exam_recipient_email: str = ""
    # Flows (d) grading result + (e) transcript — 2 recipients, comma-separated.
    results_recipient_emails_raw: str = ""

    # ── Curriculum upload ─────────────────────────────────────────────────────
    # How often the "resources still missing" reminder re-fires for a held
    # Midterm. The daily recheck job itself always runs daily regardless —
    # this only throttles the email.
    pending_hold_reminder_interval_days: int = Field(default=7, ge=1)
    # Entry-only reminder timing (flow b): counts back from due_date (the
    # deadline), unlike standalone's reminder which counts back from
    # scheduled_at (the send date) and stays hardcoded to 1 day — see
    # AssessmentService.build_assessment_dates vs.
    # CurriculumUploadService's entry-specific date math.
    entry_reminder_hours_before_deadline: int = Field(default=24, ge=1)
    # Flow (e) transcript, secondary copy: removed from the per-event
    # trigger entirely (that now sends only to user_email, the primary) and
    # sent instead by a standalone biweekly job. Empty = the biweekly job
    # sends nothing — set once a real secondary address is confirmed.
    transcript_secondary_recipient_email: str = ""
    transcript_secondary_interval_days: int = Field(default=14, ge=1)
    # Self-healing sweep for send_assessment_job's one-shot-job failure mode
    # (see recheck_stuck_assessments_job) — how far past scheduled_at an
    # assessment must sit still `scheduled` before the sweep treats it as
    # stuck rather than legitimately mid-flight.
    stuck_assessment_grace_minutes: int = Field(default=30, ge=1)
    # Ceiling on automatic retries: once a row has been stuck past
    # scheduled_at for longer than this, the sweep stops calling the LLM for
    # it automatically and just logs loudly instead. Without this, a
    # persistent failure (bad credentials, insufficient API credit, a broken
    # prompt template) gets retried every 15 minutes forever — real API
    # spend on autopilot, indefinitely, for a cause a retry can't fix. A
    # human must resolve the real cause and use /resend manually past this
    # point.
    stuck_assessment_max_auto_retry_hours: int = Field(default=3, ge=1)

    @property
    def results_recipient_emails(self) -> list[str]:
        parsed = [e.strip() for e in self.results_recipient_emails_raw.split(",") if e.strip()]
        return parsed or [self.user_email]

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 3
    # Circuit breaker for the 5 tool-enabled (web_search/web_fetch) call
    # sites specifically — see AnthropicLLMAdapter._CallBudget. Cumulative
    # input+output tokens across every attempt within one logical
    # generate_*/grade_*_submission call; once crossed, the next attempt
    # aborts immediately instead of making another expensive request.
    # Defense-in-depth on top of the tool-round-trip cap and the reduced
    # tool-path attempt count — not the primary fix, a backstop.
    llm_tool_call_budget_tokens: int = Field(default=200_000, ge=1000)
    # ── Test / isolated mode ─────────────────────────────────────────────────────
    # Single flag for standing up a fully isolated server (e.g. to verify a
    # UI/API change) with ZERO real outbound side effects — neither a real
    # LLM call nor a real email send, regardless of whether ANTHROPIC_API_KEY
    # / RESEND_API_KEY happen to be present in the environment (both are
    # normally read straight from .env, so an "isolated" server pointed at a
    # throwaway DB would otherwise still fire real API calls using whatever
    # keys .env happens to have — this is what actually neuters them).
    # app.dependencies._build_llm()/_build_email() check this FIRST, before
    # their key-presence checks: true -> FakeLLMAdapter / FakeEmailAdapter
    # (canned/no-op, no network call). Must stay unset in production; real
    # app deployments never set this env var. This is distinct from a
    # deliberate dry-run against real integrations (isolated DB, real LLM
    # and email, to prove real-world behavior) — that kind of run leaves
    # this flag OFF on purpose.
    test_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
