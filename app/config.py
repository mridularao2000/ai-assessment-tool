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

    # ── Email (Resend) ────────────────────────────────────────────────────────
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

    @property
    def results_recipient_emails(self) -> list[str]:
        parsed = [e.strip() for e in self.results_recipient_emails_raw.split(",") if e.strip()]
        return parsed or [self.user_email]

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
