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

    # ── Assessment scheduling ─────────────────────────────────────────────────
    assessment_window_min_days: int = Field(default=1, ge=1)
    assessment_window_max_days: int = Field(default=3, ge=1)
    reminder_hours_before: int = Field(default=24, ge=1)
    assessment_due_days: int = Field(default=5, ge=1)

    # ── File storage ──────────────────────────────────────────────────────────
    uploads_dir: str = "uploads"

    # ── Mastery ───────────────────────────────────────────────────────────────
    mastery_threshold: float = Field(default=85.0, ge=0.0, le=100.0)

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
