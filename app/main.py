import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# APScheduler logs every job execution and any exceptions at INFO/ERROR.
logging.getLogger("apscheduler").setLevel(logging.INFO)

from app.api.v1 import (
    assessment,
    curriculum,
    grading,
    health,
    reschedule,
    submission,
)
from app.dependencies import get_scheduler_adapter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from app.database import Base, SessionLocal, engine
    from app.db.seed import check_missing_templates, seed_prompt_templates

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_prompt_templates(db)
        missing = check_missing_templates(db)
        if missing:
            logging.getLogger(__name__).warning(
                "Required prompt templates still missing after seed: %s — "
                "run `python -m app.db.seed` to fix.",
                missing,
            )
    finally:
        db.close()

    from app.config import get_settings
    _settings = get_settings()
    if "localhost" in _settings.app_base_url:
        logging.getLogger(__name__).warning(
            "APP_BASE_URL is set to %r which contains 'localhost'. "
            "Assessment emails will contain broken links in production. "
            "Set APP_BASE_URL=https://<your-render-app>.onrender.com in the Render dashboard.",
            _settings.app_base_url,
        )

    get_scheduler_adapter().start()
    yield
    get_scheduler_adapter().shutdown()


app = FastAPI(
    title="AI Assessment Tool",
    version="0.1.0",
    lifespan=lifespan,
)

_STATIC = os.path.join(os.path.dirname(__file__), "static")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")

@app.get("/logo.svg", include_in_schema=False)
def logo() -> FileResponse:
    return FileResponse(os.path.join(_STATIC, "logo.svg"), media_type="image/svg+xml")

@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(os.path.join(_STATIC, "index.html"))


app.include_router(curriculum.router, prefix="/api/v1/curriculum", tags=["curriculum"])
app.include_router(assessment.router, prefix="/api/v1/assessments", tags=["assessments"])
app.include_router(submission.router, prefix="/api/v1/submissions", tags=["submissions"])
app.include_router(grading.router, prefix="/api/v1/submissions", tags=["grading"])
app.include_router(reschedule.router, prefix="/api/v1/assessments", tags=["reschedule"])
app.include_router(health.router, prefix="/health", tags=["health"])
