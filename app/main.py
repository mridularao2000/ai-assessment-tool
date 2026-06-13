import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse

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
    reschedule,
    submission,
)
from app.dependencies import get_scheduler_adapter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from app.database import Base, engine

    Base.metadata.create_all(bind=engine)

    get_scheduler_adapter().start()
    yield
    get_scheduler_adapter().shutdown()


app = FastAPI(
    title="AI Assessment Tool",
    version="0.1.0",
    lifespan=lifespan,
)

_STATIC = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(os.path.join(_STATIC, "index.html"))


app.include_router(curriculum.router, prefix="/api/v1/curriculum", tags=["curriculum"])
app.include_router(assessment.router, prefix="/api/v1/assessments", tags=["assessments"])
app.include_router(submission.router, prefix="/api/v1/submissions", tags=["submissions"])
app.include_router(grading.router, prefix="/api/v1/submissions", tags=["grading"])
app.include_router(reschedule.router, prefix="/api/v1/assessments", tags=["reschedule"])
