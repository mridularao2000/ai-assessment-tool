from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

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

app.include_router(curriculum.router, prefix="/api/v1/curriculum", tags=["curriculum"])
app.include_router(assessment.router, prefix="/api/v1/assessments", tags=["assessments"])
app.include_router(submission.router, prefix="/api/v1/submissions", tags=["submissions"])
app.include_router(grading.router, prefix="/api/v1/submissions", tags=["grading"])
app.include_router(reschedule.router, prefix="/api/v1/assessments", tags=["reschedule"])
