"""Lessons as the learner reads them (the student web app's lesson player)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.authorization import owned_lesson
from app.api.deps import container
from app.schemas.reading import LessonReading
from app.services.container import Container

router = APIRouter(tags=["lessons"])


@router.get("/lessons/{lesson_id}/reading", response_model=LessonReading, dependencies=[Depends(owned_lesson)])
def lesson_reading(lesson_id: str, c: Container = Depends(container)) -> LessonReading:
    """The lesson's teaching text, built by code from the stored lesson (which is never served): teaching sections
    only, and no sentence that states an answer the learner is graded on, or the answer of a question open for them
    now. `lesson_id`: the LESSON artifact id or the id of the completed lesson task (409 while it is not complete)."""
    return c.lesson_reading.reading(lesson_id)
