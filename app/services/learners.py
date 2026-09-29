"""Learner application service."""

from __future__ import annotations

from app.learner.memory import LearnerMemoryService
from app.schemas.learner import LearnerProfile, LearnerProfileInput, LearnerProgress


class LearnerService:
    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    def upsert(self, learner_id: str, data: LearnerProfileInput) -> LearnerProfile:
        return self._memory.upsert(learner_id, data)

    def get(self, learner_id: str) -> LearnerProfile:
        return self._memory.get(learner_id)

    def progress(self, learner_id: str) -> LearnerProgress:
        return self._memory.progress(learner_id)
