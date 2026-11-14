"""Learner data deletion (MVP item 9): `DELETE /learners/{id}` removes everything stored about a learner.

Everything happens in one database transaction (`SqlLearnerDataRepository.erase`). It locks the learner's rows, refuses
while the learner has work in progress (`LearnerDataBusy`, 409, nothing deleted: a worker writing their lesson at
that moment would keep producing data), deletes every row of the learner, then, still under the locks, the stored
objects only this learner's artifacts point at (lesson text, images, narration, slides, video), and commits. If the
store refuses one object, the transaction rolls back: no row is deleted and the request fails (503), so a retry finds
every object still left through the rows that point at it. A request writing for the learner at the same time
(grading an answer, queueing a lesson) either committed before the locks, and is deleted with the rest, or is refused
after (`learner_writes`).

Kept on purpose: the spend ledger (ids and amounts, never content),
so budgets and daily caps still count what was spent (app/schemas/privacy.py)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.schemas.common import utcnow
from app.schemas.privacy import LearnerDataDeletion, ObjectDeletionFailed
from app.storage.repositories import SqlLearnerDataRepository

logger = logging.getLogger(__name__)


class LearnerDeletionService:
    def __init__(self, repository: SqlLearnerDataRepository, artifacts: ArtifactService,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.repository = repository
        self._artifacts = artifacts
        self._clock = clock

    def delete(self, learner_id: str) -> LearnerDataDeletion:
        """Delete the learner's data; LearnerDataBusy (nothing deleted) while they have work in progress,
        ObjectDeletionFailed (no row deleted) when the object store refuses a deletion. Deleting a learner with no
        data succeeds with nothing to report, so a retry after a lost response is safe."""
        records, objects = self.repository.erase(learner_id, self._delete_objects)
        # The learner id is not logged: the log line records that a deletion happened, not whose data it was.
        logger.info("learner data deleted: %d records, %d objects, %d shared objects kept",
                    sum(records.values()), len(objects.delete), objects.shared)
        return LearnerDataDeletion(learner_id=learner_id, deleted_at=self._clock(), records=records,
                                   objects_deleted=len(objects.delete), objects_kept_shared=objects.shared)

    def _delete_objects(self, uris: tuple[str, ...]) -> None:
        for uri in uris:
            try:
                self._artifacts.delete_object(uri)
            except Exception:  # the store's own error types differ (OSError, S3 errors): any of them stops here
                logger.exception("learner data deletion: an object could not be deleted; no rows were deleted")
                raise ObjectDeletionFailed() from None
