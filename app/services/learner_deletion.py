"""Learner data deletion (MVP item 9): `DELETE /learners/{id}` removes everything stored about a learner.

The order makes a failure safe to retry. First the stored objects only this learner's artifacts point at are deleted
(lesson text, images, narration, slides, video); if one cannot be, nothing else is touched and the request fails, so
a retry finds every object again through the rows that still reference it. Then every database row of the learner is
deleted in one transaction (`SqlLearnerDataRepository.erase`). Doing it the other way round could leave files that no
row references any more, which nobody could find to delete.

Both steps refuse while the learner has work in progress (`LearnerDataBusy`, 409), because a worker writing their
lesson at that moment would keep producing data. Kept on purpose: the spend ledger (ids and amounts, never content),
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
        objects = self.repository.objects(learner_id)
        for uri in objects.delete:
            try:
                self._artifacts.delete_object(uri)
            except Exception:  # the store's own error types differ (OSError, S3 errors): any of them stops here
                logger.exception("learner data deletion: an object could not be deleted; no rows were deleted")
                raise ObjectDeletionFailed() from None
        records = self.repository.erase(learner_id)
        # The learner id is not logged: the log line records that a deletion happened, not whose data it was.
        logger.info("learner data deleted: %d records, %d objects, %d shared objects kept",
                    sum(records.values()), len(objects.delete), objects.shared)
        return LearnerDataDeletion(learner_id=learner_id, deleted_at=self._clock(), records=records,
                                   objects_deleted=len(objects.delete), objects_kept_shared=objects.shared)
