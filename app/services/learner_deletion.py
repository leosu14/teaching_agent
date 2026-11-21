"""Learner data deletion (MVP item 9): `DELETE /learners/{id}` removes everything stored about a learner.

Two steps. First one database transaction (`SqlLearnerDataRepository.erase`): it locks the learner's rows, refuses
while the learner has work in progress (`LearnerDataBusy`, 409, nothing deleted: a worker writing their lesson at
that moment would keep producing data), deletes every row of the learner and queues the stored objects their
artifacts and holds point at (lesson text, images, narration, slides, video), and commits. A request writing for the
learner at the same time (grading an answer, queueing a lesson) either committed before the locks, and is deleted
with the rest, or is refused after (`learner_writes`). If anything fails before the commit, nothing is deleted, no
row and no file.

Then, after the commit, the queued objects (`SqlLearnerDataRepository.purge`), each under its own lock: one that
someone else's artifact or hold still points at (shared media) is kept, every other one is deleted. One the store
refuses stays queued and the sweeper tries it again until it is done, so the deletion completes without the learner
having to retry; the receipt says how many are still pending.

Kept on purpose: the spend ledger (ids and amounts, never content), so budgets and daily caps still count what was
spent (app/schemas/privacy.py)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.schemas.common import utcnow
from app.schemas.privacy import LearnerDataDeletion
from app.storage.repositories import ObjectPurge, SqlLearnerDataRepository

logger = logging.getLogger(__name__)


class LearnerDeletionService:
    def __init__(self, repository: SqlLearnerDataRepository, artifacts: ArtifactService,
                 clock: Callable[[], datetime] = utcnow,
                 forget: Sequence[Callable[[str], None]] = ()) -> None:
        self.repository = repository
        # Process-local stores that hold learner data outside the database (the in-memory research cache of a
        # single-process deployment), told after the erase commits. The shared ones are erased by the transaction.
        self._forget = tuple(forget)
        self._artifacts = artifacts
        self._clock = clock

    def delete(self, learner_id: str) -> LearnerDataDeletion:
        """Delete the learner's data; LearnerDataBusy (nothing deleted) while they have work in progress. Deleting a
        learner with no data succeeds with nothing to report, so a retry after a lost response is safe."""
        records, queued = self.repository.erase(learner_id)
        for forget in self._forget:
            forget(learner_id)
        objects = self.repository.purge(self._artifacts.delete_object, queued)
        # The learner id is not logged: the log line records that a deletion happened, not whose data it was.
        logger.info("learner data deleted: %d records, %d objects, %d shared objects kept, %d objects pending",
                    sum(records.values()), objects.deleted, objects.kept_shared, objects.pending)
        return LearnerDataDeletion(learner_id=learner_id, deleted_at=self._clock(), records=records,
                                   objects_deleted=objects.deleted, objects_kept_shared=objects.kept_shared,
                                   objects_pending=objects.pending)

    def purge_due(self) -> ObjectPurge:
        """The sweeper's step: the queued object deletions that are due (a store failure earlier)."""
        result = self.repository.purge(self._artifacts.delete_object)
        if result.deleted or result.pending:
            logger.info("learner data deletion: %d queued objects deleted, %d kept shared, %d still pending",
                        result.deleted, result.kept_shared, result.pending)
        return result
