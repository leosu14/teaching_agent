"""Learner data deletion (MVP item 9): what `DELETE /learners/{id}` removed, and what it kept.

A deletion removes every record of the learner's learning data from the database (profile, mastery and evidence,
goals and curricula, lessons and their artifacts, sessions, evaluations, assessment attempts and grades, learning
cycles, the events about them, class memberships) and the stored objects only they reference. It keeps the spend
ledger: ids and amounts of the paid provider calls made for them, with no content, so that the budgets and daily
spending caps still hold after a deletion (see docs/privacy.md)."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import Schema

# What a deletion keeps, by table: the spend ledger (ids and amounts, never content) that the spending caps count.
RETAINED_RECORDS = ("spend_entries", "spend_counters")


class LearnerDataBusy(Exception):
    """The learner has work in progress (a queued or running lesson, or a learning cycle being driven): their data
    is not deleted while a worker may still write it (409). Wait for it to finish, or cancel it, then delete again."""

    def __init__(self) -> None:
        super().__init__("the learner has work in progress; wait for it to finish or cancel it, then delete again")


class ObjectDeletionFailed(Exception):
    """A stored object could not be deleted (503). Nothing was removed from the database, so a retry finds every
    object again; the objects already deleted stay deleted."""

    def __init__(self) -> None:
        super().__init__("the stored files could not all be deleted; nothing else was deleted, try again")


class LearnerDataDeletion(Schema):
    """The receipt of a deletion: counts only, nothing of the deleted data."""

    learner_id: str
    deleted_at: datetime
    records: dict[str, int] = Field(description="Deleted rows by table (tables with none are left out).")
    objects_deleted: int = Field(ge=0, description="Stored files (lesson text, images, audio, slides, video) deleted.")
    objects_kept_shared: int = Field(
        ge=0, description="Files left in place because another learner's lesson holds the identical bytes.")
    retained: list[str] = Field(default_factory=lambda: list(RETAINED_RECORDS))
