# Learner data and privacy

MVP item 9: a learner can delete everything stored about them, and the apps link a privacy notice that says what
is kept, who sees it and how to delete it. This page is for whoever runs the service.

## The privacy notice

`/app/privacy.html` (`app/api/web/privacy.html`) is served like the apps' files: public, no data in it, the same
strict headers (`app/api/routes/web.py`), and no script at all. The student app links it on the welcome screen, in
its footer and on the Progress page; the teacher dashboard links it in its footer.

The notice describes this codebase as it is. Before a deployment goes live, the operator should read it against
their own setup and change it where it differs, in particular:

- **Who runs the service.** The notice says "the school or organisation that gave you access"; a deployment should
  name itself and give a contact for privacy questions.
- **Providers.** The notice lists the kinds of AI and media services used. A deployment should name its vendors
  (`docs/production.md`, Configuration) and check their data-processing terms.
- **Backups.** The notice says operator backups may hold copies until they expire. A deployment should state its
  backup retention. Deleting a learner does not reach into database backups or object-store versioning: if the
  bucket keeps old versions, deleted objects stay there until those versions expire.
- **Legal basis, age and region.** Learners who are minors, or in the EU or UK, need more than this notice (consent
  from a guardian, a lawful basis, the right of access). This release has deletion only; exporting a learner's data
  and a retention policy are not part of it.

The notice is a file in the image, so changing it means changing `app/api/web/privacy.html` and building the image.

## What is stored about a learner

Everything below is keyed by the learner id the gateway asserts (`X-Auth-Learners`), or by ids that lead back to it.

| Data | Tables | Objects |
|---|---|---|
| Profile: display name, preferences, subjects, mastery, assessment and lesson history | `learners` | |
| Evidence and learning events | `learning_evidence`, `learning_events` | |
| Goals and curricula | `learning_goals`, `curricula`, `curriculum_versions`, `curriculum_progress` | |
| Lessons and evaluations (tasks), their jobs, artifacts and events | `tasks`, `task_jobs`, `artifacts`, `events` | lesson text and plans under the task's keys; images, narration, slides and video (content-addressed) |
| Teaching sessions | `teaching_sessions`, `teaching_turns`, `interaction_evidence`, `teaching_requests`, `teaching_outbox` | |
| Assessment items on the learner's lessons, attempts and grades | `assessment_items`, `assessment_attempts`, `assessment_grades` | |
| Learning cycles | `learning_cycles`, `learning_cycle_slots`, `learning_cycle_requests`, their artifacts and events | the cycle's records |
| Class memberships | `class_members` | |
| Events outside a task that name the learner (goal created or changed) | `events` | |
| Spend ledger: ids and amounts of paid provider calls | `spend_entries`, `spend_counters` | |

Not learner data: `classes` (a teacher's), `assessment_rubrics` (an author's). Every table is in one of two lists in
`app/storage/repositories.py`, `ERASED_TABLES` and `KEPT_TABLES`, and `tests/unit/test_learner_deletion.py` fails
when a table is in neither, so a new table cannot hold learner data without deciding how it is deleted.

## Deleting a learner

`DELETE /learners/{learner_id}` deletes everything in the table above except the spend ledger, and answers `200` with
a receipt: rows deleted by table, objects deleted, objects kept because another learner shares them, and what was
retained. The receipt holds counts only.

| Caller | Answer |
|---|---|
| The learner's own scope (the learner, or a guardian who acts for them) | `200` and the receipt; `200` with an empty receipt when nothing is left, so a retry is safe |
| A teacher of a class the learner joined | `403`: a teacher reads, never changes (the same rule as every other change) |
| Anyone else | the uniform `404` |

The student app offers it on the Progress page ("Delete my data", then type DELETE). Afterwards the tab forgets the
learner; signing in again starts from the welcome screen, as a new learner.

### How it works

`LearnerDeletionService` (`app/services/learner_deletion.py`) over `SqlLearnerDataRepository`:

1. **Refuse while work is in progress.** A queued or running job of one of the learner's tasks, a task under a live
   execution lease, or a learning cycle being driven: `409 LearnerDataBusy`, and nothing is deleted. A worker writing
   the learner's lesson at that moment would keep producing data. The learner waits, or cancels the task, and
   deletes again.
2. **Delete the objects.** Every object an artifact of the learner's tasks and cycles points at, except a
   content-addressed object (`objects/sha256/...`) that an artifact of someone else also holds (identical bytes are
   stored once and shared: the same picture in two lessons). If the store refuses one, the request fails with `503
   ObjectDeletionFailed` and no row is deleted, so a retry finds every remaining object again through the rows.
   Deleting the rows first could leave objects that nothing references any more, which nobody could find to delete.
3. **Delete the rows**, all in one transaction, after the same in-progress check. The learner's task and cycle rows
   are locked first (`SELECT ... FOR UPDATE` on PostgreSQL), so a worker's claim either happened before (and is seen)
   or finds nothing to claim afterwards.

A worker that still has the learner's task in memory cannot bring data back: its writes are fenced by the task row,
which is gone (`LeaseLost`), and profile and curriculum saves are version-checked against rows that no longer exist.
The deletion's log line records counts, not the learner id.

### What is kept, and why

- **The spend ledger** (`spend_entries`, `spend_counters`): ids and amounts of paid calls, never content. The
  per-learner daily cap counts `learner:{id}:{day}`; deleting it would let a learner reset their cap by deleting
  their data. It is kept, and the caps still hold.
- **Shared media.** A content-addressed object that another learner's artifact holds stays, because that learner's
  lesson uses the identical bytes.

### Known limits

- **Unused image candidates.** The visual step stores the images it fetches or generates before it chooses one; the
  ones it does not choose are content-addressed pictures of the topic that no row ties to anyone, so a deletion
  cannot find them. They are not linked to the learner.
- **The generated-video ledger** (`records/video-generations/`, only when generated clips are on) is keyed by the
  clip request, not by learner, and is reused across lessons so a paid clip is not generated twice. The ledger and
  the provider's original clip files it records are not deleted with a learner; the learner's own artifacts (the
  lesson video, the clips in it) are, and a ledger entry whose object is gone is generated again when next needed.
- **A request racing the deletion.** A lesson the learner starts in another tab at the same moment as the deletion
  can be created just after it and survive it; deleting again removes it.
- **A shared object deleted at the moment another lesson stores the same bytes** could leave that lesson's artifact
  pointing at a missing object; the artifact service then reports it missing. Identical bytes are rare outside the
  mock providers.
- **No export and no automatic retention.** Data is kept until the learner deletes it. Backups and object-store
  versions are the operator's to manage (above).
- **The sign-in account** belongs to the identity provider; deleting a learner's data does not touch it.
