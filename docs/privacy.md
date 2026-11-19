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
| Teacher reviews: lessons held for a teacher's release and their decisions (with the teacher's note), teacher overrides of UNCERTAIN grades | `lesson_releases`, `grade_overrides` | |
| Class assignments given to the learner: which assignment, their goal for it, when and how they got it (their goal itself is with the goals) | `assignment_learners` | |
| Events outside a task that name the learner (goal created or changed, an assignment given to them) | `events` | |
| Spend ledger: ids and amounts of paid provider calls | `spend_entries`, `spend_counters` | |

Not learner data: `classes` and `class_assignments` (a teacher's: an assignment names no learner),
`assessment_rubrics` (an author's). Every table is in one of two lists in
`app/storage/repositories.py`, `ERASED_TABLES` and `KEPT_TABLES`, and `tests/unit/test_learner_deletion.py` fails
when a table is in neither, so a new table cannot hold learner data without deciding how it is deleted.

## Deleting a learner

`DELETE /learners/{learner_id}` deletes everything in the table above except the spend ledger, and answers `200` with
a receipt: rows deleted by table, objects deleted, objects kept because another learner shares them, objects left
for the sweeper to retry, and what was retained. The receipt holds counts only.

| Caller | Answer |
|---|---|
| The learner's own scope (the learner, or a guardian who acts for them) | `200` and the receipt; `200` with an empty receipt when nothing is left, so a retry is safe |
| A teacher of a class the learner joined | `403`: a teacher reads, never changes (the same rule as every other change) |
| Anyone else | the uniform `404` |

The student app offers it on the Progress page ("Delete my data", then type DELETE). Afterwards the tab forgets the
learner; signing in again starts from the welcome screen, as a new learner.

### How it works

`LearnerDeletionService` (`app/services/learner_deletion.py`) over `SqlLearnerDataRepository`:

The rows go in one database transaction, the files after it commits:

1. **Lock.** First a lock on the learner's id that every write for the learner takes too (shared) and the deletion
   takes alone (exclusive), so the two take turns and two deletions of one learner wait for each other (a PostgreSQL
   advisory lock; on SQLite the first statement is a write, which holds the database's write lock). Then the
   learner's task rows, their learner row and their cycle rows (`SELECT ... FOR UPDATE`).
2. **Refuse while work is in progress.** A queued or running job of one of the learner's tasks, a task under a live
   execution lease, or a learning cycle being driven: `409 LearnerDataBusy`, and nothing is deleted, no row and no
   file. A worker writing the learner's lesson at that moment would keep producing data. The learner waits, or
   cancels the task, and deletes again.
3. **Delete the rows** of every table above but the spend ledger, including the events outside any task that name
   the learner, one of their goals or one of their curricula (objective started, learning action selected), and the
   learner's object holds (below).
4. **Queue the files**: every object the learner's artifacts and holds point at goes in `object_deletions`, in the
   same transaction. Then **commit**. If anything fails before the commit, nothing is deleted, no row and no file.
5. **Delete the files**, after the commit, one by one, each in a transaction of its own under that object's lock
   (below): if nothing references the object any more, it is deleted and its queue entry removed; if something does
   (another learner's lesson uses the same bytes), it is kept and the entry removed. If the store refuses one, the
   entry stays and the request still answers `200`, with `objects_pending` in the receipt; the worker's sweeper
   retries it (after 30 s, then doubling up to an hour) until it goes. Deleting an object that is already gone counts
   as done, so a retry is safe.

**Shared objects.** Lesson media are content-addressed (`objects/sha256/...`): identical bytes are stored once and
shared, the same picture in two lessons. A lesson records a hold on such an object (`object_holds`: the object and
the task) before it puts or reuses it, and adds the artifact row only after checking that the object is there; both
under the object's lock, the same lock the file deletion takes to re-check the references (a PostgreSQL advisory lock
on the object's uri, shared by every API and worker process and every app instance; on SQLite the database's write
lock). So a deletion and another learner's lesson using the same object take turns: either the hold commits first,
and the deletion sees it and keeps the object, or the deletion's check and delete commit first, and the lesson's put
then finds the object gone and writes it again. No artifact row points at a deleted object.

**Requests at the same moment.** Grading an answer, a teaching session, queueing a lesson, a learning cycle, a goal
and joining a class write for the learner without a task lease, so the in-progress check cannot see them. Each of
their writes checks, inside its own transaction and before anything else, that the lesson task and the learner rows
still exist, and locks them in the same order as the deletion (`learner_writes` in `app/schemas/lease.py`, `_guard`
in `app/storage/repositories.py`). So either the write commits first, and the deletion waits for it and deletes it
with the rest, or the deletion commits first, and the write and every later one of that request are refused: the
request answers the uniform `404` and stores nothing. A file the request had already put for an artifact whose row
is then refused is deleted again at once, so no file holds what the learner wrote. A lesson queued at the same moment
is either seen (the deletion answers `409` and deletes nothing) or refused after it (`404`). The same holds for
registering an assessment item, for the next action (which stores curriculum progress and publishes events naming
the goal), and for the sweeper's recovery of a task left mid-run: none writes once the deletion has committed.

A lesson may be queued for a learner who has not sent their profile yet (no learner row); its writes are then checked
against its task row only, which the deletion deletes with the rest.

**Lock order.** Every write for a learner takes the learner's id lock first (shared), then the task row and the
learner row, then, for an object, the object's lock; the deletion takes the id lock (exclusive) first, then the same
rows. The file deletion takes only the object's lock, after the rows' transaction has committed. Because everyone
takes the locks in that order, two of them cannot wait for each other (no deadlock); a change that takes a lock out
of this order must say why it cannot deadlock.

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

- **The generated-video ledger** (`records/video-generations/`, only when generated clips are on) is keyed by the
  clip request, not by learner, and is reused across lessons so a paid clip is not generated twice. Its entries are
  not deleted with a learner. The provider's clip files it records are held by every lesson that used them (as are
  the image candidates the visual step stores before it chooses one), and deleted with the last learner holding them;
  a ledger entry whose object is gone is generated again when next needed.
- **Files after the commit.** The files go just after the rows, so for a moment (or, when the store refuses, until
  the sweeper's retry succeeds) a file can still be in the store although nothing leads to it: every request for it
  already answers `404`. The receipt's `objects_pending` says how many are left to the sweeper.
- **Reads during the deletion.** A page of the learner's open in another tab while the deletion runs reads the rows
  until the deletion commits, and the uniform `404` after it.
- **No export and no automatic retention.** Data is kept until the learner deletes it. Backups and object-store
  versions are the operator's to manage (above).
- **The sign-in account** belongs to the identity provider; deleting a learner's data does not touch it.
