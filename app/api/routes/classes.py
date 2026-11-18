"""Classes and rosters (MVP item 4), and the teacher dashboard over them (MVP item 7). A teacher (the `teacher` role) creates classes and hands out their join codes;
a learner joins with a code, and from then on the teacher may read, never change, that learner's data through every
learner route (app/api/authorization.py). Either side can end the membership: the teacher removes the learner, or
the learner leaves.

A class of another teacher, or of another tenant, gets the same 404 as one that does not exist, and so does a wrong
or retired join code.

Teacher review (Important item 2): the class's teacher may require approving lessons before its learners see them,
previews and approves or rejects them, and overrides the UNCERTAIN grades of its learners. These are the only changes
a teacher makes for a learner, and every one is a class route through `taught_class`, for a learner still on that
class's roster: `Access.check` (a teacher reads a learner's routes, never changes them) is unchanged."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from app.api.auth import TEACHER, Principal
from app.api.authorization import ResourceNotFound, authorize, principal, require_learner, require_role, taught_class
from app.api.deps import container
from app.schemas.classroom import ClassJoinCode, Classroom, ClassView, CreateClass, Enrollment, JoinClass
from app.schemas.dashboard import ClassDashboard, UncertainGrade
from app.schemas.reading import LessonReading
from app.schemas.review import (
    ClassSettings,
    LessonReleaseView,
    OverrideGrade,
    OverrideResult,
    ReleaseDecision,
    ReleaseStatus,
)
from app.services.container import Container

router = APIRouter(tags=["classes"])


@router.post("/classes", response_model=ClassJoinCode, status_code=201)
def create_class(body: CreateClass, p: Principal = Depends(principal),
                 c: Container = Depends(container)) -> ClassJoinCode:
    """A new class of the calling teacher, with its join code. The code is shown here and on rotation only."""
    require_role(p, TEACHER)
    return c.classes.create(teacher_id=p.user_id, tenant_id=p.tenant_id, name=body.name)


@router.get("/classes", response_model=list[ClassView])
def list_classes(p: Principal = Depends(principal), c: Container = Depends(container)) -> list[ClassView]:
    """The calling teacher's classes, with their rosters."""
    require_role(p, TEACHER)
    return c.classes.for_teacher(teacher_id=p.user_id, tenant_id=p.tenant_id)


@router.get("/classes/{class_id}", response_model=ClassView)
def get_class(classroom: Classroom = Depends(taught_class), c: Container = Depends(container)) -> ClassView:
    return c.classes.view(classroom)


@router.patch("/classes/{class_id}", response_model=ClassView)
def update_class(body: ClassSettings, classroom: Classroom = Depends(taught_class),
                 c: Container = Depends(container)) -> ClassView:
    """The class's settings. `lesson_approval`: lessons requested from now on for its learners wait for the
    teacher's approval before the learners see them (lessons requested before keep what was decided then)."""
    if body.lesson_approval is None:
        return c.classes.view(classroom)
    return c.classes.set_lesson_approval(classroom, body.lesson_approval)


@router.get("/classes/{class_id}/lessons", response_model=list[LessonReleaseView])
def class_lessons(classroom: Classroom = Depends(taught_class), status: ReleaseStatus | None = Query(default=None),
                  c: Container = Depends(container)) -> list[LessonReleaseView]:
    """The lessons requested for this class's learners while it required approval, newest first (`status`: only
    PENDING, APPROVED or REJECTED). Learners who left the class are not listed."""
    return c.lesson_releases.for_class(classroom, status)


@router.get("/classes/{class_id}/lessons/{task_id}", response_model=LessonReleaseView)
def class_lesson(task_id: str, classroom: Classroom = Depends(taught_class),
                 c: Container = Depends(container)) -> LessonReleaseView:
    return c.lesson_releases.view(classroom, task_id)


@router.get("/classes/{class_id}/lessons/{task_id}/preview", response_model=LessonReading)
def preview_lesson(task_id: str, classroom: Classroom = Depends(taught_class),
                   c: Container = Depends(container)) -> LessonReading:
    """The lesson as its learner's reading will show it once released: the same answer-isolated text and images
    (never the exercises, answers, narration or deck). 409 while the lesson is still being generated."""
    c.lesson_releases.release_for(classroom, task_id)
    return c.lesson_reading.reading(task_id, preview=True)


@router.post("/classes/{class_id}/lessons/{task_id}/approve", response_model=LessonReleaseView)
def approve_lesson(task_id: str, body: ReleaseDecision | None = None, classroom: Classroom = Depends(taught_class),
                   p: Principal = Depends(principal), c: Container = Depends(container)) -> LessonReleaseView:
    """Release the lesson to its learner. Final: approving again returns it as it is; a rejected lesson is 409."""
    return c.lesson_releases.decide(classroom, task_id, ReleaseStatus.APPROVED, teacher_id=p.user_id,
                                    note=body.note if body else None)


@router.post("/classes/{class_id}/lessons/{task_id}/reject", response_model=LessonReleaseView)
def reject_lesson(task_id: str, body: ReleaseDecision | None = None, classroom: Classroom = Depends(taught_class),
                  p: Principal = Depends(principal), c: Container = Depends(container)) -> LessonReleaseView:
    """Do not release the lesson: it stays held from its learner, and a learning cycle on it fails (the learner
    starts a new one, which generates a new lesson). Final: rejecting again returns it; an approved lesson is 409."""
    return c.lesson_releases.decide(classroom, task_id, ReleaseStatus.REJECTED, teacher_id=p.user_id,
                                    note=body.note if body else None)


@router.post("/classes/{class_id}/attempts/{attempt_id}/override", response_model=OverrideResult)
async def override_grade(attempt_id: str, body: OverrideGrade, classroom: Classroom = Depends(taught_class),
                         p: Principal = Depends(principal), c: Container = Depends(container)) -> OverrideResult:
    """The teacher's outcome for an answer the grader could not decide (UNCERTAIN), of a learner on this class's
    roster. It becomes one piece of learning evidence (mastery changes through the learner's MasteryUpdater); the
    grade itself stays as it was. Once per attempt: the same override again is a replay, a different one 409."""
    owner = c.ownership.assessment_attempt(attempt_id)
    if owner is None or c.classes.member(classroom.class_id, owner) is None:
        raise ResourceNotFound()
    return await c.assessment_service.override(attempt_id, body, class_id=classroom.class_id, teacher_id=p.user_id)


@router.get("/classes/{class_id}/dashboard", response_model=ClassDashboard)
def class_dashboard(classroom: Classroom = Depends(taught_class),
                    c: Container = Depends(container)) -> ClassDashboard:
    """The teacher dashboard (MVP item 7): every learner on the roster with their mastery, goals and curriculum
    progress, learning-cycle state and open UNCERTAIN grades. Read-only: nothing is reconciled or recorded."""
    return c.class_dashboard.dashboard(classroom)


@router.get("/classes/{class_id}/uncertain-grades", response_model=list[UncertainGrade])
def uncertain_grades(classroom: Classroom = Depends(taught_class), include_resolved: bool = Query(default=False),
                     c: Container = Depends(container)) -> list[UncertainGrade]:
    """The roster's answers the grader could not decide, newest first: the question as the learner saw it and their
    answer, never the answer key or the grader's reason. By default only those that no later graded attempt on the
    same item settled; `include_resolved=true` lists all."""
    return c.class_dashboard.uncertain_grades(classroom, include_resolved=include_resolved)


@router.post("/classes/{class_id}/join-code", response_model=ClassJoinCode)
def rotate_join_code(classroom: Classroom = Depends(taught_class),
                     c: Container = Depends(container)) -> ClassJoinCode:
    """A new join code for the class. The old code stops working at once; learners already on the roster stay."""
    return c.classes.rotate_join_code(classroom)


@router.delete("/classes/{class_id}/learners/{learner_id}", status_code=204, response_class=Response)
def remove_learner(class_id: str, learner_id: str, p: Principal = Depends(principal),
                   c: Container = Depends(container)) -> Response:
    """Take a learner off the roster: the class's teacher removes them, or the learner (whoever acts for them in
    their own scope) leaves. The teacher's read access to them through this class ends at once."""
    if p.may_access(learner_id):
        found = c.classes.repository.get(class_id)
        if found is None or found.tenant_id != p.tenant_id:
            raise ResourceNotFound()
    else:
        taught_class(class_id, p, c)
    if c.classes.member(class_id, learner_id) is None:
        raise ResourceNotFound()
    c.classes.remove(class_id, learner_id)
    return Response(status_code=204)


@router.post("/learners/{learner_id}/classes", response_model=Enrollment, status_code=201,
             dependencies=[Depends(require_learner)])
def join_class(learner_id: str, body: JoinClass, response: Response, p: Principal = Depends(principal),
               c: Container = Depends(container)) -> Enrollment:
    """Join the class the code names: the learner consents to its teacher reading their data. Joining again
    returns the same enrollment (200)."""
    enrollment, created = c.classes.join(learner_id, body.join_code, tenant_id=p.tenant_id)
    if not created:
        response.status_code = 200
    return enrollment


@router.get("/learners/{learner_id}/classes", response_model=list[Enrollment])
def learner_classes(learner_id: str, p: Principal = Depends(principal),
                    c: Container = Depends(container)) -> list[Enrollment]:
    """The classes the learner is in. For the learner's own scope only: a teacher does not see their learners'
    other classes."""
    authorize(p, learner_id)
    return c.classes.enrollments(learner_id, tenant_id=p.tenant_id)
