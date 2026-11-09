"""Classes and rosters (MVP item 4). A teacher (the `teacher` role) creates classes and hands out their join codes;
a learner joins with a code, and from then on the teacher may read, never change, that learner's data through every
learner route (app/api/authorization.py). Either side can end the membership: the teacher removes the learner, or
the learner leaves.

A class of another teacher, or of another tenant, gets the same 404 as one that does not exist, and so does a wrong
or retired join code."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.auth import TEACHER, Principal
from app.api.authorization import ResourceNotFound, authorize, principal, require_learner, require_role, taught_class
from app.api.deps import container
from app.schemas.classroom import ClassJoinCode, Classroom, ClassView, CreateClass, Enrollment, JoinClass
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
