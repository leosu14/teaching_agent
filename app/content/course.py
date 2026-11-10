"""Course loader and validator: authored course files -> validated knowledge-base documents (MVP item 5).

`load_courses` reads every configured course, validates each one and all of them together, and returns the documents
the local retriever indexes. Every problem is reported at once, each with the path of the field it is about, so an
author can fix a course in one pass. A course with any problem is never loaded: the application refuses to start.

What is checked, beyond the field types of `app/schemas/course.py`:

- the framework is known and the course's levels are distinct levels of it, in its order;
- unit ids, topics (case-insensitively) and concept ids are unique; every level has concepts; a unit holds at most
  `MAX_CONCEPTS_PER_UNIT` concepts (a lesson's concept map looks up at most that many for its topic);
- every concept's level is one of the course's levels;
- prerequisites are known concepts of the course, listed once, not the concept itself, never at a higher level than
  the concept (a goal "up to A2" must not pull in a B1 concept), and the graph has no cycle (each cycle is named);
- every concept has at least `MIN_PROBES` probes with at least two difficulties (the adaptive diagnostic follows a
  missed question with an easier one), distinct prompts, and accepted answers that differ from the answer and from
  each other;
- across courses: course ids, concept ids, and (subject, topic) pairs are unique.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.learner.frameworks import FrameworkRegistry, UnknownFramework
from app.pedagogy.graph import ConceptGraph
from app.schemas.concepts import Concept
from app.schemas.course import Course, CourseSummary
from app.utils.text import normalize

MIN_PROBES = 2
MAX_CONCEPTS_PER_UNIT = 12  # rag.concept_map's default k


class CourseError(ValueError):
    """One or more courses are invalid. `problems` holds every problem, each prefixed with its file and field."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{len(problems)} course problem(s):\n" + "\n".join(f"- {p}" for p in problems))


@dataclass
class LoadedCourses:
    courses: list[Course] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)

    def summaries(self) -> list[CourseSummary]:
        return [summarize(c) for c in self.courses]


# --- parsing ----------------------------------------------------------------------------------------------------


def _loc(parts: Iterable[object]) -> str:
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "(course)"


def parse_course(raw: object, source: str) -> Course:
    """The course in `raw` (decoded JSON). Raises CourseError with one problem per invalid field."""
    try:
        return Course.model_validate(raw)
    except ValidationError as exc:
        raise CourseError([f"{source}: {_loc(e['loc'])}: {e['msg']}" for e in exc.errors()]) from None


def read_course(path: Path) -> Course:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CourseError([f"{path}: no such course file"]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise CourseError([f"{path}: cannot read the course file ({exc.__class__.__name__})"]) from None
    except json.JSONDecodeError as exc:
        raise CourseError([f"{path}: not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})"]) from None
    return parse_course(raw, str(path))


# --- validation -------------------------------------------------------------------------------------------------


def validate_course(course: Course, frameworks: FrameworkRegistry, source: str = "") -> list[str]:
    """Every problem of one (well-typed) course; empty when it is valid."""
    where = f"{source}: " if source else ""
    problems: list[str] = []

    def problem(path: str, message: str) -> None:
        problems.append(f"{where}{path}: {message}")

    try:
        framework_levels = frameworks.get(course.framework).levels
    except UnknownFramework:
        problem("framework", f"unknown level framework {course.framework!r} (known: {frameworks.ids()})")
        framework_levels = None
    if len(set(course.levels)) != len(course.levels):
        problem("levels", "levels are listed more than once")
    if framework_levels is not None:
        unknown = [lv for lv in course.levels if lv not in framework_levels]
        if unknown:
            problem("levels", f"{unknown} are not levels of {course.framework} ({framework_levels})")
        known = [lv for lv in course.levels if lv in framework_levels]
        if known != sorted(known, key=framework_levels.index):
            problem("levels", f"levels must be in the framework's order ({framework_levels})")
    course_levels = set(course.levels)
    rank = {lv: i for i, lv in enumerate(course.levels)}

    units: dict[str, int] = {}
    topics: dict[str, int] = {}
    located: dict[str, str] = {}  # concept id -> its path
    level_of: dict[str, str] = {}
    for u, unit in enumerate(course.units):
        upath = f"units[{u}]"
        if unit.unit_id in units:
            problem(f"{upath}.unit_id", f"duplicate unit id {unit.unit_id!r} (also units[{units[unit.unit_id]}])")
        units.setdefault(unit.unit_id, u)
        topic = normalize(unit.topic)
        if not topic:
            problem(f"{upath}.topic", "the topic has no words")
        elif topic in topics:
            problem(f"{upath}.topic", f"topic {unit.topic!r} is also the topic of units[{topics[topic]}]")
        topics.setdefault(topic, u)
        if unit.level not in course_levels:
            problem(f"{upath}.level", f"{unit.level!r} is not one of the course's levels {course.levels}")
        if not unit.concepts:
            problem(f"{upath}.concepts", "a unit needs at least one concept")
        if len(unit.concepts) > MAX_CONCEPTS_PER_UNIT:
            problem(f"{upath}.concepts", f"{len(unit.concepts)} concepts: a unit holds at most "
                                         f"{MAX_CONCEPTS_PER_UNIT} (split it into more units)")
        for c, concept in enumerate(unit.concepts):
            cpath = f"{upath}.concepts[{c}]"
            if concept.concept_id in located:
                problem(f"{cpath}.concept_id",
                        f"duplicate concept id {concept.concept_id!r} (also {located[concept.concept_id]})")
                continue
            located[concept.concept_id] = cpath
            level = course.level_of(unit, concept)
            level_of[concept.concept_id] = level
            if concept.level is not None and concept.level not in course_levels:
                problem(f"{cpath}.level", f"{concept.level!r} is not one of the course's levels {course.levels}")
            problems.extend(f"{where}{cpath}.{p}" for p in _probe_problems(concept.probes))

    for lv in course.levels:
        if lv in course_levels and not any(level == lv for level in level_of.values()):
            problem("levels", f"level {lv!r} has no concepts")

    edges: dict[str, list[str]] = {}
    for unit, concept in course.concepts():
        cpath = located.get(concept.concept_id)
        if cpath is None or edges.get(concept.concept_id) is not None:
            continue
        edges[concept.concept_id] = []
        seen: set[str] = set()
        for p, pre in enumerate(concept.prerequisites):
            ppath = f"{cpath}.prerequisites[{p}]"
            if pre == concept.concept_id:
                problem(ppath, "a concept cannot be its own prerequisite")
            elif pre in seen:
                problem(ppath, f"{pre!r} is listed more than once")
            elif pre not in located:
                problem(ppath, f"unknown concept {pre!r}")
            else:
                edges[concept.concept_id].append(pre)
                mine, theirs = level_of[concept.concept_id], level_of[pre]
                if mine in rank and theirs in rank and rank[theirs] > rank[mine]:
                    problem(ppath, f"{pre!r} is {theirs}, above this {mine} concept: a prerequisite cannot be at a "
                                   "higher level")
            seen.add(pre)
    for cycle in _cycles(edges):
        problem(located[cycle[0]] + ".prerequisites", "prerequisite cycle " + " -> ".join([*cycle, cycle[0]]))
    return problems


def _probe_problems(probes) -> list[str]:
    out: list[str] = []
    if len(probes) < MIN_PROBES:
        out.append(f"probes: {len(probes)} probe(s): a concept needs at least {MIN_PROBES}")
    elif len({p.difficulty for p in probes}) < 2:
        out.append("probes: every probe has the same difficulty: the diagnostic needs an easier follow-up question")
    prompts: dict[str, int] = {}
    for i, probe in enumerate(probes):
        prompt = normalize(probe.prompt)
        if not prompt:
            out.append(f"probes[{i}].prompt: the prompt has no words")
        elif prompt in prompts:
            out.append(f"probes[{i}].prompt: same prompt as probes[{prompts[prompt]}]")
        prompts.setdefault(prompt, i)
        answer = normalize(probe.answer)
        if not answer:
            out.append(f"probes[{i}].answer: the answer has no words")
        seen = {answer}
        for a, accepted in enumerate(probe.accepted):
            value = normalize(accepted)
            if not value:
                out.append(f"probes[{i}].accepted[{a}]: the accepted answer has no words")
            elif value in seen:
                out.append(f"probes[{i}].accepted[{a}]: {accepted!r} repeats the answer or another accepted answer")
            seen.add(value)
    return out


def _cycles(edges: dict[str, list[str]]) -> list[list[str]]:
    """Each prerequisite cycle once, starting at its smallest id."""
    found: set[tuple[str, ...]] = set()
    state: dict[str, int] = {}  # 1: on the stack, 2: done
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in edges.get(node, []):
            if state.get(nxt) == 1:
                cycle = stack[stack.index(nxt):]
                start = cycle.index(min(cycle))
                found.add(tuple(cycle[start:] + cycle[:start]))
            elif state.get(nxt) is None:
                visit(nxt)
        stack.pop()
        state[node] = 2

    for node in sorted(edges):
        if node not in state:
            visit(node)
    return [list(c) for c in sorted(found)]


# --- compilation ------------------------------------------------------------------------------------------------


def concept_doc_id(course: Course, concept_id: str) -> str:
    return f"{course.course_id}/{concept_id}"


def reference_doc_id(course: Course, concept_id: str) -> str:
    return f"{course.course_id}/{concept_id}/reference"


def compile_course(course: Course) -> list[dict]:
    """The knowledge-base documents of a valid course, in the retriever's format: per concept, a `concept` document
    (the concept graph node and its probes) and a `reference` document (the teaching material research quotes, as
    'Explanation. Example: ... Practice: ... Answer: ...')."""
    docs: list[dict] = []
    for unit, concept in course.concepts():
        level = course.level_of(unit, concept)
        common = {"subject": course.subject, "course_id": course.course_id, "unit_id": unit.unit_id,
                  "publisher": course.publisher, "language": course.language}
        examples = " ".join(concept.examples)
        docs.append({
            "doc_id": concept_doc_id(course, concept.concept_id),
            "title": concept.name,
            "text": f"{concept.name}: {concept.explanation} Examples: {examples} {unit.title}, {level}.",
            "metadata": {
                "kind": "concept", **common, "topic": unit.topic, "level": level,
                "concept": {"concept_id": concept.concept_id, "name": concept.name, "level": level,
                            "prerequisites": list(concept.prerequisites), "description": concept.description},
                "probes": [p.model_dump(mode="json") for p in concept.probes],
            },
        })
        docs.append({
            "doc_id": reference_doc_id(course, concept.concept_id),
            "title": f"Reference: {concept.name}",
            "text": f"{concept.name}: {concept.explanation} Example: {examples} Practice: {concept.practice.prompt} "
                    f"Answer: {concept.practice.answer}",
            "metadata": {"kind": "reference", **common, "topic": unit.topic, "level": level,
                         "concept_id": concept.concept_id},
        })
    return docs


def concepts_of(course: Course) -> list[Concept]:
    return [Concept(concept_id=c.concept_id, name=c.name, level=course.level_of(u, c),
                    prerequisites=list(c.prerequisites), description=c.description, domain=course.subject,
                    topic=u.topic) for u, c in course.concepts()]


def summarize(course: Course) -> CourseSummary:
    per_level = {lv: 0 for lv in course.levels}
    for unit, concept in course.concepts():
        level = course.level_of(unit, concept)
        per_level[level] = per_level.get(level, 0) + 1
    return CourseSummary(course_id=course.course_id, title=course.title, subject=course.subject,
                         framework=course.framework, levels=list(course.levels), units=len(course.units),
                         concepts=sum(len(u.concepts) for u in course.units),
                         probes=sum(len(c.probes) for _, c in course.concepts()), concepts_per_level=per_level,
                         topics=[u.topic for u in course.units])


# --- loading ----------------------------------------------------------------------------------------------------


def load_courses(paths: Iterable[Path], frameworks: FrameworkRegistry) -> LoadedCourses:
    """Read, validate and compile every course; raise CourseError with every problem of every course."""
    problems: list[str] = []
    courses: list[tuple[Course, str]] = []
    for path in paths:
        try:
            course = read_course(path)
        except CourseError as exc:
            problems += exc.problems
            continue
        found = validate_course(course, frameworks, str(path))
        problems += found
        if not found:
            courses.append((course, str(path)))
    problems += _cross_course_problems(courses)
    if problems:
        raise CourseError(problems)
    loaded = LoadedCourses(courses=[c for c, _ in courses])
    for course, _ in courses:
        # A final check with the engine's own graph: what the engine loads is exactly what was validated.
        ConceptGraph(concepts_of(course))
        loaded.documents += compile_course(course)
    return loaded


def _cross_course_problems(courses: list[tuple[Course, str]]) -> list[str]:
    problems: list[str] = []
    ids: dict[str, str] = {}
    concepts: dict[str, str] = {}
    topics: dict[tuple[str, str], str] = {}
    for course, source in courses:
        if course.course_id in ids:
            problems.append(f"{source}: course_id: course {course.course_id!r} is also loaded from "
                            f"{ids[course.course_id]}")
            continue
        ids[course.course_id] = source
        for unit, concept in course.concepts():
            if concept.concept_id in concepts:
                problems.append(f"{source}: concept {concept.concept_id!r} is also defined in "
                                f"{concepts[concept.concept_id]}")
            concepts.setdefault(concept.concept_id, source)
        for unit in course.units:
            key = (normalize(course.subject), normalize(unit.topic))
            if key in topics:
                problems.append(f"{source}: {course.subject} topic {unit.topic!r} is also a topic of "
                                f"{topics[key]}")
            topics.setdefault(key, source)
    return problems
