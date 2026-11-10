"""Authored courses (MVP item 5): the course format, the validator (every rule, every problem at once, with the field
path), compilation to knowledge-base documents, TA_COURSES, the CLI, and the bundled Spanish A1 to B1 course."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.config.routing import ConfigError
from app.config.settings import COURSES_DIR, Settings, course_path
from app.content.__main__ import main as cli
from app.content.course import (
    MAX_CONCEPTS_PER_UNIT,
    CourseError,
    compile_course,
    concepts_of,
    load_courses,
    parse_course,
    read_course,
    summarize,
    validate_course,
)
from app.learner.frameworks import CEFR, default_frameworks
from app.pedagogy.graph import ConceptGraph
from app.providers.retrieval.local import LocalKnowledgeBase
from app.schemas.course import Course

BUNDLED = COURSES_DIR / "spanish-a1-b1.json"
FRAMEWORKS = default_frameworks()


def probe(prompt: str, answer: str, difficulty: float) -> dict:
    return {"prompt": prompt, "answer": answer, "difficulty": difficulty}


def concept(cid: str, *, prerequisites=(), level=None) -> dict:
    out = {"concept_id": cid, "name": cid.split(".")[-1].title(), "prerequisites": list(prerequisites),
           "description": f"About {cid}.", "explanation": f"How {cid} works.", "examples": [f"Example of {cid}."],
           "practice": {"prompt": f"Practise {cid}.", "answer": "yes"},
           "probes": [probe(f"Easy {cid}?", "a", 0.3), probe(f"Hard {cid}?", "b", 0.6)]}
    if level:
        out["level"] = level
    return out


def raw_course(**overrides) -> dict:
    course = {
        "course_id": "test-course", "title": "Test course", "subject": "testing", "framework": "cefr",
        "levels": ["A1", "A2"], "publisher": "Tests",
        "units": [
            {"unit_id": "u1", "title": "One", "topic": "first topic", "level": "A1",
             "concepts": [concept("t.a"), concept("t.b", prerequisites=["t.a"])]},
            {"unit_id": "u2", "title": "Two", "topic": "second topic", "level": "A2",
             "concepts": [concept("t.c", prerequisites=["t.b"])]},
        ],
    }
    course.update(overrides)
    return course


def problems_of(raw: dict) -> list[str]:
    return validate_course(parse_course(raw, "c.json"), FRAMEWORKS)


def write(tmp_path: Path, raw: dict, name: str = "course.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# --- the bundled course ----------------------------------------------------------------------------------------


def test_the_bundled_spanish_course_is_valid_and_covers_a1_to_b1() -> None:
    course = read_course(BUNDLED)
    assert validate_course(course, FRAMEWORKS, str(BUNDLED)) == []
    s = summarize(course)
    assert (s.subject, s.framework, s.levels) == ("spanish", "cefr", ["A1", "A2", "B1"])
    assert s.units >= 15 and s.concepts >= 40 and all(n >= 10 for n in s.concepts_per_level.values())
    assert s.probes >= 3 * s.concepts and len(set(s.topics)) == s.units
    # the engine's own graph accepts it, and every concept's prerequisites are at its level or below
    graph = ConceptGraph(concepts_of(course))
    order = CEFR.levels
    for c in graph.concepts():
        assert all(order.index(graph.concept(p).level) <= order.index(c.level) for p in c.prerequisites)


def test_bundled_course_compiles_to_one_concept_and_one_reference_document_per_concept() -> None:
    course = read_course(BUNDLED)
    docs = compile_course(course)
    kinds = [d["metadata"]["kind"] for d in docs]
    assert kinds.count("concept") == kinds.count("reference") == summarize(course).concepts
    assert len({d["doc_id"] for d in docs}) == len(docs)
    for d in docs:
        assert d["metadata"]["subject"] == "spanish" and d["metadata"]["course_id"] == "spanish-a1-b1"
        if d["metadata"]["kind"] == "reference":
            # the material research quotes: 'Explanation. Example: ... Practice: ... Answer: ...'
            assert " Example: " in d["text"] and " Practice: " in d["text"] and " Answer: " in d["text"]


# --- compilation -----------------------------------------------------------------------------------------------


def test_compiled_documents_carry_the_concept_node_probes_and_unit_topic() -> None:
    course = parse_course(raw_course(), "c.json")
    docs = {d["doc_id"]: d for d in compile_course(course)}
    assert sorted(docs) == ["test-course/t.a", "test-course/t.a/reference", "test-course/t.b",
                            "test-course/t.b/reference", "test-course/t.c", "test-course/t.c/reference"]
    b = docs["test-course/t.b"]["metadata"]
    assert b["kind"] == "concept" and b["topic"] == "first topic" and b["level"] == "A1" and b["unit_id"] == "u1"
    assert b["concept"] == {"concept_id": "t.b", "name": "B", "level": "A1", "prerequisites": ["t.a"],
                            "description": "About t.b."}
    assert [p["answer"] for p in b["probes"]] == ["a", "b"]
    ref = docs["test-course/t.b/reference"]
    assert ref["metadata"]["kind"] == "reference" and ref["metadata"]["concept_id"] == "t.b"
    assert "probes" not in ref["metadata"] and "Easy" not in ref["text"]  # probe keys stay out of research material


def test_a_concept_level_defaults_to_its_unit_and_may_be_set() -> None:
    raw = raw_course()
    raw["units"][1]["concepts"].append(concept("t.d", level="A1"))
    course = parse_course(raw, "c.json")
    assert {c.concept_id: c.level for c in concepts_of(course)} == {"t.a": "A1", "t.b": "A1", "t.c": "A2",
                                                                   "t.d": "A1"}
    assert validate_course(course, FRAMEWORKS) == []


# --- validation rules ------------------------------------------------------------------------------------------


def test_a_valid_course_has_no_problems() -> None:
    assert problems_of(raw_course()) == []


def test_field_errors_name_the_file_and_the_field_path() -> None:
    raw = raw_course()
    raw["units"][0]["concepts"][1]["concept_id"] = "Bad Id"
    del raw["units"][1]["concepts"][0]["practice"]
    raw["surprise"] = 1
    with pytest.raises(CourseError) as err:
        parse_course(raw, "c.json")
    text = "\n".join(err.value.problems)
    assert "c.json: units[0].concepts[1].concept_id: String should match pattern" in text
    assert "c.json: units[1].concepts[0].practice: Field required" in text
    assert "c.json: surprise: Extra inputs are not permitted" in text


@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r.update(framework="nope"), "framework: unknown level framework 'nope'"),
    (lambda r: r.update(levels=["A1", "A2", "Z9"]), "levels: ['Z9'] are not levels of cefr"),
    (lambda r: r.update(levels=["A2", "A1"]), "levels: levels must be in the framework's order"),
    (lambda r: r.update(levels=["A1", "A1", "A2"]), "levels: levels are listed more than once"),
    (lambda r: r.update(levels=["A1", "A2", "B1"]), "levels: level 'B1' has no concepts"),
    (lambda r: r["units"][1].update(unit_id="u1"), "units[1].unit_id: duplicate unit id 'u1' (also units[0])"),
    (lambda r: r["units"][1].update(topic="First  Topic!"),
     "units[1].topic: topic 'First  Topic!' is also the topic of units[0]"),
    (lambda r: r["units"][1].update(topic="?!"), "units[1].topic: the topic has no words"),
    (lambda r: r["units"][1].update(level="B2"), "units[1].level: 'B2' is not one of the course's levels"),
    (lambda r: r["units"][1].update(concepts=[]), "units[1].concepts: a unit needs at least one concept"),
    (lambda r: r["units"][1]["concepts"].append(concept("t.a")),
     "units[1].concepts[1].concept_id: duplicate concept id 't.a' (also units[0].concepts[0])"),
    (lambda r: r["units"][1]["concepts"][0].update(level="C1"),
     "units[1].concepts[0].level: 'C1' is not one of the course's levels"),
    (lambda r: r["units"][0]["concepts"][1].update(prerequisites=["t.zz"]),
     "units[0].concepts[1].prerequisites[0]: unknown concept 't.zz'"),
    (lambda r: r["units"][0]["concepts"][1].update(prerequisites=["t.b"]),
     "units[0].concepts[1].prerequisites[0]: a concept cannot be its own prerequisite"),
    (lambda r: r["units"][0]["concepts"][1].update(prerequisites=["t.a", "t.a"]),
     "units[0].concepts[1].prerequisites[1]: 't.a' is listed more than once"),
    (lambda r: r["units"][0]["concepts"][1].update(prerequisites=["t.c"]),
     "units[0].concepts[1].prerequisites[0]: 't.c' is A2, above this A1 concept"),
    (lambda r: r["units"][0]["concepts"][0].update(probes=[probe("Only?", "x", 0.5)]),
     "units[0].concepts[0].probes: 1 probe(s): a concept needs at least 2"),
    (lambda r: r["units"][0]["concepts"][0].update(probes=[probe("One?", "x", 0.5), probe("Two?", "y", 0.5)]),
     "units[0].concepts[0].probes: every probe has the same difficulty"),
    # The diagnostic opens on the probe nearest 0.5 and follows a miss with an easier one: there must be one.
    (lambda r: r["units"][0]["concepts"][0].update(probes=[probe("One?", "x", 0.5), probe("Two?", "y", 0.65)]),
     "units[0].concepts[0].probes: no probe is easier than the opening question probes[0] (difficulty 0.5"),
    # A tie for the opening question counts the easier probe as the opening one.
    (lambda r: r["units"][0]["concepts"][0].update(probes=[probe("One?", "x", 0.45), probe("Two?", "y", 0.55)]),
     "units[0].concepts[0].probes: no probe is easier than the opening question probes[0] (difficulty 0.45"),
    (lambda r: r["units"][0]["concepts"][0].update(probes=[probe("Same?", "x", 0.3), probe("same", "y", 0.6)]),
     "units[0].concepts[0].probes[1].prompt: same prompt as probes[0]"),
    (lambda r: r["units"][0]["concepts"][0]["probes"][0].update(accepted=["A!"]),
     "units[0].concepts[0].probes[0].accepted[0]: 'A!' repeats the answer"),
    (lambda r: r["units"][0]["concepts"][0]["probes"][0].update(answer="..."),
     "units[0].concepts[0].probes[0].answer: the answer has no words"),
])
def test_each_rule_reports_the_field_it_is_about(mutate, expected) -> None:
    raw = copy.deepcopy(raw_course())
    mutate(raw)
    problems = problems_of(raw)
    assert any(expected in p for p in problems), problems
    assert all(p.startswith("units") or p.startswith("levels") or p.startswith("framework") for p in problems)


def test_a_prerequisite_cycle_is_named() -> None:
    raw = raw_course(levels=["A1"])
    raw["units"][1]["level"] = "A1"
    raw["units"][0]["concepts"][0]["prerequisites"] = ["t.c"]  # t.a -> t.c -> t.b -> t.a
    problems = problems_of(raw)
    assert problems == ["units[0].concepts[0].prerequisites: prerequisite cycle t.a -> t.c -> t.b -> t.a"]


def test_a_unit_holds_at_most_the_concepts_a_concept_map_returns() -> None:
    raw = raw_course()
    raw["units"][1]["concepts"] = [concept(f"t.many{i}") for i in range(MAX_CONCEPTS_PER_UNIT + 1)]
    assert any(f"{MAX_CONCEPTS_PER_UNIT + 1} concepts: a unit holds at most" in p for p in problems_of(raw))
    raw["units"][1]["concepts"].pop()
    assert problems_of(raw) == []


def test_every_problem_is_reported_at_once() -> None:
    raw = raw_course(framework="cefr", levels=["A1", "A2", "B1"])
    raw["units"][1]["unit_id"] = "u1"
    raw["units"][0]["concepts"][1]["prerequisites"] = ["t.zz"]
    raw["units"][0]["concepts"][0]["probes"] = raw["units"][0]["concepts"][0]["probes"][:1]
    assert len(problems_of(raw)) == 4


# --- loading ---------------------------------------------------------------------------------------------------


def test_load_reports_unreadable_files_and_problems_of_every_course(tmp_path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ not json", encoding="utf-8")
    broken = raw_course(course_id="broken")
    broken["units"][0]["concepts"][1]["prerequisites"] = ["t.zz"]
    with pytest.raises(CourseError) as err:
        load_courses([tmp_path / "missing.json", bad_json, write(tmp_path, broken)], FRAMEWORKS)
    problems = err.value.problems
    assert problems[0] == f"{tmp_path / 'missing.json'}: no such course file"
    assert problems[1].startswith(f"{bad_json}: not valid JSON (line 1")
    assert problems[2] == f"{tmp_path / 'course.json'}: units[0].concepts[1].prerequisites[0]: unknown concept 't.zz'"
    assert "3 course problem(s)" in str(err.value)


def test_courses_loaded_together_must_not_collide(tmp_path) -> None:
    first = write(tmp_path, raw_course(), "one.json")
    same_id = write(tmp_path, raw_course(), "two.json")
    with pytest.raises(CourseError) as err:
        load_courses([first, same_id], FRAMEWORKS)
    assert err.value.problems == [f"{same_id}: course_id: course 'test-course' is also loaded from {first}"]

    other = raw_course(course_id="other")
    other["units"][1]["topic"] = "another topic"
    with pytest.raises(CourseError) as err:
        load_courses([first, write(tmp_path, other, "three.json")], FRAMEWORKS)
    text = "\n".join(err.value.problems)
    assert "concept 't.a' is also defined in" in text and "testing topic 'first topic' is also a topic of" in text


def test_load_returns_the_compiled_documents_of_valid_courses(tmp_path) -> None:
    other = raw_course(course_id="other", subject="other subject")
    for unit in other["units"]:
        for c in unit["concepts"]:
            c["concept_id"] = c["concept_id"].replace("t.", "o.")
            c["prerequisites"] = [p.replace("t.", "o.") for p in c["prerequisites"]]
    loaded = load_courses([write(tmp_path, raw_course(), "a.json"), write(tmp_path, other, "b.json")], FRAMEWORKS)
    assert [c.course_id for c in loaded.courses] == ["test-course", "other"]
    assert len(loaded.documents) == 12
    assert [s.concepts for s in loaded.summaries()] == [3, 3]


async def test_the_retriever_indexes_compiled_documents() -> None:
    kb = LocalKnowledgeBase(documents=compile_course(parse_course(raw_course(), "c.json")))
    found = await kb.retrieve("first topic", 10, {"kind": "concept", "subject": "testing", "topic": "First Topic"})
    assert sorted(p.doc_id for p in found) == ["test-course/t.a", "test-course/t.b"]
    with pytest.raises(ValueError):
        LocalKnowledgeBase()
    with pytest.raises(ValueError):
        LocalKnowledgeBase(BUNDLED, documents=[])


def test_course_model_rejects_ids_that_could_collide_in_document_ids() -> None:
    for bad in ("a/b", "a:b", "nodot", "A.b"):
        raw = raw_course()
        raw["units"][0]["concepts"][0]["concept_id"] = bad
        with pytest.raises(CourseError):
            parse_course(raw, "c.json")
    with pytest.raises(CourseError):
        parse_course(raw_course(course_id="has.dot"), "c.json")
    assert isinstance(parse_course(raw_course(), "c.json"), Course)


# --- settings and startup --------------------------------------------------------------------------------------


def test_course_entries_are_bundled_names_or_paths(tmp_path) -> None:
    assert course_path("spanish-a1-b1") == BUNDLED
    assert course_path(" spanish-a1-b1 ") == BUNDLED
    assert course_path("my.json") == Path("my.json")
    assert course_path(str(tmp_path / "c")) == tmp_path / "c"
    s = Settings(data_dir=tmp_path, courses="spanish-a1-b1, ./other.json")
    assert s.course_paths() == [BUNDLED, Path("./other.json")]
    assert Settings(data_dir=tmp_path).course_paths() == []


def test_with_courses_the_fixture_knowledge_base_is_not_needed(tmp_path) -> None:
    empty = tmp_path / "corpus"
    empty.mkdir()
    for name in ("web_corpus.json", "image_catalog.json"):
        (empty / name).write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing knowledge_base.json"):
        Settings(data_dir=tmp_path, corpus_dir=empty).validate_runtime()
    Settings(data_dir=tmp_path, corpus_dir=empty, courses="spanish-a1-b1").validate_runtime()
    assert "TA_COURSES" in Settings(data_dir=tmp_path, courses="spanish-a1-b1").knowledge_source()


def test_a_missing_course_file_stops_startup(tmp_path) -> None:
    with pytest.raises(ConfigError, match="TA_COURSES names course files that do not exist"):
        Settings(data_dir=tmp_path, courses="no-such-course").validate_runtime()


def test_an_invalid_course_stops_the_container_with_every_problem(tmp_path) -> None:
    from app.services.container import build_container

    raw = raw_course()
    raw["units"][0]["concepts"][1]["prerequisites"] = ["t.zz"]
    raw["units"][1]["level"] = "C2"  # not a course level, so A2 also has no concepts: three problems
    path = write(tmp_path, raw)
    with pytest.raises(ConfigError) as err:
        build_container(Settings(data_dir=tmp_path / "data", courses=str(path), log_json=False))
    assert "TA_COURSES: 3 course problem(s)" in str(err.value) and "unknown concept 't.zz'" in str(err.value)
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").iterdir())  # nothing was created


# --- CLI -------------------------------------------------------------------------------------------------------


def test_cli_validates_and_builds(tmp_path, capsys) -> None:
    assert cli(["validate", "spanish-a1-b1"]) == 0
    assert capsys.readouterr().out.startswith("OK spanish-a1-b1 (spanish, cefr A1-B1): ")
    out = tmp_path / "kb.json"
    assert cli(["build", str(BUNDLED), "--out", str(out)]) == 0
    docs = json.loads(out.read_text(encoding="utf-8"))
    assert docs == compile_course(read_course(BUNDLED))
    assert f"wrote {len(docs)} documents" in capsys.readouterr().out


def test_cli_prints_every_problem_and_fails(tmp_path, capsys) -> None:
    raw = raw_course()
    raw["units"][0]["concepts"][1]["prerequisites"] = ["t.zz"]
    assert cli(["validate", str(write(tmp_path, raw))]) == 1
    err = capsys.readouterr().err
    assert err.startswith("INVALID: 1 problem(s)") and "unknown concept 't.zz'" in err
