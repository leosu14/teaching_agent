# Courses

A course is the knowledge base of one subject, authored as one JSON file. `spanish-a1-b1.json` is the bundled pilot
course: Spanish for English speakers from A1 to B1, 18 units, 46 concepts, 138 diagnostic probes.

Turn courses on with `TA_COURSES` (comma-separated bundled names or file paths):

```bash
TA_COURSES=spanish-a1-b1                                # the bundled course
TA_COURSES=spanish-a1-b1,/srv/courses/french-a1.json    # several courses
```

When `TA_COURSES` is set the courses are the knowledge base: `TA_CORPUS_DIR/knowledge_base.json` is not read. Every
course is validated at startup and an invalid course stops the application with every problem listed. Unset, the
fixture knowledge base in `TA_CORPUS_DIR` is used, as before.

## Checking a course

```bash
python -m app.content validate spanish-a1-b1 my-course.json   # the same checks as startup; exit 1 on a problem
python -m app.content build spanish-a1-b1 --out kb.json       # also write the compiled knowledge-base documents
```

Problems name the file and the field, for example
`my-course.json: units[3].concepts[1].prerequisites[0]: unknown concept 'es.a1.verbs'`.

## Format

```json
{
  "schema_version": 1,
  "course_id": "spanish-a1-b1",            // lowercase letters, digits and hyphens
  "title": "Spanish A1 to B1",
  "subject": "spanish",                    // the domain goals and lessons name
  "framework": "cefr",                     // a level framework: cefr or mastery
  "levels": ["A1", "A2", "B1"],            // the framework's levels the course covers, lowest first
  "language": "en",                        // language of the explanations
  "publisher": "Course knowledge base",
  "description": "...",
  "units": [{
    "unit_id": "a1-greetings",
    "title": "Greetings and introductions",
    "topic": "greetings and introductions",  // what a lesson asks for; unique in the subject
    "level": "A1",
    "summary": "...",
    "concepts": [{
      "concept_id": "es.a1.greetings",       // lowercase, dotted: unique across all courses
      "name": "Greetings and goodbyes",
      "level": "A1",                         // optional: the unit's level
      "prerequisites": [],                   // concept ids of this course, at this level or below
      "description": "One line for planners and objectives.",
      "explanation": "What a lesson teaches, quoted by research.",
      "examples": ["¡Hola! Buenos días, señor García."],
      "practice": {"prompt": "Complete: ___ noches, mamá. (good)", "answer": "Buenas"},
      "probes": [                            // diagnostic questions with their keys
        {"prompt": "How do you say 'good morning' in Spanish?", "answer": "buenos días", "accepted": [],
         "difficulty": 0.5},
        {"prompt": "Which greeting do you use late in the evening: buenas noches or buenos días?",
         "answer": "buenas noches", "difficulty": 0.25}
      ]
    }]
  }]
}
```

(The comments are for this page only: JSON has none.)

## Rules

The validator (`app/content/course.py`) checks, beyond the field types:

- the framework exists, and the levels are distinct levels of it in its order; every level has concepts;
- unit ids, topics (ignoring case and punctuation) and concept ids are unique; a unit has 1 to 12 concepts (a
  lesson's concept map looks up at most 12 per topic);
- a concept's level is one of the course's levels;
- prerequisites are concepts of the course, listed once, not the concept itself, never at a higher level than the
  concept (a goal "up to A2" must not pull in a B1 concept), and without cycles (each cycle is named);
- every concept has at least 2 probes, and one of them is easier than the opening probe: the diagnostic opens on the
  probe nearest difficulty 0.5 (on a tie, the easier one) and after a miss asks an easier one; prompts are distinct,
  and accepted answers differ from the answer and each other;
- across courses: course ids, concept ids and (subject, topic) pairs are unique.

Grading keeps accents (`hablé` and `hable` are different words), so list an accepted answer only when it is really
correct, never as a looser spelling.

## What the application does with a course

Each concept compiles to two knowledge-base documents: a `concept` document (the prerequisite-graph node, its unit's
topic and level, and the probes) and a `reference` document (`explanation`, `examples` and `practice` as
"... Example: ... Practice: ... Answer: ..."), which research cites as `kb://<course>/<concept>/reference`. Goals for
a level target the course's concepts up to that level; a curriculum orders them prerequisites first; a lesson's
diagnostic asks the probes of its topic's concepts; the production preflight refuses a topic no unit covers.
