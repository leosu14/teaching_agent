"""Course authoring CLI.

    python -m app.content validate courses/spanish-a1-b1.json [more.json ...]
    python -m app.content build courses/spanish-a1-b1.json --out knowledge_base.json

`validate` checks the courses together, as the application does at startup, and prints a summary or every problem
(exit 1). `build` also writes the compiled knowledge-base documents, for inspection or for TA_CORPUS_DIR.
Course names without a path (`spanish-a1-b1`) are the bundled courses in `courses/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config.settings import course_path
from app.content.course import CourseError, load_courses
from app.learner.frameworks import default_frameworks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.content", description="Validate and build course files.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "build"):
        cmd = sub.add_parser(name)
        cmd.add_argument("courses", nargs="+", help="course files, or names of bundled courses")
        if name == "build":
            cmd.add_argument("--out", type=Path, required=True, help="where to write the knowledge-base JSON")
    args = parser.parse_args(argv)
    paths = [course_path(c) for c in args.courses]
    try:
        loaded = load_courses(paths, default_frameworks())
    except CourseError as exc:
        print(f"INVALID: {len(exc.problems)} problem(s)", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    for s in loaded.summaries():
        levels = ", ".join(f"{lv}: {n}" for lv, n in s.concepts_per_level.items())
        print(f"OK {s.course_id} ({s.subject}, {s.framework} {'-'.join([s.levels[0], s.levels[-1]])}): "
              f"{s.units} units, {s.concepts} concepts ({levels}), {s.probes} probes")
    if args.command == "build":
        args.out.write_text(json.dumps(loaded.documents, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(loaded.documents)} documents to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
