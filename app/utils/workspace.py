"""Controlled scratch space for media work.

Every temporary file a composition or validation needs lives in a directory created here, under one configured
root, with a sanitised name. A directory is deleted when the work succeeds; when it fails it is kept under
`<root>/failed/` for diagnostics (unless `keep_failed` is off). Nothing outside the root is ever created or deleted.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SAFE = re.compile(r"[^A-Za-z0-9_-]+")


class ScratchSpace:
    def __init__(self, root: Path, *, keep_failed: bool = True) -> None:
        self.root = root.resolve()
        self.keep_failed = keep_failed

    @contextmanager
    def session(self, label: str) -> Iterator[Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        prefix = SAFE.sub("_", label)[:48] + "-"
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.root))
        try:
            yield path
        except BaseException:
            if self.keep_failed:
                failed = self.root / "failed"
                failed.mkdir(exist_ok=True)
                shutil.move(str(path), failed / path.name)
            else:
                shutil.rmtree(path, ignore_errors=True)
            raise
        else:
            shutil.rmtree(path, ignore_errors=True)

    def failed_sessions(self) -> list[Path]:
        failed = self.root / "failed"
        return sorted(p for p in failed.iterdir() if p.is_dir()) if failed.exists() else []
