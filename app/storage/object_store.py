"""Filesystem object store for large artifacts (text, audio, video). Swappable for S3-style stores."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO


class FilesystemObjectStore:
    scheme = "file"

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path.as_uri()

    def put_if_absent(self, key: str, data: bytes) -> tuple[str, bool]:
        """Write `data` unless `key` already exists. Returns (uri, created). For content-addressed keys."""
        path = self._path(key)
        if path.exists():
            return path.as_uri(), False
        return self.put(key, data), True

    def put_file_if_absent(self, key: str, source: Path) -> tuple[str, bool]:
        """Copy a file into the store unless `key` exists, streaming it (large media never sits in memory)."""
        path = self._path(key)
        if path.exists():
            return path.as_uri(), False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        tmp.replace(path)
        return path.as_uri(), True

    def get(self, uri: str) -> bytes:
        return self._resolve(uri).read_bytes()

    def get_key(self, key: str) -> bytes | None:
        """The object stored at `key`, or None (for small records whose key, not uri, is known)."""
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def exists(self, uri: str) -> bool:
        return self._resolve(uri).is_file()

    def delete(self, uri: str) -> None:
        """Remove an object (a corrupt one, so it can be written again). Missing objects are ignored."""
        self._resolve(uri).unlink(missing_ok=True)

    def open(self, uri: str) -> BinaryIO:
        """A binary reader for an object, for streaming large media."""
        return self._resolve(uri).open("rb")

    def _resolve(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise ValueError(f"unsupported uri {uri}")
        path = Path(uri.removeprefix("file://")).resolve()
        if self._root not in path.parents:
            raise ValueError("uri outside object store root")
        return path

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root not in path.parents:
            raise ValueError(f"invalid object key {key}")
        return path
