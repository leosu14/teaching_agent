"""Filesystem object store for large artifacts (text, audio, video), for development and single-machine use.

A production deployment uses the S3 store (`app/providers/storage/s3.py`) behind the same `ObjectStore` protocol.
Both record objects as store-relative `object:<key>` URIs (app/utils/object_uri.py). This store also reads the
absolute `file://` URIs that rows written before that scheme hold, as long as they lie under its root.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from app.utils.object_uri import is_object_uri, key_of, object_uri, validate_key


class FilesystemObjectStore:
    scheme = "file"

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return object_uri(key)

    def put_if_absent(self, key: str, data: bytes) -> tuple[str, bool]:
        """Write `data` unless `key` already exists. Returns (uri, created). Atomic: of two concurrent writers of one
        key exactly one creates it, and an existing object is never replaced."""
        path = self._path(key)
        if path.exists():
            return object_uri(key), False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        try:
            os.link(tmp, path)  # fails if the key exists, whoever created it meanwhile
        except FileExistsError:
            return object_uri(key), False
        finally:
            tmp.unlink(missing_ok=True)
        return object_uri(key), True

    def put_file_if_absent(self, key: str, source: Path) -> tuple[str, bool]:
        """Copy a file into the store unless `key` exists, streaming it (large media never sits in memory)."""
        path = self._path(key)
        if path.exists():
            return object_uri(key), False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, tmp)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return object_uri(key), False
        finally:
            tmp.unlink(missing_ok=True)
        return object_uri(key), True

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

    def local_path(self, uri: str) -> Path:
        """Where an object's file is (diagnosis and tests; the S3 store has no such thing)."""
        return self._resolve(uri)

    def check(self) -> None:
        """Readiness: the root is a writable directory (raises OSError otherwise)."""
        probe = self._root / f".probe-{uuid.uuid4().hex}"
        probe.write_bytes(b"ok")
        probe.unlink()

    def _resolve(self, uri: str) -> Path:
        if is_object_uri(uri):
            return self._path(key_of(uri))
        if not uri.startswith("file://"):  # a legacy absolute URI, accepted only under this store's root
            raise ValueError("unsupported object uri")
        path = Path(uri.removeprefix("file://")).resolve()
        if self._root not in path.parents:
            raise ValueError("uri outside object store root")
        return path

    def _path(self, key: str) -> Path:
        path = (self._root / validate_key(key)).resolve()
        if self._root not in path.parents:
            raise ValueError("invalid object key")
        return path
