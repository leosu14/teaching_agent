"""Object URIs: where an artifact's bytes are, relative to the configured object store.

An object is recorded as `object:<key>`, a key relative to the store (`<task>/<name>/v1.json`,
`objects/sha256/ab/ab...png`), never as a location on one machine. The same database row therefore resolves on any
replica and against either store (the filesystem under `TA_OBJECT_STORE_DIR`, or an S3 bucket), and moving the
objects means copying the keys. Rows written before this scheme hold absolute `file://` URIs; the filesystem store
still reads those when they lie under its root.
"""

from __future__ import annotations

import re

SCHEME = "object:"
# Keys are what the artifact service builds: ids, names, versions, digests and extensions. Nothing absolute, no
# parent or empty segments, no backslashes or control characters.
_FORBIDDEN = re.compile(r"[\\\x00-\x1f\x7f]")


def is_object_uri(uri: str) -> bool:
    return uri.startswith(SCHEME)


def validate_key(key: str) -> str:
    parts = key.split("/")
    if not key or len(key) > 1024 or _FORBIDDEN.search(key) or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"invalid object key {key!r}")
    return key


def object_uri(key: str) -> str:
    return SCHEME + validate_key(key)


def key_of(uri: str) -> str:
    """The key of an `object:` URI (ValueError for anything else)."""
    if not is_object_uri(uri):
        raise ValueError("not an object uri")
    return validate_key(uri[len(SCHEME):])
