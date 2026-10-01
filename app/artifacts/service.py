"""Artifact service: versioning, content-hash deduplication and the artifact dependency graph.

Media objects (images, audio, presentations) are content-addressed: `put_object` stores identical bytes once, and any number of
artifacts can point at the same object.
"""

from __future__ import annotations

import hashlib
import mimetypes
from typing import Protocol

from app.observability.scope import ExecutionScope
from app.schemas.artifact import Artifact, ArtifactDraft, ArtifactType, StoredArtifacts, StoredObject
from app.schemas.common import new_id
from app.schemas.events import EventType

EXTENSIONS = {
    "application/json": ".json",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "image/svg+xml": ".svg",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


class ArtifactRepository(Protocol):
    def add(self, artifact: Artifact) -> None: ...

    def get(self, artifact_id: str) -> Artifact: ...

    def list_for_task(self, task_id: str) -> list[Artifact]: ...

    def latest(self, task_id: str, name: str) -> Artifact | None: ...


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...

    def put_if_absent(self, key: str, data: bytes) -> tuple[str, bool]: ...

    def get(self, uri: str) -> bytes: ...


class ArtifactGraphError(ValueError):
    pass


class ArtifactService:
    def __init__(self, repository: ArtifactRepository, store: ObjectStore) -> None:
        self._repo = repository
        self._store = store

    def store(
        self,
        *,
        task_id: str,
        name: str,
        type: ArtifactType,
        media_type: str,
        content: bytes,
        provider: str,
        parent_ids: list[str],
        metadata: dict,
        scope: ExecutionScope,
    ) -> Artifact:
        digest = hashlib.sha256(content).hexdigest()
        reused = self._reusable(task_id, name, digest, parent_ids)
        if reused is not None:
            return reused
        latest = self._repo.latest(task_id, name)
        version = latest.version + 1 if latest else 1
        uri = self._store.put(f"{task_id}/{name}/v{version}{_extension(media_type)}", content)
        return self._add(Artifact(
            artifact_id=new_id("art"), task_id=task_id, type=type, name=name, uri=uri, media_type=media_type,
            content_hash=digest, size_bytes=len(content), version=version, provider=provider,
            parent_ids=parent_ids, metadata=metadata,
        ), scope)

    def put_object(self, content: bytes, media_type: str) -> StoredObject:
        """Store bytes content-addressed by their sha256. Identical content is written once and reused."""
        digest = hashlib.sha256(content).hexdigest()
        key = f"objects/sha256/{digest[:2]}/{digest}{_extension(media_type)}"
        uri, created = self._store.put_if_absent(key, content)
        return StoredObject(uri=uri, checksum=digest, media_type=media_type, size_bytes=len(content),
                            reused=not created, key=key)

    def read_object(self, uri: str) -> bytes:
        return self._store.get(uri)

    def store_object(
        self,
        *,
        task_id: str,
        name: str,
        type: ArtifactType,
        obj: StoredObject,
        provider: str,
        parent_ids: list[str],
        metadata: dict,
        scope: ExecutionScope,
    ) -> Artifact:
        """An artifact that points at an object already in the store (see `put_object`); nothing is copied."""
        reused = self._reusable(task_id, name, obj.checksum, parent_ids)
        if reused is not None:
            return reused
        latest = self._repo.latest(task_id, name)
        return self._add(Artifact(
            artifact_id=new_id("art"), task_id=task_id, type=type, name=name, uri=obj.uri,
            media_type=obj.media_type, content_hash=obj.checksum, size_bytes=obj.size_bytes,
            version=latest.version + 1 if latest else 1, provider=provider, parent_ids=parent_ids,
            metadata=metadata,
        ), scope)

    def _reusable(self, task_id: str, name: str, digest: str, parent_ids: list[str]) -> Artifact | None:
        for pid in parent_ids:
            self._repo.get(pid)  # every parent must exist
        latest = self._repo.latest(task_id, name)
        if latest is not None and latest.content_hash == digest and sorted(latest.parent_ids) == sorted(parent_ids):
            return latest  # identical content: reuse instead of creating a new version
        return None

    def _add(self, artifact: Artifact, scope: ExecutionScope) -> Artifact:
        self._repo.add(artifact)
        scope.emit(EventType.ARTIFACT_CREATED, artifact_id=artifact.artifact_id, artifact_type=artifact.type.value,
                   name=artifact.name, version=artifact.version, parent_ids=artifact.parent_ids, uri=artifact.uri)
        return artifact

    def store_batch(self, task_id: str, drafts: list[ArtifactDraft], scope: ExecutionScope) -> StoredArtifacts:
        """Store drafts in dependency order. `parent_keys` refer to other drafts in the same batch."""
        by_key = {d.key: d for d in drafts}
        if len(by_key) != len(drafts):
            raise ArtifactGraphError("duplicate draft keys")
        stored: dict[str, Artifact] = {}
        visiting: set[str] = set()

        def visit(key: str) -> Artifact:
            if key in stored:
                return stored[key]
            if key in visiting:
                raise ArtifactGraphError(f"cycle through draft '{key}'")
            if key not in by_key:
                raise ArtifactGraphError(f"unknown parent key '{key}'")
            visiting.add(key)
            draft = by_key[key]
            parents = [visit(pk).artifact_id for pk in draft.parent_keys] + list(draft.parent_ids)
            stored[key] = self.store(
                task_id=task_id, name=draft.name, type=draft.type, media_type=draft.media_type,
                content=draft.content.encode("utf-8"), provider=draft.provider, parent_ids=parents,
                metadata=draft.metadata, scope=scope,
            )
            visiting.discard(key)
            return stored[key]

        for draft in drafts:
            visit(draft.key)
        return StoredArtifacts(artifacts=[stored[d.key] for d in drafts],
                               by_key={k: a.artifact_id for k, a in stored.items()})

    def get(self, artifact_id: str) -> Artifact:
        return self._repo.get(artifact_id)

    def read(self, artifact_id: str) -> bytes:
        return self._store.get(self._repo.get(artifact_id).uri)

    def list_for_task(self, task_id: str) -> list[Artifact]:
        return self._repo.list_for_task(task_id)

    def find(self, task_id: str, name: str) -> Artifact | None:
        """Latest version of a named artifact of a task."""
        return self._repo.latest(task_id, name)

    def graph(self, task_id: str) -> dict[str, list[str]]:
        """artifact_id -> parent artifact ids."""
        return {a.artifact_id: a.parent_ids for a in self._repo.list_for_task(task_id)}

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """All ancestors of an artifact, nearest first."""
        seen: dict[str, Artifact] = {}
        frontier = list(self._repo.get(artifact_id).parent_ids)
        while frontier:
            pid = frontier.pop(0)
            if pid in seen:
                continue
            seen[pid] = self._repo.get(pid)
            frontier.extend(seen[pid].parent_ids)
        return list(seen.values())


def _extension(media_type: str) -> str:
    return EXTENSIONS.get(media_type) or mimetypes.guess_extension(media_type) or ".bin"
