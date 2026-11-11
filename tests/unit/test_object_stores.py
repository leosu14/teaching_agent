"""Object stores (MVP item 8): store-relative `object:` URIs, the filesystem store's legacy `file://` URIs, and the
S3 store (AWS Signature Version 4 over httpx, conditional writes, streaming, sanitized errors)."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone

import pytest

from app.artifacts.service import ArtifactService
from app.config.routing import ConfigError
from app.config.storage import S3Settings
from app.providers.storage.s3 import ObjectStoreError, S3ObjectStore, sign_v4
from app.schemas.artifact import ArtifactType
from app.storage.db import create_db
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.utils.object_uri import key_of, object_uri, validate_key
from tests.fake_s3 import FakeS3
from tests.unit.helpers import scope


# --- object URIs ----------------------------------------------------------------------------------------------------

def test_object_uris_are_relative_keys() -> None:
    assert object_uri("t1/lesson/v1.json") == "object:t1/lesson/v1.json"
    assert key_of("object:objects/sha256/ab/ab.png") == "objects/sha256/ab/ab.png"
    for bad in ("", "/abs", "a/../b", "a//b", "a/./b", "a\\b", "a\x00b", "a/", "x" * 1025):
        with pytest.raises(ValueError):
            validate_key(bad)
    with pytest.raises(ValueError):
        key_of("file:///srv/objects/x")


def test_the_filesystem_store_records_relative_uris_and_still_reads_legacy_ones(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    uri = store.put("t1/lesson/v1.txt", b"body")
    assert uri == "object:t1/lesson/v1.txt" and store.get(uri) == b"body"
    legacy = (tmp_path / "objects" / "t1" / "lesson" / "v1.txt").resolve().as_uri()  # a row from before
    assert store.get(legacy) == b"body" and store.exists(legacy)
    moved = FilesystemObjectStore(tmp_path / "elsewhere")  # the objects copied to another root still resolve
    (tmp_path / "elsewhere" / "t1" / "lesson").mkdir(parents=True)
    (tmp_path / "elsewhere" / "t1" / "lesson" / "v1.txt").write_bytes(b"body")
    assert moved.get(uri) == b"body"
    with pytest.raises(ValueError):
        moved.get(legacy)  # an absolute path outside the store's root is never read
    with pytest.raises(ValueError):
        store.get("object:../escape")
    store.check()


def test_the_filesystem_store_creates_a_key_once_under_concurrency(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path / "o")
    created = []
    threads = [threading.Thread(target=lambda i=i: created.append(store.put_if_absent("k/v1", f"{i}".encode())[1]))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert created.count(True) == 1


# --- AWS Signature Version 4 -----------------------------------------------------------------------------------------

def test_sign_v4_matches_the_aws_documented_example() -> None:
    """The worked example of AWS's Signature Version 4 documentation (IAM ListUsers, 2015-08-30T12:36:00Z)."""
    headers = sign_v4("GET", "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
                      {"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
                      hashlib.sha256(b"").hexdigest(), access_key="AKIDEXAMPLE",
                      secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", region="us-east-1", service="iam",
                      now=datetime(2015, 8, 30, 12, 36, tzinfo=timezone.utc))
    assert headers["x-amz-date"] == "20150830T123600Z"
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, "
        "SignedHeaders=content-type;host;x-amz-date, "
        "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7")


# --- the S3 store ----------------------------------------------------------------------------------------------------

@pytest.fixture
def s3():
    fake = FakeS3()
    config = S3Settings(bucket=fake.bucket, access_key_id=fake.access_key, secret_access_key=fake.secret_key,
                        endpoint_url=f"https://{fake.host}", prefix="pilot")
    store = S3ObjectStore(config, transport=fake.transport())
    yield fake, store
    store.close()


def test_the_s3_store_round_trips_signed_requests_under_a_prefix(s3) -> None:
    fake, store = s3
    uri = store.put("t1/lesson/v1.json", b'{"a": 1}')
    assert uri == "object:t1/lesson/v1.json"  # the bucket and the prefix are never in the recorded URI
    assert fake.objects == {"pilot/t1/lesson/v1.json": b'{"a": 1}'}
    assert store.get(uri) == b'{"a": 1}' and store.exists(uri)
    assert store.get_key("t1/lesson/v1.json") == b'{"a": 1}' and store.get_key("nope") is None
    store.delete(uri)
    assert not store.exists(uri) and fake.objects == {}
    with pytest.raises(FileNotFoundError):
        store.get(uri)
    store.delete(uri)  # deleting a missing object is not an error
    assert all(r.url.scheme == "https" for r in fake.requests)


def test_the_s3_store_encodes_keys_in_the_signed_path(s3) -> None:
    fake, store = s3
    uri = store.put("t1/a name+with&chars/v1.json", b"x")  # the fake recomputes the signature over the path
    assert store.get(uri) == b"x" and "pilot/t1/a name+with&chars/v1.json" in fake.objects


def test_put_if_absent_is_a_conditional_write(s3) -> None:
    fake, store = s3
    assert store.put_if_absent("k/v1.bin", b"first") == ("object:k/v1.bin", True)
    assert store.put_if_absent("k/v1.bin", b"second") == ("object:k/v1.bin", False)
    assert fake.objects["pilot/k/v1.bin"] == b"first"  # never replaced
    assert fake.requests[-1].headers["if-none-match"] == "*"


def test_put_file_if_absent_streams_with_a_length_and_an_unsigned_payload(s3, tmp_path) -> None:
    fake, store = s3
    source = tmp_path / "video.mp4"
    source.write_bytes(b"v" * (3 << 20))
    assert store.put_file_if_absent("objects/sha256/aa/aa.mp4", source) == ("object:objects/sha256/aa/aa.mp4", True)
    put = fake.requests[-1]
    assert put.headers["content-length"] == str(3 << 20) and put.headers["x-amz-content-sha256"] == "UNSIGNED-PAYLOAD"
    assert store.put_file_if_absent("objects/sha256/aa/aa.mp4", source)[1] is False
    with store.open("object:objects/sha256/aa/aa.mp4") as reader:  # read back in chunks, as the API streams it
        chunks = []
        while chunk := reader.read(1 << 20):
            chunks.append(chunk)
    assert b"".join(chunks) == source.read_bytes()


def test_s3_errors_are_typed_and_carry_no_secret(s3) -> None:
    fake, store = s3
    fake.fail_next = 500
    with pytest.raises(ObjectStoreError) as failed:
        store.put("k", b"x")
    assert str(failed.value) == "object store PUT answered 500 (InternalError)"
    wrong = S3ObjectStore(S3Settings(bucket=fake.bucket, access_key_id=fake.access_key,
                                     secret_access_key="a-wrong-secret-key-0000000000",
                                     endpoint_url=f"https://{fake.host}"), transport=fake.transport())
    with pytest.raises(ObjectStoreError) as refused:
        wrong.put("k", b"x")
    assert "SignatureDoesNotMatch" in str(refused.value) and "a-wrong-secret" not in str(refused.value)
    with pytest.raises(ValueError):
        store.get("file:///etc/passwd")  # only object: URIs exist for this store


def test_the_s3_check_writes_reads_and_deletes_a_probe(s3) -> None:
    fake, store = s3
    store.check()
    assert fake.objects == {} and [r.method for r in fake.requests] == ["PUT", "GET", "DELETE"]


def test_s3_settings_are_validated() -> None:
    base = {"bucket": "b", "access_key_id": "k", "secret_access_key": "s"}
    with pytest.raises(ConfigError, match="TA_S3_ALLOW_HTTP"):
        S3Settings(**base, endpoint_url="http://minio:9000")
    assert S3Settings(**base, endpoint_url="http://minio:9000", allow_http=True).uses_path_style
    with pytest.raises(ConfigError, match="credentials"):
        S3Settings(**base, endpoint_url="https://user:pw@s3.example")
    with pytest.raises(ConfigError, match="TA_S3_SECRET_ACCESS_KEY"):
        S3Settings(bucket="b", access_key_id="k", secret_access_key="")
    aws = S3Settings(**base, region="eu-west-1")
    assert aws.base == "https://s3.eu-west-1.amazonaws.com" and not aws.uses_path_style


def test_virtual_hosted_urls_for_aws() -> None:
    store = S3ObjectStore(S3Settings(bucket="lessons", access_key_id="k", secret_access_key="s", region="eu-west-1"))
    try:
        assert store._url("t/v1.json") == "https://lessons.s3.eu-west-1.amazonaws.com/t/v1.json"
    finally:
        store.close()


def test_artifacts_are_stored_and_deduplicated_in_s3(s3, tmp_path) -> None:
    fake, store = s3
    artifacts = ArtifactService(SqlArtifactRepository(create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")), store)
    s, _ = scope("t1")
    a = artifacts.store(task_id="t1", name="lesson", type=ArtifactType.LESSON, media_type="application/json",
                        content=b"{}", provider="test", parent_ids=[], metadata={}, scope=s)
    assert a.uri == "object:t1/lesson/v1.json" and artifacts.read(a.artifact_id) == b"{}"
    first = artifacts.put_object(b"png-bytes", "image/png")
    again = artifacts.put_object(b"png-bytes", "image/png")
    assert not first.reused and again.reused and first.uri == again.uri
    assert artifacts.object_checksum(first.uri) == hashlib.sha256(b"png-bytes").hexdigest()
