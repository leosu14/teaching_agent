"""An in-process S3 fake for the S3 object store's tests: an httpx transport that keeps objects in a dict.

It does what the store relies on and checks what S3 would refuse: every request must carry a valid AWS Signature
Version 4 for the configured key pair (recomputed here from the headers it names as signed, with the payload hash
the request declares, which must match the body unless it is UNSIGNED-PAYLOAD), a PUT needs a Content-Length, and
`If-None-Match: *` makes a PUT of an existing key fail with 412. The requests are recorded for assertions.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import httpx

from app.providers.storage.s3 import sign_v4

AUTH = re.compile(r"^AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<day>\d{8})/(?P<region>[^/]+)/s3/aws4_request, "
                  r"SignedHeaders=(?P<signed>[a-z0-9;-]+), Signature=(?P<sig>[0-9a-f]{64})$")


class FakeS3:
    def __init__(self, *, access_key: str = "AKIDTEST", secret_key: str = "fake-secret-access-key-0123456789",
                 bucket: str = "bucket", region: str = "us-east-1", host: str = "s3.test") -> None:
        self.access_key, self.secret_key, self.bucket, self.region, self.host = (access_key, secret_key, bucket,
                                                                                 region, host)
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        self.fail_next: int | None = None  # answer the next request with this status

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _key(self, request: httpx.Request) -> str:
        path = request.url.raw_path.decode().split("?", 1)[0]
        prefix = f"/{self.bucket}/"
        assert request.url.host == self.host and path.startswith(prefix), request.url
        from urllib.parse import unquote

        return unquote(path[len(prefix):])

    def _verify(self, request: httpx.Request, body: bytes) -> httpx.Response | None:
        match = AUTH.match(request.headers.get("authorization", ""))
        if not match or match["key"] != self.access_key or match["region"] != self.region:
            return httpx.Response(403, text="<Error><Code>InvalidAccessKeyId</Code></Error>")
        declared = request.headers.get("x-amz-content-sha256", "")
        if declared != "UNSIGNED-PAYLOAD" and declared != hashlib.sha256(body).hexdigest():
            return httpx.Response(400, text="<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
        names = match["signed"].split(";")
        assert "host" in names and "x-amz-date" in names and "x-amz-content-sha256" in names
        headers = {n: request.headers[n] for n in names if n not in {"host", "x-amz-date"}}
        when = datetime.strptime(request.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expected = sign_v4(request.method, str(request.url), headers, declared, access_key=self.access_key,
                           secret_key=self.secret_key, region=self.region, now=when)["Authorization"]
        if expected != request.headers["authorization"]:
            return httpx.Response(403, text="<Error><Code>SignatureDoesNotMatch</Code></Error>")
        return None

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        self.requests.append(request)
        if self.fail_next is not None:
            status, self.fail_next = self.fail_next, None
            return httpx.Response(status, text="<Error><Code>InternalError</Code><Message>secret-ish</Message></Error>")
        refused = self._verify(request, body)
        if refused is not None:
            return refused
        key = self._key(request)
        if request.method == "PUT":
            if "content-length" not in request.headers:
                return httpx.Response(411, text="<Error><Code>MissingContentLength</Code></Error>")
            if request.headers.get("if-none-match") == "*" and key in self.objects:
                return httpx.Response(412, text="<Error><Code>PreconditionFailed</Code></Error>")
            self.objects[key] = body
            return httpx.Response(200)
        if request.method in {"GET", "HEAD"}:
            if key not in self.objects:
                return httpx.Response(404, text="<Error><Code>NoSuchKey</Code></Error>")
            return httpx.Response(200, content=b"" if request.method == "HEAD" else self.objects[key])
        if request.method == "DELETE":
            self.objects.pop(key, None)
            return httpx.Response(204)
        return httpx.Response(405)
