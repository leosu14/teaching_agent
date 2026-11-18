"""The HTTP side of the progress streams (app/services/progress_stream.py): a `text/event-stream` response that
always closes its stream, and so frees its slot, however the response ends (finished, client gone, cancelled)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import anyio

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from app.api.auth import Principal
from app.services.progress_stream import ProgressStream

EVENT_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",  # nginx and compatible ingresses: do not buffer this response
}


def principal_key(p: Principal) -> str:
    """Whose streams count together against the per-principal cap."""
    return f"{p.tenant_id or ''}\x1f{p.user_id}"


class EventStreamResponse(StreamingResponse):
    def __init__(self, stream: ProgressStream) -> None:
        self._stream = stream
        super().__init__(self._encoded(stream), media_type="text/event-stream", headers=EVENT_STREAM_HEADERS)

    @staticmethod
    async def _encoded(stream: ProgressStream) -> AsyncIterator[bytes]:
        async for chunk in stream:
            yield chunk.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stream until the stream ends or the client disconnects, whichever comes first. The disconnect is watched
        whatever the server's ASGI version: an idle stream writes only a heartbeat now and then, and a write to a
        closed connection is not always an error, so without the watch a stream could outlive its client."""
        try:
            async with anyio.create_task_group() as group:

                async def stream() -> None:
                    await self.stream_response(send)
                    group.cancel_scope.cancel()

                group.start_soon(stream)
                await self.listen_for_disconnect(receive)
                group.cancel_scope.cancel()
        except OSError:
            pass  # the client went away mid-write
        finally:
            await self._stream.aclose()
