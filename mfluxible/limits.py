"""A ceiling on request body size, as a pure ASGI middleware.

Without one, the image-bearing endpoints read whatever they are sent into memory --
`image`/`mask` are base64 strings in a JSON body, and `/v1/images/edits` does
`await image.read()` -- and only afterwards look at what it was. The VLM endpoint's own
32MB check (vlm.MAX_IMAGE_BYTES) runs after the whole body has been parsed and decoded.

Pure ASGI for the reason auth.py is: a body-reading `BaseHTTPMiddleware` would wrap the
streaming responses this server depends on. The declared Content-Length is checked up
front (the cheap, common case); a body with none, or one that lies, is counted as it is
read. Mounted *inside* auth, so an unauthenticated caller is turned away before this ever
reads a byte of what it sent.
"""

import json
import os

from mfluxible.config import as_int

# 64MB: a 32MB image (the VLM endpoint's own cap) is ~43MB as base64, and a mask rides
# beside an image in one request.
DEFAULT_MAX_BODY_MB = 64


def max_body_bytes_from_env() -> int | None:
    """MFLUXIBLE_MAX_BODY_MB in bytes; 0 (or `none`) turns the limit off."""
    raw = os.environ.get("MFLUXIBLE_MAX_BODY_MB", "").strip()
    if not raw:
        return DEFAULT_MAX_BODY_MB * 1024 * 1024
    if raw.lower() == "none":
        return None
    mb = as_int("MFLUXIBLE_MAX_BODY_MB", raw)
    return mb * 1024 * 1024 if mb > 0 else None


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app = app
        # A callable, like auth's credentials, so tests can swap it per request.
        self.limit = limit

    async def __call__(self, scope, receive, send):
        limit = self.limit()
        if scope["type"] != "http" or limit is None:
            await self.app(scope, receive, send)
            return

        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # malformed; the server's own parsing will reject it
                if declared > limit:
                    await _send_too_large(send, limit)
                    return
                break

        seen = 0
        started = False
        rejected = False

        async def counting_receive():
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    # Answered here rather than by raising: FastAPI turns any exception
                    # out of a body read into its own 400, which would hide this. The
                    # app is told the client went away, and whatever it then tries to
                    # send is dropped by tracking_send.
                    rejected = True
                    if not started:
                        await _send_too_large(send, limit)
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message):
            nonlocal started
            if rejected:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, counting_receive, tracking_send)


async def _send_too_large(send, limit: int) -> None:
    body = json.dumps(
        {
            "error": {
                "message": f"request body is larger than the {limit // (1024 * 1024)}MB limit "
                "(MFLUXIBLE_MAX_BODY_MB).",
                "type": "invalid_request_error",
                "param": None,
                "code": "request_too_large",
            }
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
