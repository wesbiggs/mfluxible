"""Optional HTTP Basic auth, and the startup warning for running without it.

Off unless both MFLUXIBLE_BASIC_AUTH_USERNAME and MFLUXIBLE_BASIC_AUTH_PASSWORD
are set, so an existing localhost setup keeps working untouched.

This is a *pure ASGI* middleware rather than a Starlette BaseHTTPMiddleware on
purpose: BaseHTTPMiddleware wraps the response in an anyio task group and
buffers it through a memory stream, which is exactly the wrong shape for this
server -- nearly every interesting endpoint here returns a StreamingResponse
that must reach the client event by event (the SSE `thinking` events carry
per-step previews, and a client renders them live). A pure ASGI middleware
forwards send/receive untouched, so a stream stays a stream.

The `except binascii.Error` below is the only one in the request path outside
engine.py, and it is compatible with the invariant CLAUDE.md describes rather
than an exception to it: it *discards* the exception and returns None, so a
malformed header becomes the same fixed 401 as a wrong password. Nothing from
the exception reaches a response body, which is the property being protected.
"""

import base64
import binascii
import ipaddress
import json
import os
import secrets
import sys

REALM = "mfluxible"


def credentials_from_env() -> tuple[str, str] | None:
    """The configured username/password, or None if auth is off.

    Both variables are required: setting only one is far more likely to be a
    half-finished deployment than a deliberate blank, and silently serving
    unauthenticated in that case is the failure worth avoiding.
    """
    username = os.environ.get("MFLUXIBLE_BASIC_AUTH_USERNAME", "")
    password = os.environ.get("MFLUXIBLE_BASIC_AUTH_PASSWORD", "")
    if not username or not password:
        return None
    return username, password


def _decoded_credentials(header: str | None) -> tuple[str, str] | None:
    """The username/password in an Authorization header, or None if it doesn't
    carry a well-formed Basic credential. Returned rather than raised -- see the
    module docstring."""
    if not header:
        return None
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        raw = base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    username, sep, password = decoded.partition(":")
    if not sep:
        return None
    return username, password


def is_authorized(header: str | None, expected: tuple[str, str]) -> bool:
    supplied = _decoded_credentials(header)
    # compare_digest on both halves unconditionally: a plain `==`, or bailing out
    # early once the username mismatches, leaks how much of each was right
    # through timing. The dummy compare keeps a missing header on the same path.
    username, password = supplied if supplied is not None else ("", "")
    username_ok = secrets.compare_digest(username, expected[0])
    password_ok = secrets.compare_digest(password, expected[1])
    return bool(supplied) and username_ok and password_ok


class BasicAuthMiddleware:
    """Gates every HTTP request when `credentials()` returns a pair.

    `credentials` is a callable, not a value, so the configuration is read per
    request instead of being frozen at construction -- which is what lets the
    tests swap it the same way they swap server.engine.
    """

    def __init__(self, app, credentials):
        self.app = app
        self.credentials = credentials

    async def __call__(self, scope, receive, send):
        # Lifespan and websocket scopes have no Authorization header to check and
        # no response to reject with; this server has no websockets anyway.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        expected = self.credentials()
        if expected is None:
            await self.app(scope, receive, send)
            return

        header = None
        for key, value in scope.get("headers", []):
            if key == b"authorization":
                header = value.decode("latin-1")
                break

        if is_authorized(header, expected):
            await self.app(scope, receive, send)
            return

        await _send_unauthorized(send)


async def _send_unauthorized(send) -> None:
    # OpenAI's error envelope, because that is the shape the widest range of
    # third-party tooling pointed at this server already knows how to read, and
    # a middleware cannot tell which endpoint was being called to pick another.
    # No detail about what was wrong with the credentials -- a 401 saying "no
    # such user" and one saying "wrong password" are an enumeration oracle.
    body = json.dumps(
        {
            "error": {
                "message": (
                    "authentication required -- this server has "
                    "MFLUXIBLE_BASIC_AUTH_USERNAME/PASSWORD set; send HTTP Basic credentials."
                ),
                "type": "invalid_request_error",
                "param": None,
                "code": "unauthorized",
            }
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                # Prompts a browser for credentials, which is what makes the
                # harness at GET / usable at all once auth is on.
                (b"www-authenticate", f'Basic realm="{REALM}", charset="UTF-8"'.encode("latin-1")),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def resolve_bind_host(argv: list[str] | None = None, environ: dict | None = None) -> str:
    """The host uvicorn will bind to, resolved the way uvicorn itself resolves it.

    uvicorn's CLI is a click command with `auto_envvar_prefix="UVICORN"` and
    `--host` defaulting to "127.0.0.1" (verified against uvicorn 0.52.4), so the
    precedence is an explicit flag, then UVICORN_HOST, then loopback.

    Best-effort by design: the app is handed to uvicorn rather than starting it,
    so there is no binding to interrogate, and a caller embedding this app some
    other way won't be reflected here. It only decides whether to print a
    warning, so being wrong is a missing or spurious warning, not a wrong bind.
    """
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ

    for i, arg in enumerate(argv):
        if arg == "--host" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]

    return environ.get("UVICORN_HOST") or "127.0.0.1"


def is_loopback(host: str) -> bool:
    """True if `host` can only be reached from this machine."""
    host = (host or "").strip().strip("[]")
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname this can't parse -- resolving it would be a DNS lookup at
        # startup to decide the wording of a warning. Treat as non-loopback: the
        # useful direction to be wrong in is warning about a safe bind rather
        # than staying quiet about an exposed one.
        return False


def startup_warning(bind_host: str, auth_configured: bool) -> str | None:
    """The warning to log, or None if this configuration doesn't warrant one."""
    if auth_configured or is_loopback(bind_host):
        return None
    return (
        f"listening on {bind_host} with no authentication -- every endpoint, including "
        f"the browser harness at GET /, is reachable by anything that can route to this "
        f"host. Set MFLUXIBLE_BASIC_AUTH_USERNAME and MFLUXIBLE_BASIC_AUTH_PASSWORD, or "
        f"bind to 127.0.0.1 and reach it over a tunnel."
    )
