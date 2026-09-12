"""Coverage for the optional HTTP Basic auth in server/auth.py.

Auth is configured by a module-level `BASIC_AUTH` in server.py, read back through
a callable per request, so these swap that attribute the same way conftest's
`client` fixture swaps `engine` -- no environment juggling, and no reimport.
"""

import base64

import pytest

from auth import is_authorized, is_loopback, resolve_bind_host, startup_warning

CREDS = ("tavern", "hunter2")


def _header(username: str, password: str) -> dict:
    raw = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


@pytest.fixture
def auth_client(client, monkeypatch):
    """The app fixture, with auth switched on."""
    import server as server_module

    monkeypatch.setattr(server_module, "BASIC_AUTH", CREDS)
    return client


# ──────────────────────────────────────────────
# Off by default
# ──────────────────────────────────────────────

def test_endpoints_are_open_when_no_credentials_are_configured(client):
    # The default, and the whole existing localhost story: setting neither env
    # var must leave every endpoint exactly as it was.
    for path in ("/health", "/v1/models", "/"):
        assert client.get(path).status_code == 200


def test_credentials_from_env_needs_both_halves(monkeypatch):
    import auth

    monkeypatch.setenv("MFLUXIBLE_BASIC_AUTH_USERNAME", "tavern")
    monkeypatch.delenv("MFLUXIBLE_BASIC_AUTH_PASSWORD", raising=False)
    assert auth.credentials_from_env() is None, "half-configured must not mean half-open"

    monkeypatch.setenv("MFLUXIBLE_BASIC_AUTH_PASSWORD", "hunter2")
    assert auth.credentials_from_env() == CREDS


# ──────────────────────────────────────────────
# On
# ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/"),               # the harness UI -- the reason this covers everything
        ("get", "/health"),
        ("get", "/v1/models"),
        ("post", "/mfluxible/v1/images/generations"),
        ("post", "/v1/images/generations"),
        ("post", "/v1/chat/completions"),
        ("post", "/sdapi/v1/txt2img"),
        # FastAPI's own routes are routes like any other, so the middleware
        # covers them -- docs/server.md says so, so it gets asserted.
        ("get", "/docs"),
        ("get", "/openapi.json"),
    ],
)
def test_every_endpoint_401s_without_credentials(auth_client, method, path):
    kwargs = {"json": {}} if method == "post" else {}
    resp = getattr(auth_client, method)(path, **kwargs)
    assert resp.status_code == 401
    # Without this header a browser never offers a login box, so GET / would be
    # unreachable rather than merely protected.
    assert resp.headers["www-authenticate"].startswith('Basic realm="mfluxible"')


def test_correct_credentials_pass_through(auth_client):
    resp = auth_client.get("/health", headers=_header(*CREDS))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_a_generation_still_streams_under_auth(auth_client):
    # A pure ASGI middleware was chosen so SSE stays event-by-event; a
    # BaseHTTPMiddleware would buffer it. This is what that choice protects.
    with auth_client.stream(
        "POST",
        "/mfluxible/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2, "stream": True},
        headers=_header(*CREDS),
    ) as resp:
        assert resp.status_code == 200
        kinds = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert len(kinds) > 1, "more than one SSE event must arrive"


@pytest.mark.parametrize(
    "header",
    [
        None,
        {"Authorization": "Basic " + base64.b64encode(b"tavern:wrong").decode()},
        {"Authorization": "Basic " + base64.b64encode(b"wrong:hunter2").decode()},
        {"Authorization": "Bearer " + base64.b64encode(b"tavern:hunter2").decode()},
        {"Authorization": "Basic !!!not-base64!!!"},
        {"Authorization": "Basic " + base64.b64encode(b"no-colon-here").decode()},
        {"Authorization": "Basic "},
        {"Authorization": ""},
    ],
)
def test_bad_credentials_are_rejected_the_same_way(auth_client, header):
    # Malformed base64 in particular: b64decode raises, and the point of routing
    # that through a return value is that the 401 is identical to a wrong
    # password -- no exception text, no hint about which part failed.
    resp = auth_client.get("/health", headers=header or {})
    assert resp.status_code == 401
    assert "not-base64" not in resp.text
    assert resp.json()["error"]["code"] == "unauthorized"


def test_the_401_body_does_not_say_which_half_was_wrong(auth_client):
    wrong_user = auth_client.get("/health", headers=_header("nobody", "hunter2")).text
    wrong_pass = auth_client.get("/health", headers=_header("tavern", "nope")).text
    assert wrong_user == wrong_pass, "differing bodies would enumerate valid usernames"


def test_is_authorized_accepts_a_password_containing_a_colon(auth_client):
    # RFC 7617 splits on the *first* colon; passwords may contain more.
    assert is_authorized(
        "Basic " + base64.b64encode(b"tavern:hunter2:extra").decode(),
        ("tavern", "hunter2:extra"),
    )


# ──────────────────────────────────────────────
# Interaction with CORS
# ──────────────────────────────────────────────

def test_cors_preflight_is_answered_without_credentials(auth_client):
    # The middleware ordering in server.py depends on CORS wrapping auth rather
    # than the reverse. A preflight carries no credentials by spec, so a 401 here
    # would break every cross-origin browser client with no way to fix it from
    # the page. This fails if someone swaps the two add_middleware calls.
    resp = auth_client.options(
        "/v1/images/generations",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_the_sillytavern_ping_is_not_a_preflight_and_is_gated(auth_client):
    # A bare OPTIONS carries no Access-Control-Request-Method, so CORS passes it
    # through to auth. That means enabling auth does block SillyTavern's sdcpp
    # source, which sends no credentials on any of its three calls -- documented
    # in docs/clients.md rather than worked around.
    assert auth_client.options("/v1/images/generations").status_code == 401
    assert auth_client.options("/v1/images/generations", headers=_header(*CREDS)).status_code == 204


# ──────────────────────────────────────────────
# The startup warning
# ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "host,loopback",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("127.0.0.53", True),
        ("0.0.0.0", False),
        ("192.168.4.86", False),
        ("::", False),
        ("mfluxible.local", False),  # unresolvable here; warn rather than stay quiet
        ("", False),
    ],
)
def test_is_loopback(host, loopback):
    assert is_loopback(host) is loopback


@pytest.mark.parametrize(
    "argv,environ,expected",
    [
        (["uvicorn", "server:app", "--host", "0.0.0.0"], {}, "0.0.0.0"),
        (["uvicorn", "server:app", "--host=0.0.0.0"], {}, "0.0.0.0"),
        (["uvicorn", "server:app"], {"UVICORN_HOST": "0.0.0.0"}, "0.0.0.0"),
        # An explicit flag beats the env var, the way click resolves it.
        (["uvicorn", "server:app", "--host", "127.0.0.1"], {"UVICORN_HOST": "0.0.0.0"}, "127.0.0.1"),
        # uvicorn's own default when neither is given.
        (["uvicorn", "server:app"], {}, "127.0.0.1"),
        # A trailing --host with nothing after it must not IndexError.
        (["uvicorn", "server:app", "--host"], {}, "127.0.0.1"),
    ],
)
def test_resolve_bind_host(argv, environ, expected):
    assert resolve_bind_host(argv, environ) == expected


def test_startup_warning_only_fires_when_exposed_and_unauthenticated():
    assert startup_warning("127.0.0.1", auth_configured=False) is None
    assert startup_warning("0.0.0.0", auth_configured=True) is None
    assert startup_warning("127.0.0.1", auth_configured=True) is None

    warning = startup_warning("0.0.0.0", auth_configured=False)
    assert warning is not None
    assert "0.0.0.0" in warning
    # Naming GET / matters: it is the surface a passerby can use with no
    # knowledge of the API at all.
    assert "GET /" in warning
    assert "MFLUXIBLE_BASIC_AUTH_USERNAME" in warning


def _boot(monkeypatch, caplog, argv, basic_auth):
    """Run the real app's lifespan under a given bind/auth configuration and
    return what it logged. Uses the toy engine, so no weights are touched."""
    import logging

    from fastapi.testclient import TestClient

    import server as server_module
    from engine import MfluxEngine
    from tests.doubles.toy_model import TOY_MODEL_SPEC

    # auth.resolve_bind_host reads sys.argv at call time, so patching it here reaches it.
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(server_module, "BASIC_AUTH", basic_auth)
    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    with caplog.at_level(logging.WARNING, logger="mfluxible.server"):
        with TestClient(server_module.app):
            pass
    return caplog.text


def test_lifespan_warns_when_exposed_without_auth(monkeypatch, caplog):
    # The wiring, not the predicate: startup_warning is unit-tested above, but
    # nothing else proves lifespan actually calls it and logs the result.
    text = _boot(monkeypatch, caplog, ["uvicorn", "server:app", "--host", "0.0.0.0"], None)
    assert "0.0.0.0" in text
    assert "no authentication" in text


def test_lifespan_is_quiet_when_bound_to_loopback(monkeypatch, caplog):
    text = _boot(monkeypatch, caplog, ["uvicorn", "server:app", "--host", "127.0.0.1"], None)
    assert "no authentication" not in text


def test_lifespan_is_quiet_when_exposed_with_auth(monkeypatch, caplog):
    text = _boot(monkeypatch, caplog, ["uvicorn", "server:app", "--host", "0.0.0.0"], CREDS)
    assert "no authentication" not in text
