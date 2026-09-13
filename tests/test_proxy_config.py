"""Checks that Caddyfile.example still matches the routes it's gating.

The file is a deployment example rather than code, which normally puts it outside a
test's reach. It's here because its failure mode is silent and one-directional: an
endpoint added to server.py is protected automatically (the bearer block catches
everything the public list doesn't), but a path *added to the public list* -- or one
that quietly stops being harmless -- turns into an open generation endpoint that
nothing else would notice. The same reasoning as test_docs.py, one risk level up.
"""

import pathlib
import re

import pytest

CADDYFILE = pathlib.Path(__file__).resolve().parents[1] / "Caddyfile.example"


def _public_paths() -> list[str]:
    """The path list from the `@public path ...` matcher."""
    body = CADDYFILE.read_text()
    line = re.search(r"^\s*@public\s+path\s+(.+)$", body, re.M)
    assert line, f"no `@public path` matcher in {CADDYFILE}"
    return line.group(1).split()


def _matches(pattern: str, path: str) -> bool:
    """Caddy's `path` matcher: exact, unless the pattern ends in `*`, which makes it a
    prefix. Case-insensitive, as Caddy's is."""
    if pattern.endswith("*"):
        return path.lower().startswith(pattern[:-1].lower())
    return path.lower() == pattern.lower()


def _app_routes():
    import server as server_module

    return [
        (route.path, methods)
        for route in server_module.app.routes
        if (methods := getattr(route, "methods", None))
    ]


def test_every_public_path_is_a_route_the_server_actually_serves():
    # A stale or mistyped entry here doesn't fail loudly -- it just 401s a path the
    # harness needs, and the page comes up looking broken for no visible reason.
    served = {path for path, _ in _app_routes()}
    for pattern in _public_paths():
        prefix = pattern[:-1] if pattern.endswith("*") else pattern
        assert any(p == prefix or p.startswith(prefix) for p in served), (
            f"Caddyfile.example serves {pattern} without credentials, but no route in "
            f"server.py matches it"
        )


def test_no_endpoint_that_generates_is_reachable_without_a_token():
    # The invariant the whole file exists for. Every POST here costs GPU time on a
    # machine with one model and one in-flight generation, so leaving one in the public
    # list would hand a stranger the ability to occupy it indefinitely.
    public = _public_paths()
    for path, methods in _app_routes():
        if "POST" not in methods:
            continue
        assert not any(_matches(p, path) for p in public), (
            f"POST {path} generates, but Caddyfile.example's @public list exempts it "
            f"from the bearer check"
        )


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/", True),
        ("/health", True),
        ("/docs", True),
        ("/docs/oauth2-redirect", True),
        ("/openapi.json", True),
        ("/mfluxible/v1/images/generations", False),
        ("/v1/images/generations", False),
        ("/sdapi/v1/txt2img", False),
        # Verified against a live Caddy 2.11.4: it normalizes before matching, so none
        # of these reach the public branch. Pinned here so the helper above can't drift
        # into a laxer prefix check than the real matcher.
        ("/health/../sdapi/v1/txt2img", False),
        ("/healthz", False),
        ("/docs-private", False),
    ],
)
def test_path_matcher_agrees_with_caddy(path, expected):
    assert any(_matches(p, path) for p in _public_paths()) is expected


def test_the_example_still_carries_a_placeholder_token():
    # Catches the obvious accident: editing this file to run it, then committing it.
    assert "REPLACE-WITH-A-LONG-RANDOM-TOKEN" in CADDYFILE.read_text()
