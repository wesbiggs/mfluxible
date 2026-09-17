"""Checks that the sample payloads in docs/ still describe what the code emits.

Documentation drift normally isn't a test's business, but a response sample is a
special case: it's the thing readers copy, and nothing else notices when it goes
stale. `available` on `/health` is the concrete instance -- it went from listing four
models to listing fifteen, and the docs kept showing the old four (then, briefly, a
truncated list with a "..." element that no client would ever receive).
"""

import json
import pathlib
import re

from mfluxible.models import MODELS

API_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "api.md"


def _health_sample() -> dict:
    """The first ```json block under `## `GET /health``, parsed."""
    body = API_DOC.read_text()
    section = re.search(r"^## `GET /health`$(.*?)^## ", body, re.S | re.M)
    assert section, f"no `GET /health` section in {API_DOC}"
    block = re.search(r"```json\n(.*?)```", section.group(1), re.S)
    assert block, "the `GET /health` section has no ```json sample"
    # A parse failure here is the point, not an accident: the sample is meant to be a
    # real response, so anything elided into it (a trailing "...", a comment) fails.
    return json.loads(block.group(1))


def test_documented_health_sample_lists_every_model_in_table_order():
    # Order matters as well as membership: `available` is built by iterating MODELS, and
    # server.md's model table is written in that same order, so a reader comparing the
    # two should see them line up.
    assert _health_sample()["available"] == [spec.key for spec in MODELS]


def test_the_documented_health_sample_reports_the_current_version():
    """The sample carries a literal version string, so it goes stale at every release.

    Cheap to pin and easy to miss otherwise: nothing else reads that field, so a wrong
    one would sit in the docs indefinitely, quietly telling readers they are looking at
    a response from an older build than the one they installed.
    """
    from mfluxible import __version__

    assert _health_sample()["version"] == __version__
