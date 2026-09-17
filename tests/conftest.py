"""Shared fixtures for the mfluxible test suite.

Nothing here needs to touch sys.path itself -- pytest.ini's `pythonpath = .` setting
puts the repo root on sys.path before collection, so `import mfluxible.engine` resolves
to the working tree whether or not the package is installed.

The MFLUXIBLE_CONFIG line below has to come before that first import and is not
housekeeping. Importing `mfluxible` applies a config file (see mfluxible/__init__.py),
and discovery looks in the working directory and then ~/.config/mfluxible -- so without
this, a developer's own config file would set MFLUXIBLE_MODEL for the whole suite and
the failures would point anywhere but at the file that caused them. `none` turns
discovery off; test_config.py drives the loader's functions directly instead of relying
on that import-time side effect.
"""

import os

os.environ.setdefault("MFLUXIBLE_CONFIG", "none")

import pytest  # noqa: E402  -- must follow the line above
from fastapi.testclient import TestClient  # noqa: E402

from tests.doubles.toy_model import TOY_MODEL_SPEC  # noqa: E402


@pytest.fixture
async def toy_engine():
    """A loaded MfluxEngine running ToyModel: no weights, no network, no MLX/GPU work
    beyond trivial array ops. Passing a ModelSpec instance directly (rather than a
    model-name string) bypasses models.py's registry entirely -- see toy_model.py's
    module docstring -- so this needs no change to production code."""
    from mfluxible.engine import MfluxEngine

    engine = MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None)
    await engine.load()
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def client(monkeypatch):
    """A TestClient for the real FastAPI app, with its module-level `engine` swapped
    for a toy one before the ASGI lifespan's startup (engine.load()) runs. server.py
    looks up `engine` from its own module globals each time lifespan()/generate() run,
    so swapping the attribute is all it takes -- no dependency-injection wiring needed
    in server.py itself."""
    import mfluxible.server as server_module
    from mfluxible.engine import MfluxEngine

    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    with TestClient(server_module.app) as test_client:
        yield test_client
