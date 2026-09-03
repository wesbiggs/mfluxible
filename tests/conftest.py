"""Shared fixtures for the mfluxible test suite.

Nothing here needs to touch sys.path itself -- pytest.ini's `pythonpath = server`
setting puts server/ on sys.path before collection, matching how the app is actually
run (`uvicorn server:app --app-dir server`).
"""

import pytest
from fastapi.testclient import TestClient

from tests.doubles.toy_model import TOY_MODEL_SPEC


@pytest.fixture
async def toy_engine():
    """A loaded MfluxEngine running ToyModel: no weights, no network, no MLX/GPU work
    beyond trivial array ops. Passing a ModelSpec instance directly (rather than a
    model-name string) bypasses models.py's registry entirely -- see toy_model.py's
    module docstring -- so this needs no change to production code."""
    from engine import MfluxEngine

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
    import server as server_module
    from engine import MfluxEngine

    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    with TestClient(server_module.app) as test_client:
        yield test_client
