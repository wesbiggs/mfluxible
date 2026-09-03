"""Pure request-validation logic -- none of this touches self.model, so it doesn't
need a loaded engine (see MfluxEngine.check_request in server/engine.py)."""

import pytest

from engine import MfluxEngine
from schemas import GenerateRequest
from tests.doubles.toy_model import TOY_MODEL_SPEC


@pytest.fixture
def bare_engine():
    return MfluxEngine(model=TOY_MODEL_SPEC, model_cache_dir=None)


def test_check_request_rejects_guidance_when_unsupported(bare_engine):
    with pytest.raises(ValueError, match="guidance"):
        bare_engine.check_request(GenerateRequest(prompt="x", guidance=3.5))


def test_check_request_rejects_negative_prompt_when_unsupported(bare_engine):
    with pytest.raises(ValueError, match="negative"):
        bare_engine.check_request(GenerateRequest(prompt="x", negative_prompt="blurry"))


def test_check_request_accepts_a_plain_request(bare_engine):
    bare_engine.check_request(GenerateRequest(prompt="x"))  # must not raise
