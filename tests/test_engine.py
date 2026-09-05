"""Pure request-validation logic -- none of this touches self.model, so it doesn't
need a loaded engine (see MfluxEngine.check_request in server/engine.py)."""

import base64
import io

import pytest
from PIL import Image

from engine import MfluxEngine
from schemas import GenerateRequest
from tests.doubles.toy_model import TOY_MODEL_SPEC


@pytest.fixture
def bare_engine():
    return MfluxEngine(model=TOY_MODEL_SPEC, model_cache_dir=None)


def _b64_png(size=(8, 8), color=(255, 0, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_check_request_rejects_guidance_when_unsupported(bare_engine):
    with pytest.raises(ValueError, match="guidance"):
        bare_engine.check_request(GenerateRequest(prompt="x", guidance=3.5))


def test_check_request_rejects_negative_prompt_when_unsupported(bare_engine):
    with pytest.raises(ValueError, match="negative"):
        bare_engine.check_request(GenerateRequest(prompt="x", negative_prompt="blurry"))


def test_check_request_accepts_a_plain_request(bare_engine):
    bare_engine.check_request(GenerateRequest(prompt="x"))  # must not raise


def test_check_request_rejects_image_strength_without_image(bare_engine):
    with pytest.raises(ValueError, match="image_strength requires image"):
        bare_engine.check_request(GenerateRequest(prompt="x", image_strength=0.5))


def test_check_request_rejects_out_of_range_image_strength(bare_engine):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        bare_engine.check_request(GenerateRequest(prompt="x", image=_b64_png(), image_strength=1.5))


def test_check_request_rejects_invalid_base64(bare_engine):
    with pytest.raises(ValueError, match="not valid base64"):
        bare_engine.check_request(GenerateRequest(prompt="x", image="not!base64!!"))


def test_check_request_rejects_base64_that_is_not_an_image(bare_engine):
    not_an_image = base64.b64encode(b"just some bytes, not a png").decode("ascii")
    with pytest.raises(ValueError, match="could not be decoded as an image"):
        bare_engine.check_request(GenerateRequest(prompt="x", image=not_an_image))


def test_check_request_accepts_a_valid_image_with_strength(bare_engine):
    bare_engine.check_request(GenerateRequest(prompt="x", image=_b64_png(), image_strength=0.6))  # must not raise


def test_check_request_accepts_image_without_strength(bare_engine):
    # image_strength is optional -- engine.py falls back to mflux's own CLI default
    # (DEFAULT_IMAGE_STRENGTH) when it's omitted, see generate_stream.
    bare_engine.check_request(GenerateRequest(prompt="x", image=_b64_png()))  # must not raise
