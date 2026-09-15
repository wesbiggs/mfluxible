"""Pure request-validation logic -- none of this touches self.model, so it doesn't
need a loaded engine (see MfluxEngine.check_request in server/engine.py)."""

import base64
import dataclasses
import io

import pytest
from PIL import Image

from engine import MfluxEngine
from models import CFG_GUIDANCE_FLOOR
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


# --- Guards that only bite on models added alongside the Tier 1 table -------------
#
# The toy spec is linear-by-default and guidance-free, so these build variants of it
# with dataclasses.replace rather than pulling in a real model: check_request reads
# nothing but self.spec, and the point under test is the flag, not the weights.


def _spec(**overrides):
    return dataclasses.replace(TOY_MODEL_SPEC, **overrides)


def _engine(**overrides):
    return MfluxEngine(model=_spec(**overrides), model_cache_dir=None)


@pytest.mark.parametrize("scheduler", ["flow_match_euler_discrete", "er_sde"])
def test_check_request_rejects_fractional_start_on_a_non_linear_model(scheduler):
    # SCHEDULER_PATH replaces the variant's own scheduler rather than adding to it, so
    # off the linear schedule it is a sampler swap (silent, wrong images) or a
    # ValueError raised on the worker thread with the SSE headers already sent. Neither
    # is acceptable mid-stream, hence the up-front 400.
    engine = _engine(default_scheduler=scheduler)
    with pytest.raises(ValueError, match="fractional_start only applies"):
        engine.check_request(
            GenerateRequest(prompt="x", image=_b64_png(), fractional_start=True)
        )


def test_check_request_allows_fractional_start_on_a_linear_model(bare_engine):
    bare_engine.check_request(
        GenerateRequest(prompt="x", image=_b64_png(), fractional_start=True)
    )  # must not raise


def test_check_request_rejects_a_negative_prompt_at_the_cfg_floor():
    # Krea-2's shape: a real negative branch, but mflux's own default guidance sits
    # exactly on the floor where the unconditional prompt stops being encoded. Accepting
    # it would be the same silent drop the supports_* flags exist to prevent.
    engine = _engine(
        supports_guidance=True,
        default_guidance=CFG_GUIDANCE_FLOOR,
        supports_negative_prompt=True,
    )
    with pytest.raises(ValueError, match="only encodes a negative prompt above guidance"):
        engine.check_request(GenerateRequest(prompt="x", negative_prompt="blurry"))


def test_a_negative_prompt_is_accepted_once_the_request_raises_guidance():
    # Same model, guidance lifted past the floor by the request itself: now CFG is on
    # and the negative prompt genuinely does something, so it must not be rejected.
    engine = _engine(
        supports_guidance=True,
        default_guidance=CFG_GUIDANCE_FLOOR,
        supports_negative_prompt=True,
    )
    engine.check_request(
        GenerateRequest(prompt="x", negative_prompt="blurry", guidance=3.5)
    )  # must not raise


def test_a_request_lowering_guidance_to_the_floor_loses_its_negative_prompt():
    # The mirror case: a model whose default clears the floor, undercut by the request.
    # The check has to read the effective guidance, not the spec's default.
    engine = _engine(
        supports_guidance=True,
        default_guidance=4.0,
        supports_negative_prompt=True,
    )
    engine.check_request(GenerateRequest(prompt="x", negative_prompt="blurry"))  # fine
    with pytest.raises(ValueError, match="only encodes a negative prompt above guidance"):
        engine.check_request(
            GenerateRequest(prompt="x", negative_prompt="blurry", guidance=CFG_GUIDANCE_FLOOR)
        )


# --- Masks ------------------------------------------------------------------------


def _b64_mask(size=(8, 8), white=((0, 0), (4, 8))) -> str:
    """A mask with a white rectangle in it -- white being "regenerate here"."""
    mask = Image.new("L", size, 0)
    if white is not None:
        (x0, y0), (x1, y1) = white
        mask.paste(255, (x0, y0, x1, y1))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_check_request_accepts_a_mask_alongside_an_image(bare_engine):
    bare_engine.check_request(
        GenerateRequest(prompt="x", image=_b64_png(), mask=_b64_mask(), mask_feather=4)
    )  # must not raise


def test_check_request_rejects_a_mask_without_an_image(bare_engine):
    with pytest.raises(ValueError, match="mask requires image"):
        bare_engine.check_request(GenerateRequest(prompt="x", mask=_b64_mask()))


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"mask_feather": 8}, "mask_feather requires mask"),
        ({"mask_composite": False}, "mask_composite requires mask"),
    ],
)
def test_check_request_rejects_mask_options_with_no_mask(bare_engine, extra, match):
    # Both only describe how a mask is applied, so with no mask they name a behaviour
    # that can't happen -- the same reason image_strength alone is a 400. Only a
    # non-default value counts: every maskless request in the suite sends the defaults.
    with pytest.raises(ValueError, match=match):
        bare_engine.check_request(GenerateRequest(prompt="x", image=_b64_png(), **extra))


def test_check_request_rejects_a_mask_that_is_a_different_size_to_the_image(bare_engine):
    # The silent-failure this prevents: mflux resizes the input image to width/height
    # with no aspect-ratio handling, so a mismatched mask would be stretched to fit and
    # select a region other than the one that was painted.
    with pytest.raises(ValueError, match="must be the same size"):
        bare_engine.check_request(
            GenerateRequest(prompt="x", image=_b64_png(size=(16, 16)), mask=_b64_mask(size=(8, 8)))
        )


def test_check_request_rejects_an_all_black_mask(bare_engine):
    # Valid, decodable, right size, and asks for nothing to change -- which would burn
    # a whole generation reproducing the input.
    with pytest.raises(ValueError, match="entirely black"):
        bare_engine.check_request(
            GenerateRequest(prompt="x", image=_b64_png(), mask=_b64_mask(white=None))
        )


def test_check_request_rejects_a_mask_that_is_not_an_image(bare_engine):
    not_an_image = base64.b64encode(b"not a png at all").decode("ascii")
    with pytest.raises(ValueError, match="mask could not be decoded"):
        bare_engine.check_request(GenerateRequest(prompt="x", image=_b64_png(), mask=not_an_image))


def test_a_broken_image_is_reported_before_the_mask_is_looked_at(bare_engine):
    # Order matters: _input_mask_problem measures the mask against the image, so it
    # needs one it can open. Checking the mask first would report the wrong field.
    with pytest.raises(ValueError, match="image could not be decoded"):
        bare_engine.check_request(
            GenerateRequest(
                prompt="x",
                image=base64.b64encode(b"not a png").decode("ascii"),
                mask=_b64_mask(),
            )
        )


@pytest.mark.parametrize("scheduler", ["flow_match_euler_discrete", "er_sde"])
def test_check_request_rejects_a_mask_on_a_non_linear_model(scheduler):
    # Same root cause as fractional_start above: holding the kept region in place means
    # handing mflux a scheduler, which *replaces* the one the variant would have picked.
    engine = _engine(default_scheduler=scheduler)
    with pytest.raises(ValueError, match="mask needs the linear schedule"):
        engine.check_request(GenerateRequest(prompt="x", image=_b64_png(), mask=_b64_mask()))


def test_supports_mask_tracks_the_default_scheduler():
    assert _spec(default_scheduler="linear").supports_mask is True
    assert _spec(default_scheduler="flow_match_euler_discrete").supports_mask is False
