"""End-to-end coverage of MfluxEngine.generate_stream() against ToyModel: the SSE
event sequence, step timing/callback wiring, preview decoding, and final-image
encoding, all without a real model or GPU work.
"""

import base64
import io
import os

import pytest
from PIL import Image

from schemas import GenerateRequest


async def _collect(engine, req):
    return [event async for event in engine.generate_stream(req)]


async def test_event_sequence_and_final_image(toy_engine):
    req = GenerateRequest(prompt="a cat", width=32, height=32, steps=3, seed=123)
    events = await _collect(toy_engine, req)

    assert events[0]["type"] == "start"
    assert events[0]["seed"] == 123
    assert events[0]["total_steps"] == 3

    thinking = events[1:-1]
    assert [e["type"] for e in thinking] == ["thinking"] * 3
    assert [e["step"] for e in thinking] == [1, 2, 3]
    # elapsed_ms is measured from the same start_ts each step, so it can't decrease
    # (step_ms values are each independently truncated to an int and so don't
    # necessarily sum to it exactly -- see _StreamCallback.call_in_loop).
    elapsed = [e["elapsed_ms"] for e in thinking]
    assert elapsed == sorted(elapsed)
    assert all(e["step_ms"] >= 0 for e in thinking)

    final = events[-1]
    assert final["type"] == "image"
    assert final["seed"] == 123
    assert final["mime_type"] == "image/png"

    image = Image.open(io.BytesIO(base64.b64decode(final["data"])))
    assert image.size == (32, 32)
    assert len(set(image.get_flattened_data())) == 1  # a single solid color


async def test_random_seed_is_used_when_omitted(toy_engine):
    req = GenerateRequest(prompt="a cat", width=32, height=32, steps=1)
    events = await _collect(toy_engine, req)
    assert isinstance(events[0]["seed"], int)


async def test_same_seed_reproduces_the_same_pixels(toy_engine):
    # Not a byte-for-byte comparison: the final PNG embeds a generation timestamp via
    # mflux's real metadata pipeline (see engine.py's _encode_final_png_with_metadata),
    # so identical seeds still produce different bytes -- only the pixels are
    # deterministic.
    req = GenerateRequest(prompt="a cat", width=16, height=16, steps=1, seed=99)
    first = await _collect(toy_engine, req)
    second = await _collect(toy_engine, req)
    first_image = Image.open(io.BytesIO(base64.b64decode(first[-1]["data"])))
    second_image = Image.open(io.BytesIO(base64.b64decode(second[-1]["data"])))
    assert list(first_image.get_flattened_data())[0] == list(second_image.get_flattened_data())[0]


async def test_preview_included_only_on_requested_steps(toy_engine):
    req = GenerateRequest(prompt="a cat", width=32, height=32, steps=4, preview_every=2)
    events = await _collect(toy_engine, req)
    thinking = [e for e in events if e["type"] == "thinking"]
    with_preview = [e["step"] for e in thinking if "preview" in e]
    assert with_preview == [2, 4]

    preview_b64 = next(e["preview"] for e in thinking if e["step"] == 2)
    preview = Image.open(io.BytesIO(base64.b64decode(preview_b64)))
    assert preview.size == (32, 32)


async def test_no_preview_key_when_preview_every_is_zero(toy_engine):
    req = GenerateRequest(prompt="a cat", width=16, height=16, steps=2, preview_every=0)
    events = await _collect(toy_engine, req)
    thinking = [e for e in events if e["type"] == "thinking"]
    assert all("preview" not in e for e in thinking)


def _b64_png(size=(8, 8), color=(0, 255, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def test_image_to_image_passes_a_real_file_and_gets_cleaned_up(toy_engine):
    req = GenerateRequest(
        prompt="a cat", width=32, height=32, steps=2, image=_b64_png(), image_strength=0.7
    )
    events = await _collect(toy_engine, req)
    assert events[-1]["type"] == "image"

    # ToyModel recorded what engine.py actually handed to generate_image(): a real
    # file that existed while generation was running (see toy_model.py), and the
    # image_strength from the request passed straight through.
    assert toy_engine.model.last_image_path is not None
    assert toy_engine.model.last_image_path_existed is True
    assert toy_engine.model.last_image_strength == 0.7

    # engine.py's finally block must have removed the temp file once generation
    # finished -- it must not leak one input-image temp file per request.
    assert not os.path.exists(toy_engine.model.last_image_path)


async def test_image_to_image_defaults_strength_when_omitted(toy_engine):
    req = GenerateRequest(prompt="a cat", width=16, height=16, steps=1, image=_b64_png())
    await _collect(toy_engine, req)
    assert toy_engine.model.last_image_strength == 0.4  # engine.DEFAULT_IMAGE_STRENGTH


async def test_check_request_failure_propagates_before_any_event(toy_engine):
    # check_request runs synchronously at the top of generate_stream and isn't
    # caught there -- callers that want a clean 400 instead of a raised exception
    # (the /mfluxible/v1/images/generations endpoint) call check_request themselves first.
    # See test_server_api.py::test_generate_rejects_unsupported_guidance.
    with pytest.raises(ValueError, match="guidance"):
        await _collect(toy_engine, GenerateRequest(prompt="x", guidance=1.0))
