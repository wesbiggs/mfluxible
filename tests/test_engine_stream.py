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
    # No input image, so nothing is skipped and there's no strength bucket to report.
    assert events[0]["start_step"] == 0
    assert events[0]["effective_image_strength"] is None

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


async def test_image_to_image_reports_the_steps_it_skips(toy_engine):
    # mflux starts img2img partway down the schedule: init_time_step = max(1,
    # int(steps * strength)), so 8 steps at 0.5 skips 4 and runs steps 5..8. The
    # `start` event has to say so, or a client showing "step 5/8" first looks like it
    # dropped events, and progress against total_steps alone opens at 50%.
    req = GenerateRequest(
        prompt="a cat", width=16, height=16, steps=8, image=_b64_png(), image_strength=0.5
    )
    events = await _collect(toy_engine, req)

    assert events[0]["start_step"] == 4
    assert events[0]["total_steps"] == 8
    assert events[0]["effective_image_strength"] == 0.5

    thinking = events[1:-1]
    assert [e["step"] for e in thinking] == [5, 6, 7, 8]


async def test_effective_image_strength_reports_the_bucket_not_the_request(toy_engine):
    # image_strength reaches the model only as an int (init_time_step), so it's
    # quantized to 1/steps: at 8 steps, 0.5 and 0.55 both floor to 4 and produce the
    # same image for the same seed. Both must report the same effective strength --
    # that's the whole point of the field, and it's the bucket's lower edge (4/8),
    # not whatever the caller happened to send.
    async def start_event(strength):
        req = GenerateRequest(
            prompt="a cat", width=16, height=16, steps=8, image=_b64_png(), image_strength=strength
        )
        return (await _collect(toy_engine, req))[0]

    assert (await start_event(0.55))["effective_image_strength"] == 0.5
    assert (await start_event(0.5))["effective_image_strength"] == 0.5
    # ...and a strength one bucket up is reported as a different one, not rounded back.
    assert (await start_event(0.625))["effective_image_strength"] == 0.625


async def test_image_strength_of_one_runs_no_steps_at_all(toy_engine):
    # int(steps * 1.0) == steps, so the loop range is empty: the output is the input
    # image round-tripped through the VAE, with no denoising. Degenerate, but valid
    # input (check_request allows 0.0-1.0 inclusive), so the stream still has to be
    # well-formed -- a start event, no thinking events, and a real final image.
    req = GenerateRequest(
        prompt="a cat", width=16, height=16, steps=4, image=_b64_png(), image_strength=1.0
    )
    events = await _collect(toy_engine, req)

    assert events[0]["start_step"] == 4
    assert events[0]["effective_image_strength"] == 1.0
    assert [e["type"] for e in events] == ["start", "image"]


async def test_fractional_start_selects_the_scheduler_and_reports_an_exact_strength(toy_engine):
    # 0.25 at 10 steps falls halfway between rungs 2 and 3. The loop still starts at 2
    # and still runs 8 steps -- only the noise level moves -- so what changes in the
    # stream is effective_image_strength: 0.25 rather than the floored 0.2.
    req = GenerateRequest(
        prompt="a cat",
        width=16,
        height=16,
        steps=10,
        image=_b64_png(),
        image_strength=0.25,
        fractional_start=True,
    )
    events = await _collect(toy_engine, req)

    assert toy_engine.model.last_scheduler == "schedulers.FractionalStartLinearScheduler"
    assert events[0]["start_step"] == 2
    assert events[0]["effective_image_strength"] == 0.25
    assert [e["step"] for e in events[1:-1]] == [3, 4, 5, 6, 7, 8, 9, 10]


async def test_fractional_start_is_off_by_default(toy_engine):
    # An ordinary img2img request must not pass a scheduler at all -- the variant picks
    # its own -- and must still report the floored bucket it actually used.
    req = GenerateRequest(
        prompt="a cat", width=16, height=16, steps=10, image=_b64_png(), image_strength=0.25
    )
    events = await _collect(toy_engine, req)

    assert toy_engine.model.last_scheduler is None
    assert events[0]["effective_image_strength"] == 0.2


async def test_fractional_start_without_an_image_is_rejected(toy_engine):
    # Same treatment as image_strength without image: a knob that cannot do anything is
    # a 400, not a silently dropped field.
    with pytest.raises(ValueError, match="fractional_start"):
        await _collect(toy_engine, GenerateRequest(prompt="x", fractional_start=True))


async def test_check_request_failure_propagates_before_any_event(toy_engine):
    # check_request runs synchronously at the top of generate_stream and isn't
    # caught there -- callers that want a clean 400 instead of a raised exception
    # (the /mfluxible/v1/images/generations endpoint) call check_request themselves first.
    # See test_server_api.py::test_generate_rejects_unsupported_guidance.
    with pytest.raises(ValueError, match="guidance"):
        await _collect(toy_engine, GenerateRequest(prompt="x", guidance=1.0))
