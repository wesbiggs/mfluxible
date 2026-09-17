"""End-to-end coverage of MfluxEngine.generate_stream() against ToyModel: the SSE
event sequence, step timing/callback wiring, preview decoding, and final-image
encoding, all without a real model or GPU work.
"""

import base64
import io
import os

import pytest
from PIL import Image

from mfluxible.schemas import GenerateRequest


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

    assert toy_engine.model.last_scheduler == "mfluxible.schedulers.FractionalStartLinearScheduler"
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


# The message on a failed generation is the one thing here a *client* sees that the
# engine doesn't otherwise curate, so it gets its own coverage: mflux, MLX and HF-hub
# exceptions quote absolute cache paths, and an `except Exception as exc: str(exc)`
# would put the host's home directory in an HTTP response body. The stand-in below is
# shaped like the real thing (CodeQL flagged this path as py/stack-trace-exposure).
_LEAKY_MESSAGE = "No such file: '/Users/someone/.cache/huggingface/hub/models--x/transformer.safetensors'"


async def test_generation_failure_reports_without_quoting_the_exception(toy_engine, caplog):
    def boom(*_args, **_kwargs):
        raise FileNotFoundError(_LEAKY_MESSAGE)

    toy_engine.model.generate_image = boom
    with caplog.at_level("ERROR", logger="mfluxible.engine"):
        events = await _collect(toy_engine, GenerateRequest(prompt="x", width=32, height=32, steps=2))

    assert events[-1]["type"] == "error"
    assert "/Users/someone" not in events[-1]["message"]
    assert "server log" in events[-1]["message"]
    # ...and the operator, who does need it, still gets the whole thing.
    assert _LEAKY_MESSAGE in caplog.text
    assert "FileNotFoundError" in caplog.text


def test_an_interruption_still_says_which_step_it_stopped_on():
    # The counterexample to the test above: this message is mfluxible's own, describes
    # the request rather than the host, and so is deliberately still reported verbatim.
    # Driven through the callback directly -- mflux only ever reaches its interrupt path
    # from a literal KeyboardInterrupt on the server process (see CLAUDE.md), which is
    # not something a test can stage around a generation.
    from mfluxible.engine import _StreamCallback

    emitted = []
    callback = _StreamCallback(None, 0, emitted.append, fractional_start=False)
    callback.call_interrupt(4, seed=1, prompt="x", latents=None, config=None, time_steps=None)

    assert emitted == [{"type": "error", "message": "generation interrupted at step 5"}]


# --- Masked inpainting --------------------------------------------------------------


def _b64_mask(size=(32, 32)) -> str:
    """White down the left half: regenerate there, keep the right half."""
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, size[0] // 2, size[1]))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _masked_request(**overrides):
    base = dict(
        prompt="a cat",
        width=32,
        height=32,
        steps=3,
        seed=5,
        image=_b64_png(size=(32, 32), color=(10, 200, 30)),
        image_strength=0.0,
        mask=_b64_mask(),
    )
    return GenerateRequest(**{**base, **overrides})


async def test_a_masked_request_runs_the_masked_scheduler_and_clears_its_job(toy_engine):
    from mfluxible import schedulers

    events = await _collect(toy_engine, _masked_request())
    assert events[-1]["type"] == "image"
    # The whole wiring is this string: engine.py picks it from schedulers.scheduler_path
    # and mflux imports it. ToyModel steps through config.scheduler, and _MaskedBlend
    # raises when no job is set -- so a completed generation is also proof the job was
    # there for every step.
    assert toy_engine.model.last_scheduler == "mfluxible.schedulers.MaskedBlendLinearScheduler"
    # Cleared in run()'s finally, not after_loop's: a job surviving the call would be
    # picked up by whatever generates next.
    assert schedulers.active_mask_job() is None


async def test_a_masked_request_defaults_strength_to_zero_rather_than_0_4(toy_engine):
    """The maskless default is wrong here, and wrong in the worst possible shape.

    0.4 starts the masked region 40% along the schedule, so the object that was meant to
    be replaced comes back very nearly intact -- while the stream is well-formed and the
    generation reports success. Measured on a 768x768 Z-Image-Turbo run, identical seed
    and mask: 5.1/255 of change inside the mask against 41.6 at 0.0. Since omitting an
    optional field is the ordinary thing to do, the default has to follow the mask.
    """
    await _collect(toy_engine, _masked_request(image_strength=None))
    assert toy_engine.model.last_image_strength == 0.0  # engine.DEFAULT_MASKED_IMAGE_STRENGTH


async def test_an_explicit_strength_still_wins_over_the_masked_default(toy_engine):
    # Raising it restyles what is inside the region instead of replacing it, which is a
    # real request -- so this has to stay a default rather than becoming a coercion.
    await _collect(toy_engine, _masked_request(image_strength=0.35))
    assert toy_engine.model.last_image_strength == 0.35


async def test_the_kept_region_comes_back_byte_identical(toy_engine):
    events = await _collect(toy_engine, _masked_request())
    out = Image.open(io.BytesIO(base64.b64decode(events[-1]["data"])))

    # Right half is the input, exactly -- that's mask_composite, on by default.
    assert out.getpixel((24, 16)) == (10, 200, 30)
    # Left half is whatever the model produced, which for ToyModel is its seed colour.
    assert out.getpixel((8, 16)) != (10, 200, 30)


async def test_mask_composite_off_leaves_the_model_s_own_decode_alone(toy_engine):
    # ToyModel's VAE paints one colour across the whole frame, so with the composite
    # off there is nothing left to make the two halves differ.
    events = await _collect(toy_engine, _masked_request(mask_composite=False))
    out = Image.open(io.BytesIO(base64.b64decode(events[-1]["data"])))
    assert out.getpixel((8, 16)) == out.getpixel((24, 16))


async def test_a_mask_and_a_fractional_start_select_the_combined_scheduler(toy_engine):
    await _collect(toy_engine, _masked_request(image_strength=0.35, fractional_start=True))
    assert toy_engine.model.last_scheduler == "mfluxible.schedulers.MaskedFractionalStartScheduler"


async def test_an_unmasked_request_still_passes_no_scheduler_at_all(toy_engine):
    # The regression this guards: once masking made a scheduler routine, it would be
    # easy to start passing one unconditionally -- which is a silent sampler swap on
    # every model whose own default isn't linear.
    await _collect(toy_engine, GenerateRequest(prompt="a cat", width=16, height=16, steps=1))
    assert toy_engine.model.last_scheduler is None


async def test_the_mask_job_is_cleared_even_when_generation_fails(toy_engine, monkeypatch):
    from mfluxible import schedulers

    # Fails at the VAE decode, i.e. *after* before_loop has set the job and the loop
    # has run -- which is the case that matters. A failure before the job exists would
    # pass this assertion without exercising anything. after_loop never runs on this
    # path either, which is exactly why the clear lives in run()'s finally.
    def boom(latent):
        raise RuntimeError("mflux fell over")

    monkeypatch.setattr(toy_engine.model.vae, "decode", boom)
    events = await _collect(toy_engine, _masked_request())
    assert events[-1]["type"] == "error"
    assert schedulers.active_mask_job() is None
