"""Tests for clients/mcp_server.py, which is a client of the HTTP API rather than part
of the server -- so most of what matters here is whether it builds requests the server
will actually accept, and whether everything it refuses comes back as a `ToolError`.

That second one is not a style preference. mcp 2.x replaces any exception that isn't a
ToolError with a bare "Error executing tool generate_image" and drops the text (see
CLAUDE.md), so a message the calling model could have acted on -- a box given in pixels,
a mask that doesn't fit -- reaches it as a failure it can learn nothing from. Every
rejection test below asserts the type as well as the text.
"""

import base64
import io

import pytest
from PIL import Image

import mcp_server
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import TextContent
from models import MODELS
from schemas import GenerateRequest

BOX = [0.385, 0.195, 0.964, 0.794]
"""The cup in a 768x768 test scene, as the tool takes it. Its pixel form (296, 150,
740, 610) is what a hand-drawn mask of the same region measured, so it doubles as a
fixed point for the rasterizer."""


def _png_bytes(size: tuple[int, int], color=(128, 128, 128), mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_array(boxes, size):
    """The tool's rasterized mask for `boxes`, as a PIL image."""
    png = mcp_server._mask_png_from_boxes(mcp_server._checked_mask_boxes(boxes), size)
    return Image.open(io.BytesIO(png))


@pytest.fixture
def base_image(tmp_path):
    """A 768x768 image on disk, the size the BOX fractions are quoted against."""
    path = tmp_path / "base.png"
    path.write_bytes(_png_bytes((768, 768)))
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may reach the HTTP server, including when the code under
    test is broken.

    Most tests here assert a refusal, and a refusal that stops happening doesn't just
    fail -- it falls through to a real generation against whatever MFLUXIBLE_URL points
    at, which on a developer's machine is a loaded model and minutes of GPU time. Found
    the honest way: deleting the supports_mask precheck took one run from 1.5s to 45s.

    Deliberately not a coroutine. _run is called to build the coroutine that
    create_task schedules, so raising here surfaces at the call site in generate_image
    rather than inside a task nobody awaits -- where job.done would stay False and the
    test would instead block for WAIT_SECONDS.
    """

    def _forbidden(job, body):
        raise AssertionError("this test started a generation; it should never reach _run")

    monkeypatch.setattr(mcp_server, "_run", _forbidden)


@pytest.fixture
def offline(monkeypatch):
    """No /health read. Returning None is the tool's own "unknown server" path, which
    skips the capability prechecks and leaves the local validation under test."""

    async def _none():
        return None

    monkeypatch.setattr(mcp_server, "_model_info", _none)


# --------------------------------------------------------------------------
# Rasterizing
# --------------------------------------------------------------------------


def test_a_box_is_rasterized_at_the_pixels_its_fractions_name():
    """The fractions->pixels conversion, pinned against a region measured by hand.

    Worth an exact assertion rather than a tolerance: these coordinates are the whole
    interface between a model looking at an image and the region that actually gets
    regenerated, and an off-by-a-few-percent here would read as the model having
    localized badly rather than as an arithmetic bug.
    """
    mask = _mask_array([BOX], (768, 768))
    assert mask.mode == "L"
    assert mask.getbbox() == (296, 150, 741, 611)  # getbbox()'s bounds are half-open


def test_a_full_frame_box_stays_inside_the_image():
    # 1.0 * width is one past the last valid index. PIL would clip it, but clipping
    # here is what keeps the rectangle drawn identical to the one the numbers describe.
    mask = _mask_array([[0.0, 0.0, 1.0, 1.0]], (64, 64))
    assert mask.getextrema() == (255, 255)


def test_boxes_accumulate_and_leave_the_gap_between_them_black():
    mask = _mask_array([[0.0, 0.0, 0.25, 1.0], [0.75, 0.0, 1.0, 1.0]], (64, 64))
    assert mask.getpixel((5, 32)) == 255
    assert mask.getpixel((58, 32)) == 255
    assert mask.getpixel((32, 32)) == 0


def test_a_mask_is_black_wherever_no_box_covers_it():
    # The server rejects an all-black mask, so the inverse -- an all-white one from a
    # small box -- would be a silent whole-frame regeneration rather than an error.
    mask = _mask_array([[0.4, 0.4, 0.6, 0.6]], (100, 100))
    assert mask.getpixel((0, 0)) == 0
    assert mask.getextrema() == (0, 255)


# --------------------------------------------------------------------------
# Box validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boxes, expected",
    [
        ([], "empty"),
        ([[0.1, 0.1, 0.5]], "exactly 4 numbers"),
        ([[0.1, 0.1, 0.5, 0.5, 0.9]], "exactly 4 numbers"),
        ([["a", "b", "c", "d"]], "isn't a number"),
        ([[None, 0.1, 0.5, 0.5]], "isn't a number"),
        ([[296, 150, 740, 610]], "not pixels"),  # the mistake this phrasing exists for
        ([[-0.1, 0.1, 0.5, 0.5]], "out of range"),
        ([[0.8, 0.1, 0.2, 0.5]], "no area"),  # right <= left
        ([[0.1, 0.8, 0.5, 0.2]], "no area"),  # bottom <= top
        ([[0.5, 0.1, 0.5, 0.5]], "no area"),  # zero width
    ],
)
def test_an_unusable_box_is_a_tool_error_that_says_which(boxes, expected):
    with pytest.raises(ToolError) as excinfo:
        mcp_server._checked_mask_boxes(boxes)
    assert expected in str(excinfo.value)


def test_a_bad_box_names_itself_so_the_model_can_tell_which_one():
    with pytest.raises(ToolError, match=r"0\.8"):
        mcp_server._checked_mask_boxes([[0.0, 0.0, 0.5, 0.5], [0.8, 0.1, 0.2, 0.5]])


def test_numeric_strings_are_accepted_rather_than_rejected_on_a_technicality():
    # The SDK coerces arguments against the annotation before they arrive, so this is
    # about not second-guessing it: a box that survived validation there shouldn't die
    # here over the type it arrived as.
    assert mcp_server._checked_mask_boxes([["0.1", "0.2", "0.3", "0.4"]]) == [(0.1, 0.2, 0.3, 0.4)]


# --------------------------------------------------------------------------
# EXIF orientation
# --------------------------------------------------------------------------


def _jpeg_with_orientation(size, orientation: int) -> bytes:
    img = Image.new("RGB", size, (200, 120, 60))
    exif = img.getexif()
    exif[274] = orientation  # ExifTags.Base.Orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_orientation_is_applied_so_a_mask_is_measured_in_the_frame_it_is_applied_in():
    """A phone photo is stored one way round and displayed the other.

    mflux rotates an input image before encoding it, so the server's size check compares
    *oriented* sizes. Rasterizing at the stored size would produce a mask rejected for a
    mismatch the caller never saw -- the image it looked at was the rotated one.
    """
    raw = _jpeg_with_orientation((40, 20), orientation=6)
    assert Image.open(io.BytesIO(raw)).size == (40, 20)  # as stored
    assert mcp_server._oriented(raw).size == (20, 40)  # as displayed, and as mflux sees it


def test_the_client_and_the_server_orient_an_image_identically():
    """mcp_server._oriented and engine._oriented are separate implementations of one
    rule, and the size check that spans them only works while they agree. They are in
    different dependency sets -- the MCP client must never import the server -- so this
    is the only place the two can be held together."""
    import engine

    for orientation in (1, 3, 6, 8):
        raw = _jpeg_with_orientation((40, 20), orientation)
        assert mcp_server._oriented(raw).size == engine._oriented(raw).size, orientation


# --------------------------------------------------------------------------
# Argument validation on the tool itself
# --------------------------------------------------------------------------


async def test_the_two_mask_arguments_are_refused_together(base_image, tmp_path, offline):
    mask = tmp_path / "mask.png"
    mask.write_bytes(_png_bytes((768, 768), 255, "L"))
    with pytest.raises(ToolError, match="not both"):
        await mcp_server.generate_image(
            prompt="x", image_path=str(base_image), mask_boxes=[BOX], mask_path=str(mask)
        )


@pytest.mark.parametrize("kwargs", [{"mask_boxes": [BOX]}, {"mask_path": "/nonexistent.png"}])
async def test_a_mask_without_an_image_is_refused(kwargs, offline):
    with pytest.raises(ToolError, match="requires image_path"):
        await mcp_server.generate_image(prompt="x", **kwargs)


async def test_a_feather_without_a_mask_is_refused_here_rather_than_as_a_400(base_image, offline):
    """request_problem rejects a non-default mask_feather with no mask, so this would
    otherwise be a round trip ending in a 400 that names a field the caller did set."""
    with pytest.raises(ToolError, match="requires mask_boxes or mask_path"):
        await mcp_server.generate_image(prompt="x", image_path=str(base_image), mask_feather=16)


async def test_a_mask_of_the_wrong_size_is_refused_with_both_sizes(base_image, tmp_path, offline):
    mask = tmp_path / "mask.png"
    mask.write_bytes(_png_bytes((100, 100), 255, "L"))
    with pytest.raises(ToolError, match=r"100x100.*768x768"):
        await mcp_server.generate_image(
            prompt="x", image_path=str(base_image), mask_path=str(mask)
        )


async def test_a_mask_sized_against_the_stored_frame_of_a_rotated_photo_is_refused(
    tmp_path, offline
):
    """The other half of the orientation rule, at the tool's edge: a mask drawn to the
    photo's stored dimensions doesn't match the frame the photo is displayed and encoded
    in, and the error says so in the displayed frame's terms."""
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(_jpeg_with_orientation((40, 20), orientation=6))
    mask = tmp_path / "mask.png"
    mask.write_bytes(_png_bytes((40, 20), 255, "L"))
    with pytest.raises(ToolError, match=r"40x20 but image is 20x40"):
        await mcp_server.generate_image(prompt="x", image_path=str(photo), mask_path=str(mask))


async def test_an_unreadable_mask_does_not_quote_pillows_exception(base_image, tmp_path, offline):
    """Pillow's message for an undecodable file is a repr of the in-memory buffer,
    memory address included. Nothing upstream should reach a caller through a message
    written here -- the same rule engine.py's decode checks follow, and the reason
    _input_image_problem splits _decode_input_image off at all."""
    junk = tmp_path / "notanimage.png"
    junk.write_bytes(b"this is not a PNG")
    with pytest.raises(ToolError) as excinfo:
        await mcp_server.generate_image(
            prompt="x", image_path=str(base_image), mask_path=str(junk)
        )
    assert "could not be decoded" in str(excinfo.value)
    assert "BytesIO" not in str(excinfo.value)
    assert "0x" not in str(excinfo.value)


async def test_a_missing_mask_file_names_the_path(base_image, tmp_path, offline):
    with pytest.raises(ToolError, match="could not read mask_path"):
        await mcp_server.generate_image(
            prompt="x", image_path=str(base_image), mask_path=str(tmp_path / "absent.png")
        )


async def test_a_model_without_mask_support_is_refused_before_a_generation_starts(
    base_image, monkeypatch
):
    async def _flow_match_model():
        return {"label": "FLUX.2 Klein", "supports_mask": False}

    monkeypatch.setattr(mcp_server, "_model_info", _flow_match_model)
    with pytest.raises(ToolError, match="FLUX.2 Klein"):
        await mcp_server.generate_image(
            prompt="x", image_path=str(base_image), mask_boxes=[BOX]
        )


# --------------------------------------------------------------------------
# What actually goes on the wire
# --------------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch):
    """Runs generate_image without any HTTP, and hands back the request body it built.

    _run is replaced rather than the transport mocked because the body handed to it is
    exactly the thing under test: everything after that point is httpx's business.
    """
    bodies = []

    async def fake_run(job, body):
        bodies.append(body)
        job.content = [TextContent(type="text", text="ok")]

    async def fake_await_job(job, ctx, seconds):
        # The real one polls on a 0.25s tick, which would dominate a suite that runs in
        # under a second. Awaiting the task is the same wait with no sleeping.
        if job.task is not None:
            await job.task

    monkeypatch.setattr(mcp_server, "_run", fake_run)
    monkeypatch.setattr(mcp_server, "_await_job", fake_await_job)
    return bodies


async def test_a_maskless_call_sends_no_feather_whatever_the_default_is(base_image, sent, offline):
    """The one place this tool's defaults and the API's disagree, and the disagreement
    has a sharp edge: mask_feather defaults to 8 here and 0 there, and request_problem
    rejects a non-default feather with no mask. So a plain text-to-image call has to
    send 0 rather than the default it was handed, or every unmasked generation 400s."""
    await mcp_server.generate_image(prompt="x")
    assert sent[0]["mask"] is None
    assert sent[0]["mask_feather"] == 0

    await mcp_server.generate_image(prompt="x", image_path=str(base_image), image_strength=0.4)
    assert sent[1]["mask"] is None
    assert sent[1]["mask_feather"] == 0


async def test_a_masked_call_sends_the_feather_and_a_mask_matching_the_image(
    base_image, sent, offline
):
    await mcp_server.generate_image(prompt="x", image_path=str(base_image), mask_boxes=[BOX])
    body = sent[0]
    assert body["mask_feather"] == mcp_server.DEFAULT_MASK_FEATHER
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask"])))
    assert mask.size == Image.open(base_image).size
    assert mask.getbbox() == (296, 150, 741, 611)


async def test_a_masked_call_defaults_image_strength_to_zero_rather_than_the_api_s_0_4(
    base_image, sent, offline
):
    """The sharpest footgun in the whole feature, closed in code rather than in prose.

    Omitting image_strength is the ordinary thing to do with an optional argument, and
    it used to mean the server applied its own 0.4 -- which with a mask starts the
    region 40% along the schedule and hands back the original object very nearly
    untouched, having spent a full generation and reported success. Measured on a
    768x768 Z-Image-Turbo run: 5.1/255 of change inside the mask at 0.4, against 41.6
    for the identical request at 0.0.

    Sending 0.0 explicitly is what makes this hold -- None would leave the server to
    apply DEFAULT_IMAGE_STRENGTH, which is exactly the value being avoided.
    """
    await mcp_server.generate_image(prompt="x", image_path=str(base_image), mask_boxes=[BOX])
    assert sent[0]["image_strength"] == 0.0


async def test_the_zero_default_does_not_leak_onto_a_maskless_call(base_image, sent, offline):
    # Plain image-to-image keeps deferring to the server, where 0.4 is the right default
    # and 0.0 would mean the input image had no influence at all.
    await mcp_server.generate_image(prompt="x", image_path=str(base_image))
    assert sent[0]["image_strength"] is None


async def test_an_explicit_image_strength_survives_a_masked_call(base_image, sent, offline):
    # Restyling what is in the region rather than replacing it is a real request, so the
    # default must be a default and not a coercion.
    await mcp_server.generate_image(
        prompt="x", image_path=str(base_image), mask_boxes=[BOX], image_strength=0.35
    )
    assert sent[0]["image_strength"] == 0.35


async def test_an_explicit_feather_overrides_the_default(base_image, sent, offline):
    await mcp_server.generate_image(
        prompt="x", image_path=str(base_image), mask_boxes=[BOX], mask_feather=16
    )
    assert sent[0]["mask_feather"] == 16


async def test_a_supplied_mask_is_forwarded_unchanged(base_image, tmp_path, sent, offline):
    """mask_path exists for a mask something else drew, so the bytes that arrive are the
    bytes that go out -- no re-encode, no threshold, no reinterpretation of its greys."""
    raw = _png_bytes((768, 768), 255, "L")
    mask = tmp_path / "mask.png"
    mask.write_bytes(raw)
    await mcp_server.generate_image(
        prompt="x", image_path=str(base_image), mask_path=str(mask)
    )
    assert base64.b64decode(sent[0]["mask"]) == raw


async def test_the_mcp_tool_never_builds_a_request_its_own_server_rejects(
    base_image, sent, monkeypatch
):
    """The tool's local checks duplicate request_problem's rules rather than sharing code
    with them -- they can't share any, since the MCP client must not import the server --
    so this pins the two together across every model in the real table.

    It is the same guarantee test_a1111_shim_never_builds_a_request_its_own_model_rejects
    gives the SillyTavern endpoint, and it matters more here: a 400 from a tool call is
    reported to the model as a failed generation, minutes after the user asked for one.

    No weights are touched -- MfluxEngine resolves its spec in __init__ and downloads
    nothing until load().
    """
    from engine import MfluxEngine

    for spec in MODELS:
        async def _health(spec=spec):
            return {
                "label": spec.label,
                "supports_guidance": spec.supports_guidance,
                "supports_negative_prompt": spec.supports_negative_prompt,
                "supports_fractional_start": spec.supports_fractional_start,
                "supports_mask": spec.supports_mask,
            }

        monkeypatch.setattr(mcp_server, "_model_info", _health)
        engine = MfluxEngine(model=spec.key, quantize=None, model_cache_dir=None)
        try:
            sent.clear()
            await mcp_server.generate_image(prompt="a cat")
            await mcp_server.generate_image(
                prompt="a cat", image_path=str(base_image), image_strength=0.4
            )

            if spec.supports_mask:
                await mcp_server.generate_image(
                    prompt="a cat",
                    image_path=str(base_image),
                    image_strength=0.0,
                    mask_boxes=[BOX],
                )
            else:
                # Refused up front, so it never reaches a body to check.
                with pytest.raises(ToolError, match="masking"):
                    await mcp_server.generate_image(
                        prompt="a cat", image_path=str(base_image), mask_boxes=[BOX]
                    )

            for body in sent:
                problem = engine.request_problem(GenerateRequest(**body))
                assert problem is None, f"{spec.key}: {problem}"
        finally:
            engine.shutdown()
