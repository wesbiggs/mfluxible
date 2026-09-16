"""MCP server exposing mfluxible's image generation as a tool Claude can call directly.

Runs over stdio (the transport Claude Code/Desktop use to launch local MCP servers)
and proxies to a already-running mfluxible HTTP server, forwarding step progress via
MCP's progress-reporting mechanism and returning the final image as inline content.

Requires the mfluxible HTTP server (server.py) to already be running separately --
this doesn't load the model itself, just calls the API.

Three constraints from the MCP host shape this file, and none of them is something
it can lift on its own -- see the comments on MAX_RESULT_BYTES, on WAIT_SECONDS, and
on the width/height defaults below.
"""

import asyncio
import base64
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import urllib.parse

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import Annotations, ImageContent, TextContent
from PIL import Image as PILImage, ImageDraw, ImageOps

MFLUXIBLE_URL = os.environ.get("MFLUXIBLE_URL", "http://127.0.0.1:8420/mfluxible/v1/images/generations")

# Which model the server loaded is the server's business, but this tool has to know two
# things about it: what to tell the caller it generated, and whether guidance /
# negative_prompt are arguments it will accept at all. /health answers both, and lives
# alongside the generations endpoint, so it's derived from the same URL by default.
MFLUXIBLE_HEALTH_URL = os.environ.get(
    "MFLUXIBLE_HEALTH_URL",
    urllib.parse.urlunsplit(urllib.parse.urlsplit(MFLUXIBLE_URL)._replace(path="/health", query="")),
)

# Sent as a bearer token on every call when set, for a server behind an auth proxy
# (see Caddyfile.example). Environment-only because there is no other channel here:
# an MCP host launches this as a stdio subprocess, so there is no command line to
# pass and no terminal to prompt at. Empty means send no Authorization header at all
# rather than an empty one, which a proxy could only reject.
MFLUXIBLE_BEARER_TOKEN = os.environ.get("MFLUXIBLE_BEARER_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {MFLUXIBLE_BEARER_TOKEN}"} if MFLUXIBLE_BEARER_TOKEN else {}

# Claude caps a single tool result at ~1MB, and MCP ships images as base64, which
# inflates bytes by 4/3. So the *raw* image has to come in around 750KB to clear
# the cap once encoded; 700KB leaves room for the surrounding JSON. A full-res
# 1024x1280 PNG off this model runs ~1.8MB raw / ~2.4MB base64 -- roughly 2.4x
# over -- so anything at that size has to be re-encoded before it can be returned.
MAX_RESULT_BYTES = int(os.environ.get("MFLUXIBLE_MCP_MAX_BYTES", 700_000))

# How long one tool call may block before handing back a handle instead of an image.
#
# MCP hosts time each tool call out on their own schedule, and this server can neither
# negotiate nor extend that. Reporting progress is *not* a reliable workaround: measured
# against a live Claude Code session on 2026-08-31, a generate_image call died at ~60s
# with the MCP SDK's default "Request timed out" even though this server had sent a
# progress notification every ~8s throughout. (Some hosts do reset on progress -- the
# Claude Code CLI, for one, runs a separate 30-minute idle watchdog that progress
# notifications rearm -- but a server that only streams progress is betting on the host.)
#
# So generation runs in a background task and the tool waits at most WAIT_SECONDS for it:
# a quick generation still returns the image from the first call, and a slow one returns
# a handle that check_image() picks up. Keep this a little under the shortest host timeout
# you care about (60s, hence 45); raise it if your host is more generous.
WAIT_SECONDS = float(os.environ.get("MFLUXIBLE_MCP_WAIT_SECONDS", 45))

# Deliberately smaller than the HTTP API's own 1024x1024 default. Generation is linear
# in pixel count -- measured on an M2 Pro, 768x768 at 9 steps runs 77-115s against 249s
# for 1024x1280 -- and resolution, not step count, is the lever (most of a short run is
# fixed cost outside the denoising loop). Since WAIT_SECONDS above means a long run costs
# extra polling round-trips rather than a failure, this is now a latency default rather
# than a correctness one. Ask for explicit width/height in the call to override.
DEFAULT_WIDTH = int(os.environ.get("MFLUXIBLE_MCP_WIDTH", 768))
DEFAULT_HEIGHT = int(os.environ.get("MFLUXIBLE_MCP_HEIGHT", 768))

# Unset on purpose: with no step count in the request the server uses whichever default
# suits the model it actually loaded (9 for Z-Image-Turbo, 4 for FLUX.1-schnell, 25 for
# FLUX.1-dev, 20 for Qwen-Image). Hardcoding one here would have meant every model
# running at Z-Image-Turbo's step count. Set it only to override that per-model default.
_raw_steps = os.environ.get("MFLUXIBLE_MCP_STEPS", "").strip()
DEFAULT_STEPS = int(_raw_steps) if _raw_steps else None

# Feather applied to a mask unless a call overrides it. The HTTP API defaults this to
# 0, and that difference is deliberate: a caller writing JSON by hand has read the
# field's documentation, while the model calling this tool is working from a docstring
# and will mostly not pass it at all. 8 is the low end of the useful range -- enough to
# hide the latent grid's 8px staircase along the mask edge, far short of the width where
# a feather stops being an edge treatment and starts cross-fading old content with new.
DEFAULT_MASK_FEATHER = int(os.environ.get("MFLUXIBLE_MCP_MASK_FEATHER", 8))

# The returned image may be downscaled/recompressed to fit MAX_RESULT_BYTES, so the
# untouched full-resolution PNG (metadata and all) is always written here first.
# Deliberately not under ~/.cache alongside MFLUXIBLE_MODEL_DIR: cache directories are
# reasonably treated as disposable, and this is the only full-quality copy that exists.
SAVE_DIR = Path(os.environ.get("MFLUXIBLE_MCP_SAVE_DIR", "~/Pictures/mfluxible")).expanduser()

# Finished jobs stay retrievable for a while (a host that timed out mid-generation may
# come back for the image minutes later), but their content holds a whole base64 image,
# so both the age and the count are bounded.
JOB_RETENTION_S = float(os.environ.get("MFLUXIBLE_MCP_JOB_RETENTION_S", 900))
MAX_FINISHED_JOBS = 8

server = MCPServer(name="mfluxible")


@dataclass
class _Job:
    """One in-flight or finished generation, addressed by `handle`."""

    handle: str
    width: int
    height: int
    steps: int | None  # None = whatever the server's model defaults to
    model: str  # label from /health, "" if it couldn't be read
    started: float  # time.monotonic()
    step: int = 0
    total_steps: int = 0
    step_ms: int = 0
    content: list[TextContent | ImageContent] | None = None
    error: str | None = None
    finished: float | None = None  # time.monotonic()
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.content is not None or self.error is not None

    def status(self) -> str:
        elapsed = (self.finished or time.monotonic()) - self.started
        if self.error is not None:
            return f"failed after {elapsed:.0f}s: {self.error}"
        if self.content is not None:
            return f"done in {elapsed:.0f}s"
        if self.total_steps == 0:
            # No `start` event yet: the server takes one generation at a time, so this
            # is either model warm-up or a queue behind someone else's request.
            return f"starting, {elapsed:.0f}s elapsed"
        return f"step {self.step}/{self.total_steps}, {elapsed:.0f}s elapsed"


_JOBS: dict[str, _Job] = {}


def _prune_jobs() -> None:
    now = time.monotonic()
    for handle, job in list(_JOBS.items()):
        if job.finished is not None and now - job.finished > JOB_RETENTION_S:
            del _JOBS[handle]

    finished = sorted(
        (j for j in _JOBS.values() if j.finished is not None), key=lambda j: j.finished
    )
    for job in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
        del _JOBS[job.handle]


def _fit_result(png: bytes) -> tuple[bytes, str, str]:
    """Shrink `png` until it fits MAX_RESULT_BYTES. Returns (bytes, mime_type, note)."""
    if len(png) <= MAX_RESULT_BYTES:
        return png, "image/png", ""

    img = PILImage.open(io.BytesIO(png))
    img = img.convert("RGB")  # JPEG has no alpha channel
    smallest = None

    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        frame = img if scale == 1.0 else img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            PILImage.LANCZOS,
        )
        for quality in (85, 70, 55, 40):
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = buf.getvalue()
            if smallest is None or len(candidate) < len(smallest[0]):
                smallest = (candidate, frame.width, frame.height, quality)
            if len(candidate) <= MAX_RESULT_BYTES:
                note = (
                    f"shown as JPEG q{quality} at {frame.width}x{frame.height} "
                    f"to fit the ~1MB tool-result cap"
                )
                return candidate, "image/jpeg", note

    data, width, height, quality = smallest
    note = f"shown as JPEG q{quality} at {width}x{height}; still {len(data) / 1e6:.1f}MB"
    return data, "image/jpeg", note


def _oriented(raw: bytes) -> PILImage.Image:
    """`raw` decoded with its EXIF Orientation applied, matching what the server does.

    Load-bearing for the size check below rather than a nicety. mflux rotates an input
    image before encoding it, so the server compares a mask against the image's
    *oriented* size -- and a phone photo is exactly where raw and oriented differ
    (4032x3024 bytes carrying an Orientation tag, displayed 3024x4032). A mask
    rasterized at the raw size would be rejected for a mismatch the caller can't see,
    and one drawn by hand at the displayed size would look wrong here but be right.
    """
    with PILImage.open(io.BytesIO(raw)) as img:
        return ImageOps.exif_transpose(img)


def _checked_mask_boxes(boxes: list[list[float]]) -> list[tuple[float, float, float, float]]:
    """`boxes` as plain floats, or a ToolError naming the box that couldn't be used.

    Every failure here has to leave as a ToolError specifically: mcp 2.x replaces any
    other exception with a bare "Error executing tool generate_image" and drops the
    text, so a box the model could have fixed would come back as a failure it can learn
    nothing from. That includes the coercion itself -- the SDK validates arguments
    against this function's annotations first and would normally reject a non-number
    before it arrives, but a float() raising ValueError here is precisely the case that
    would reach the model stripped.
    """
    if not boxes:
        raise ToolError("mask_boxes is empty; pass at least one [x0, y0, x1, y1] box, or omit it.")
    checked = []
    for box in boxes:
        if len(box) != 4:
            raise ToolError(f"each mask box needs exactly 4 numbers, [x0, y0, x1, y1]; got {box!r}.")
        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError):
            raise ToolError(f"mask box {box!r} has a value that isn't a number.")
        if not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
            raise ToolError(
                f"mask box {box!r} is out of range: these are fractions of the image, 0.0 to 1.0, "
                "not pixels."
            )
        if x1 <= x0 or y1 <= y0:
            raise ToolError(
                f"mask box {box!r} encloses no area; it reads [left, top, right, bottom], so it "
                "needs right > left and bottom > top."
            )
        checked.append((x0, y0, x1, y1))
    return checked


def _mask_png_from_boxes(boxes: list[tuple[float, float, float, float]], size: tuple[int, int]) -> bytes:
    """Rasterize normalized boxes into a mask PNG: white inside them, black outside.

    Normalized rather than pixel coordinates because of what the model picking them has
    actually seen. _fit_result downscales the inline copy of any image past the host's
    result cap, so for anything much larger than this tool's default size the frame the
    model looked at is smaller than the PNG on disk that image_path points back to --
    and pixel coordinates read off the former would select the wrong region of the
    latter, silently and by a factor nothing in the response states outright. A fraction
    of the frame means the same thing in both.
    """
    width, height = size
    mask = PILImage.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in boxes:
        # Clamped to the last valid index: a box ending at 1.0 would otherwise land one
        # pixel past the edge. PIL would clip it anyway; doing it here keeps the
        # rectangle that gets drawn the same one the coordinates describe.
        draw.rectangle(
            (
                max(0, min(width - 1, round(x0 * width))),
                max(0, min(height - 1, round(y0 * height))),
                max(0, min(width - 1, round(x1 * width))),
                max(0, min(height - 1, round(y1 * height))),
            ),
            fill=255,
        )
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def _build_content(job: _Job, event: dict) -> list[TextContent | ImageContent]:
    full = base64.b64decode(event["data"])

    # A generation costs minutes; never let a disk problem throw that away.
    # Worst case the caller loses the full-res copy, not the image itself.
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = SAVE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{event['seed']}.png"
        path.write_bytes(full)
        saved = f"Full-resolution PNG saved to {path}"
    except OSError as exc:
        saved = f"Full-resolution PNG could not be saved to {SAVE_DIR} ({exc})"

    data, mime, note = _fit_result(full)
    # total_steps comes off the server's own `start` event, so it is right even when
    # the request left `steps` unset and the model's default decided it.
    caption = (
        f"{job.model + ', ' if job.model else ''}"
        f"{job.width}x{job.height}, {job.total_steps or job.steps} steps, seed {event['seed']}, "
        f"{event['generation_time']:.1f}s. {saved}"
    )
    if note:
        caption += f" ({note})"

    return [
        # audience/priority are MCP's display hints: they ask the host to
        # surface this content to the user rather than bury it in the
        # collapsed tool-result block. Honoring them is up to the host.
        ImageContent(
            type="image",
            data=base64.b64encode(data).decode(),
            mime_type=mime,
            annotations=Annotations(audience=["user", "assistant"], priority=1.0),
        ),
        TextContent(
            type="text",
            text=caption,
            annotations=Annotations(audience=["user", "assistant"], priority=0.4),
        ),
    ]


async def _run(job: _Job, body: dict) -> None:
    """Stream one generation to completion, independent of any single tool call.

    This deliberately outlives the call that started it: when a host gives up on a
    tool call, only the request is cancelled, not this task -- so the generation the
    user already paid minutes for still finishes, still lands on disk, and is still
    waiting under its handle when the model comes back for it.
    """
    try:
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("POST", MFLUXIBLE_URL, json=body, headers=AUTH_HEADERS) as resp,
        ):
            if resp.status_code != 200:
                # Not raise_for_status(): the server refuses guidance/negative_prompt on
                # a model that can't act on them and says which in the body, which
                # raise_for_status() would throw away. A streamed response has to be
                # read before its body is available at all.
                await resp.aread()
                try:
                    detail = resp.json().get("message") or resp.text
                except ValueError:
                    detail = resp.text
                raise RuntimeError(f"mfluxible returned {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])

                if event["type"] == "start":
                    job.total_steps = event["total_steps"]
                elif event["type"] == "thinking":
                    job.step = event["step"]
                    job.total_steps = event["total_steps"]
                    job.step_ms = event["step_ms"]
                elif event["type"] == "image":
                    job.content = _build_content(job, event)
                    return
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])
        raise RuntimeError("mfluxible stream ended without producing an image")
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller via job.error
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished = time.monotonic()


async def _await_job(job: _Job, ctx: Context | None, seconds: float) -> None:
    """Block up to `seconds` for `job`, forwarding each step as a progress notification.

    Progress can't be relied on to hold a host's timeout open (see WAIT_SECONDS), but
    hosts that do render it show the user something moving, so it's still worth sending.
    """
    deadline = time.monotonic() + seconds
    reported = -1
    while not job.done and time.monotonic() < deadline:
        if ctx is not None and job.step != reported:
            reported = job.step
            await ctx.report_progress(
                job.step,
                job.total_steps or None,
                f"step {job.step}/{job.total_steps} ({job.step_ms}ms)",
            )
        await asyncio.sleep(0.25)


def _result(job: _Job) -> list[TextContent | ImageContent]:
    """Whatever the caller should get right now: the image, the failure, or a handle."""
    if job.error is not None:
        # ToolError, not a bare exception: mcp 2.x reports anything else as a flat
        # "Error executing tool generate_image" with the message stripped, so the
        # actual failure (mflux blew up, HTTP server not running) never reaches the
        # model. ToolError is the SDK's "a failure you anticipated" and keeps the text.
        raise ToolError(job.error)
    if job.content is not None:
        return job.content

    return [
        TextContent(
            type="text",
            text=(
                f'Still generating ({job.status()}). Call check_image with handle "{job.handle}" '
                f"to collect it -- that call blocks up to {WAIT_SECONDS:.0f}s and returns the image "
                f"the moment it's ready, so if it reports the job is still running, just call it "
                f"again. Generation continues either way, and the full-resolution PNG is saved to "
                f"disk even if nobody collects it."
            ),
        )
    ]


_MODEL: dict | None = None


async def _model_info() -> dict | None:
    """The server's /health `model` block, cached after the first successful read.

    Best-effort by design: an unreachable server, or one predating this field, just
    means no local check and no model name in the caption -- the generation request
    that follows reports the real problem. Only a successful read is cached, so a
    server that starts later is picked up on the next call.
    """
    global _MODEL
    if _MODEL is None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(MFLUXIBLE_HEALTH_URL, headers=AUTH_HEADERS)
                resp.raise_for_status()
                _MODEL = resp.json().get("model")
        except Exception:  # noqa: BLE001 -- advisory only; the generation call reports failures
            return None
    return _MODEL


@server.tool()
async def generate_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int | None = DEFAULT_STEPS,
    seed: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    image_path: str | None = None,
    image_strength: float | None = None,
    fractional_start: bool = False,
    mask_boxes: list[list[float]] | None = None,
    mask_path: str | None = None,
    mask_feather: int = DEFAULT_MASK_FEATHER,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt, optionally seeded from an existing image.

    Which model runs is the server's choice, not this tool's: Z-Image-Turbo by
    default, or any other checkpoint it was started with, across Z-Image, FLUX.1,
    FLUX.2 Klein, Qwen-Image, Krea-2 and ERNIE-Image.

    Leave steps unset and the server uses that model's own default, which runs
    anywhere from 4 to 50 depending on the checkpoint -- prefer that to guessing,
    since a step count that suits one model is several times too small for another.

    guidance, negative_prompt and fractional_start are supported per checkpoint, not
    per family: a turbo variant and the base model it came from often differ, so
    knowing the family is not enough to know whether an argument applies. Sending one
    to a model that cannot act on it is an error naming that model, so leave all three
    unset unless the user asked for them.

    width/height must be divisible by 16 (the server floors them to a multiple of 16,
    so anything else silently generates up to 15px smaller). Generation time scales with
    pixel count and can take minutes, so prefer the defaults unless the user asks for a
    specific size. Leave seed unset for a random one.

    image_path is a local file path (read from disk, not a URL) to an image-to-image
    input -- unlike guidance/negative_prompt, every model this server can run accepts
    it. image_strength (0.0-1.0, only meaningful alongside image_path, default 0.4)
    follows mflux's own convention, which is the *inverse* of the "denoising strength"
    used by tools like Stable Diffusion/A1111: HIGHER means the input image constrains
    the output MORE (closer to unchanged, possibly with few or even zero denoising
    steps actually run), and 0.0 means the image has no influence at all (plain
    text-to-image). Don't assume the opposite-convention meaning.

    fractional_start (only with image_path) makes image_strength continuous. Without
    it, strength is quantized to 1/steps -- at 9 steps, 0.35 and 0.4 are the same
    image -- so set it when the user wants to tune a strength finely, or is asking why
    a small change to it did nothing. It costs no extra time.

    mask_boxes and mask_path inpaint: they regenerate one region and hold the rest of
    the frame to the input. Pass one or the other, never both, and only with image_path.

    prompt describes the WHOLE FINISHED FRAME, not the masked region on its own, and
    this matters more the smaller the mask is. The model composes for the entire image
    and the region is a window onto that composition, so a prompt naming only the new
    object gets composed at full-frame scale and the window lands somewhere inside it --
    which on a small mask returns a giant cropped fragment of the thing that was asked
    for. Replacing a chest emblem across 6% of a frame, "a stylized bold letter F emblem,
    heroic shield badge shape" gave a letterform several times too large, cut off flat at
    the mask edge; the identical seed and box prompted "a cartoon superhero frog with a
    red cape, a bold letter F emblem on its chest, pale cream background" gave a properly
    sized badge. Across a third of a frame the difference is cosmetic. Describe the frame
    either way -- it costs nothing and it is the only control over how the new content is
    scaled and placed within the region.

    Check the box against what is actually being replaced, rather than trusting a first
    estimate. In that same case the box was offset about 13% of the frame to the left of
    the emblem and clipped its right edge, which on its own cost a flat-cut badge -- a
    smaller error than the prompt, but the two compound.

    mask_boxes is the one to reach for -- a list of [x0, y0, x1, y1] rectangles covering
    what should be replaced, given as fractions of the image from 0.0 to 1.0, reading
    left, top, right, bottom. Fractions rather than pixels because the copy of an image
    returned inline may have been downscaled to fit the host's size cap, so pixel
    coordinates taken off it can address a different part of the full-resolution file
    that image_path points to. mask_path is a local path to a mask image already drawn,
    white where the model may change the image and black where it may not; it has to be
    exactly the same pixel size as the image, so it is the option for a mask some other
    step produced rather than one to author from scratch.

    Three things about inpainting are counterintuitive enough to be worth stating
    outright:

      - image_strength means something different here, and this tool defaults it to 0.0
        when a mask is present rather than to the 0.4 above. With a mask it no longer
        decides how much of the *frame* survives -- the mask decides that -- only how
        much of the old content *inside the region* survives. At 0.4 the thing that was
        meant to be replaced comes back very nearly intact while the call still reports
        success, so 0.4 is never what is wanted here.

        The range worth knowing is 0.0 to about 0.2, and it is a composition anchor
        rather than a strength. At 0.0 the region is reinvented from pure noise and
        where the new content lands inside the box is the prompt's decision alone. One
        rung in -- 0.1 to 0.2, with fractional_start to tell them apart -- it starts
        from the encoded original instead, so the new content inherits the old one's
        position, scale and outline. Reach for that when the prompt cannot reproduce the
        original composition, which is the normal case for a photograph: "a man reading
        a book" will not be framed the way the photo was, and anchoring is the only thing
        that holds the replacement where the original sat. The cost is bleed-through --
        replacing an S emblem with an F, 0.2 keeps the shield exactly in place and lets
        the old S contaminate the letterform. Anchor when the new content should share
        the old one's geometry; don't when it shouldn't. Nothing does both.
      - Give the box room for what the new content needs, shadow included. The mask edge
        is a hard boundary and anything crossing it is cut off flat at it. Room is not
        free, though: a box that takes in a swathe of flat background can come back a
        visibly different shade of it, so bound the subject rather than the wall behind.
      - It fills, it does not erase. Nothing can say "leave this region empty", so
        masking an object and prompting for bare background yields an object-shaped
        something instead of the background closing over it.

    mask_feather (pixels, default 8) hides the 8px staircase the latent grid leaves
    along a mask edge. It is not a way to blend two regions: a grey mask value holds
    that pixel partway to the original at every step, so a wide feather cross-fades the
    old content with the new into a double exposure. 8-16 is the whole useful range.

    Masking is per-checkpoint in the same way guidance is, and for the same underlying
    reason -- it needs the model's own scheduler to be the linear one -- so a mask sent
    to a model that cannot take it is an error naming that model.

    Returns the image directly if it finishes quickly. Otherwise it returns a handle
    and keeps generating in the background: call check_image with that handle to
    collect the image, repeating until it comes back.

    The full-resolution PNG is always saved to disk and its path returned; the inline
    copy may be recompressed to fit the host's tool-result size cap.
    """
    # Checked here rather than left to the server so a request that was never going to
    # work fails immediately, with the model's name in the message, instead of after a
    # round trip. Missing keys mean an unknown server -- defer to it and let it decide.
    info = await _model_info()
    if info is not None:
        if guidance is not None and not info.get("supports_guidance", True):
            raise ToolError(f"{info.get('label', 'this model')} does not use guidance; omit the guidance argument.")
        if negative_prompt is not None and not info.get("supports_negative_prompt", True):
            raise ToolError(
                f"{info.get('label', 'this model')} has no negative-prompt branch; omit the negative_prompt argument."
            )
        if fractional_start and not info.get("supports_fractional_start", True):
            raise ToolError(
                f"{info.get('label', 'this model')} does not run the linear schedule fractional_start "
                "extends; omit the fractional_start argument (image_strength still works)."
            )
        if (mask_boxes is not None or mask_path is not None) and not info.get("supports_mask", True):
            raise ToolError(
                f"{info.get('label', 'this model')} does not run the linear schedule masking needs; "
                "omit mask_boxes/mask_path (image_path on its own still works, on the whole frame)."
            )

    image_b64 = None
    image_raw = None
    if image_path is not None:
        if image_strength is not None and not (0.0 <= image_strength <= 1.0):
            raise ToolError("image_strength must be between 0.0 and 1.0.")
        try:
            image_raw = Path(image_path).expanduser().read_bytes()
        except OSError as exc:
            raise ToolError(f"could not read image_path {image_path!r}: {exc}")
        image_b64 = base64.b64encode(image_raw).decode("ascii")
    elif image_strength is not None:
        raise ToolError("image_strength requires image_path to also be set.")
    elif fractional_start:
        raise ToolError("fractional_start requires image_path to also be set.")

    # Built here rather than left to the server so a mask that was never going to fit
    # fails before a generation is started, and -- for mask_boxes -- so the rasterizing
    # happens against the image's real oriented size, which only this side knows.
    mask_b64 = None
    if mask_boxes is not None and mask_path is not None:
        raise ToolError("pass either mask_boxes or mask_path, not both: they are two ways to say one thing.")
    if mask_boxes is not None or mask_path is not None:
        if image_raw is None:
            raise ToolError("a mask requires image_path to also be set -- there is nothing to mask a region of.")
        try:
            image_size = _oriented(image_raw).size
        except Exception:  # noqa: BLE001 -- Pillow raises several unrelated types here
            # Deliberately not interpolating the exception, the way server.py's own
            # decode checks don't: Pillow's message for an undecodable file is a repr
            # of the in-memory buffer, memory address and all, which tells the caller
            # nothing it can act on and the path already says which file was meant.
            raise ToolError(f"image_path {image_path!r} could not be decoded as an image.")

        if mask_boxes is not None:
            mask_raw = _mask_png_from_boxes(_checked_mask_boxes(mask_boxes), image_size)
        else:
            try:
                mask_raw = Path(mask_path).expanduser().read_bytes()
            except OSError as exc:
                raise ToolError(f"could not read mask_path {mask_path!r}: {exc}")
            try:
                mask_size = _oriented(mask_raw).size
            except Exception:  # noqa: BLE001 -- as above
                raise ToolError(f"mask_path {mask_path!r} could not be decoded as an image.")
            if mask_size != image_size:
                raise ToolError(
                    f"mask is {mask_size[0]}x{mask_size[1]} but image is {image_size[0]}x{image_size[1]}; "
                    "they must be the same size, or the mask selects a different region than it looks like."
                )
        mask_b64 = base64.b64encode(mask_raw).decode("ascii")
    elif mask_feather != DEFAULT_MASK_FEATHER:
        raise ToolError("mask_feather requires mask_boxes or mask_path to also be set.")

    # With a mask, 0.4 is not a neutral choice: it starts the masked region 40% of the way
    # along the schedule, leaving the object that was meant to be replaced very nearly
    # untouched -- and reporting success, having spent a full generation. Measured on a
    # 768x768 Z-Image-Turbo run against the same request at 0.0: 5.1/255 of change inside
    # the mask, against 41.6.
    #
    # engine.py now defaults this as well (DEFAULT_MASKED_IMAGE_STRENGTH), so this is a
    # second copy on purpose rather than the only one. This file has to keep working
    # against a server that predates that -- the same reason it treats a missing /health
    # field as "unknown, let the server decide" rather than an error. The two cannot
    # disagree, and a literal 0.0 is what has to go on the wire: None is the value that
    # routes to 0.4 on an older server.
    if mask_b64 is not None and image_strength is None:
        image_strength = 0.0

    _prune_jobs()
    job = _Job(
        handle=uuid.uuid4().hex[:8],
        width=width,
        height=height,
        steps=steps,
        model=(info or {}).get("label", ""),
        started=time.monotonic(),
    )
    _JOBS[job.handle] = job
    job.task = asyncio.create_task(
        _run(
            job,
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "guidance": guidance,
                "negative_prompt": negative_prompt,
                "preview_every": 0,
                "stream": True,
                "image": image_b64,
                "image_strength": image_strength,
                "fractional_start": fractional_start,
                "mask": mask_b64,
                # Both only describe how a mask is applied, so the server rejects a
                # non-default either side of one. This tool's own mask_feather default
                # is non-zero, which makes a maskless call the case that needs the care.
                "mask_feather": mask_feather if mask_b64 is not None else 0,
            },
        )
    )

    await _await_job(job, ctx, WAIT_SECONDS)
    return _result(job)


@server.tool()
async def check_image(
    handle: str,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Collect an image from a generate_image call that returned a handle.

    Blocks until the image is ready or the wait window elapses, whichever comes first,
    so a "still generating" answer means it genuinely isn't done yet -- call again with
    the same handle. Handles stay collectable for a while after they finish.
    """
    job = _JOBS.get(handle)
    if job is None:
        known = ", ".join(f"{h} ({j.status()})" for h, j in _JOBS.items()) or "none"
        raise ToolError(f"No generation with handle {handle!r}. Known handles: {known}")

    await _await_job(job, ctx, WAIT_SECONDS)
    return _result(job)


if __name__ == "__main__":
    server.run()
