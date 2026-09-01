"""MCP server exposing mfluxible's Z-Image-Turbo generation as a tool Claude can call directly.

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

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import Annotations, ImageContent, TextContent
from PIL import Image as PILImage

MFLUXIBLE_URL = os.environ.get("MFLUXIBLE_URL", "http://127.0.0.1:8420/v1/images/generations")

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
DEFAULT_STEPS = int(os.environ.get("MFLUXIBLE_MCP_STEPS", 9))

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
    steps: int
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
    caption = (
        f"{job.width}x{job.height}, {job.steps} steps, seed {event['seed']}, "
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
        async with httpx.AsyncClient(timeout=None) as client, client.stream("POST", MFLUXIBLE_URL, json=body) as resp:
            resp.raise_for_status()
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


@server.tool()
async def generate_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt using Z-Image-Turbo.

    width/height must be divisible by 8. Generation time scales with pixel count and
    can take minutes, so prefer the defaults unless the user asks for a specific size.
    Z-Image-Turbo's normal step range is single digits (default 9). Leave seed unset
    for a random one.

    Returns the image directly if it finishes quickly. Otherwise it returns a handle
    and keeps generating in the background: call check_image with that handle to
    collect the image, repeating until it comes back.

    The full-resolution PNG is always saved to disk and its path returned; the inline
    copy may be recompressed to fit the host's tool-result size cap.
    """
    _prune_jobs()
    job = _Job(
        handle=uuid.uuid4().hex[:8],
        width=width,
        height=height,
        steps=steps,
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
                "preview_every": 0,
                "stream": True,
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
