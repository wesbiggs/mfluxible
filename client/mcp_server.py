"""MCP server exposing mfluxible's Z-Image-Turbo generation as a tool Claude can call directly.

Runs over stdio (the transport Claude Code/Desktop use to launch local MCP servers)
and proxies to a already-running mfluxible HTTP server, forwarding step progress via
MCP's progress-reporting mechanism and returning the final image as inline content.

Requires the mfluxible HTTP server (server.py) to already be running separately --
this doesn't load the model itself, just calls the API.

Two constraints from the MCP host shape the defaults here, and neither is something
this file can lift on its own -- see the comments on MAX_RESULT_BYTES and on the
width/height defaults below.
"""

import base64
import io
import json
import os
import time
from pathlib import Path

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp_types import Annotations, ImageContent, TextContent
from PIL import Image as PILImage

MFLUXIBLE_URL = os.environ.get("MFLUXIBLE_URL", "http://127.0.0.1:8420/v1/images/generations")

# Claude caps a single tool result at ~1MB, and MCP ships images as base64, which
# inflates bytes by 4/3. So the *raw* image has to come in around 750KB to clear
# the cap once encoded; 700KB leaves room for the surrounding JSON. A full-res
# 1024x1280 PNG off this model runs ~1.8MB raw / ~2.4MB base64 -- roughly 2.4x
# over -- so anything at that size has to be re-encoded before it can be returned.
MAX_RESULT_BYTES = int(os.environ.get("MFLUXIBLE_MCP_MAX_BYTES", 700_000))

# Deliberately smaller than the HTTP API's own 1024x1024 default. Generation is
# linear in pixel count, and MCP hosts apply a wall-clock timeout to each tool
# call that this server cannot negotiate or extend -- it can only report progress
# and hope the host resets on it. 768x768 is ~2.2x fewer pixels than 1024x1280,
# which is the difference between comfortably inside a typical timeout and well
# outside it. Raise these if your host's timeout is generous; ask for explicit
# width/height in the call to override per-request.
DEFAULT_WIDTH = int(os.environ.get("MFLUXIBLE_MCP_WIDTH", 768))
DEFAULT_HEIGHT = int(os.environ.get("MFLUXIBLE_MCP_HEIGHT", 768))
DEFAULT_STEPS = int(os.environ.get("MFLUXIBLE_MCP_STEPS", 9))

# The returned image may be downscaled/recompressed to fit MAX_RESULT_BYTES, so the
# untouched full-resolution PNG (metadata and all) is always written here first.
# Deliberately not under ~/.cache alongside MFLUXIBLE_MODEL_DIR: cache directories are
# reasonably treated as disposable, and this is the only full-quality copy that exists.
SAVE_DIR = Path(os.environ.get("MFLUXIBLE_MCP_SAVE_DIR", "~/Pictures/mfluxible")).expanduser()

server = MCPServer(name="mfluxible")


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

    The full-resolution PNG is always saved to disk and its path returned; the inline
    copy may be downscaled to fit the host's tool-result size cap.
    """
    body = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "preview_every": 0,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client, client.stream("POST", MFLUXIBLE_URL, json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])

            if event["type"] == "thinking" and ctx is not None:
                # Hosts that honor progress notifications extend their tool timeout on
                # each one; this is the only lever this server has over that timeout.
                await ctx.report_progress(
                    event["step"],
                    event["total_steps"],
                    f"step {event['step']}/{event['total_steps']} ({event['step_ms']}ms)",
                )
            elif event["type"] == "image":
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
                    f"{width}x{height}, {steps} steps, seed {event['seed']}, "
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
            elif event["type"] == "error":
                raise RuntimeError(event["message"])

    raise RuntimeError("mfluxible stream ended without producing an image")


if __name__ == "__main__":
    server.run()
