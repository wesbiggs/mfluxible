"""MCP server exposing mfluxible's Z-Image-Turbo generation as a tool Claude can call directly.

Runs over stdio (the transport Claude Code/Desktop use to launch local MCP servers)
and proxies to a already-running mfluxible HTTP server, forwarding step progress via
MCP's progress-reporting mechanism and returning the final image as inline content.

Requires the mfluxible HTTP server (server.py) to already be running separately --
this doesn't load the model itself, just calls the API.
"""

import base64
import json
import os

import httpx
from mcp.server.mcpserver import Context, Image, MCPServer

MFLUXIBLE_URL = os.environ.get("MFLUXIBLE_URL", "http://127.0.0.1:8420/v1/images/generations")

server = MCPServer(name="mfluxible")


@server.tool()
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    seed: int | None = None,
    ctx: Context | None = None,
) -> Image:
    """Generate an image from a text prompt using Z-Image-Turbo.

    width/height must be divisible by 8. Z-Image-Turbo's normal step range is
    single digits (default 9). Leave seed unset for a random one.
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
                await ctx.report_progress(
                    event["step"],
                    event["total_steps"],
                    f"step {event['step']}/{event['total_steps']}",
                )
            elif event["type"] == "image":
                return Image(data=base64.b64decode(event["data"]), format="png")
            elif event["type"] == "error":
                raise RuntimeError(event["message"])

    raise RuntimeError("mfluxible stream ended without producing an image")


if __name__ == "__main__":
    server.run()
