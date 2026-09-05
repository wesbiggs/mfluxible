# MCP tool (generate images from within Claude)

`clients/mcp_server.py` exposes two tools over MCP's stdio transport: `generate_image(prompt, width, height, steps, seed, guidance, negative_prompt, image_path, image_strength, fractional_start)`, which forwards each `thinking` step as an MCP progress update and returns the image as inline content, and `check_image(handle)`, which collects an image from a `generate_image` call that outlived its tool-call timeout (see below).

`image_path` is a local file path (read from disk by the tool, not a URL) for image-to-image; `image_strength` (0.0–1.0, only meaningful alongside `image_path`, server default 0.4 if omitted) follows mflux's own convention — see [Image-to-image](api.md#image-to-image) — which is the *inverse* of "denoising strength" in some other tools, so the tool's own docstring spells this out for the model calling it. `fractional_start` is the same [flag](api.md#fractional-start) the API takes, described there as the thing to reach for when a user is tuning strength finely or asking why a small change to it did nothing.

```bash
uv pip install -r clients/requirements-mcp.txt
```

You still need the server from the [Quickstart](../README.md#quickstart) running separately with the model loaded — this only proxies to it.

Two limits imposed by the MCP host shape this tool, and neither is something the server can lift on its own:

- **A wall-clock timeout on each tool call**, shorter than a generation. Measured against a live Claude Code session on 2026-08-31, a `generate_image` call died at ~60s with the MCP SDK's default `Request timed out` — *despite* this server sending a progress notification every ~8s throughout. Progress notifications are worth sending (some hosts do reset on them; the Claude Code CLI runs a separate 30-minute idle watchdog that they rearm) but a server that relies on them is betting on the host, and that bet loses on at least one real host today.

  So generation runs in a background task and `generate_image` blocks on it for at most `MFLUXIBLE_MCP_WAIT_SECONDS` (default 45, comfortably under that 60s). A generation that beats the window returns its image from the first call, exactly as before. A slower one returns a handle instead, and `check_image(handle)` collects it — that call blocks for the same window and returns the image the moment it's ready, so the model calls it again if it comes back "still generating" rather than spinning. **The generation itself is never cancelled by a host giving up**: it keeps running, the full-resolution PNG still lands on disk, and the result stays collectable under its handle for 15 minutes (`MFLUXIBLE_MCP_JOB_RETENTION_S`).

  How many round-trips that costs is a question of resolution more than step count: a chunk of fixed cost outside the per-step loop (text encoding, VAE decode) scales with pixel count, not step count, so trimming resolution buys back more wall-clock time than trimming steps does — and at less cost to output quality than cutting steps. Hence the **768×768** default rather than the HTTP API's 1024×1024 — a latency default, not a correctness one, since a long generation here costs extra `check_image` calls rather than failing outright. Raise `MFLUXIBLE_MCP_WIDTH`/`_HEIGHT` (or `MFLUXIBLE_MCP_WAIT_SECONDS`, if your host's timeout is generous) to trade round-trips back for size, or pass explicit `width`/`height` per call. Actual timings are worth measuring on your own machine and model rather than trusting a hardcoded figure here — watch `step_ms`/`elapsed_ms` on `thinking` events (see [Streaming response](api.md#streaming-response-stream-true-default)) for a live read.
- **A ~1MB cap on a single tool result.** MCP ships images as base64, which inflates bytes by 4/3, so the raw image has to land near 750KB. A full-resolution 1024×1280 PNG off this model is ~1.8MB (~2.4MB base64) — about 2.4× over. The tool now re-encodes to fit: PNG is returned untouched when it's already small enough, otherwise it steps down JPEG quality first and only then resolution. In practice quality alone is enough and resolution is never touched: a real 1024×1280 generation measured 1.75MB as PNG and 0.21MB as JPEG q85 at unchanged dimensions — comfortably inside the budget — and even a pathological 3.9MB noise PNG still fits at full size, at q70. So images come back at the resolution you asked for, just recompressed.

Because the inline copy may be recompressed, the untouched full-resolution PNG (mflux metadata intact) is always written to `MFLUXIBLE_MCP_SAVE_DIR` (default `~/Pictures/mfluxible`) first, and the tool returns that path alongside the image.

The image is tagged with MCP's `annotations.audience`/`priority` display hints, which ask the host to surface it to the user rather than bury it in the collapsed tool-result block. Claude Desktop has been observed honoring them and rendering the image in the main transcript. This is a hint, though, not a guarantee: the protocol has no way to *require* main-transcript rendering, so other hosts may still collapse it.

Claude Code and Claude Desktop each keep their own MCP configuration and don't share a registry, so registering the tool with one has no effect on the other. Set up whichever you use, or both.

## Claude Code

```bash
claude mcp add mfluxible --scope user -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/clients/mcp_server.py
```

(`--scope user` makes it available in every project; drop it to register for just the current project.) This writes to `~/.claude.json`; `claude mcp list` shows what got registered and whether it connects. Claude Code starts `mcp_server.py` itself when needed.

## Claude Desktop

`claude mcp add` does **not** register anything with Desktop — Desktop reads its own file, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS. Open it via **Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "mfluxible": {
      "command": "/path/to/mfluxible/.venv/bin/python",
      "args": ["/path/to/mfluxible/clients/mcp_server.py"]
    }
  }
}
```

Merge that entry into the existing `mcpServers` object if the file already lists other servers. Both paths have to be absolute — Desktop launches stdio servers with a minimal environment that doesn't inherit your shell's `PATH`, so a bare `python` or a relative path fails to resolve. Then quit Desktop completely (⌘Q; closing the window leaves the process running) and reopen it.

The tool appears under Developer/Extensions and in the composer's tool menu — not under **Connectors**, which lists remote OAuth connectors only, so a local stdio server like this one will never show up there.

Desktop also reports `mfluxible` as connected as soon as `mcp_server.py` starts, which says nothing about whether the HTTP server it proxies to is up. If that server isn't running, you'll only find out when a `generate_image` call fails.

Either way, if the server is on a different host/port, point the proxy at it with `MFLUXIBLE_URL` (defaults to `http://127.0.0.1:8420/mfluxible/v1/images/generations`): as `-e MFLUXIBLE_URL=...` on the `claude mcp add` command, or as an `"env"` object alongside `command`/`args` in Desktop's config.

## Configuration

Environment variables for `clients/mcp_server.py`, all optional. Set them where the tool is registered (`-e` on `claude mcp add`, or an `"env"` object in Claude Desktop's config).

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_URL` | `http://127.0.0.1:8420/mfluxible/v1/images/generations` | Which mfluxible server to proxy to |
| `MFLUXIBLE_HEALTH_URL` | `/health` on the same host | Read once to name the model and reject arguments it can't act on; only needed if `/health` isn't alongside the generations endpoint |
| `MFLUXIBLE_MCP_WIDTH` | `768` | Default width, kept below the API's own default so most generations finish in one round-trip |
| `MFLUXIBLE_MCP_HEIGHT` | `768` | Default height, same reason |
| `MFLUXIBLE_MCP_STEPS` | unset | Step count to send. Unset lets the server use its model's own default, which is normally what you want |
| `MFLUXIBLE_MCP_WAIT_SECONDS` | `45` | How long one tool call blocks before handing back a `check_image` handle; keep it under the host's tool-call timeout |
| `MFLUXIBLE_MCP_JOB_RETENTION_S` | `900` | How long a finished generation stays collectable by handle |
| `MFLUXIBLE_MCP_MAX_BYTES` | `700000` | Raw-byte budget for the inline image, sized so base64 clears the host's ~1MB result cap |
| `MFLUXIBLE_MCP_SAVE_DIR` | `~/Pictures/mfluxible` | Where the untouched full-resolution PNG is written |
