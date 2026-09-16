# MCP tool (generate images from an MCP client)

`clients/mcp_server.py` exposes two tools over MCP's stdio transport: `generate_image(prompt, width, height, steps, seed, guidance, negative_prompt, image_path, image_strength, fractional_start, mask_boxes, mask_path, mask_feather)`, which forwards each `thinking` step as an MCP progress update and returns the image as inline content, and `check_image(handle)`, which collects an image from a `generate_image` call that outlived its tool-call timeout (see below).

`image_path` is a local file path (read from disk by the tool, not a URL) for image-to-image; `image_strength` (0.0–1.0, only meaningful alongside `image_path`, server default 0.4 if omitted) follows mflux's own convention — see [Image-to-image](api.md#image-to-image) — which is the *inverse* of "denoising strength" in some other tools, so the tool's own docstring spells this out for the model calling it. `fractional_start` is the same [flag](api.md#fractional-start) the API takes, described there as the thing to reach for when a user is tuning strength finely or asking why a small change to it did nothing.

### Masks

`mask_boxes` and `mask_path` are the two ways to [inpaint](api.md#inpainting-a-region) — regenerate one region and hold the rest of the frame — and a call passes one or the other, never both, always alongside `image_path`. Everything else about masking is the API's behaviour unchanged, including `mask_composite`, which this tool doesn't expose and the server therefore leaves on.

**Two defaults differ from the API's, both because an omitted argument is the common case here.** A caller writing JSON has read that page; the model calling this tool has read a docstring and will mostly pass neither field at all. `mask_feather` is `8` rather than `0`. And `image_strength` is **`0.0` when a mask is present** rather than the server's `0.4` — which is not a tuning preference but the difference between inpainting and a no-op. At `0.4` the masked region starts 40% of the way along the schedule and the object meant to be replaced comes back very nearly intact, after a full generation that reports success. Measured on a 768×768 Z-Image-Turbo run: 5.1/255 of change inside the mask at `0.4`, against 41.6 for the identical request at `0.0`. An explicit `image_strength` is still honoured, for restyling a region rather than replacing it.

**The `prompt` describes the whole finished frame, not the masked region.** The model denoises the entire image and can see the part it is holding, so the prompt is what decides how the new content sits in the scene it is being dropped into — `"a toy dinosaur on a wooden desk beside a notebook"`, not `"a toy dinosaur"`. This one is a preference rather than a rule, and measurably so: the same run with a bare region prompt produced a perfectly good image, just a different one (38.4/255 apart inside the mask). Enough is inferred from the held context that it usually works; what it gives up is the caller's control over which image comes back.

**`mask_boxes` takes fractions of the image, not pixels** — a list of `[x0, y0, x1, y1]` rectangles reading left, top, right, bottom, each between `0.0` and `1.0`, which the tool rasterizes into a mask at the input image's own size. That it's normalized is the load-bearing part, and the reason is [`MFLUXIBLE_MCP_MAX_BYTES`](#configuration): the inline copy of a returned image is downscaled to fit the host's result cap, so a model that generates an image and then masks part of it has been *looking at* a smaller frame than the full-resolution PNG `image_path` points back at. Pixel coordinates read off the former would address the wrong region of the latter, quietly, and scaled by a factor only the caption's note ever mentions. A fraction means the same thing in both frames.

This is what makes masking usable from a host where the model has no filesystem of its own: picking a rectangle off an image it can see is something it can do unaided, whereas authoring a mask PNG needs tools this server doesn't provide. `mask_path` covers the other case — a mask some other step already drew, `harness.html` included — and must match the input image's pixel size exactly, measured *after* EXIF orientation, since that's the frame mflux rotates the input into before encoding it.

Masks are per-checkpoint in the same way guidance is (`supports_mask` on `/health`), so the tool rejects one up front on a model that can't take it, with the same local precheck it uses for `guidance` and `negative_prompt`.

```bash
uv pip install -r clients/requirements-mcp.txt
```

You still need the server from the [Quickstart](../README.md#quickstart) running separately with the model loaded — this only proxies to it.

## What the host imposes

Two limits come from the MCP client, and neither is something this server can lift on its own.

- **A wall-clock timeout on each tool call**, shorter than a generation. Measured against a live Claude Code session on 2026-08-31, a `generate_image` call died at ~60s with the MCP SDK's default `Request timed out` — *despite* this server sending a progress notification every ~8s throughout. Progress notifications are worth sending (some hosts do reset on them; the Claude Code CLI runs a separate 30-minute idle watchdog that they rearm) but a server that relies on them is betting on the host, and that bet loses on at least one real host today.

  So generation runs in a background task and `generate_image` blocks on it for at most `MFLUXIBLE_MCP_WAIT_SECONDS` (default 45). A generation that beats the window returns its image from the first call. A slower one returns a handle instead, and `check_image(handle)` collects it — that call blocks for the same window and returns the image the moment it's ready, so the model calls it again if it comes back "still generating" rather than spinning. **The generation itself is never cancelled by a host giving up**: it keeps running, the full-resolution PNG still lands on disk, and the result stays collectable under its handle for 15 minutes (`MFLUXIBLE_MCP_JOB_RETENTION_S`).

  That 45 suits a host that gives up around 60s. Hosts differ by a lot, and some are much less generous — see [Tested clients](#tested-clients) for the value each one needs. Keep it under the *shortest* timeout you care about; blocking longer only converts a working handle into a failed call.

  How many round-trips that costs is a question of resolution more than step count: a chunk of fixed cost outside the per-step loop (text encoding, VAE decode) scales with pixel count, not step count, so trimming resolution buys back more wall-clock time than trimming steps does — and at less cost to output quality than cutting steps. Hence the **768×768** default rather than the HTTP API's 1024×1024 — a latency default, not a correctness one, since a long generation here costs extra `check_image` calls rather than failing outright. Raise `MFLUXIBLE_MCP_WIDTH`/`_HEIGHT` (or `MFLUXIBLE_MCP_WAIT_SECONDS`, if your host's timeout is generous) to trade round-trips back for size, or pass explicit `width`/`height` per call. Actual timings are worth measuring on your own machine and model rather than trusting a hardcoded figure here — watch `step_ms`/`elapsed_ms` on `thinking` events (see [Streaming response](api.md#streaming-response-stream-true-default)) for a live read.
- **A ~1MB cap on a single tool result.** MCP ships images as base64, which inflates bytes by 4/3, so the raw image has to land near 750KB. A full-resolution 1024×1280 PNG off this model is ~1.8MB (~2.4MB base64) — about 2.4× over. The tool now re-encodes to fit: PNG is returned untouched when it's already small enough, otherwise it steps down JPEG quality first and only then resolution. In practice quality alone is enough and resolution is never touched: a real 1024×1280 generation measured 1.75MB as PNG and 0.21MB as JPEG q85 at unchanged dimensions — comfortably inside the budget — and even a pathological 3.9MB noise PNG still fits at full size, at q70. So images come back at the resolution you asked for, just recompressed.

Because the inline copy may be recompressed, the untouched full-resolution PNG (mflux metadata intact) is always written to `MFLUXIBLE_MCP_SAVE_DIR` (default `~/Pictures/mfluxible`) first, and the tool returns that path alongside the image.

The image is tagged with MCP's `annotations.audience`/`priority` display hints, which ask the host to surface it to the user rather than bury it in the collapsed tool-result block. This is a hint, not a guarantee: the protocol has no way to *require* main-transcript rendering, hosts are free to ignore it, and at least one drops the image entirely (see oMLX below). The saved PNG is the reliable copy.

## Tested clients

Each client keeps its own MCP registry, and none of them share one — registering the tool with one has no effect on the others. Set up whichever you use.

| Client | Where its MCP config lives | `MFLUXIBLE_MCP_WAIT_SECONDS` | Inline image |
|---|---|---|---|
| [Claude Code](#claude-code) | `~/.claude.json`, via `claude mcp add` | default (45) | yes, in the tool result |
| [Claude Desktop](#claude-desktop) | `claude_desktop_config.json` | default (45) | yes, in the main transcript |
| [oMLX](#omlx) | `~/.omlx/mcp.json` | **20** — its chat UI aborts at 30s | no, not as of 0.6.4 |

### Claude Code

```bash
claude mcp add mfluxible --scope user -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/clients/mcp_server.py
```

(`--scope user` makes it available in every project; drop it to register for just the current project.) This writes to `~/.claude.json`; `claude mcp list` shows what got registered and whether it connects. Claude Code starts `mcp_server.py` itself when needed.

The image comes back as inline tool content and renders in the transcript — a 512×512 run took 62s here, so it went out as a handle and `check_image` collected it on the first poll, which is the path worth exercising once before you trust a long generation to it.

The ~60s the default `MFLUXIBLE_MCP_WAIT_SECONDS` is sized against was measured here, not documented anywhere, and it belongs to one build: reading the CLI binary (Homebrew cask 2.1.236) also shows a separate idle watchdog that progress notifications *do* rearm, and auto-backgrounding of any call still running at 120s. Treat the numbers as this-build-today. If you set `MCP_TOOL_TIMEOUT`, note it lowers that idle watchdog too, and lower `MFLUXIBLE_MCP_WAIT_SECONDS` to match.

### Claude Desktop

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

Desktop honors the display hints above: the image has been observed rendering in the main transcript rather than inside the collapsed tool result.

Desktop also reports `mfluxible` as connected as soon as `mcp_server.py` starts, which says nothing about whether the HTTP server it proxies to is up. If that server isn't running, you'll only find out when a `generate_image` call fails.

### oMLX

[oMLX](https://omlx.ai/) is a local LLM inference server for Apple Silicon whose admin UI includes a chat that can call MCP tools. Tested against 0.6.4 (Homebrew).

It searches `./mcp.json`, `~/.config/omlx/mcp.json`, `$OMLX_MCP_CONFIG` and `--mcp-config` for its config; the menu-bar app instead records a path in `~/.omlx/settings.json` under `mcp.config_path`, normally `~/.omlx/mcp.json`. Either its own `servers` key or Claude Desktop's `mcpServers` is accepted:

```json
{
  "default_timeout": 300.0,
  "mcpServers": {
    "mfluxible": {
      "command": "/path/to/mfluxible/.venv/bin/python",
      "args": ["/path/to/mfluxible/clients/mcp_server.py"],
      "env": { "MFLUXIBLE_MCP_WAIT_SECONDS": "20" }
    }
  }
}
```

Three things differ from the Claude hosts:

- **`MFLUXIBLE_MCP_WAIT_SECONDS` has to be under 30.** The admin chat page aborts each tool call *in the browser* after 30 seconds (`TOOL_TIMEOUT_MS` in its `chat.html`) — hardcoded in the page, not read from `mcp.json`, and not exposed as a setting. The default 45 fails every call. 20 leaves room for `generate_image` to hand back a handle, and for each `check_image` poll to answer inside the window.
- **Raise `default_timeout` as well.** It defaults to 30.0 and is what bounds oMLX's own server-side wait on the tool. The per-server `timeout` key parses, but the admin chat path passes the top-level `default_timeout`, so that's the one to set.
- **Raise Max Tool Rounds** (chat settings, default 10) if you generate at larger sizes — each `check_image` poll spends one round.

Tools are namespaced by server, so the model sees `mfluxible__generate_image`. After editing `mcp.json`, restart oMLX or toggle MCP off and on in Settings: the `env` block is applied when the subprocess is spawned, so a reconnect is what picks it up.

As of 0.6.4 the admin chat doesn't display images returned by an MCP tool at all — tool results are kept out of the transcript, and an image block loses its media type on the way through, reaching the model as bare base64. The full-resolution PNG still lands in `MFLUXIBLE_MCP_SAVE_DIR` and the caption names that path, so the model can at least tell you where it is. [jundot/omlx#3596](https://github.com/jundot/omlx/pull/3596) proposes rendering them; [#3575](https://github.com/jundot/omlx/issues/3575) tracks feeding them to a vision model.

### Other clients

Nothing here is specific to the clients above — any MCP host that launches stdio servers can run this tool. Four things are worth checking on a new one:

1. **Where its MCP config lives**, and whether it uses `servers` or `mcpServers`. Most accept Claude Desktop's shape.
2. **Absolute paths.** Hosts commonly launch stdio servers with a minimal environment, so point at `/path/to/mfluxible/.venv/bin/python` rather than a bare `python`.
3. **Its per-call timeout**, which is the number `MFLUXIBLE_MCP_WAIT_SECONDS` has to stay under. If you can't find it documented, time a call that you know takes minutes and watch when it gives up — and remember a host may enforce one in its UI as well as in its MCP client, as oMLX does.
4. **Whether it renders inline images.** If it doesn't, the tool still works: the saved PNG path comes back in the caption.

Wherever the server is registered, if the HTTP server is on a different host or port, point the proxy at it with `MFLUXIBLE_URL` (defaults to `http://127.0.0.1:8420/mfluxible/v1/images/generations`): as `-e MFLUXIBLE_URL=...` on `claude mcp add`, or as an `"env"` object alongside `command`/`args` in a JSON config.

## Configuration

Environment variables for `clients/mcp_server.py`, all optional. Set them where the tool is registered (`-e` on `claude mcp add`, or an `"env"` object in a JSON config).

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_URL` | `http://127.0.0.1:8420/mfluxible/v1/images/generations` | Which mfluxible server to proxy to |
| `MFLUXIBLE_BEARER_TOKEN` | unset | Sent as `Authorization: Bearer <token>` on every call, for a server behind an auth proxy (see [Authentication](server.md#authentication)). Environment-only by necessity: an MCP host launches this as a stdio subprocess, so there's no command line to pass and no terminal to prompt at |
| `MFLUXIBLE_HEALTH_URL` | `/health` on the same host | Read once to name the model and reject arguments it can't act on; only needed if `/health` isn't alongside the generations endpoint |
| `MFLUXIBLE_MCP_WIDTH` | `768` | Default width, kept below the API's own default so most generations finish in one round-trip |
| `MFLUXIBLE_MCP_HEIGHT` | `768` | Default height, same reason |
| `MFLUXIBLE_MCP_STEPS` | unset | Step count to send. Unset lets the server use its model's own default, which is normally what you want |
| `MFLUXIBLE_MCP_WAIT_SECONDS` | `45` | How long one tool call blocks before handing back a `check_image` handle; keep it under the host's tool-call timeout (see [Tested clients](#tested-clients)) |
| `MFLUXIBLE_MCP_JOB_RETENTION_S` | `900` | How long a finished generation stays collectable by handle |
| `MFLUXIBLE_MCP_MAX_BYTES` | `700000` | Raw-byte budget for the inline image, sized so base64 clears the host's ~1MB result cap |
| `MFLUXIBLE_MCP_SAVE_DIR` | `~/Pictures/mfluxible` | Where the untouched full-resolution PNG is written |
| `MFLUXIBLE_MCP_MASK_FEATHER` | `8` | Feather applied to a mask when a call doesn't say; 8–16 is the useful range, and it is not a region blend (see [Inpainting a region](api.md#inpainting-a-region)) |
