# mfluxible

A minimal streaming HTTP API for Z-Image-Turbo image generation on Apple Silicon, built on [mflux](https://github.com/filipstrand/mflux).

Rather than a full node-graph tool (ComfyUI) or a proprietary format (Draw Things), this exposes a small API in the same spirit as a chat-completions endpoint: each denoising step streams as a "thinking" event while generation happens, with optional in-progress preview images (Draw Things-style), followed by the final image.

## Layout

```
server/   the model + HTTP API (FastAPI)
client/   everything that talks to it: terminal scripts, a browser harness, an MCP tool for Claude
```

Nothing in `client/` needs `server/`'s dependencies (mflux, PyTorch, etc.) or vice versa — install only what you need for what you're doing.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

uvicorn server:app --app-dir server --host 127.0.0.1 --port 8420
```

The model loads on startup, before the server accepts any requests. On first run this downloads the Z-Image-Turbo weights from Hugging Face ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)) — expect a sizable one-time download — then quantizes them and caches the quantized copy (see [Model cache](#model-cache) below); both only happen once.

Once it's running:

```bash
curl -N -X POST http://127.0.0.1:8420/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a puffin on a cliff at sunset", "preview_every": 2}'
```

or use one of the clients below for something more visual.

## Clients

All three live in `client/` and talk to the server over HTTP — none of them load the model themselves, so the server above must already be running.

### Terminal

`stream_client.py` (install `client/requirements.txt` first — just `requests`) or the dependency-free `stream_client.js` (Node 18+) render step progress and previews inline in the terminal as they stream, saving the final image to disk:

```bash
python client/stream_client.py "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
# or
node client/stream_client.js "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
```

Both render the exact full-resolution bytes returned by the server — no downscaling, no recompression, nothing client-side touches the image data. They use the iTerm2 inline-image protocol's chunked `MultipartFile` variant (also works in WezTerm; in an unsupported terminal the escape codes are just ignored, and the saved file and progress text still work either way), the same variant iTerm2's own [`imgcat`](https://github.com/gnachman/iTerm2-shell-integration/blob/master/utilities/imgcat) reference tool uses by default: the base64 payload is split into `FilePart=` sequences behind a metadata-only header and a `FileEnd` marker, rather than one giant `File=...:<base64>` sequence.

This matters because iTerm2's own source caps how much data it'll accumulate for a *single* OSC escape sequence at 1,048,576 bytes ([`VT100XtermParser.m`](https://github.com/gnachman/iTerm2/blob/master/sources/VT100/VT100XtermParser.m)) — past that it truncates rather than cleanly dropping the sequence, which can corrupt what renders afterward too, not just fail to show the one image. Diffusion output is detailed/photographic content that a full-resolution PNG can realistically approach or cross that limit for. Chunking (500,000 bytes/chunk here — `imgcat`'s own 200-byte default exists specifically to survive tmux, which doesn't apply since neither script wraps for tmux) means no image, at any size or detail level, can hit that cap.

Images render at `width=auto` (height defaults to auto too) — the same default `imgcat` uses: native pixel dimensions divided by the display's backing scale factor (e.g. a 1024px image renders at 512pt on a 2x/Retina display), rather than a fixed cell-count width that would scale with the terminal's font size instead of the image's actual dimensions.

Not tmux-aware — iTerm2's protocol needs extra passthrough wrapping inside tmux that these scripts don't do.

### Browser

`client/harness.html` is a small, dependency-free page (plain HTML/CSS/JS, no build step) with a form for prompt/width/height/steps/seed/preview_every that calls the streaming endpoint directly from the browser via `fetch`, parsing the SSE stream the same way the terminal clients do, and renders previews and the final image as `<img>` elements (via `data:` URLs) plus a download link for the final PNG.

It needs to be served over HTTP, not opened as a `file://` URL — the browser's `Origin` header for a local file is `null`, which the server's default CORS config won't match:

```bash
cd client && python3 -m http.server 8000
# then open http://localhost:8000/harness.html
```

(CORS is on by default and reflects back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin, so this works with no server-side configuration — see [CORS](#cors) below if you need something different.)

### MCP tool (generate images from within Claude)

`client/mcp_server.py` exposes a single tool, `generate_image(prompt, width, height, steps, seed)`, over MCP's stdio transport, forwarding each `thinking` step as an MCP progress update and returning the final image as inline content.

```bash
pip install -r client/requirements-mcp.txt
```

You still need the server from Quickstart running separately with the model loaded — this only proxies to it.

Two limits imposed by the MCP host shape this tool's defaults, and neither is something the server can lift on its own:

- **A wall-clock timeout on each tool call.** Generation time scales with pixel count, and step count is not the useful lever — measured on an M2 Pro at 9 steps:

  | Size | Time | PNG |
  |---|---|---|
  | 1024×1280 | 249s (~27.7s/step) | 1.75MB |
  | 1024×1024 | 212s (~23.6s/step) | 1.50MB |
  | 768×768 | 115s (~12.8s/step) | 0.85MB |

  At ~27s/step even a 4-step run at 1024×1280 lands near 110s, so trimming steps doesn't rescue a large image. The tool therefore defaults to **768×768** rather than the HTTP API's 1024×1024. Each `thinking` step is forwarded as an MCP progress notification, which is the only lever the server has here — hosts that reset their timeout on progress will tolerate much longer runs, but that's the host's choice, not the server's. Raise `MFLUXIBLE_MCP_WIDTH`/`_HEIGHT` if your host is generous, or pass explicit `width`/`height` per call.
- **A ~1MB cap on a single tool result.** MCP ships images as base64, which inflates bytes by 4/3, so the raw image has to land near 750KB. A full-resolution 1024×1280 PNG off this model is ~1.8MB (~2.4MB base64) — about 2.4× over. The tool now re-encodes to fit: PNG is returned untouched when it's already small enough, otherwise it steps down JPEG quality first and only then resolution. In practice quality alone is enough and resolution is never touched: a real 1024×1280 generation measured 1.75MB as PNG and 0.21MB as JPEG q85 at unchanged dimensions — comfortably inside the budget — and even a pathological 3.9MB noise PNG still fits at full size, at q70. So images come back at the resolution you asked for, just recompressed.

Because the inline copy may be recompressed, the untouched full-resolution PNG (mflux metadata intact) is always written to `MFLUXIBLE_MCP_SAVE_DIR` (default `~/.cache/mfluxible/outputs`) first, and the tool returns that path alongside the image.

The image is tagged with MCP's `annotations.audience`/`priority` display hints, which ask the host to surface it to the user rather than bury it in the collapsed tool-result block. Whether a host honors those hints is up to the host — the protocol has no way to *require* main-transcript rendering.

Claude Code and Claude Desktop each keep their own MCP configuration and don't share a registry, so registering the tool with one has no effect on the other. Set up whichever you use, or both.

#### Claude Code

```bash
claude mcp add mfluxible --scope user -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/client/mcp_server.py
```

(`--scope user` makes it available in every project; drop it to register for just the current project.) This writes to `~/.claude.json`; `claude mcp list` shows what got registered and whether it connects. Claude Code starts `mcp_server.py` itself when needed.

#### Claude Desktop

`claude mcp add` does **not** register anything with Desktop — Desktop reads its own file, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS. Open it via **Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "mfluxible": {
      "command": "/path/to/mfluxible/.venv/bin/python",
      "args": ["/path/to/mfluxible/client/mcp_server.py"]
    }
  }
}
```

Merge that entry into the existing `mcpServers` object if the file already lists other servers. Both paths have to be absolute — Desktop launches stdio servers with a minimal environment that doesn't inherit your shell's `PATH`, so a bare `python` or a relative path fails to resolve. Then quit Desktop completely (⌘Q; closing the window leaves the process running) and reopen it.

The tool appears under Developer/Extensions and in the composer's tool menu — not under **Connectors**, which lists remote OAuth connectors only, so a local stdio server like this one will never show up there.

Desktop also reports `mfluxible` as connected as soon as `mcp_server.py` starts, which says nothing about whether the HTTP server it proxies to is up. If that server isn't running, you'll only find out when a `generate_image` call fails.

Either way, if the server is on a different host/port, point the proxy at it with `MFLUXIBLE_URL` (defaults to `http://127.0.0.1:8420/v1/images/generations`): as `-e MFLUXIBLE_URL=...` on the `claude mcp add` command, or as an `"env"` object alongside `command`/`args` in Desktop's config.

## API

FastAPI auto-generates interactive docs for all of this at `/docs` (Swagger UI) and `/redoc` once the server is running.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true|false}`. Useful for waiting on startup (weight download + quantization can take a while the first time) before sending a generation request.

### `POST /v1/images/generations`

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | |
| `width` | int | 1024 | must be divisible by 8 |
| `height` | int | 1024 | must be divisible by 8 |
| `steps` | int | 9 | Z-Image-Turbo's normal range is single digits |
| `seed` | int or null | random | echoed back in the response so a run can be reproduced |
| `preview_every` | int | 0 | decode and stream an in-progress preview every N steps; 0 disables previews. Each preview is a full VAE decode, so this trades speed for visibility |
| `stream` | bool | true | SSE stream vs a single JSON response |

### Streaming response (`stream: true`, default)

`text/event-stream`, one JSON object per `data:` line:

```
data: {"type": "start", "seed": 123, "total_steps": 9}

data: {"type": "thinking", "step": 1, "total_steps": 9, "elapsed_ms": 210}

data: {"type": "thinking", "step": 2, "total_steps": 9, "elapsed_ms": 400, "preview": "<base64 png>"}

...

data: {"type": "image", "mime_type": "image/png", "data": "<base64 png>", "seed": 123, "generation_time": 14.2}
```

`preview` is only present on steps where `preview_every` divides the step number, and — like `data` on the final `image` event — is always the full requested resolution; the server never downscales anything (that's a client concern — see [Clients](#clients) above). An `{"type": "error", "message": "..."}` event replaces the final `image` event if generation fails or is interrupted.

The final image's PNG bytes (`data` on the `image` event) carry embedded metadata — prompt, seed, steps, model, quantization, LoRA config (if any), and generation time — as EXIF (`UserComment`), XMP, and IPTC all at once, for broad tool compatibility (readable with e.g. `exiftool output.png`; macOS's built-in `sips`/`mdls` don't surface it). This is mflux's own metadata pipeline (`GeneratedImage.save()`), not something reimplemented here. Step previews aren't touched — only the final image is worth the overhead.

### Non-streaming response (`stream: false`)

Returns the same `image` event object as a single JSON body (or a 500 with the `error` object on failure).

## Configuration

Environment variables for `server.py`, all optional.

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_QUANTIZE` | `8` | Quantization bits; try `4` for less memory, `none` for full precision |
| `MFLUXIBLE_MODEL_DIR` | `~/.cache/mfluxible` | Where quantized weights are cached (see [Model cache](#model-cache)) |
| `MFLUXIBLE_LORA_PATHS` | unset | Comma-separated local LoRA `.safetensors` files to bake in (see [LoRAs](#loras)) |
| `MFLUXIBLE_LORA_SCALES` | `1.0` each | Comma-separated scales matching `MFLUXIBLE_LORA_PATHS` |
| `MFLUXIBLE_CORS_ORIGIN_REGEX` | `https?://(localhost\|127\.0\.0\.1)(:\d+)?` | Origins to reflect back in CORS (see [CORS](#cors)) |
| `MFLUXIBLE_CORS_ORIGINS` | unset | Comma-separated exact-match origins, in addition to the regex |

### MCP tool

Environment variables for `client/mcp_server.py`, all optional. Set them where the tool is registered (`-e` on `claude mcp add`, or an `"env"` object in Claude Desktop's config).

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_URL` | `http://127.0.0.1:8420/v1/images/generations` | Which mfluxible server to proxy to |
| `MFLUXIBLE_MCP_WIDTH` | `768` | Default width, kept below the API's own default to fit host tool-call timeouts |
| `MFLUXIBLE_MCP_HEIGHT` | `768` | Default height, same reason |
| `MFLUXIBLE_MCP_STEPS` | `9` | Default step count |
| `MFLUXIBLE_MCP_MAX_BYTES` | `700000` | Raw-byte budget for the inline image, sized so base64 clears the host's ~1MB result cap |
| `MFLUXIBLE_MCP_SAVE_DIR` | `~/.cache/mfluxible/outputs` | Where the untouched full-resolution PNG is written |

### Model cache

Startup quantizes the raw downloaded weights and caches the result to `MFLUXIBLE_MODEL_DIR/z-image-turbo-q<bits>/` — a real, separate step from the Hugging Face download: HF already caches the raw weights locally, but quantizing them into MLX's packed format is nontrivial per-layer compute that would otherwise happen on every startup (mflux's `mflux-save` mechanism does the same thing via its own CLI; this just does it automatically here). Every startup after the first loads the pre-quantized weights directly and skips that step.

This means a second, smaller copy of the weights lives on disk alongside HF's cache of the original — once you've confirmed a cached startup works, the original HF cache entry (under `~/.cache/huggingface`) is no longer needed and can be deleted to reclaim space.

### LoRAs

```bash
MFLUXIBLE_LORA_PATHS="/path/to/style.safetensors" MFLUXIBLE_LORA_SCALES="0.8" \
  uvicorn server:app --app-dir server --host 127.0.0.1 --port 8420
```

LoRA weights are applied and permanently merged ("baked") into the model at load time — this is mflux's own design, not a limitation added here. That makes it a server-startup choice, not a per-request one: one running server has one fixed LoRA configuration (or none), and switching LoRAs means restarting the server with different env vars, the same way `MFLUXIBLE_QUANTIZE` works. Each distinct combination of LoRA paths/scales gets its own model-cache directory (named with a hash of that exact config), so switching between a few LoRA setups doesn't require re-quantizing each time you switch back.

### CORS

On by default, reflecting back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin — so a local static server on either hostname, any port (e.g. for `harness.html`) can call the API with no extra configuration. `MFLUXIBLE_CORS_ORIGIN_REGEX` overrides the pattern entirely; `MFLUXIBLE_CORS_ORIGINS` adds specific exact-match origins on top of it (e.g. for a deployed frontend on a real domain).

## Running on a dedicated machine

The server doesn't have to run on the same machine as the clients. Point it at a spare Apple Silicon box (a Mac Mini, say) and bind it to the network instead of loopback:

```bash
uvicorn server:app --app-dir server --host 0.0.0.0 --port 8420
```

Then everything else just points at that host instead of `127.0.0.1`, no code changes needed:

- `stream_client.py` / `stream_client.js`: `--url http://mac-mini.local:8420/v1/images/generations`
- `mcp_server.py`: set `MFLUXIBLE_URL=http://mac-mini.local:8420/v1/images/generations` when registering it, e.g. `claude mcp add mfluxible --scope user -e MFLUXIBLE_URL=http://mac-mini.local:8420/v1/images/generations -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/client/mcp_server.py` — or, in Claude Desktop's config, `"env": {"MFLUXIBLE_URL": "http://mac-mini.local:8420/v1/images/generations"}` alongside `command`/`args`

There's no authentication on the API — only bind it to `0.0.0.0` on a network you trust (home LAN, Tailscale/VPN), never expose it directly to the internet.

## How it works

mflux's `generate_image()` is synchronous: it runs the whole denoising loop in one thread and invokes registered callbacks at each step (`InLoopCallback`). The server runs that call on a dedicated single-worker thread (not just any worker thread — see the comment at the top of `server/engine.py` for why that matters with MLX) and bridges each callback invocation back to the event loop as an SSE event via `call_soon_threadsafe`, so a single async server can stream progress out of an otherwise blocking call.

Only one generation runs at a time (there's a lock) — MLX/Metal generation against one shared model instance isn't set up here for concurrency.

Preview images are decoded the same way mflux's own `--stepwise-image-output-dir` CLI flag does internally (see `mflux.callbacks.instances.stepwise_handler.StepwiseHandler`), just streamed instead of written to disk.

## Model

Currently hardcoded to Z-Image-Turbo (`mflux.models.z_image.variants.z_image.ZImage`). mflux also supports FLUX and Qwen-Image; adding one here means swapping the model class in `server/engine.py` — not done yet since only Z-Image-Turbo was needed for this project.

## License

[Apache 2.0](LICENSE).
