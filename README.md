# mfluxible

A minimal streaming HTTP API for image generation on Apple Silicon, built on [mflux](https://github.com/filipstrand/mflux). Runs Z-Image-Turbo (the default), FLUX.1-schnell, FLUX.1-dev, or Qwen-Image — one model per server process, picked at startup (see [Models](#models)).

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

The model loads on startup, before the server accepts any requests. On first run this downloads its weights from Hugging Face — expect a sizable one-time download — then quantizes them and caches the quantized copy (see [Model cache](#model-cache) below); both only happen once.

The default model is Z-Image-Turbo ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)). To run something else, set `MFLUXIBLE_MODEL` — `flux-schnell`, `flux-dev`, or `qwen-image` — before starting the server; only the model you select is ever downloaded. See [Models](#models) for what differs between them.

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

Both take `--steps`, `--seed`, `--guidance` and `--negative-prompt`, and leave all four to the server when you don't pass them — so `--steps` is only worth setting to override the loaded model's own default. `--guidance` and `--negative-prompt` are refused (with a message naming the model) on models that can't act on them; see [Models](#models).

Both render the exact full-resolution bytes returned by the server — no downscaling, no recompression, nothing client-side touches the image data. They use the iTerm2 inline-image protocol's chunked `MultipartFile` variant (also works in WezTerm; in an unsupported terminal the escape codes are just ignored, and the saved file and progress text still work either way), the same variant iTerm2's own [`imgcat`](https://github.com/gnachman/iTerm2-shell-integration/blob/master/utilities/imgcat) reference tool uses by default: the base64 payload is split into `FilePart=` sequences behind a metadata-only header and a `FileEnd` marker, rather than one giant `File=...:<base64>` sequence.

This matters because iTerm2's own source caps how much data it'll accumulate for a *single* OSC escape sequence at 1,048,576 bytes ([`VT100XtermParser.m`](https://github.com/gnachman/iTerm2/blob/master/sources/VT100/VT100XtermParser.m)) — past that it truncates rather than cleanly dropping the sequence, which can corrupt what renders afterward too, not just fail to show the one image. Diffusion output is detailed/photographic content that a full-resolution PNG can realistically approach or cross that limit for. Chunking (500,000 bytes/chunk here — `imgcat`'s own 200-byte default exists specifically to survive tmux, which doesn't apply since neither script wraps for tmux) means no image, at any size or detail level, can hit that cap.

Images render at `width=auto` (height defaults to auto too) — the same default `imgcat` uses: native pixel dimensions divided by the display's backing scale factor (e.g. a 1024px image renders at 512pt on a 2x/Retina display), rather than a fixed cell-count width that would scale with the terminal's font size instead of the image's actual dimensions.

Not tmux-aware — iTerm2's protocol needs extra passthrough wrapping inside tmux that these scripts don't do.

### Browser

`client/harness.html` is a small, dependency-free page (plain HTML/CSS/JS, no build step) with a form for prompt/width/height/steps/seed/preview_every that calls the streaming endpoint directly from the browser via `fetch`, reads [`/health`](#get-health) on load to show which model the server is running (leaving Steps blank uses that model's default, and Guidance / Negative prompt appear only if it accepts them), parsing the SSE stream the same way the terminal clients do, and renders previews and the final image as `<img>` elements (via `data:` URLs) plus a download link for the final PNG.

It needs to be served over HTTP, not opened as a `file://` URL — the browser's `Origin` header for a local file is `null`, which the server's default CORS config won't match:

```bash
cd client && python3 -m http.server 8000
# then open http://localhost:8000/harness.html
```

(CORS is on by default and reflects back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin, so this works with no server-side configuration — see [CORS](#cors) below if you need something different.)

### MCP tool (generate images from within Claude)

`client/mcp_server.py` exposes two tools over MCP's stdio transport: `generate_image(prompt, width, height, steps, seed, guidance, negative_prompt)`, which forwards each `thinking` step as an MCP progress update and returns the image as inline content, and `check_image(handle)`, which collects an image from a `generate_image` call that outlived its tool-call timeout (see below).

```bash
pip install -r client/requirements-mcp.txt
```

You still need the server from Quickstart running separately with the model loaded — this only proxies to it.

Two limits imposed by the MCP host shape this tool, and neither is something the server can lift on its own:

- **A wall-clock timeout on each tool call**, shorter than a generation. Measured against a live Claude Code session on 2026-08-31, a `generate_image` call died at ~60s with the MCP SDK's default `Request timed out` — *despite* this server sending a progress notification every ~8s throughout. Progress notifications are worth sending (some hosts do reset on them; the Claude Code CLI runs a separate 30-minute idle watchdog that they rearm) but a server that relies on them is betting on the host, and that bet loses on at least one real host today.

  So generation runs in a background task and `generate_image` blocks on it for at most `MFLUXIBLE_MCP_WAIT_SECONDS` (default 45, comfortably under that 60s). A generation that beats the window returns its image from the first call, exactly as before. A slower one returns a handle instead, and `check_image(handle)` collects it — that call blocks for the same window and returns the image the moment it's ready, so the model calls it again if it comes back "still generating" rather than spinning. **The generation itself is never cancelled by a host giving up**: it keeps running, the full-resolution PNG still lands on disk, and the result stays collectable under its handle for 15 minutes (`MFLUXIBLE_MCP_JOB_RETENTION_S`).

  How many round-trips that costs is a question of resolution, not step count — measured on an M2 Pro at 9 steps:

  | Size | Steps | Time | PNG |
  |---|---|---|---|
  | 1024×1280 | 9 | 249s | 1.75MB |
  | 1024×1024 | 9 | 212s | 1.50MB |
  | 768×768 | 9 | 115s | 0.85MB |
  | 768×768 | 3 | 77s | |

  Times vary somewhat with prompt content and thermal state — treat them as ballpark, not benchmarks. Dividing them by step count overstates what a step costs. The two 768×768 rows pin the split: six fewer steps saved 38s, so the loop runs ~6.4s/step and ~57s is fixed cost outside it (text encoding, VAE decode) that no step count reduces — and 57 + 3 × 6.4 lands on the measured 77s. **Resolution, not step count, is the lever**: dropping 1024×1280 to 768×768 saves 134s, while going 9 steps to 3 saves 38s and costs quality. Hence the **768×768** default rather than the HTTP API's 1024×1024 — now a latency default rather than a correctness one, since a long generation costs extra `check_image` calls instead of failing. Raise `MFLUXIBLE_MCP_WIDTH`/`_HEIGHT` (or `MFLUXIBLE_MCP_WAIT_SECONDS`, if your host's timeout is generous) to trade round-trips back for size, or pass explicit `width`/`height` per call.
- **A ~1MB cap on a single tool result.** MCP ships images as base64, which inflates bytes by 4/3, so the raw image has to land near 750KB. A full-resolution 1024×1280 PNG off this model is ~1.8MB (~2.4MB base64) — about 2.4× over. The tool now re-encodes to fit: PNG is returned untouched when it's already small enough, otherwise it steps down JPEG quality first and only then resolution. In practice quality alone is enough and resolution is never touched: a real 1024×1280 generation measured 1.75MB as PNG and 0.21MB as JPEG q85 at unchanged dimensions — comfortably inside the budget — and even a pathological 3.9MB noise PNG still fits at full size, at q70. So images come back at the resolution you asked for, just recompressed.

Because the inline copy may be recompressed, the untouched full-resolution PNG (mflux metadata intact) is always written to `MFLUXIBLE_MCP_SAVE_DIR` (default `~/Pictures/mfluxible`) first, and the tool returns that path alongside the image.

The image is tagged with MCP's `annotations.audience`/`priority` display hints, which ask the host to surface it to the user rather than bury it in the collapsed tool-result block. Claude Desktop has been observed honoring them and rendering the image in the main transcript. This is a hint, though, not a guarantee: the protocol has no way to *require* main-transcript rendering, so other hosts may still collapse it.

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

Returns:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": {
    "name": "z-image-turbo",
    "label": "Z-Image-Turbo",
    "repo": "Tongyi-MAI/Z-Image-Turbo",
    "quantize": 8,
    "default_steps": 9,
    "supports_guidance": false,
    "default_guidance": null,
    "supports_negative_prompt": false,
    "available": ["z-image-turbo", "flux-schnell", "flux-dev", "qwen-image"]
  },
  "memory": {"active_bytes": 10307921920, "cache_bytes": 1073741824, "peak_bytes": 12884901888}
}
```

`model_loaded` is useful for waiting on startup (weight download + quantization can take a while the first time) before sending a generation request.

`model` describes what this process is running and which request fields it will accept, so a client can fill in sensible defaults without being told how the server was configured: `default_steps` is what `steps` falls back to, and `supports_guidance` / `supports_negative_prompt` say whether `guidance` / `negative_prompt` are accepted or rejected with a 400. `available` lists every model this build knows how to run — all but `name` would need a restart (and a download) to use.

`memory` reports MLX's own byte counters for the server process. `active_bytes` is memory backing live arrays — near zero until the first generation, since weights are quantized lazily and only materialize when something first forces evaluation. `cache_bytes` is buffers MLX has freed but retains for reuse: reclaimable, but it counts toward the process's memory footprint just the same, so on a memory-tight machine it is worth watching between generations. `peak_bytes` is the high-water mark of active memory. All three are plain counters, so polling `/health` mid-generation is cheap and does not disturb the run.

### `POST /v1/images/generations`

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | |
| `width` | int | 1024 | must be divisible by 8 |
| `height` | int | 1024 | must be divisible by 8 |
| `steps` | int or null | the model's own default | 9 for Z-Image-Turbo, 4 for FLUX.1-schnell, 25 for FLUX.1-dev, 20 for Qwen-Image; see `default_steps` on [`/health`](#get-health) |
| `seed` | int or null | random | echoed back in the response so a run can be reproduced |
| `guidance` | float or null | the model's own default | only on models that use guidance (3.5 for FLUX.1-dev and Qwen-Image). A **400** on guidance-distilled models rather than a silently ignored field — see [Models](#models) |
| `negative_prompt` | string or null | unset | Qwen-Image only; a **400** elsewhere, for the same reason |
| `preview_every` | int | 0 | decode and stream an in-progress preview every N steps; 0 disables previews. Each preview is a full VAE decode, so this trades speed for visibility |
| `stream` | bool | true | SSE stream vs a single JSON response |

### Streaming response (`stream: true`, default)

`text/event-stream`, one JSON object per `data:` line:

```
data: {"type": "start", "seed": 123, "total_steps": 9}

data: {"type": "thinking", "step": 1, "total_steps": 9, "step_ms": 210, "elapsed_ms": 210}

data: {"type": "thinking", "step": 2, "total_steps": 9, "step_ms": 190, "elapsed_ms": 400, "preview": "<base64 png>"}

...

data: {"type": "image", "mime_type": "image/png", "data": "<base64 png>", "seed": 123, "generation_time": 14.2}
```

A `thinking` event is emitted once step `step` has finished, carrying both `step_ms` (how long that one step took) and `elapsed_ms` (cumulative time since the denoising loop started, so the `step_ms` values sum to it). The first step's `step_ms` is normally much larger than the rest, and this is **per generation, not a one-time startup cost**: MLX defers all compute until something forces evaluation, so the first step's `mx.eval` pays for everything built lazily ahead of it (paging in and materializing the quantized weights, encoding the prompt) on top of its own denoising. Measured on an M2 Pro at 512×512, the first step reports 55–73s against ~3.3s for each later step, on every run, and is insensitive to both prompt length and step count. `generation_time` on the final `image` event is mflux's own measurement of the denoising loop, so it lands within a few ms of the last `elapsed_ms` — the final VAE decode and PNG encoding happen after it and are counted in neither. A preview decode is charged to the *next* step's `step_ms`, not the step it was requested on.

`preview` is only present on steps where `preview_every` divides the step number, and — like `data` on the final `image` event — is always the full requested resolution; the server never downscales anything (that's a client concern — see [Clients](#clients) above). An `{"type": "error", "message": "..."}` event replaces the final `image` event if generation fails or is interrupted.

The final image's PNG bytes (`data` on the `image` event) carry embedded metadata — prompt, seed, steps, model, quantization, LoRA config (if any), and generation time — as EXIF (`UserComment`), XMP, and IPTC all at once, for broad tool compatibility (readable with e.g. `exiftool output.png`; macOS's built-in `sips`/`mdls` don't surface it). This is mflux's own metadata pipeline (`GeneratedImage.save()`), not something reimplemented here. Step previews aren't touched — only the final image is worth the overhead.

### Non-streaming response (`stream: false`)

Returns the same `image` event object as a single JSON body (or a 500 with the `error` object on failure). A request the configured model cannot honour — `guidance` or `negative_prompt` where it has no effect — is rejected up front with a 400 carrying the same `error` object, in both modes: for a stream that check has to happen before the first byte, since by then the status line is already sent.

## Configuration

Environment variables for `server.py`, all optional.

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_MODEL` | `z-image-turbo` | Which model to run: `z-image-turbo`, `flux-schnell`, `flux-dev`, `qwen-image` (see [Models](#models)) |
| `MFLUXIBLE_QUANTIZE` | `8` | Quantization bits; try `4` for less memory, `none` for full precision |
| `MFLUXIBLE_MODEL_DIR` | `~/.cache/mfluxible` | Where quantized weights are cached (see [Model cache](#model-cache)) |
| `MFLUXIBLE_MLX_CACHE_LIMIT_MB` | `1024` | Cap on MLX's reusable buffer cache; `none` for MLX's own default (see [Memory](#memory)) |
| `MFLUXIBLE_MLX_WIRED_LIMIT_MB` | unset | Wire this much memory so the OS cannot page the weights out (see [Memory](#memory)) |
| `MFLUXIBLE_LORA_PATHS` | unset | Comma-separated local LoRA `.safetensors` files to bake in (see [LoRAs](#loras)) |
| `MFLUXIBLE_LORA_SCALES` | `1.0` each | Comma-separated scales matching `MFLUXIBLE_LORA_PATHS` |
| `MFLUXIBLE_CORS_ORIGIN_REGEX` | `https?://(localhost\|127\.0\.0\.1)(:\d+)?` | Origins to reflect back in CORS (see [CORS](#cors)) |
| `MFLUXIBLE_CORS_ORIGINS` | unset | Comma-separated exact-match origins, in addition to the regex |

### MCP tool

Environment variables for `client/mcp_server.py`, all optional. Set them where the tool is registered (`-e` on `claude mcp add`, or an `"env"` object in Claude Desktop's config).

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_URL` | `http://127.0.0.1:8420/v1/images/generations` | Which mfluxible server to proxy to |
| `MFLUXIBLE_HEALTH_URL` | `/health` on the same host | Read once to name the model and reject arguments it can't act on; only needed if `/health` isn't alongside the generations endpoint |
| `MFLUXIBLE_MCP_WIDTH` | `768` | Default width, kept below the API's own default so most generations finish in one round-trip |
| `MFLUXIBLE_MCP_HEIGHT` | `768` | Default height, same reason |
| `MFLUXIBLE_MCP_STEPS` | unset | Step count to send. Unset lets the server use its model's own default, which is normally what you want |
| `MFLUXIBLE_MCP_WAIT_SECONDS` | `45` | How long one tool call blocks before handing back a `check_image` handle; keep it under the host's tool-call timeout |
| `MFLUXIBLE_MCP_JOB_RETENTION_S` | `900` | How long a finished generation stays collectable by handle |
| `MFLUXIBLE_MCP_MAX_BYTES` | `700000` | Raw-byte budget for the inline image, sized so base64 clears the host's ~1MB result cap |
| `MFLUXIBLE_MCP_SAVE_DIR` | `~/Pictures/mfluxible` | Where the untouched full-resolution PNG is written |

### Memory

The model is the smaller half of what this process holds. MLX also keeps buffers it has freed, so it can reuse them rather than asking Metal for new ones, and that cache counts toward the process footprint even though it is reclaimable. Left uncapped it roughly doubles the resident set, which is what pushes a machine with limited RAM into swapping — and once that happens, every generation pays to fault its weights back in before it can compute anything.

Measured on a 32GB M2 Pro, q8 (10.1GB of weights), three consecutive 512×512 single-step generations in one server process:

| | gen 1 | gen 2 | gen 3 | idle footprint |
|---|---|---|---|---|
| uncapped cache | 6.9s | 80.6s | 104.6s | 20 GB |
| `MFLUXIBLE_MLX_CACHE_LIMIT_MB=1024` | 5.2s | 5.3s | 4.9s | 12 GB |
| q4 + 1GB cache limit | 4.7s | 4.5s | 4.5s | 7 GB |

The first generation is fast either way — a fresh process has not yet grown into the pressure. What the cap prevents is the process becoming slow, so it is the default. Note what this is *not*: it does not make the model faster, and on a machine with ample headroom for the weights there is nothing here to fix. Watch `memory` on [`/health`](#get-health) to see which situation you are in — if `cache_bytes` climbs while generations get slower, this is your knob.

`MFLUXIBLE_MLX_WIRED_LIMIT_MB` goes further and asks the OS to keep that much memory unpageable, protecting the weights from pressure created by *other* processes. On an otherwise-quiet machine it measured no different from the cache cap alone (4.8–5.2s), so it is off by default; reach for it only if generations still degrade under load from elsewhere on the system. Keep any value above the model's resident size and well under `mx.device_info()["max_recommended_working_set_size"]` — wiring too much starves everything else.

### Model cache

Startup quantizes the raw downloaded weights and caches the result to `MFLUXIBLE_MODEL_DIR/<model>-q<bits>/` (e.g. `z-image-turbo-q8`, `qwen-image-q4`) — a real, separate step from the Hugging Face download: HF already caches the raw weights locally, but quantizing them into MLX's packed format is nontrivial per-layer compute that would otherwise happen on every startup (mflux's `mflux-save` mechanism does the same thing via its own CLI; this just does it automatically here). Every startup after the first loads the pre-quantized weights directly and skips that step.

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

## Models

One server process runs one model, chosen at startup with `MFLUXIBLE_MODEL`:

| `MFLUXIBLE_MODEL` | Weights | Default steps | Guidance | Negative prompt |
|---|---|---|---|---|
| `z-image-turbo` (default) | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 9 | — | — |
| `flux-schnell` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | 4 | — | — |
| `flux-dev` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | 25 | default 3.5 | — |
| `qwen-image` | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | 20 | default 3.5 | yes |

mflux's own aliases work too (`schnell`, `dev`, `qwen`, `zimage`, …), and an unrecognised name fails at startup with the list of valid ones — before anything is downloaded.

**Only the model you select is ever fetched.** All four are named in `server/models.py`, but an entry there is inert data: its mflux imports are deferred into a loader function that runs at load time, and mflux downloads weights inside the model's constructor, not at import. The three you aren't running cost nothing beyond their row in that table.

Switching models means restarting the server, the same way `MFLUXIBLE_QUANTIZE` and LoRAs do. Each model + quantization + LoRA combination keeps its own quantized cache directory, so switching back doesn't re-quantize.

What differs per model, beyond the step count:

- **Guidance.** Z-Image-Turbo and FLUX.1-schnell are guidance-distilled: mflux forces guidance to 0 on the former and builds no guidance embedder at all on the latter, so a value has nowhere to go. Sending `guidance` to those is a 400 rather than a field quietly dropped. FLUX.1-dev takes distilled guidance; Qwen-Image runs true classifier-free guidance.
- **Negative prompts.** Only Qwen-Image has a negative branch. That branch is also why its steps are expensive: it runs the transformer twice per step, conditional and unconditional, whether or not you send a `negative_prompt`.
- **Size.** Z-Image-Turbo is the smallest of the four and Qwen-Image much the largest (a ~20B transformer alongside a multimodal text encoder). For scale, Z-Image-Turbo alone occupies 10GB of quantized weights at `MFLUXIBLE_QUANTIZE=8` and 5.5GB at `4`. On a 32GB machine, expect to want `4` or lower for Qwen-Image, and read [Memory](#memory) first — running out of headroom doesn't fail loudly, it just makes every generation slow.

All the bundled clients leave `steps` to the server unless you set it, so they follow whichever model is loaded without reconfiguration. The MCP tool and the browser harness go further and read [`/health`](#get-health): the harness only shows Guidance and Negative prompt when the model accepts them, and the MCP tool refuses those arguments up front rather than spending a round trip to be told no.

Adding another mflux model is a new entry in `server/models.py` and nothing else: the three variants already share a constructor signature, a `generate_image()` signature and a `save_model()`, which is what keeps `server/engine.py` free of per-model branching.

## License

[Apache 2.0](LICENSE).
