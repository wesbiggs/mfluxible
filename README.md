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
curl -N -X POST http://127.0.0.1:8420/mfluxible/v1/images/generations \
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

Both also take `--image PATH` for image-to-image (read from disk and base64-encoded, not a URL) and `--image-strength` (0.0–1.0, only valid alongside `--image`; the server's own default, 0.4, applies if you omit it) — see [Image-to-image](#image-to-image) for what `image_strength` actually controls (mflux's convention is the inverse of some other tools'). `--fractional-start` (also only valid alongside `--image`) makes that strength continuous instead of quantized to `1/steps`, at no extra cost — see [Fractional start](#fractional-start).

Both render the exact full-resolution bytes returned by the server — no downscaling, no recompression, nothing client-side touches the image data. They use the iTerm2 inline-image protocol's chunked `MultipartFile` variant (also works in WezTerm; in an unsupported terminal the escape codes are just ignored, and the saved file and progress text still work either way), the same variant iTerm2's own [`imgcat`](https://github.com/gnachman/iTerm2-shell-integration/blob/master/utilities/imgcat) reference tool uses by default: the base64 payload is split into `FilePart=` sequences behind a metadata-only header and a `FileEnd` marker, rather than one giant `File=...:<base64>` sequence.

This matters because iTerm2's own source caps how much data it'll accumulate for a *single* OSC escape sequence at 1,048,576 bytes ([`VT100XtermParser.m`](https://github.com/gnachman/iTerm2/blob/master/sources/VT100/VT100XtermParser.m)) — past that it truncates rather than cleanly dropping the sequence, which can corrupt what renders afterward too, not just fail to show the one image. Diffusion output is detailed/photographic content that a full-resolution PNG can realistically approach or cross that limit for. Chunking (500,000 bytes/chunk here — `imgcat`'s own 200-byte default exists specifically to survive tmux, which doesn't apply since neither script wraps for tmux) means no image, at any size or detail level, can hit that cap.

Images render at `width=auto` (height defaults to auto too) — the same default `imgcat` uses: native pixel dimensions divided by the display's backing scale factor (e.g. a 1024px image renders at 512pt on a 2x/Retina display), rather than a fixed cell-count width that would scale with the terminal's font size instead of the image's actual dimensions.

Not tmux-aware — iTerm2's protocol needs extra passthrough wrapping inside tmux that these scripts don't do.

### Browser

`client/harness.html` is a small, dependency-free page (plain HTML/CSS/JS, no build step) with a form for prompt/width/height/steps/seed/preview_every that calls the streaming endpoint directly from the browser via `fetch`, reads [`/health`](#get-health) on load to show which model the server is running (leaving Steps blank uses that model's default, and Guidance / Negative prompt appear only if it accepts them), parsing the SSE stream the same way the terminal clients do, and renders previews and the final image as `<img>` elements (via `data:` URLs) plus a download link for the final PNG.

It also does image-to-image. Get a base image onto the page either by dragging an image file onto it from anywhere (drop targets are the whole page, not just the drop-zone box — that's just where the visual highlight and preview show up) or by clicking the drop zone to pick a file; or click **Use last result** to feed the most recently generated image straight back in as the next input, for chaining edits without a round trip through disk. **Clear image** drops back to plain text-to-image. Loading a base image any of those ways also sets Width and Height to the image's own dimensions, rounded down to a multiple of 16 — mflux resizes the input to whatever `width`/`height` the request carries with a plain `resize()` and no aspect-ratio handling, so a portrait image against the 1024×1024 default would be stretched, not letterboxed, and that distorted image is what the generation is seeded from. 16 rather than 8 because that's what the server itself does with the number (mflux floors both dimensions to a multiple of 16), so the box says the size you'll actually get; the fields step by 16 to match. The Image strength field controls how strongly that input constrains the output (default 0.4) — see [Image-to-image](#image-to-image) for what the number actually means; the page's own hint text is a reminder that it's the inverse of some other tools' "denoising strength". The **Fractional start** checkbox next to it is [the same flag](#fractional-start) the API takes, and is sent only while an image is loaded. If the chosen image is a PNG this server generated, its embedded prompt (read straight out of the PNG's `eXIf` metadata, client-side, no server round trip) is loaded into the Prompt box automatically — a photo with no such metadata just leaves the box alone.

The server itself serves this page, at `GET /harness.html` — just open `http://localhost:8420/harness.html` (or whichever host/port `server.py` is bound to) once it's up. The Server URL field defaults to the relative path `/mfluxible/v1/images/generations`, which resolves against whatever origin served the page, so no configuration is needed for this same-origin case.

If you'd rather host the page separately (e.g. to point one harness at multiple servers, or to exercise the CORS path), it still works opened from any static file server — just not as a `file://` URL, since the browser's `Origin` header for a local file is `null`, which the server's default CORS config won't match:

```bash
cd client && python3 -m http.server 8000
# then open http://localhost:8000/harness.html and point Server URL at the API host
```

(CORS is on by default and reflects back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin, so this works with no server-side configuration — see [CORS](#cors) below if you need something different.)

### MCP tool (generate images from within Claude)

`client/mcp_server.py` exposes two tools over MCP's stdio transport: `generate_image(prompt, width, height, steps, seed, guidance, negative_prompt, image_path, image_strength, fractional_start)`, which forwards each `thinking` step as an MCP progress update and returns the image as inline content, and `check_image(handle)`, which collects an image from a `generate_image` call that outlived its tool-call timeout (see below).

`image_path` is a local file path (read from disk by the tool, not a URL) for image-to-image; `image_strength` (0.0–1.0, only meaningful alongside `image_path`, server default 0.4 if omitted) follows mflux's own convention — see [Image-to-image](#image-to-image) — which is the *inverse* of "denoising strength" in some other tools, so the tool's own docstring spells this out for the model calling it. `fractional_start` is the same [flag](#fractional-start) the API takes, described there as the thing to reach for when a user is tuning strength finely or asking why a small change to it did nothing.

```bash
pip install -r client/requirements-mcp.txt
```

You still need the server from Quickstart running separately with the model loaded — this only proxies to it.

Two limits imposed by the MCP host shape this tool, and neither is something the server can lift on its own:

- **A wall-clock timeout on each tool call**, shorter than a generation. Measured against a live Claude Code session on 2026-08-31, a `generate_image` call died at ~60s with the MCP SDK's default `Request timed out` — *despite* this server sending a progress notification every ~8s throughout. Progress notifications are worth sending (some hosts do reset on them; the Claude Code CLI runs a separate 30-minute idle watchdog that they rearm) but a server that relies on them is betting on the host, and that bet loses on at least one real host today.

  So generation runs in a background task and `generate_image` blocks on it for at most `MFLUXIBLE_MCP_WAIT_SECONDS` (default 45, comfortably under that 60s). A generation that beats the window returns its image from the first call, exactly as before. A slower one returns a handle instead, and `check_image(handle)` collects it — that call blocks for the same window and returns the image the moment it's ready, so the model calls it again if it comes back "still generating" rather than spinning. **The generation itself is never cancelled by a host giving up**: it keeps running, the full-resolution PNG still lands on disk, and the result stays collectable under its handle for 15 minutes (`MFLUXIBLE_MCP_JOB_RETENTION_S`).

  How many round-trips that costs is a question of resolution more than step count: a chunk of fixed cost outside the per-step loop (text encoding, VAE decode) scales with pixel count, not step count, so trimming resolution buys back more wall-clock time than trimming steps does — and at less cost to output quality than cutting steps. Hence the **768×768** default rather than the HTTP API's 1024×1024 — a latency default, not a correctness one, since a long generation here costs extra `check_image` calls rather than failing outright. Raise `MFLUXIBLE_MCP_WIDTH`/`_HEIGHT` (or `MFLUXIBLE_MCP_WAIT_SECONDS`, if your host's timeout is generous) to trade round-trips back for size, or pass explicit `width`/`height` per call. Actual timings are worth measuring on your own machine and model rather than trusting a hardcoded figure here — watch `step_ms`/`elapsed_ms` on `thinking` events (see [Streaming response](#streaming-response-stream-true-default)) for a live read.
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

Either way, if the server is on a different host/port, point the proxy at it with `MFLUXIBLE_URL` (defaults to `http://127.0.0.1:8420/mfluxible/v1/images/generations`): as `-e MFLUXIBLE_URL=...` on the `claude mcp add` command, or as an `"env"` object alongside `command`/`args` in Desktop's config.

### OpenAI-compatible frontends

None of the above are this — they're the bundled clients, and they all speak mfluxible's own native API. But `POST /v1/images/generations` and `POST /v1/images/edits` (see [API](#api) below) are genuine [OpenAI Images API](https://platform.openai.com/docs/api-reference/images/create)-compatible endpoints, so any tool built against that API can point at this server directly, with no code changes on its side. For example, [Open WebUI](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/openai/)'s Settings → Admin → Images panel takes an arbitrary `IMAGES_OPENAI_API_BASE_URL` and a free-text model name — set the base URL to `http://127.0.0.1:8420/v1` and the model name to whatever `model.name` reports on [`/health`](#get-health) (e.g. `z-image-turbo`), and Open WebUI's own chat UI becomes a frontend for this server.

Open WebUI's *Native* (agentic) mode also needs an actual chat model behind the connection to decide when to call the image tool — normally a separate LLM. If you'd rather not run one just for that, point Open WebUI's chat connection at mfluxible's own `POST /v1/chat/completions` (same base URL) too — see [that section](#post-v1chatcompletions) below for what it does and, importantly, doesn't do.

## API

FastAPI auto-generates interactive docs for all of this at `/docs` (Swagger UI) and `/redoc` once the server is running.

### `GET /health`

Returns:

```json
{
  "status": "ok",
  "model_loaded": true,
  "available": ["z-image-turbo", "flux-schnell", "flux-dev", "qwen-image"],
  "model": {
    "name": "z-image-turbo",
    "label": "Z-Image-Turbo",
    "repo": "Tongyi-MAI/Z-Image-Turbo",
    "quantize": 8,
    "default_steps": 9,
    "supports_guidance": false,
    "default_guidance": null,
    "supports_negative_prompt": false
  },
  "memory": {"active_bytes": 10307921920, "cache_bytes": 1073741824, "peak_bytes": 12884901888}
}
```

`model_loaded` is useful for waiting on startup (weight download + quantization can take a while the first time) before sending a generation request.

`model` describes what this process is running and which request fields it will accept, so a client can fill in sensible defaults without being told how the server was configured: `default_steps` is what `steps` falls back to, and `supports_guidance` / `supports_negative_prompt` say whether `guidance` / `negative_prompt` are accepted or rejected with a 400. `available` lists every model this build knows how to run — all but `model.name` would need a restart (and a download) to use.

`memory` reports MLX's own byte counters for the server process. `active_bytes` is memory backing live arrays — near zero until the first generation, since weights are quantized lazily and only materialize when something first forces evaluation. `cache_bytes` is buffers MLX has freed but retains for reuse: reclaimable, but it counts toward the process's memory footprint just the same, so on a memory-tight machine it is worth watching between generations. `peak_bytes` is the high-water mark of active memory. All three are plain counters, so polling `/health` mid-generation is cheap and does not disturb the run.

### `GET /harness.html`

Serves [`client/harness.html`](#browser) as-is — see the Browser section above. Lets you open the harness straight from the running server (`http://localhost:8420/harness.html`) instead of standing up a separate static file server for it.

### `POST /mfluxible/v1/images/generations`

mfluxible's own, native endpoint — everything below (step-by-step `thinking` events, previews, `guidance`, `negative_prompt`) is specific to it. Every bundled client targets this path. See [`POST /v1/images/generations`](#post-v1imagesgenerations) below for the separate OpenAI-compatible endpoint.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | |
| `width` | int | 1024 | should be divisible by 16 — mflux floors it to a multiple of 16 rather than rejecting it, so anything else generates up to 15px smaller than asked |
| `height` | int | 1024 | same as `width` |
| `steps` | int or null | the model's own default | 9 for Z-Image-Turbo, 4 for FLUX.1-schnell, 25 for FLUX.1-dev, 20 for Qwen-Image; see `default_steps` on [`/health`](#get-health) |
| `seed` | int or null | random | echoed back in the response so a run can be reproduced |
| `guidance` | float or null | the model's own default | only on models that use guidance (3.5 for FLUX.1-dev and Qwen-Image). A **400** on guidance-distilled models rather than a silently ignored field — see [Models](#models) |
| `negative_prompt` | string or null | unset | Qwen-Image only; a **400** elsewhere, for the same reason |
| `preview_every` | int | 0 | decode and stream an in-progress preview every N steps; 0 disables previews. Each preview is a full VAE decode, so this trades speed for visibility |
| `stream` | bool | true | SSE stream vs a single JSON response |
| `image` | string or null | unset | base64-encoded input image (no `data:` URI prefix) for image-to-image. Accepted by every model this server can run — see [Image-to-image](#image-to-image) below |
| `image_strength` | float or null | 0.4 if `image` is set | how strongly `image` constrains the output, `0.0`–`1.0`; only meaningful, and only accepted, alongside `image` — a **400** if set without it |
| `fractional_start` | bool | `false` | start image-to-image *between* two steps of the sigma schedule instead of flooring to one, making `image_strength` continuous at no extra compute; only accepted alongside `image` — a **400** otherwise. See [Fractional start](#fractional-start) |

### Streaming response (`stream: true`, default)

`text/event-stream`, one JSON object per `data:` line:

```
data: {"type": "start", "seed": 123, "total_steps": 9, "start_step": 0, "effective_image_strength": null}

data: {"type": "thinking", "step": 1, "total_steps": 9, "step_ms": 210, "elapsed_ms": 210}

data: {"type": "thinking", "step": 2, "total_steps": 9, "step_ms": 190, "elapsed_ms": 400, "preview": "<base64 png>"}

...

data: {"type": "image", "mime_type": "image/png", "data": "<base64 png>", "seed": 123, "generation_time": 14.2}
```

The `start` event's `start_step` is the number of leading denoising steps this generation skips, and it is `0` for everything except image-to-image — so the first `thinking` event is always step `start_step + 1`, and `total_steps - start_step` steps actually run. Progress is therefore `(step - start_step) / (total_steps - start_step)`, which reduces to `step / total_steps` when nothing is skipped. `effective_image_strength` is the `image_strength` bucket the request landed in (`start_step / total_steps`), or `null` when no `image` was sent — see [Image-to-image](#image-to-image) for why the value you sent and the value that took effect aren't always the same. Both fields are display/progress information: `effective_image_strength` is the exact *lower edge* of its bucket as a fraction, but as a float it lands a hair under it, so feeding it back into a later request can floor to the next bucket down. Under [`fractional_start`](#fractional-start) there are no buckets and it reports the strength that actually took effect.

A `thinking` event is emitted once step `step` has finished, carrying both `step_ms` (how long that one step took) and `elapsed_ms` (cumulative time since the denoising loop started, so the `step_ms` values sum to it). The first step's `step_ms` is normally much larger than the rest, and this is **per generation, not a one-time startup cost**: MLX defers all compute until something forces evaluation, so the first step's `mx.eval` pays for everything built lazily ahead of it (paging in and materializing the quantized weights, encoding the prompt) on top of its own denoising. Measured on an M2 Pro at 512×512, the first step reports 55–73s against ~3.3s for each later step, on every run, and is insensitive to both prompt length and step count. `generation_time` on the final `image` event is mflux's own measurement of the denoising loop, so it lands within a few ms of the last `elapsed_ms` — the final VAE decode and PNG encoding happen after it and are counted in neither. A preview decode is charged to the *next* step's `step_ms`, not the step it was requested on.

`preview` is only present on steps where `preview_every` divides the step number, and — like `data` on the final `image` event — is always the full requested resolution; the server never downscales anything (that's a client concern — see [Clients](#clients) above). An `{"type": "error", "message": "..."}` event replaces the final `image` event if generation fails or is interrupted.

The final image's PNG bytes (`data` on the `image` event) carry embedded metadata — prompt, seed, steps, model, quantization, LoRA config (if any), and generation time — as EXIF (`UserComment`), XMP, and IPTC all at once, for broad tool compatibility (readable with e.g. `exiftool output.png`; macOS's built-in `sips`/`mdls` don't surface it). This is mflux's own metadata pipeline (`GeneratedImage.save()`), not something reimplemented here. Step previews aren't touched — only the final image is worth the overhead.

### Non-streaming response (`stream: false`)

Returns the same `image` event object as a single JSON body (or a 500 with the `error` object on failure). A request the configured model cannot honour — `guidance` or `negative_prompt` where it has no effect, or `image_strength` without `image` — is rejected up front with a 400 carrying the same `error` object, in both modes: for a stream that check has to happen before the first byte, since by then the status line is already sent. A malformed `image` (not valid base64, or not a decodable image) gets the same 400 treatment.

### Image-to-image

Set `image` (and, optionally, `image_strength`) to seed generation from an existing image instead of pure noise, using mflux's own `image_path`/`image_strength` support. Unlike `guidance`/`negative_prompt`, this isn't model-specific — mflux's `ZImage`, `Flux1` and `QwenImage` all share the same `generate_image(image_path=..., image_strength=...)` parameters, so it works on every model this server can run, with no `/health` check needed first.

`image_strength` follows **mflux's own convention, not the "denoising strength" convention** used by tools like Stable Diffusion/A1111/diffusers, where a *higher* value means *more* change from the input. mflux's is the other way round: it's how strongly the input image constrains the output. `0.0` means the image has no influence at all (equivalent to plain text-to-image); `1.0` means maximum influence, which can mean very few — or even zero — denoising steps actually run, so the output stays close to the input. The default, `0.4`, is a middle ground (and matches mflux's own CLI default). If you're used to the inverted convention, mentally flip the slider.

`image_strength` is also **quantized to `1/steps`**, because it only ever reaches the model as an integer: mflux turns it into `init_time_step = max(1, int(steps * image_strength))`, which both picks the step the denoising loop starts at and indexes the noise level blended into the input image. Nothing downstream sees the original float. At the 9 steps Z-Image-Turbo defaults to, that's ten distinct settings — `0.35` and `0.4` both floor to `3` and produce identical pixels for the same seed, while `0.3` and `0.35` straddle a boundary and don't. The `start` event's `effective_image_strength` reports the bucket actually used, so a nudge that changed nothing is visible rather than looking like the server ignored it. Raising `steps` gives a finer dial (20 steps → 21 buckets), but it also runs proportionally more steps — so you move start-noise and trajectory length together — and on a few-step distilled model like Z-Image-Turbo or FLUX.1-schnell it's off-distribution rather than simply slower. Whether the finer grid even *contains* the coarse one is model-specific: for Z-Image-Turbo and FLUX it does exactly (doubling `steps` interpolates the rungs you had and leaves them where they were), but Qwen-Image sets `sigma_shift_terminal`, whose stretch is scaled by the last raw sigma `1/steps`, so its schedules at different step counts don't line up at all. [`fractional_start`](#fractional-start) is the way to get granularity without any of that.

Note the degenerate end: `image_strength: 1.0` makes `init_time_step` equal `steps`, so **no denoising steps run at all** and the noise level is zero. The stream is well-formed — a `start` event, no `thinking` events, then the final image — but that image is just your input scaled to `width`/`height` and round-tripped through the VAE. It's accepted rather than rejected because it's the honest limit of the scale, not a mistake the server can distinguish from intent.

#### Fractional start

`fractional_start: true` removes the quantization instead of working around it, at the same step count and the same cost.

It works because `init_time_step` is doing two jobs. Job one — which step the loop starts at — genuinely has to be an integer: every step integrates between adjacent grid points, `dt = sigmas[t+1] - sigmas[t]`, so there is no such thing as starting halfway through one. Job two — the noise level the input image is blended to, `sigmas[init_time_step]` — is just an array lookup, and nothing requires it to land on a grid point. Sharing one integer between them is what quantizes the dial.

So [`server/schedulers.py`](server/schedulers.py) moves the rung rather than the index: it takes the schedule the request's own `steps` produces and replaces `sigmas[init_time_step]` with a point interpolated toward its neighbour, at the exact position `image_strength` names. The loop still starts on the same whole step and still runs `steps - start_step` of them; only its first step is shorter. At 10 steps, `0.25` lands halfway between the rungs `0.2` and `0.3` reach, `0.22` lands a fifth of the way, and so on — with no change to how many steps run, so a strength sweep isn't also a step-count sweep.

This is safe rather than a mismatch between the latents and what the model thinks it is denoising because **every variant conditions the transformer on `sigmas[t]` itself, not on the step index** — `ZImage` computes `timestep = 1 - sigmas[t]` inline, and the FLUX and Qwen transformers read `config.scheduler.sigmas[...]` for their time embedding. Moving the rung moves the conditioning with it. That's the assumption to re-check if a future mflux version makes images from this path come out wrong.

Mechanically it's one of mflux's own extension points: `Config` resolves a scheduler given as a dotted import path, and all three variants take `scheduler=` on `generate_image()`. The API exposes a **bool, not that path** — a caller-supplied dotted path would be an arbitrary module import in the server process — so the server picks the class and the request only chooses whether to use it.

It's off by default because it changes the pixels a given strength produces: an existing seed/strength pair keeps reproducing its old image unless you ask. With it on, `effective_image_strength` on the `start` event reports the strength that actually took effect rather than a floored bucket, so the two modes stay distinguishable from the stream alone. Two edge cases still report a fraction of `0` and leave the schedule untouched, since mflux's own clamps have already moved the start off the position the strength names: a strength below `1/steps` (floored *up* to rung 1), and `1.0` (starts past the last rung, with no steps to run).

Two caveats worth stating. The interpolation is done on the request's own already-shifted schedule rather than by re-deriving mflux's sigma-shift math, so a half-step lands *near*, not exactly on, the rung that twice as many steps would have given — within 0.001 across the low-index region img2img actually uses, growing to about 0.013 at the very tail (measured for Z-Image-Turbo at 1024×1024; for FLUX.1-schnell, whose schedule isn't shifted at all, it's exact). Every bundled client exposes it: `--fractional-start` on both terminal clients, a checkbox beside Image strength in the harness, and a `fractional_start` argument on the MCP `generate_image` tool.

```bash
python client/stream_client.py "a lighthouse" --steps 10 --image input.png --image-strength 0.25 --fractional-start
```

The image is scaled to the request's `width`/`height` before use, so it need not match them. Internally, the base64 payload is decoded to a temp file for the duration of one generation (mflux's `image_path` wants an actual path, not bytes) and removed once that generation finishes — nothing is written that outlives the request.

### `POST /v1/images/generations`

A genuine [OpenAI Images API](https://platform.openai.com/docs/api-reference/images/create)-compatible endpoint, so an existing OpenAI-client-based tool can point its base URL at this server unmodified — e.g. [Open WebUI](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/openai/)'s "OpenAI" image-generation engine, which already takes an arbitrary `IMAGES_OPENAI_API_BASE_URL` and a free-text model name. This is a separate, additive endpoint — it does not replace [`/mfluxible/v1/images/generations`](#post-mfluxiblev1imagesgenerations) above, which every bundled client still uses.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | |
| `model` | string | required | must equal the loaded model's key (see `model.name` on [`/health`](#get-health)) — one model runs per process, so this is validated, not routed on; a mismatch is a 400 naming what's actually loaded |
| `size` | string | `"1024x1024"` | `"<width>x<height>"`, both ideally divisible by 16 (see [`width`](#post-mfluxiblev1imagesgenerations)); `"auto"` maps to `1024x1024` |
| `n` | int | 1 | must be 1 — a 400 otherwise, rather than silently generating just one |
| `response_format` | string | `"b64_json"` | only `"b64_json"` is supported — a 400 for `"url"`, since this server doesn't host images |
| `stream` | bool | false | |
| `partial_images` | int | 0 | 0–3; how many in-progress previews to emit while streaming. OpenAI's semantics are a *total count*; mflux's `preview_every` is a *stride*, so this is translated by dividing it into the loaded model's `default_steps` — an approximation, not an exact per-step mapping |

Every other field OpenAI's schema defines (`quality`, `style`, `background`, `output_format`, `output_compression`, `moderation`, `user`) is accepted and silently ignored — there's no mflux equivalent to act on it with, and OpenAI clients send these with their own defaults regardless of whether the server does anything with them.

Non-streaming response:

```json
{"created": 1735689600, "data": [{"b64_json": "<base64 png>"}]}
```

Streaming response (`stream: true`), same SSE transport as the native endpoint but OpenAI's event names and payload shape:

```
data: {"type": "image_generation.partial_image", "b64_json": "<base64 png>", "partial_image_index": 0, "created_at": 1735689600}

data: {"type": "image_generation.completed", "b64_json": "<base64 png>", "created_at": 1735689600}
```

`partial_image` events only appear when `partial_images` > 0. The stream ends after `image_generation.completed` with no trailing `[DONE]` sentinel — OpenAI's own raw wire format for this isn't documented anywhere publicly to confirm one either way, so none is fabricated here.

Errors use OpenAI's envelope rather than the native endpoint's: `{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}`, with a 400 for a bad request (wrong `model`, `n != 1`, unsupported `response_format`, malformed `size`) and a 500 (`type: "api_error"`) if generation itself fails.

### `POST /v1/images/edits`

A genuine [OpenAI Images-Edit-API](https://platform.openai.com/docs/api-reference/images/createEdit)-compatible endpoint for image-to-image, so an existing OpenAI-client-based image-editing tool can point at this server unmodified. Unlike every other endpoint here, the request is `multipart/form-data`, not JSON — matching OpenAI's own request shape for this one.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string (form field) | required | |
| `model` | string (form field) | required | same validation as [`/v1/images/generations`](#post-v1imagesgenerations) |
| `image` | file | required | the input image |
| `mask` | file | — | **not supported** — a 400 if present, rather than silently ignored. mflux has no masked-region inpainting pipeline wired up here; this endpoint does whole-image edits only (see [Image-to-image](#image-to-image)), and a caller expecting only the masked region to change would otherwise get a silently wrong result |
| `n` | int (form field) | 1 | same as `/v1/images/generations` |
| `size` | string (form field) | `"1024x1024"` | same as `/v1/images/generations` |
| `response_format` | string (form field) | `"b64_json"` | same as `/v1/images/generations` |
| `stream` | bool (form field) | false | same SSE transport and event shape as `/v1/images/generations` |
| `partial_images` | int (form field) | 0 | same as `/v1/images/generations` |
| `image_strength` | float (form field) | 0.4 | **not part of OpenAI's request shape** — accepted as an extension the same way `partial_images` already is. See [Image-to-image](#image-to-image) for what it controls (mflux's convention, the inverse of some other tools') |

Response shapes, streaming behavior, and the error envelope are all identical to [`/v1/images/generations`](#post-v1imagesgenerations) above.

### `GET /v1/models`

For frontends that discover models rather than taking a free-text name — [list format](https://platform.openai.com/docs/api-reference/models/list) with exactly one entry, the loaded model:

```json
{
  "object": "list",
  "data": [
    {"id": "z-image-turbo", "object": "model", "created": 1735689600, "owned_by": "mfluxible"}
  ]
}
```

`id` is what `model` on [`POST /v1/images/generations`](#post-v1imagesgenerations) must equal. `created` is when this process finished loading the model (there's no meaningful weight-publish date to report instead).

`GET /v1/models/{id}` ([retrieve format](https://platform.openai.com/docs/api-reference/models/retrieve)) works the same way for the one `id` that list just returned, returning that same object un-wrapped:

```json
{"id": "z-image-turbo", "object": "model", "created": 1735689600, "owned_by": "mfluxible"}
```

Any other `id` is a 404 with OpenAI's own `model_not_found` error code, the same way `api.openai.com` responds to an unknown model.

### `POST /v1/chat/completions`

Not a chat model — a stub that exists so a frontend requiring an actual "chat model" behind its Native/agentic tool-calling mode (Open WebUI is the motivating case) can point that connection at mfluxible too, instead of running a separate LLM (Ollama, `llama-server`, ...) just to decide "yes, call the image tool" on every message. See [`server/chat_stub.py`](server/chat_stub.py) for the full rationale; the short version is there's no reasoning to replace, so there's nothing an LLM gets right that a hardcoded rule doesn't get right for free, at zero extra memory (it's pure Python in the same process as the diffusion model — no weights, no inference).

It plays exactly one deterministic turn of OpenAI's function-calling protocol:

- A request whose last message is a plain user turn, with a `generate_image` function offered in `tools` → responds with a `tool_calls` message invoking it, `{"prompt": "<the user's message>"}` as the arguments.
- A request whose last message is a `tool` result (i.e. the caller already ran `generate_image` and is asking for a closing reply) → responds with a short fixed acknowledgment, `finish_reason: "stop"`.
- A request with no `generate_image` tool offered at all → responds with fixed text explaining why (most likely cause in Open WebUI: Capabilities or Builtin Tools → Image Generation isn't enabled for this model).

It only ever recognizes a tool literally named `generate_image` — the name [Open WebUI's own builtin tool uses](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/tools/builtin.py) — offered in that specific request's `tools` list; it never guesses at one that wasn't offered. There's no general conversation, no other tools, no multi-turn reasoning — if that's what's needed, point the chat connection at a real model instead.

Both streaming and non-streaming (`stream: true`/`false`) are supported, in OpenAI's own chat-completion / chat-completion-chunk shapes. `model` in the request isn't validated against anything — unlike the image endpoints, there's no real model here to be inconsistent with.

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
| `MFLUXIBLE_URL` | `http://127.0.0.1:8420/mfluxible/v1/images/generations` | Which mfluxible server to proxy to |
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

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs entirely against a fake, weight-free model (`tests/doubles/toy_model.py`) that always renders a solid color instead of doing real diffusion — no download, no GPU work, and it's fast enough for every push. It's wired in by passing a `ModelSpec` instance straight to `MfluxEngine(model=...)`, which skips `models.py`'s registry entirely (see `MfluxEngine.__init__` in `server/engine.py`), so no production code has to know it exists.

Runs on GitHub Actions on every push/PR (`.github/workflows/tests.yml`). Since `mlx` (mflux's own dependency) has no Linux or Intel build, that workflow — and any other CI you point at this repo — has to run on an Apple Silicon macOS runner (`macos-14` on GitHub-hosted); `ubuntu-latest` will fail to install.

## Running on a dedicated machine

The server doesn't have to run on the same machine as the clients. Point it at a spare Apple Silicon box (a Mac Mini, say) and bind it to the network instead of loopback:

```bash
uvicorn server:app --app-dir server --host 0.0.0.0 --port 8420
```

Then everything else just points at that host instead of `127.0.0.1`, no code changes needed:

- `stream_client.py` / `stream_client.js`: `--url http://mac-mini.local:8420/mfluxible/v1/images/generations`
- `mcp_server.py`: set `MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations` when registering it, e.g. `claude mcp add mfluxible --scope user -e MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/client/mcp_server.py` — or, in Claude Desktop's config, `"env": {"MFLUXIBLE_URL": "http://mac-mini.local:8420/mfluxible/v1/images/generations"}` alongside `command`/`args`

There's no authentication on the API — only bind it to `0.0.0.0` on a network you trust (home LAN, Tailscale/VPN), never expose it directly to the internet.

## Troubleshooting

### `WARNING: Invalid HTTP request received.`

This is uvicorn's own log line (not mfluxible's) for a connection that sent bytes it couldn't parse as HTTP at all — most commonly something pointed at this server with `https://` instead of `http://` (mfluxible has no TLS of its own; a TLS ClientHello hitting the plaintext port is exactly the kind of thing that produces this), but also a health-checker or proxy speaking a different protocol at the port, or a stray port scan. uvicorn discards the actual reason and the offending bytes, logging only this fixed message, so there's no way to tell which of those it was from the log line alone — check every URL that points at this server (a reverse proxy config, a client's base-URL setting, ...) for a stray `https://` first; that's the single most common cause in practice.

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
