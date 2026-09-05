# Server

## How it works

mflux's `generate_image()` is synchronous: it runs the whole denoising loop in one thread and invokes registered callbacks at each step (`InLoopCallback`). The server runs that call on a dedicated single-worker thread (not just any worker thread — see the comment at the top of `server/engine.py` for why that matters with MLX) and bridges each callback invocation back to the event loop as an SSE event via `call_soon_threadsafe`, so a single async server can stream progress out of an otherwise blocking call.

Only one generation runs at a time (there's a lock) — MLX/Metal generation against one shared model instance isn't set up here for concurrency.

Preview images are decoded the same way mflux's own `--stepwise-image-output-dir` CLI flag does internally (see `mflux.callbacks.instances.stepwise_handler.StepwiseHandler`), just streamed instead of written to disk.

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

### Memory

The model is the smaller half of what this process holds. MLX also keeps buffers it has freed, so it can reuse them rather than asking Metal for new ones, and that cache counts toward the process footprint even though it is reclaimable. Left uncapped it roughly doubles the resident set, which is what pushes a machine with limited RAM into swapping — and once that happens, every generation pays to fault its weights back in before it can compute anything.

Measured on a 32GB M2 Pro, q8 (10.1GB of weights), three consecutive 512×512 single-step generations in one server process:

| | gen 1 | gen 2 | gen 3 | idle footprint |
|---|---|---|---|---|
| uncapped cache | 6.9s | 80.6s | 104.6s | 20 GB |
| `MFLUXIBLE_MLX_CACHE_LIMIT_MB=1024` | 5.2s | 5.3s | 4.9s | 12 GB |
| q4 + 1GB cache limit | 4.7s | 4.5s | 4.5s | 7 GB |

The first generation is fast either way — a fresh process has not yet grown into the pressure. What the cap prevents is the process becoming slow, so it is the default. Note what this is *not*: it does not make the model faster, and on a machine with ample headroom for the weights there is nothing here to fix. Watch `memory` on [`/health`](api.md#get-health) to see which situation you are in — if `cache_bytes` climbs while generations get slower, this is your knob.

`MFLUXIBLE_MLX_WIRED_LIMIT_MB` goes further and asks the OS to keep that much memory unpageable, protecting the weights from pressure created by *other* processes. On an otherwise-quiet machine it measured no different from the cache cap alone (4.8–5.2s), so it is off by default; reach for it only if generations still degrade under load from elsewhere on the system. Keep any value above the model's resident size and well under `mx.device_info()["max_recommended_working_set_size"]` — wiring too much starves everything else.

### Model cache

Startup quantizes the raw downloaded weights and caches the result to `MFLUXIBLE_MODEL_DIR/<model>-q<bits>/` (e.g. `z-image-turbo-q8`, `qwen-image-q4`) — a real, separate step from the Hugging Face download: HF already caches the raw weights locally, but quantizing them into MLX's packed format is nontrivial per-layer compute that would otherwise happen on every startup (mflux's `mflux-save` mechanism does the same thing via its own CLI; this just does it automatically here). Every startup after the first loads the pre-quantized weights directly and skips that step.

This means a second, smaller copy of the weights lives on disk alongside HF's cache of the original — once you've confirmed a cached startup works, the original HF cache entry (under `~/.cache/huggingface`) is no longer needed and can be deleted to reclaim space.

### LoRAs

```bash
MFLUXIBLE_LORA_PATHS="/path/to/style.safetensors" MFLUXIBLE_LORA_SCALES="0.8" \
  uv run uvicorn server:app --app-dir server --host 127.0.0.1 --port 8420
```

LoRA weights are applied and permanently merged ("baked") into the model at load time — this is mflux's own design, not a limitation added here. That makes it a server-startup choice, not a per-request one: one running server has one fixed LoRA configuration (or none), and switching LoRAs means restarting the server with different env vars, the same way `MFLUXIBLE_QUANTIZE` works. Each distinct combination of LoRA paths/scales gets its own model-cache directory (named with a hash of that exact config), so switching between a few LoRA setups doesn't require re-quantizing each time you switch back.

### CORS

On by default, reflecting back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin — so a local static server on either hostname, any port (e.g. for `harness.html`) can call the API with no extra configuration. `MFLUXIBLE_CORS_ORIGIN_REGEX` overrides the pattern entirely; `MFLUXIBLE_CORS_ORIGINS` adds specific exact-match origins on top of it (e.g. for a deployed frontend on a real domain).

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

All the bundled clients leave `steps` to the server unless you set it, so they follow whichever model is loaded without reconfiguration. The MCP tool and the browser harness go further and read [`/health`](api.md#get-health): the harness only shows Guidance and Negative prompt when the model accepts them, and the MCP tool refuses those arguments up front rather than spending a round trip to be told no.

Adding another mflux model is a new entry in `server/models.py` and nothing else: the three variants already share a constructor signature, a `generate_image()` signature and a `save_model()`, which is what keeps `server/engine.py` free of per-model branching.

## Running on a dedicated machine

The server doesn't have to run on the same machine as the clients. Point it at a spare Apple Silicon box (a Mac Mini, say) and bind it to the network instead of loopback:

```bash
uv run uvicorn server:app --app-dir server --host 0.0.0.0 --port 8420
```

Then everything else just points at that host instead of `127.0.0.1`, no code changes needed:

- `stream_client.py` / `stream_client.js`: `--url http://mac-mini.local:8420/mfluxible/v1/images/generations`
- `mcp_server.py`: set `MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations` when registering it, e.g. `claude mcp add mfluxible --scope user -e MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/clients/mcp_server.py` — or, in Claude Desktop's config, `"env": {"MFLUXIBLE_URL": "http://mac-mini.local:8420/mfluxible/v1/images/generations"}` alongside `command`/`args`

There's no authentication on the API — only bind it to `0.0.0.0` on a network you trust (home LAN, Tailscale/VPN), never expose it directly to the internet.

## Troubleshooting

### `WARNING: Invalid HTTP request received.`

This is uvicorn's own log line (not mfluxible's) for a connection that sent bytes it couldn't parse as HTTP at all — most commonly something pointed at this server with `https://` instead of `http://` (mfluxible has no TLS of its own; a TLS ClientHello hitting the plaintext port is exactly the kind of thing that produces this), but also a health-checker or proxy speaking a different protocol at the port, or a stray port scan. uvicorn discards the actual reason and the offending bytes, logging only this fixed message, so there's no way to tell which of those it was from the log line alone — check every URL that points at this server (a reverse proxy config, a client's base-URL setting, ...) for a stray `https://` first; that's the single most common cause in practice.
