# API

FastAPI auto-generates interactive docs for all of this at `/docs` (Swagger UI) and `/redoc` once the server is running.

## Authentication

None by default. If the server was started with `MFLUXIBLE_BASIC_AUTH_USERNAME` and `MFLUXIBLE_BASIC_AUTH_PASSWORD` set, **every** endpoint below — plus `GET /`, `/docs` and `/openapi.json` — requires HTTP Basic credentials, and answers `401` with a `WWW-Authenticate: Basic realm="mfluxible"` header otherwise. The 401 body is OpenAI's error envelope with `code: "unauthorized"`, and is identical whether the username or the password was wrong. See [Authentication](server.md#authentication) for how to turn it on and what it does and doesn't protect.

A server can instead be gated by a reverse proxy holding a bearer token, in which case none of the above applies and the endpoints below answer exactly as documented — what changes is which of them are reachable without an `Authorization: Bearer` header. See [Authentication](server.md#authentication) for both options.

CORS preflights are answered before the credential check, since a preflight carries none by spec. A *bare* `OPTIONS` (no `Access-Control-Request-Method`) is a normal request and is gated — which is what blocks SillyTavern's reachability probe when auth is on.

## `GET /health`

Returns:

```json
{
  "status": "ok",
  "model_loaded": true,
  "available": ["z-image-turbo", "z-image", "flux-schnell", "flux-dev", "krea-dev", "qwen-image", "krea-2", "krea-2-raw", "ernie-image-turbo", "ernie-image", "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-9b-kv", "flux2-klein-base-4b", "flux2-klein-base-9b"],
  "model": {
    "name": "z-image-turbo",
    "label": "Z-Image-Turbo",
    "repo": "Tongyi-MAI/Z-Image-Turbo",
    "quantize": 8,
    "default_steps": 9,
    "supports_guidance": false,
    "default_guidance": null,
    "supports_negative_prompt": false,
    "supports_fractional_start": true,
    "supports_mask": true
  },
  "memory": {"active_bytes": 10307921920, "cache_bytes": 1073741824, "peak_bytes": 12884901888}
}
```

`model_loaded` is useful for waiting on startup (weight download + quantization can take a while the first time) before sending a generation request.

`model` describes what this process is running and which request fields it will accept, so a client can fill in sensible defaults without being told how the server was configured: `default_steps` is what `steps` falls back to, and `supports_guidance` / `supports_negative_prompt` / `supports_fractional_start` / `supports_mask` say whether `guidance` / `negative_prompt` / `fractional_start` / `mask` are accepted or rejected with a 400. `available` lists every model this build knows how to run — all but `model.name` would need a restart (and a download) to use.

Treat any of those `supports_*` keys being **absent** as "unknown, let the server decide" rather than as `false` — that is what a server predating the key means, and it is how the bundled harness and MCP tool read them. Note also that `supports_negative_prompt` is necessary but not sufficient: a negative prompt also needs classifier-free guidance switched on, so on a model whose `default_guidance` is `1.0` (Krea-2) sending one without raising `guidance` is still a 400. See [Models](server.md#models).

`memory` reports MLX's own byte counters for the server process. `active_bytes` is memory backing live arrays — near zero until the first generation, since weights are quantized lazily and only materialize when something first forces evaluation. `cache_bytes` is buffers MLX has freed but retains for reuse: reclaimable, but it counts toward the process's memory footprint just the same, so on a memory-tight machine it is worth watching between generations. `peak_bytes` is the high-water mark of active memory. All three are plain counters, so polling `/health` mid-generation is cheap and does not disturb the run.

## `GET /`

Serves `clients/harness.html` as-is — see the [Browser](clients.md#browser) client. Lets you open the harness straight from the running server (`http://localhost:8420/`) instead of standing up a separate static file server for it.

## `POST /mfluxible/v1/images/generations`

mfluxible's own, native endpoint — everything below (step-by-step `thinking` events, previews, `guidance`, `negative_prompt`) is specific to it. Every bundled client targets this path. See [`POST /v1/images/generations`](#post-v1imagesgenerations) below for the separate OpenAI-compatible endpoint.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | |
| `width` | int | 1024 | should be divisible by 16 — mflux floors it to a multiple of 16 rather than rejecting it, so anything else generates up to 15px smaller than asked |
| `height` | int | 1024 | same as `width` |
| `steps` | int or null | the model's own default | 9 for Z-Image-Turbo, 4 for FLUX.1-schnell, 25 for FLUX.1-dev, 20 for Qwen-Image; see `default_steps` on [`/health`](#get-health) |
| `seed` | int or null | random | echoed back in the response so a run can be reproduced |
| `guidance` | float or null | the model's own default | only on models that use guidance (3.5 for FLUX.1-dev and Qwen-Image). A **400** on guidance-distilled models rather than a silently ignored field — see [Models](server.md#models) |
| `negative_prompt` | string or null | unset | Qwen-Image only; a **400** elsewhere, for the same reason |
| `preview_every` | int | 0 | decode and stream an in-progress preview every N steps; 0 disables previews. Each preview is a full VAE decode, so this trades speed for visibility |
| `stream` | bool | true | SSE stream vs a single JSON response |
| `image` | string or null | unset | base64-encoded input image (no `data:` URI prefix) for image-to-image. Accepted by every model this server can run — see [Image-to-image](#image-to-image) below |
| `image_strength` | float or null | 0.4 if `image` is set | how strongly `image` constrains the output, `0.0`–`1.0`; only meaningful, and only accepted, alongside `image` — a **400** if set without it |
| `mask` | string or null | unset | base64-encoded mask image (no `data:` URI prefix) selecting the region to regenerate — **white = change this, black = keep this**, greys crossfade. Must be the same pixel dimensions as `image`, and only accepted alongside it. Only on a model whose own scheduler is linear (`supports_mask` on [`/health`](#get-health)) — a **400** otherwise. See [Inpainting a region](#inpainting-a-region) |
| `mask_feather` | int | 0 | Gaussian blur radius applied to `mask`, in base-image pixels, to hide the latent grid's 8px staircase along the mask edge. Useful range is 8–16; it is **not** a region blend — see below. A **400** without `mask` |
| `mask_composite` | bool | `true` | paste the result back over the input through `mask`, so the pixels outside it are byte-identical to what was sent. A **400** without `mask` |
| `fractional_start` | bool | `false` | start image-to-image *between* two steps of the sigma schedule instead of flooring to one, making `image_strength` continuous at no extra compute; only accepted alongside `image`, and only on a model that runs mflux's linear schedule (`supports_fractional_start` on [`/health`](#get-health)) — a **400** otherwise. See [Fractional start](#fractional-start) |

## Streaming response (`stream: true`, default)

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

`preview` is only present on steps where `preview_every` divides the step number, and — like `data` on the final `image` event — is always the full requested resolution; the server never downscales anything (that's a client concern — see [Clients](clients.md)). An `{"type": "error", "message": "..."}` event replaces the final `image` event if generation fails or is interrupted.

**What `message` will and won't tell you.** Every message this API returns is one the server composed for a client to read — which for a generation that failed inside mflux means only `generation failed -- see the server log for the reason.`, not mflux's own text. The reason, with the full traceback, goes to the server's stderr instead (the terminal running `uvicorn`); mflux, MLX and Hugging Face errors routinely quote absolute cache paths, and those describe the machine serving the request rather than the request. Interruptions are the exception and still say which step they stopped on. So: a 400 can be acted on from the client, a 500 is read on the server — see [Troubleshooting](server.md#troubleshooting).

The final image's PNG bytes (`data` on the `image` event) carry embedded metadata — prompt, seed, steps, model, quantization, LoRA config (if any), and generation time — as EXIF (`UserComment`), XMP, and IPTC all at once, for broad tool compatibility (readable with e.g. `exiftool output.png`; macOS's built-in `sips`/`mdls` don't surface it). This is mflux's own metadata pipeline (`GeneratedImage.save()`), not something reimplemented here. Step previews aren't touched — only the final image is worth the overhead.

## Non-streaming response (`stream: false`)

Returns the same `image` event object as a single JSON body (or a 500 with the `error` object on failure). A request the configured model cannot honour — `guidance` or `negative_prompt` where it has no effect, `image_strength` or `mask` without `image`, or a `mask` that isn't the same size as the `image` — is rejected up front with a 400 carrying the same `error` object, in both modes: for a stream that check has to happen before the first byte, since by then the status line is already sent. A malformed `image` or `mask` (not valid base64, or not a decodable image) gets the same 400 treatment.

## Image-to-image

Set `image` (and, optionally, `image_strength`) to seed generation from an existing image instead of pure noise, using mflux's own `image_path`/`image_strength` support. Unlike `guidance`/`negative_prompt`, this isn't model-specific — mflux's `ZImage`, `Flux1` and `QwenImage` all share the same `generate_image(image_path=..., image_strength=...)` parameters, so it works on every model this server can run, with no `/health` check needed first.

`image_strength` follows **mflux's own convention, not the "denoising strength" convention** used by tools like Stable Diffusion/A1111/diffusers, where a *higher* value means *more* change from the input. mflux's is the other way round: it's how strongly the input image constrains the output. `0.0` means the image has no influence at all (equivalent to plain text-to-image); `1.0` means maximum influence, which can mean very few — or even zero — denoising steps actually run, so the output stays close to the input. The default, `0.4`, is a middle ground (and matches mflux's own CLI default). If you're used to the inverted convention, mentally flip the slider.

`image_strength` is also **quantized to `1/steps`**, because it only ever reaches the model as an integer: mflux turns it into `init_time_step = max(1, int(steps * image_strength))`, which both picks the step the denoising loop starts at and indexes the noise level blended into the input image. Nothing downstream sees the original float. At the 9 steps Z-Image-Turbo defaults to, that's ten distinct settings — `0.35` and `0.4` both floor to `3` and produce identical pixels for the same seed, while `0.3` and `0.35` straddle a boundary and don't. The `start` event's `effective_image_strength` reports the bucket actually used, so a nudge that changed nothing is visible rather than looking like the server ignored it. Raising `steps` gives a finer dial (20 steps → 21 buckets), but it also runs proportionally more steps — so you move start-noise and trajectory length together — and on a few-step distilled model like Z-Image-Turbo or FLUX.1-schnell it's off-distribution rather than simply slower. Whether the finer grid even *contains* the coarse one is model-specific: for Z-Image-Turbo and FLUX it does exactly (doubling `steps` interpolates the rungs you had and leaves them where they were), but Qwen-Image sets `sigma_shift_terminal`, whose stretch is scaled by the last raw sigma `1/steps`, so its schedules at different step counts don't line up at all. [`fractional_start`](#fractional-start) is the way to get granularity without any of that.

Note the degenerate end: `image_strength: 1.0` makes `init_time_step` equal `steps`, so **no denoising steps run at all** and the noise level is zero. The stream is well-formed — a `start` event, no `thinking` events, then the final image — but that image is just your input scaled to `width`/`height` and round-tripped through the VAE. It's accepted rather than rejected because it's the honest limit of the scale, not a mistake the server can distinguish from intent.

### Fractional start

`fractional_start: true` removes the quantization instead of working around it, at the same step count and the same cost.

It works because `init_time_step` is doing two jobs. Job one — which step the loop starts at — genuinely has to be an integer: every step integrates between adjacent grid points, `dt = sigmas[t+1] - sigmas[t]`, so there is no such thing as starting halfway through one. Job two — the noise level the input image is blended to, `sigmas[init_time_step]` — is just an array lookup, and nothing requires it to land on a grid point. Sharing one integer between them is what quantizes the dial.

So [`server/schedulers.py`](../server/schedulers.py) moves the rung rather than the index: it takes the schedule the request's own `steps` produces and replaces `sigmas[init_time_step]` with a point interpolated toward its neighbour, at the exact position `image_strength` names. The loop still starts on the same whole step and still runs `steps - start_step` of them; only its first step is shorter. At 10 steps, `0.25` lands halfway between the rungs `0.2` and `0.3` reach, `0.22` lands a fifth of the way, and so on — with no change to how many steps run, so a strength sweep isn't also a step-count sweep.

This is safe rather than a mismatch between the latents and what the model thinks it is denoising because **every variant conditions the transformer on `sigmas[t]` itself, not on the step index** — `ZImage` computes `timestep = 1 - sigmas[t]` inline, and the FLUX and Qwen transformers read `config.scheduler.sigmas[...]` for their time embedding. Moving the rung moves the conditioning with it. That's the assumption to re-check if a future mflux version makes images from this path come out wrong.

Mechanically it's one of mflux's own extension points: `Config` resolves a scheduler given as a dotted import path, and all three variants take `scheduler=` on `generate_image()`. The API exposes a **bool, not that path** — a caller-supplied dotted path would be an arbitrary module import in the server process — so the server picks the class and the request only chooses whether to use it.

It's off by default because it changes the pixels a given strength produces: an existing seed/strength pair keeps reproducing its old image unless you ask. With it on, `effective_image_strength` on the `start` event reports the strength that actually took effect rather than a floored bucket, so the two modes stay distinguishable from the stream alone. Two edge cases still report a fraction of `0` and leave the schedule untouched, since mflux's own clamps have already moved the start off the position the strength names: a strength below `1/steps` (floored *up* to rung 1), and `1.0` (starts past the last rung, with no steps to run).

Two caveats worth stating. The interpolation is done on the request's own already-shifted schedule rather than by re-deriving mflux's sigma-shift math, so a half-step lands *near*, not exactly on, the rung that twice as many steps would have given — within 0.001 across the low-index region img2img actually uses, growing to about 0.013 at the very tail (measured for Z-Image-Turbo at 1024×1024; for FLUX.1-schnell, whose schedule isn't shifted at all, it's exact). Every bundled client exposes it: `--fractional-start` on both terminal clients, a checkbox beside Image strength in the harness, and a `fractional_start` argument on the MCP `generate_image` tool.

The third caveat is which models can take it at all. `FractionalStartLinearScheduler` subclasses mflux's `LinearScheduler` and reaches the model by *replacing* the scheduler the variant would otherwise have chosen for itself — which is a no-op plus one moved rung on a model that runs the linear schedule anyway, and a different sampler entirely on one that doesn't. So it is offered only where mflux's own default is linear, and rejected with a 400 elsewhere: Z-Image (base), Krea-2, Krea-2-Raw and every FLUX.2 Klein checkpoint pick flow-match or `er_sde` instead. `supports_fractional_start` on [`/health`](#get-health) is the machine-readable form; plain `image_strength` is unaffected and works everywhere.

```bash
uv run clients/stream_client.py "a lighthouse" --steps 10 --image input.png --image-strength 0.25 --fractional-start
```

The image is scaled to the request's `width`/`height` before use, so it need not match them. Internally, the base64 payload is decoded to a temp file for the duration of one generation (mflux's `image_path` wants an actual path, not bytes) and removed once that generation finishes — nothing is written that outlives the request.


### Inpainting a region

Send a `mask` alongside `image` to regenerate only part of the frame. White in the mask means "the model may change this"; black means "hold this to the input". The mask must be the same pixel size as `image` — a mismatch is a 400 rather than a stretch-to-fit, because a stretched mask silently selects a different region than the one that was painted.

```bash
curl -N http://127.0.0.1:8000/mfluxible/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"a green potted cactus on a wooden table\",
       \"image\": \"$(base64 -i table.png)\",
       \"image_strength\": 0.0,
       \"mask\": \"$(base64 -i mask.png)\",
       \"mask_feather\": 8}"
```

**Set `image_strength` near `0`.** This is the one place mflux's inverted convention bites hardest: with a mask, `image_strength` no longer governs how much of the *frame* survives — the mask does that — it governs only how much of the original content *inside* the mask survives. The harness's usual `0.4` leaves the old object standing in the region you just painted, which reads as the mask having been ignored. `0.0` starts the masked region from rung 1 of the schedule, near pure noise, and is what actually replaces content. Because the useful range is narrow and still quantized to `1/steps`, [`fractional_start`](#fractional-start) is worth more here than it is for plain image-to-image.

How it works: the kept region is written back over the latents after **every** denoising step, re-noised to exactly the level that step reached. That is exact rather than approximate because every model here noises an image by plain interpolation — mflux's own `add_noise_by_interpolation` is `(1 - sigma) * clean + sigma * noise` — so "the original at this step's noise level" is that one line, and on the last step `sigma` is 0 and the kept region lands on the encoded original. The model still *sees* the kept region at every step, which is what lets new content match the original's lighting, perspective and grain.

`mask_composite` (on by default) then pastes the result back over the input through the mask in pixel space. The blend already holds the kept region during denoising, but the whole frame still passes through the VAE on the way out, which moves it by roughly 1.5/255 on average — at the encode/decode floor, not something the blend did wrong. Invisible on a photograph; visible on text, a logo or a flat colour. Turn it off to get the model's own decode of the whole frame.

**`mask_feather` is for the staircase, not for blending regions.** A grey mask value doesn't soften a seam in the finished image; it means "hold this pixel partway to the original *at every denoising step*", which pins that band to a ghost of the input and stops the model committing to anything there. At radius 8–16 that band is a couple of latent cells wide and the effect is what you want. At radius 96 on a 1024px image it covers most of the masked object and the result is a visible double exposure — the old content and the new one cross-faded. The bundled harness caps its field at 32 for this reason.

Three things this approach does **not** do, all inherent rather than incidental:

- **It fills, it does not erase.** Nothing tells the model the region is a hole to continue the background across; it only knows your prompt belongs there. Masking an object and prompting "an empty wooden table" gets you a table-shaped object in the hole, not a gap. On a guidance-distilled model there isn't even a negative prompt to say "nothing here" with.
- **The mask edge is a hard boundary.** Content that wants to cross it is cut off at it — a tall vase masked tightly around a mug comes back with its flowers sliced flat. Give the mask room for everything the new content needs, shadow and reflection included.
- **It doesn't have to match the tone of what it keeps.** Nothing constrains the new content to the kept region's exact exposure or colour, so on a large flat area — a wall, a sky — the model can render its own slightly different grey and the hard mask edge turns that into a visible rectangle. Measured on a 1024×1024 Z-Image-Turbo run: the wall inside the mask came back at luminance ~159 where the input's was ~149, which is the model's choice and not an artefact of `mask_composite` (the same gap is in the raw decode with compositing off). Masking tightly around the subject rather than boxing a swathe of background is the practical answer.
- **The mask is quantized to the latent grid**, one cell per 8 pixels — 128×128 cells at 1024×1024. Detail finer than that is averaged away before the model sees it, which is also why `mask_feather` below 8 does almost nothing.

Which models accept a mask is the same question as [`fractional_start`](#fractional-start)'s, for the same reason: holding the region in place means handing mflux a scheduler, and that *replaces* the one the variant would have picked for itself. So it is offered where that default is already linear and rejected with a 400 elsewhere. `supports_mask` on [`/health`](#get-health) is the machine-readable form.

The mask is read as **luminance**, and any alpha channel is ignored. This is worth knowing because OpenAI's `/v1/images/edits` uses the opposite convention — transparency marks the editable region — so a mask drawn for that API would inpaint the complement of what you meant here. That endpoint therefore still rejects `mask` outright rather than reading it with the wrong convention; see [`POST /v1/images/edits`](#post-v1imagesedits).

## `POST /v1/images/generations`

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

## `POST /v1/images/edits`

A genuine [OpenAI Images-Edit-API](https://platform.openai.com/docs/api-reference/images/createEdit)-compatible endpoint for image-to-image, so an existing OpenAI-client-based image-editing tool can point at this server unmodified. Unlike every other endpoint here, the request is `multipart/form-data`, not JSON — matching OpenAI's own request shape for this one.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string (form field) | required | |
| `model` | string (form field) | required | same validation as [`/v1/images/generations`](#post-v1imagesgenerations) |
| `image` | file | required | the input image |
| `mask` | file | — | **not supported** — a 400 if present, rather than silently ignored. The native endpoint does inpaint (see [Inpainting a region](#inpainting-a-region)), but the two APIs disagree about what a mask is: OpenAI marks the editable region with **transparency**, mfluxible's own `mask` field marks it **white**. Reading one as the other doesn't fail, it inpaints the complement — every region you meant to keep — so this refuses until the alpha convention is actually honoured. Use `POST /mfluxible/v1/images/generations` for masked work |
| `n` | int (form field) | 1 | same as `/v1/images/generations` |
| `size` | string (form field) | `"1024x1024"` | same as `/v1/images/generations` |
| `response_format` | string (form field) | `"b64_json"` | same as `/v1/images/generations` |
| `stream` | bool (form field) | false | same SSE transport and event shape as `/v1/images/generations` |
| `partial_images` | int (form field) | 0 | same as `/v1/images/generations` |
| `image_strength` | float (form field) | 0.4 | **not part of OpenAI's request shape** — accepted as an extension the same way `partial_images` already is. See [Image-to-image](#image-to-image) for what it controls (mflux's convention, the inverse of some other tools') |

Response shapes, streaming behavior, and the error envelope are all identical to [`/v1/images/generations`](#post-v1imagesgenerations) above.

## `GET /v1/models`

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

## `POST /v1/chat/completions`

Not a chat model — a stub that exists so a frontend requiring an actual "chat model" behind its Native/agentic tool-calling mode (Open WebUI is the motivating case) can point that connection at mfluxible too, instead of running a separate LLM (Ollama, `llama-server`, ...) just to decide "yes, call the image tool" on every message. See [`server/chat_stub.py`](../server/chat_stub.py) for the full rationale; the short version is there's no reasoning to replace, so there's nothing an LLM gets right that a hardcoded rule doesn't get right for free, at zero extra memory (it's pure Python in the same process as the diffusion model — no weights, no inference).

It plays exactly one deterministic turn of OpenAI's function-calling protocol:

- A request whose last message is a plain user turn, with a `generate_image` function offered in `tools` → responds with a `tool_calls` message invoking it, `{"prompt": "<the user's message>"}` as the arguments.
- A request whose last message is a `tool` result (i.e. the caller already ran `generate_image` and is asking for a closing reply) → responds with a short fixed acknowledgment, `finish_reason: "stop"`.
- A request with no `generate_image` tool offered at all → responds with fixed text explaining why (most likely cause in Open WebUI: Capabilities or Builtin Tools → Image Generation isn't enabled for this model).

It only ever recognizes a tool literally named `generate_image` — the name [Open WebUI's own builtin tool uses](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/tools/builtin.py) — offered in that specific request's `tools` list; it never guesses at one that wasn't offered. There's no general conversation, no other tools, no multi-turn reasoning — if that's what's needed, point the chat connection at a real model instead.

Both streaming and non-streaming (`stream: true`/`false`) are supported, in OpenAI's own chat-completion / chat-completion-chunk shapes. `model` in the request isn't validated against anything — unlike the image endpoints, there's no real model here to be inconsistent with.

## `OPTIONS /v1/images/generations`

`204`, with an `Allow: POST, OPTIONS` header and no body. A reachability probe, not CORS — it exists because SillyTavern's "Validate" button is a bare server-side `OPTIONS` on this path, sent without an `Access-Control-Request-Method` header and therefore never seen by the CORS middleware (which would otherwise answer it). See [`POST /sdapi/v1/txt2img`](#post-sdapiv1txt2img) below.

A browser's real preflight on this path is still handled by CORS as before; the two don't collide, because Starlette tells them apart by that header. This says nothing about whether the weights have finished loading — that's `model_loaded` on [`/health`](#get-health).

## `POST /sdapi/v1/txt2img`

An [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)-shaped `txt2img`, deliberately narrow: it exists for [SillyTavern](clients.md#sillytavern), which has no OpenAI-compatible image source, and it implements only what SillyTavern's `sdcpp` source actually sends. It is not an A1111 emulation — there is no `/sdapi/v1/options`, `/progress`, `/interrupt`, img2img, or model switching.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | `""` | |
| `negative_prompt` | string | unset | **Dropped** rather than rejected on a model that can't act on it (see below) |
| `width` / `height` | int | 512 | Floored to a multiple of 16, as everywhere else |
| `steps` | int | the model's own default | Honoured as sent — it's a visible slider in the caller's UI |
| `cfg_scale` | float | the model's own default | Maps to `guidance`. **Dropped** rather than rejected on a guidance-distilled model |
| `seed` | int | random | A1111's `-1` means random, and is mapped to that rather than used as a literal seed |
| `batch_size` / `n_iter` | int | 1 | A **400** if either is anything but 1 — mfluxible generates one image per request |
| `model` | string | unset | **Accepted and ignored.** One model runs per process; [`GET /v1/models`](#get-v1models) is the honest answer to what's loaded, and `info.sd_model_name` below reports what actually ran |

Everything else A1111's much larger payload can carry (`sampler_name`, `scheduler`, `clip_skip`, ...) is accepted and ignored — mflux picks its own sampler and has no equivalent for the rest.

Response (A1111's shape; `info` is a JSON **string**, parsed a second time by the caller):

```json
{
  "images": ["<base64 png>"],
  "parameters": {"prompt": "a puffin", "width": 1024, "...": "the request as received"},
  "info": "{\"prompt\": \"a puffin\", \"negative_prompt\": \"\", \"seed\": 42, \"all_seeds\": [42], \"width\": 1024, \"height\": 1024, \"steps\": 9, \"cfg_scale\": null, \"sd_model_name\": \"z-image-turbo\"}"
}
```

`parameters` echoes the request; `info` reports what actually took effect. That distinction is the point: a `cfg_scale` of 7 in `parameters` against `"cfg_scale": null` in `info` is how a caller sees that this model ignored the field.

### Why this endpoint drops what the native API rejects

Everywhere else, a field the loaded model can't act on is a **400** rather than a silent no-op — see [Models](server.md#models) for that reasoning. This endpoint inverts it, for one caller-specific reason: SillyTavern's `sdcpp` routes answer every upstream failure with a bare `500` and no body, logging the real message only to its own server console. A 400 from here reaches the user as a failed generation with no reason attached, so the careful wording of a rejection buys nothing — and since SillyTavern sends `cfg_scale` and `steps` on *every* request from its own sliders (whose defaults, 7 and 20, suit neither a guidance-distilled model nor a 9-step one), rejecting them would make the source unusable rather than correcting anyone.

The line between ignoring and rejecting is whether the caller could otherwise detect the difference. A dropped `cfg_scale` changes the image, but the caller named no specific image and has nothing to compare against. `batch_size: 4` answered with one image is a concretely wrong result to a request that named a number — so that one is still a 400.

A negative prompt is dropped under exactly the conditions that would otherwise reject it: no negative branch on this model, *or* classifier-free guidance not actually switched on above the floor, which is the "necessary but not sufficient" case described in [Models](server.md#models). The drops mirror those rules rather than sharing code with them, so the endpoint re-checks its own output against the real validator — a rule this ever fails to mirror surfaces as a 400 instead of a silently wrong image.
