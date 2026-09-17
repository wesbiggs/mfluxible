# Server

## How it works

mflux's `generate_image()` is synchronous: it runs the whole denoising loop in one thread and invokes registered callbacks at each step (`InLoopCallback`). The server runs that call on a dedicated single-worker thread (not just any worker thread — see the comment at the top of `server/engine.py` for why that matters with MLX) and bridges each callback invocation back to the event loop as an SSE event via `call_soon_threadsafe`, so a single async server can stream progress out of an otherwise blocking call.

Only one generation runs at a time (there's a lock) — MLX/Metal generation against one shared model instance isn't set up here for concurrency.

Preview images are decoded the same way mflux's own `--stepwise-image-output-dir` CLI flag does internally (see `mflux.callbacks.instances.stepwise_handler.StepwiseHandler`), just streamed instead of written to disk.

## Configuration

Environment variables for `server.py`, all optional.

| Variable | Default | Purpose |
|---|---|---|
| `MFLUXIBLE_MODEL` | `z-image-turbo` | Which model to run — any of the fifteen in [Models](#models) |
| `MFLUXIBLE_QUANTIZE` | `8` | Quantization bits; try `4` for less memory, `none` for full precision |
| `MFLUXIBLE_MODEL_DIR` | `~/.cache/mfluxible` | Where quantized weights are cached (see [Model cache](#model-cache)) |
| `MFLUXIBLE_MLX_CACHE_LIMIT_MB` | `1024` | Cap on MLX's reusable buffer cache; `none` for MLX's own default (see [Memory](#memory)) |
| `MFLUXIBLE_MLX_WIRED_LIMIT_MB` | unset | Wire this much memory so the OS cannot page the weights out (see [Memory](#memory)) |
| `MFLUXIBLE_LORA_PATHS` | unset | Comma-separated local LoRA `.safetensors` files to bake in (see [LoRAs](#loras)) |
| `MFLUXIBLE_LORA_SCALES` | `1.0` each | Comma-separated scales matching `MFLUXIBLE_LORA_PATHS` |
| `MFLUXIBLE_BASIC_AUTH_USERNAME` | unset | HTTP Basic username; auth is off unless this **and** the password are both set (see [Authentication](#authentication)) |
| `MFLUXIBLE_BASIC_AUTH_PASSWORD` | unset | HTTP Basic password; required alongside the username |
| `MFLUXIBLE_CORS_ORIGIN_REGEX` | `https?://(localhost\|127\.0\.0\.1)(:\d+)?` | Origins to reflect back in CORS (see [CORS](#cors)) |
| `MFLUXIBLE_CORS_ORIGINS` | unset | Comma-separated exact-match origins, in addition to the regex |
| `MFLUXIBLE_REGIONS_DIR` | unset | Enables object detection and says where images are stashed (see [Object detection](#object-detection)) |
| `MFLUXIBLE_REGIONS_TIMEOUT` | `180` | Seconds a detection waits for a worker's answer before giving up |
| `MFLUXIBLE_REGIONS_COMMAND` | `claude -p …` | What `region_worker.py` runs per detection (see [Using a different tool](#using-a-different-tool)) |
| `MFLUXIBLE_REGIONS_MODEL` | `opus` | Which model the **default** command uses; ignored if you set your own command |
| `HF_TOKEN` | unset | Not an mfluxible variable — `huggingface_hub` reads it, and gated models need it (see [Gated weights](#gated-weights-and-hf_token)) |

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

### Object detection

Off unless `MFLUXIBLE_REGIONS_DIR` names a directory. Switched on, the server gains the
three `/mfluxible/v1/regions/` endpoints ([API](api.md#post-mfluxiblev1regionsdetect))
and the harness grows a **Find objects** button that fills a row of clickable regions,
each of which sets the mask to one rectangle.

**The server does no detection.** It never loads a vision model, never holds an API key
and never makes an outbound call — it writes a file, holds one job in memory, and copies
a JSON array from one request to another. The work is done by
`server/region_worker.py`, which runs alongside (see
[Running the worker](#running-the-worker) below). With nothing running, the feature simply reports no worker attached and the harness says so.

That split is deliberate rather than tidiness: the worker runs an arbitrary configured
command with filesystem access, and an HTTP endpoint that spawned one would be by far
the most dangerous thing here. As a client it cannot be reached from the network at all,
and starting it is what consent to that command running looks like.

**Which detector runs is the worker's business, not the server's.**
`MFLUXIBLE_REGIONS_COMMAND` names the whole command line — [Claude Code](https://claude.com/claude-code)
by default, but any program that prints a JSON array of labelled boxes to stdout will
do, including a local detector or a script of your own. It is set on the worker rather
than read by the server; see [Using a different tool](#using-a-different-tool).

One variable both enables and configures because there is no useful "on, but nowhere to
put anything" state. Point the worker at the same directory: it is also the working
directory the detection command runs in, which matters twice on the default — a stashed
image inside it is already readable without widening the CLI's allowed roots, and a
directory with no `CLAUDE.md` keeps a one-shot detection from loading a project's
instructions on every call.

Stashed images are pruned after an hour, on the next detection.

#### Running the worker

`server/region_worker.py` is what makes the harness's **Find objects** button work. The
harness can see an image and drive the GPU but has no vision model; this process
supplies one. It long-polls the server for a pending detection, runs a command against
the stashed image, and posts the regions back for the harness's open stream to deliver.

By default that command is [Claude Code](https://claude.com/claude-code) — the CLI on
`PATH` and already signed in — but nothing here is specific to it; see
[Using a different tool](#using-a-different-tool). Start the server with
`MFLUXIBLE_REGIONS_DIR` (see [the section above](#object-detection)) and point
both at the same directory:

```bash
MFLUXIBLE_REGIONS_DIR=~/.cache/mfluxible/regions uv run server/region_worker.py
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `MFLUXIBLE_REGIONS_COMMAND` | `claude -p {prompt} --allowedTools Read --model $MFLUXIBLE_REGIONS_MODEL` | The command to run per detection |
| `MFLUXIBLE_REGIONS_MODEL` | `opus` | Which model the **default** command uses; ignored if you set your own command |
| `MFLUXIBLE_BEARER_TOKEN` | unset | Sent as `Authorization: Bearer …`, if a proxy gates the API |

`--url` points at the server's base URL (default `http://127.0.0.1:8420`). As with the
terminal clients, setting `MFLUXIBLE_BEARER_TOKEN` *and* putting credentials in the URL
is refused rather than resolved.

#### Using a different tool

What the worker actually requires is narrow: **a program that prints a JSON array of
labelled boxes to stdout.** Anything that can do that from an image path works — another
model's CLI, a local detector, a script of your own.

```json
[{"label": "red apple", "box": [0.305, 0.344, 0.712, 0.736]}]
```

`box` is `[x0, y0, x1, y1]` as fractions of the frame, `0,0` at the top-left. Extra prose
around the array is fine (a fenced code block is read too); entries with a malformed or
out-of-range box are dropped rather than repaired, and at most 8 are kept.

The command is a template with two placeholders:

| Placeholder | |
| --- | --- |
| `{image}` | The image's filename. The working directory is the stash directory, so a bare name resolves. |
| `{prompt}` | The worker's standard detection prompt, already naming the image. Use it for a prompt-driven tool; leave it out entirely for a detector that doesn't take one. |

```bash
# a local detector that already speaks the format
MFLUXIBLE_REGIONS_COMMAND='detect-objects --format json {image}'

# a prompt-driven CLI that takes the image as an attachment
MFLUXIBLE_REGIONS_COMMAND='llm -m some-vision-model -a {image} {prompt}'

# your own wrapper, for a tool whose coordinates need converting
MFLUXIBLE_REGIONS_COMMAND='python3 ~/bin/boxes_to_fractions.py {image}'
```

That last one is the common case, and it is deliberate that it needs a wrapper: tools
disagree about coordinates — pixels, 0–1000, `xywh` — and a conversion setting here would
turn a wrongly-scaled box into a plausible-looking one. Keeping a single reader keeps a
bad box visibly bad.

The template is split with `shlex` and run **without a shell**, so quoting behaves as you
would expect while `;` and `|` are ordinary argument characters. Splitting happens before
the placeholders are filled in, so a prompt containing quotes or newlines stays exactly
one argument and can never reshape the command.

**On the default command's model choice — and the usual trade doesn't apply here.**
Measured on the same 768×768 photograph with the same prompt, `opus` was both more
accurate *and* faster than `sonnet` — 11.4s against 66.6s. Sonnet placed a "red apple"
box at `[0.28, 0.28, 0.68, 0.62]`: about the right size, in the wrong place, clipping the
fruit's bottom third while taking in a band of forearm. Opus gave
`[0.305, 0.344, 0.712, 0.736]` against `[0.310, 0.344, 0.694, 0.729]` measured by hand —
within two percent on every edge, and quoted to three decimals rather than the round
two-decimal numbers that signal estimating on a grid rather than measuring. Sonnet had
produced a good box on an earlier run of the same image, so this is variance rather than
a fixed offset, which is worse: nothing downstream can tell a good box from a bad one.

Which is why a region is a starting point rather than a result. Clicking a chip sets the
mask to that one rectangle and opens the editor, so the box can be nudged before it
costs a generation — the same reason the MCP tool has `preview_mask`.

Detections are one at a time: starting a second supersedes the first, and the harness
says so rather than leaving the old one to time out.


## Models

One server process runs one model, chosen at startup with `MFLUXIBLE_MODEL`:

| `MFLUXIBLE_MODEL` | Weights | RAM q8 / q4 | Default steps | Guidance | Negative prompt | Fractional start |
|---|---|---|---|---|---|---|
| `z-image-turbo` (default) | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 11.0 / 5.9 GB | 9 | — | — | yes |
| `z-image` | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | 11.0 / 5.9 GB | 50 | default 4.0 | yes | — |
| `flux-schnell` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) † | 18.0 / 9.6 GB | 4 | — | — | yes |
| `flux-dev` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) † | 18.0 / 9.6 GB | 25 | default 3.5 | — | yes |
| `krea-dev` | [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) † | 18.0 / 9.6 GB | 25 | default 3.5 | — | yes |
| `qwen-image` | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | 30.8 / 16.4 GB | 20 | default 3.5 | yes | yes |
| `krea-2` | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) † | 18.8 / 10.2 GB | 8 | default 1.0 | above guidance 1.0 | — |
| `krea-2-raw` | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) † | 18.8 / 10.2 GB | 25 | default 1.0 | above guidance 1.0 | — |
| `ernie-image-turbo` | [baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) | 12.8 / 6.9 GB | 8 | — | — | yes |
| `ernie-image` | [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) | 12.8 / 6.9 GB | 50 | default 4.0 | yes | yes |
| `flux2-klein-4b` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 8.6 / 4.6 GB | 4 | — | — | — |
| `flux2-klein-9b` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) † | 18.5 / 9.9 GB | 4 | — | — | — |
| `flux2-klein-9b-kv` | [black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) † | 18.5 / 9.9 GB | 4 | — | — | — |
| `flux2-klein-base-4b` | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | 8.6 / 4.6 GB | 50 | default 1.5 | — | — |
| `flux2-klein-base-9b` | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) † | 18.5 / 9.9 GB | 50 | default 1.5 | — | — |

† Gated on Hugging Face — needs an account, an accepted licence and a token before the weights will download. See [Gated weights](#gated-weights-and-hf_token).

**RAM q8 / q4** is the quantized weights at `MFLUXIBLE_QUANTIZE=8` (the default) and at `4`, summed across the directories mflux downloads (`transformer/`, `text_encoder*/`, `vae/`). MLX quantizes in groups of 64 weights and stores an fp16 scale and bias per group, so a weight costs `bits/8 + 0.0625` bytes, and the VAE is not quantized at all — which is why **q4 lands at ~53% of q8 rather than half**: that 0.0625 is fixed, so it is a larger share of a 4-bit weight than an 8-bit one. `none` gives full precision, roughly twice the q8 figure.

Expect the process to sit roughly 2–3 GB above the figure while generating: MLX's reusable buffer cache (capped at 1 GB by default, see [Memory](#memory)) plus activations, which scale with resolution rather than step count.

These are derived from each checkpoint's published parameter count rather than measured one by one, but the formula agrees with all three quantized caches measured locally to within 2% — `flux2-klein-4b` caches at 8.59 GB at q8 and 4.63 GB at q4, `z-image-turbo` at 10.74 GB, which runs at 12–13 GB resident.

Base and turbo variants of the same family share an architecture, so they share a footprint: `z-image` and `z-image-turbo` are both 6.2B parameters, as are `ernie-image` and its turbo, and the three 9B Klein checkpoints. What separates them is step count and guidance behaviour, not size.

Disk costs more than RAM on the first run, and it's the download that dominates: every checkpoint here publishes bf16 weights, about 1.9× the q8 figure, *except* `z-image-turbo`, whose transformer ships fp32 — 33 GB downloaded to cache 10 GB. The quantized copy is then written separately, so peak disk is roughly 3× the q8 figure (4× for `z-image-turbo`). Once that quantized cache exists the Hugging Face download under `~/.cache/huggingface/hub/` can be deleted — it's only needed again if you change `MFLUXIBLE_QUANTIZE` or the LoRA configuration, both of which quantize afresh into their own cache directory.

mflux's own aliases work too (`schnell`, `dev`, `qwen`, `zimage`, `klein-4b`, `krea2`, …), and an unrecognised name fails at startup with the list of valid ones — before anything is downloaded.

**Only the model you select is ever fetched.** All fifteen are named in `server/models.py`, but an entry there is inert data: its mflux imports are deferred into a loader function that runs at load time, and mflux downloads weights inside the model's constructor, not at import. The fourteen you aren't running cost nothing beyond their row in that table.

Switching models means restarting the server, the same way `MFLUXIBLE_QUANTIZE` and LoRAs do. Each model + quantization + LoRA combination keeps its own quantized cache directory, so switching back doesn't re-quantize.

What differs per model, beyond the step count:

- **Guidance.** The distilled models — Z-Image-Turbo, FLUX.1-schnell, ERNIE-Image-Turbo and the three non-`base` FLUX.2 Klein checkpoints — have nowhere for a guidance value to go: mflux forces it to 0, builds no guidance embedder, or hard-errors on any value but 1.0. Sending `guidance` to those is a 400 rather than a field quietly dropped. FLUX.1-dev and FLUX.1-Krea-dev take distilled guidance; Z-Image, Qwen-Image, ERNIE-Image, Krea-2 and the FLUX.2 `base` checkpoints run true classifier-free guidance.
- **Negative prompts.** A negative prompt needs CFG to have any effect, and every model here builds its unconditional branch only *above* guidance 1.0. So Z-Image, Qwen-Image, ERNIE-Image and Krea-2 accept one and the rest are a 400 — and on Krea-2, whose default guidance is mflux's own 1.0, a negative prompt is also a 400 unless you raise `guidance` alongside it. FLUX.1 and FLUX.2 have no negative branch at all. That branch is also why CFG steps are expensive: the transformer runs twice per step, conditional and unconditional, whether or not you send a `negative_prompt`.
- **Fractional start.** [`fractional_start`](api.md#fractional-start) works by extending mflux's *linear* schedule, so it's only offered on models that would have run one. Z-Image (base), Krea-2 and every FLUX.2 Klein pick a different sampler for themselves — flow-match, or Krea-2's `er_sde` — and asking for a fractional start there is a 400. Plain `image_strength` still works on all of them, quantized to `1/steps` as usual.
- **Size.** Z-Image-Turbo is among the smallest here and Qwen-Image much the largest (a ~20B transformer alongside a multimodal text encoder), with Krea-2 (12B) and the 9B FLUX.2/ERNIE checkpoints in between. For scale, Z-Image-Turbo alone occupies 10GB of quantized weights at `MFLUXIBLE_QUANTIZE=8` and 5.5GB at `4`. On a 32GB machine, expect to want `4` or lower for anything above ~9B, and read [Memory](#memory) first — running out of headroom doesn't fail loudly, it just makes every generation slow.

All the bundled clients leave `steps` to the server unless you set it, so they follow whichever model is loaded without reconfiguration. The MCP tool and the browser harness go further and read [`/health`](api.md#get-health): the harness only shows Guidance, Negative prompt and Fractional start when the model accepts them, and the MCP tool refuses those arguments up front rather than spending a round trip to be told no.

### Gated weights and `HF_TOKEN`

Eight of the fifteen repos above are gated († in the table). Getting at those takes an account, an accepted licence *and* a token — the token on its own is not enough:

1. Open the model page signed in and click **Agree and access repository**. All eight are `gated: auto`, so access lands the moment you accept — there's no queue waiting on a human.
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A fine-grained token works too, but it needs *"Read access to contents of all public gated repos you can access"* ticked, or it will 401 on exactly these models.
3. Store it once with `hf auth login` (which writes `~/.cache/huggingface/token`), or export `HF_TOKEN` in the shell that starts the server. Either way it's `huggingface_hub` that reads it, not mfluxible — which is why there's no `MFLUXIBLE_*` variable for it, and why a dedicated machine needs its own copy (see [Running on a dedicated machine](#running-on-a-dedicated-machine)).

Gating is a per-repo switch with no particular relationship to the licence — FLUX.1-schnell is Apache 2.0 and still answers an anonymous request with a 401 — and it's per-checkpoint rather than per-family: `flux2-klein-4b` and `flux2-klein-base-4b` are open while all three 9B Kleins are gated. Check the row rather than generalising from the sibling you ran last.

Missing access fails at **startup**, not at request time. `engine.load()` runs in the FastAPI lifespan and mflux downloads weights inside the model's constructor, so a 401 surfaces while the server is still booting and nothing is listening on the port yet. The default `z-image-turbo` is ungated, so the [quickstart](../README.md) needs no token at all.

One thing to know before killing a download to add a token: **it will not resume.** `huggingface_hub` 1.x writes each file to a process-unique `<etag>.<uuid>.incomplete` and only renames it into place once complete, so the restart starts that file from zero and the dead run's partials are orphaned. That's deliberate — a name unique per process means a filesystem whose `flock` silently lets two writers through costs duplicated bandwidth instead of a corrupted blob. `hf cache prune` reclaims the orphans, but run it only when nothing is downloading: it globs every `*.incomplete` in the cache with no check for whether something is still writing to one.

Adding a model mfluxible doesn't already run is a [CONTRIBUTING](../CONTRIBUTING.md#adding-a-model) topic.

## Running on a dedicated machine

The server doesn't have to run on the same machine as the clients. Point it at a spare Apple Silicon box (a Mac Mini, say) and bind it to the network instead of loopback:

```bash
uv run uvicorn server:app --app-dir server --host 0.0.0.0 --port 8420
```

Then everything else just points at that host instead of `127.0.0.1`, no code changes needed:

- `stream_client.py` / `stream_client.js`: `--url http://mac-mini.local:8420/mfluxible/v1/images/generations`
- `mcp_server.py`: set `MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations` when registering it, e.g. `claude mcp add mfluxible --scope user -e MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations -- /path/to/mfluxible/.venv/bin/python /path/to/mfluxible/clients/mcp_server.py` — or, in Claude Desktop's config, `"env": {"MFLUXIBLE_URL": "http://mac-mini.local:8420/mfluxible/v1/images/generations"}` alongside `command`/`args`

### Authentication

There's none by default, and two ways to add it. They are alternatives rather than layers — running both means maintaining two secrets for one server.

| | Built-in HTTP Basic | Reverse proxy + bearer token |
|---|---|---|
| Configured in | `MFLUXIBLE_BASIC_AUTH_*` | [`Caddyfile.example`](../Caddyfile.example), or any proxy |
| Gates | every endpoint, `/` and `/docs` included | only the endpoints that cost GPU time |
| Browser harness | the browser's own credential prompt | the harness's **API token** field |
| Open WebUI | no — it sends a bearer token, not Basic | yes — its "API key" field is exactly this |
| SillyTavern | no (see below) | no (see below) |
| TLS | needs a proxy in front anyway | the proxy terminates it |
| Extra moving parts | none | a proxy process to run and keep running |

Pick the first if everything talking to the server is a terminal client or a browser and you'd rather not run a proxy. Pick the second if an OpenAI-compatible frontend is in the picture, or you want HTTPS, or you'd rather the harness stayed reachable without a credential.

#### Built-in HTTP Basic

Set **both** `MFLUXIBLE_BASIC_AUTH_USERNAME` and `MFLUXIBLE_BASIC_AUTH_PASSWORD` to require HTTP Basic credentials on every endpoint — the API, `/health`, `/docs`, and the browser harness at `GET /`:

```bash
MFLUXIBLE_BASIC_AUTH_USERNAME=tavern MFLUXIBLE_BASIC_AUTH_PASSWORD='a long random string' \
  uv run uvicorn server:app --app-dir server --host 0.0.0.0 --port 8420
```

Setting only one of the two leaves auth **off**: a half-finished deployment is far likelier than a deliberately blank username, and quietly serving unauthenticated is the failure worth avoiding. Bind to a non-loopback address without both set and the server logs a warning at startup, before it downloads anything.

Each client carries the credentials its own way — `curl -u`, `--url http://user:pass@host:8420/...` for either terminal client, and a browser will prompt you at `GET /`. The MCP tool has no Basic support; use a bearer token for it.

#### A reverse proxy with a bearer token

[`Caddyfile.example`](../Caddyfile.example) in the repo root is a working site block: leave the app on `127.0.0.1:8420` with no `MFLUXIBLE_BASIC_AUTH_*` set, replace the placeholder token, and point the proxy at it.

```bash
openssl rand -hex 32          # the token
caddy run --config Caddyfile  # or paste the site block into an existing Caddyfile
```

It deliberately carries no global `{ ... }` options block, so it pastes into a Caddyfile you already have — a second global block is a parse error, not a merge.

What it gates is the important part. `/`, `/docs`, `/openapi.json` and `/health` are served **without** a token, because none of them is anything the public repo doesn't already show, and leaving `/health` open is what lets the harness render the right fields for the loaded model before you've typed a credential. Everything that costs GPU time is gated by exclusion, so an endpoint added later is protected the day it lands. `tests/test_proxy_config.py` fails if a generating endpoint ever appears in the open list.

Clients send the token as `Authorization: Bearer <token>`:

- **Harness** — paste it into **Advanced → API token**. It's kept in `localStorage`, so it survives a reload and "Reset all"; entering it re-probes `/health`.
- **`stream_client.py` / `stream_client.js`** — `export MFLUXIBLE_BEARER_TOKEN=...`. Environment rather than a flag so the secret stays out of shell history and `ps`. Setting it *and* putting credentials in `--url` is refused rather than silently resolved.
- **`mcp_server.py`** — `MFLUXIBLE_BEARER_TOKEN` where the tool is registered; see [the MCP configuration table](mcp.md#configuration).
- **Open WebUI** — its API-key field, which sends this header already.
- **`curl`** — `-H "Authorization: Bearer $MFLUXIBLE_BEARER_TOKEN"`.

Two caveats worth stating plainly. Caddy's `header` matcher is a plain string comparison, not the constant-time one `server/auth.py` uses — not practically exploitable across a network with a high-entropy token, but a reason to use `openssl rand` rather than a passphrase. And a bearer token over plain HTTP is readable in transit exactly as Basic is, so the `tls internal` line in the example is doing real work; on a tailnet, `tls <host>.<tailnet>.ts.net` gets a publicly-trusted certificate and needs no trust step on any device already on the tailnet.

#### What neither of these is

- **Not a substitute for network placement.** Keep binding to `0.0.0.0` only on a network you trust (home LAN, Tailscale/VPN) and never expose the server directly to the internet — auth is defence in depth on that network, not permission to skip it. With a proxy this matters doubly: the app must stay on loopback, or the unauthenticated original is sitting on the network *beside* the proxy rather than behind it.
- **CORS is not access control.** It's a browser policy: `curl` and every non-browser client ignore it entirely. It also doesn't apply to the bundled harness at all, whose Server URL defaults to a relative path and is therefore same-origin. See [CORS](#cors).

#### Clients with no credential setting are blocked by either scheme

Both schemes are ordinary HTTP auth, so any client that can't be told to send an `Authorization` header 401s against both, and there's nothing configurable on this side that changes that. Worth checking per *integration* rather than per application: the same program often has a credential field on one connection type and none on another — SillyTavern as of 1.18.0 is the example to hand, where the `stable-diffusion.cpp server` source this server answers sends no credentials on any of its three calls, while its AUTOMATIC1111 source has a Basic-auth field and its chat side has its own.

If you need one of these, either keep the server unauthenticated on a trusted network, or — understanding the trade — open just the paths that client needs in your own proxy config. For SillyTavern that's `OPTIONS /v1/images/generations`, `GET /v1/models` and `POST /sdapi/v1/txt2img`; the last of those generates, so opening it means anyone who can reach the proxy can spend GPU time on it. See [SillyTavern](clients.md#sillytavern).

## Troubleshooting

### `WARNING: Invalid HTTP request received.`

This is uvicorn's own log line (not mfluxible's) for a connection that sent bytes it couldn't parse as HTTP at all — most commonly something pointed at this server with `https://` instead of `http://` (mfluxible has no TLS of its own; a TLS ClientHello hitting the plaintext port is exactly the kind of thing that produces this), but also a health-checker or proxy speaking a different protocol at the port, or a stray port scan. uvicorn discards the actual reason and the offending bytes, logging only this fixed message, so there's no way to tell which of those it was from the log line alone — check every URL that points at this server (a reverse proxy config, a client's base-URL setting, ...) for a stray `https://` first; that's the single most common cause in practice.

### `generation failed -- see the server log for the reason.`

Exactly what it says: the generation raised, and the reason is on this process's stderr — the terminal running `uvicorn` — with the full traceback, rather than in the response. That split is deliberate. mflux, MLX and Hugging Face errors are written for whoever is running the server, and they quote absolute cache paths freely (a missing weight file names `~/.cache/huggingface/...` in full, so on the default layout it hands out the account name and disk layout of the host), which tells a client nothing it can act on and tells a *remote* client something it has no business knowing.

So the client side of a 500 is never worth reading closely; the server side always is. If you're running a client on another machine (see [Running on a dedicated machine](#running-on-a-dedicated-machine)), that means going back to the host to find out what happened. A generation that was interrupted rather than failed still reports which step it stopped on, in the response, since that's mfluxible's own message and says nothing about the host.
