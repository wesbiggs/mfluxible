# Server

```bash
pip install mfluxible
mfluxible-server
```

Requires Python 3.11+ on Apple Silicon. See the [README](../README.md#quickstart) for
running it from a checkout instead, and [Configuration](#configuration) for everything
you can set.

## How it works

mflux's `generate_image()` is synchronous: it runs the whole denoising loop in one thread and invokes registered callbacks at each step (`InLoopCallback`). The server runs that call on a dedicated single-worker thread (not just any worker thread — see the comment at the top of `mfluxible/engine.py` for why that matters with MLX) and bridges each callback invocation back to the event loop as an SSE event via `call_soon_threadsafe`, so a single async server can stream progress out of an otherwise blocking call.

Only one generation runs at a time (there's a lock) — MLX/Metal generation against one shared model instance isn't set up here for concurrency.

Preview images are decoded the same way mflux's own `--stepwise-image-output-dir` CLI flag does internally (see `mflux.callbacks.instances.stepwise_handler.StepwiseHandler`), just streamed instead of written to disk.

## Configuration

Every setting below is optional, and there are two ways to set each one: an environment
variable, or a key in a [config file](#config-file). They are the same setting — a key is
its variable with `MFLUXIBLE_` dropped and the case lowered, so `model` sets
`MFLUXIBLE_MODEL`. Where both say something, **the environment wins**.

The last four are read by `mfluxible-vlm-worker` rather than the server, and only matter
under `MFLUXIBLE_VLM_BACKEND=worker`. They are listed here, with the server's, because the
worker is co-configured with it and runs on the same machine — see
[Object detection](#object-detection).

| Variable | Config key | Default | Purpose |
|---|---|---|---|
| `MFLUXIBLE_MODEL` | `model` | `z-image-turbo` | Which model to run — any of the twenty-eight in [Models](models.md) |
| `MFLUXIBLE_QUANTIZE` | `quantize` | `8` | Quantization bits; try `4` for less memory, `none` for full precision |
| `MFLUXIBLE_MODEL_DIR` | `model_dir` | `$XDG_CACHE_HOME/mfluxible` | Where quantized weights are cached (see [Model cache](#model-cache)) |
| `MFLUXIBLE_OUTPUT_DIR` | `output_dir` | `~/Pictures/mfluxible` | Directory of finished images served read-only at `GET /mfluxible/v1/output/<name>`; set it to the MCP tool's `MFLUXIBLE_MCP_SAVE_DIR` (see [`POST /v1/chat/completions`](api.md#post-v1chatcompletions)) |
| `MFLUXIBLE_MAX_BODY_MB` | `max_body_mb` | `64` | Largest request body accepted; bigger ones get a `413`. `0` or `none` disables the limit |
| `MFLUXIBLE_MLX_CACHE_LIMIT_MB` | `mlx_cache_limit_mb` | `1024` | Cap on MLX's reusable buffer cache; `none` for MLX's own default (see [Memory](#memory)) |
| `MFLUXIBLE_MLX_WIRED_LIMIT_MB` | `mlx_wired_limit_mb` | unset | Wire this much memory so the OS cannot page the weights out (see [Memory](#memory)) |
| `MFLUXIBLE_LORA_PATHS` | `lora_paths` | unset | Local LoRA `.safetensors` files to bake in (see [LoRAs](#loras)) |
| `MFLUXIBLE_LORA_SCALES` | `lora_scales` | `1.0` each | Scales matching `MFLUXIBLE_LORA_PATHS` one for one |
| `MFLUXIBLE_HOST` | `host` | `127.0.0.1` | Where `mfluxible-server` binds when `--host` isn't given |
| `MFLUXIBLE_PORT` | `port` | `8420` | Likewise for `--port` |
| `MFLUXIBLE_BASIC_AUTH_USERNAME` | `basic_auth_username` | unset | HTTP Basic username; auth is off unless this **and** the password are both set (see [Authentication](#authentication)) |
| `MFLUXIBLE_BASIC_AUTH_PASSWORD` | `basic_auth_password` | unset | HTTP Basic password; required alongside the username |
| `MFLUXIBLE_CORS_ORIGIN_REGEX` | `cors_origin_regex` | `https?://(localhost\|127\.0\.0\.1)(:\d+)?` | Origins to reflect back in CORS (see [CORS](#cors)) |
| `MFLUXIBLE_CORS_ORIGINS` | `cors_origins` | unset | Exact-match origins, in addition to the regex |
| `MFLUXIBLE_VLM_DIR` | `vlm_dir` | unset | Enables object detection and says where images are stashed (see [Object detection](#object-detection)) |
| `MFLUXIBLE_VLM_BACKEND` | `vlm_backend` | `local` | Which side runs a detection: `local` (a model in this process) or `worker` (the sidecar) |
| `MFLUXIBLE_VLM_LOCAL_MODEL` | `vlm_local_model` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | *(local)* The model to load; needs the `vlm` extra installed |
| `MFLUXIBLE_VLM_TIMEOUT` | `vlm_timeout` | `180` | Seconds a warm detection waits for its answer before giving up |
| `MFLUXIBLE_VLM_LOAD_TIMEOUT` | `vlm_load_timeout` | `900` | Seconds the **first** local detection waits instead, since it downloads the weights |
| `MFLUXIBLE_CONFIG` | — | unset | Which config file to read; `none` disables discovery entirely. Not settable from a config file, for obvious reasons |
| `HF_TOKEN` | `[env]` table | unset | Not an mfluxible variable — `huggingface_hub` reads it, and gated models need it (see [Gated weights](models.md#gated-weights-and-hf_token)) |
| `MFLUXIBLE_VLM_COMMAND` | `vlm_command` | `claude -p …` | *(worker)* What it runs per detection (see [Using a different tool](#using-a-different-tool)) |
| `MFLUXIBLE_VLM_MODEL` | `vlm_model` | `opus` | *(worker)* Which model the **default** command uses; ignored if you set your own command |
| `MFLUXIBLE_SERVER_URL` | `server_url` | `http://127.0.0.1:8420` | *(worker)* Where to find the server, when `--url` isn't given |
| `MFLUXIBLE_BEARER_TOKEN` | `bearer_token` | unset | *(worker)* Sent as `Authorization: Bearer …`, if a proxy gates the API |

### Config file

A TOML file, as an alternative to exporting the variables above. Copy
[`mfluxible.example.toml`](../mfluxible.example.toml) — every key in it is commented out
and documented — and put it wherever suits:

```toml
model = "flux-dev"
quantize = 4
port = 9000

vlm_dir = "~/.cache/mfluxible/vlm"

[env]
HF_TOKEN = "hf_..."
```

It is looked for in this order, and the first hit wins — they are never merged, because a
value that could have come from either of two files is a value you have to go hunting for:

1. `--config PATH`, on `mfluxible-server` or `mfluxible-vlm-worker`
2. `MFLUXIBLE_CONFIG`
3. `./mfluxible.toml`, in the directory you started the process from
4. `~/.config/mfluxible/config.toml` — or `$XDG_CONFIG_HOME/mfluxible/config.toml`, if you set that

Note the two filenames differ. In a working directory the file has to say whose it is,
so it is `mfluxible.toml`; inside a directory already called `mfluxible` that prefix is
noise, and `config.toml` is the shape the XDG convention uses everywhere.

Naming a file that isn't there is an error; finding nothing at 3 or 4 is not. To turn
discovery off completely — for a deployment that configures everything through the
environment and wants no file to have a say — set `MFLUXIBLE_CONFIG=none`.

Four things worth knowing:

- **The environment wins.** A variable that is already set is left alone, so
  `MFLUXIBLE_MODEL=flux-dev mfluxible-server` is the one-off override it looks like, and
  a file in your home directory can never quietly overrule a launchd plist.
- **An unrecognised key is a startup error**, not a line that does nothing. `mdoel =
  "flux-dev"` should not produce a server running the default model and reporting success.
- **Values are not interpreted.** `~` is expanded by whatever reads the setting, exactly
  as it would be from the environment, so the file and the variable always behave the
  same. The one convenience is that a list is joined with commas, so
  `lora_paths = ["a.safetensors", "b.safetensors"]` reads the way you would expect.
- **`[env]` is an escape hatch, not a section.** Names in it are exported exactly as
  written, which is how a variable belonging to some other library — `HF_TOKEN` — can
  live in the same file. Everything else in the file is flat.

`mfluxible-server` prints which file it loaded on stderr as it starts. It is deliberately
*not* reported on `/health`: that endpoint is left open by
[`Caddyfile.example`](../Caddyfile.example) precisely because it discloses no filesystem
paths.

One caution, and it belongs to `vlm_command` specifically. That setting names a program
this machine will run, and rule 3 above means a file in the current directory is picked up
without being asked for. Starting the worker somewhere you did not put that file is the
case to think about; `MFLUXIBLE_CONFIG` names one explicitly, and `none` turns discovery
off. This is the same trust boundary the worker already has — see
[Running the worker](#running-the-worker) — only now it can be reached by a file as well
as by an export.

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

`MFLUXIBLE_MODEL_DIR` defaults to `~/.cache/mfluxible`, or `$XDG_CACHE_HOME/mfluxible` when that variable is set — which matters because `huggingface_hub` reads `XDG_CACHE_HOME` too. The two directories hold the raw download and the quantized copy of the same model, so pointing that variable at a bigger disk moves both rather than half. An explicit `MFLUXIBLE_MODEL_DIR` overrides it either way.

Startup quantizes the raw downloaded weights and caches the result to `MFLUXIBLE_MODEL_DIR/<model>-q<bits>/` (e.g. `z-image-turbo-q8`, `qwen-image-q4`) — a real, separate step from the Hugging Face download: HF already caches the raw weights locally, but quantizing them into MLX's packed format is nontrivial per-layer compute that would otherwise happen on every startup (mflux's `mflux-save` mechanism does the same thing via its own CLI; this just does it automatically here). Every startup after the first loads the pre-quantized weights directly and skips that step.

This means a second, smaller copy of the weights lives on disk alongside HF's cache of the original — once you've confirmed a cached startup works, the original HF cache entry (under `~/.cache/huggingface`) is no longer needed and can be deleted to reclaim space.

### LoRAs

```bash
MFLUXIBLE_LORA_PATHS="/path/to/style.safetensors" MFLUXIBLE_LORA_SCALES="0.8" \
  mfluxible-server
```

Not every model takes a LoRA. mflux builds FIBO and Ming-Image-Design without LoRA support, so setting `MFLUXIBLE_LORA_PATHS` with one of them stops the server at startup with a message naming the model, rather than loading it without the LoRA. `supports_lora` on [`/health`](api.md#get-health) says which.

LoRA weights are applied and permanently merged ("baked") into the model at load time — this is mflux's own design, not a limitation added here. That makes it a server-startup choice, not a per-request one: one running server has one fixed LoRA configuration (or none), and switching LoRAs means restarting the server with different env vars, the same way `MFLUXIBLE_QUANTIZE` works. Each distinct combination of LoRA paths/scales gets its own model-cache directory (named with a hash of that exact config), so switching between a few LoRA setups doesn't require re-quantizing each time you switch back.

### CORS

On by default, reflecting back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin — so a local static server on either hostname, any port (e.g. for `harness.html`) can call the API with no extra configuration. `MFLUXIBLE_CORS_ORIGIN_REGEX` overrides the pattern entirely; `MFLUXIBLE_CORS_ORIGINS` adds specific exact-match origins on top of it (e.g. for a deployed frontend on a real domain).

### Object detection

Off unless `MFLUXIBLE_VLM_DIR` names a directory. Switched on, the server gains the
`/mfluxible/v1/vlm/` endpoints ([API](api.md#post-mfluxiblev1vlmdescribe))
and the harness grows a **Find objects** button that fills a row of togglable regions.
Each one is an independent toggle and the mask is the union of whatever is selected, so
a subject and the thing it is holding can be masked together.

**Two backends, and `MFLUXIBLE_VLM_BACKEND` picks one.** They exist for different
reasons rather than as alternatives to benchmark against each other:

| | `local` (default) | `worker` |
|---|---|---|
| Where the model runs | In the server process | In a sidecar you start |
| What it is | An MLX vision model, loaded on first use | Any command line, `claude -p` by default |
| Extra install | `mfluxible[vlm]` | none |
| Extra process | none | `mfluxible-vlm-worker` |
| Costs | ~3GB of weights, held while the server runs | Whatever the command costs to run |

Start with `local`. It needs no second process and no account, and a server whose users
never press the button never pays for it — nothing is downloaded or loaded until the
first detection asks.

```bash
uv pip install 'mfluxible[vlm]'
MFLUXIBLE_VLM_DIR=~/.cache/mfluxible/vlm mfluxible-server
```

The first detection downloads a few gigabytes and takes minutes; the harness says so
rather than appearing to hang, and `MFLUXIBLE_VLM_LOAD_TIMEOUT` (15 minutes) is what it
waits, separately from the `MFLUXIBLE_VLM_TIMEOUT` that applies once the model is warm.
Later detections take seconds. `/health` reports `vlm.model_loaded` if you want to know
which you are about to get.

`MFLUXIBLE_VLM_LOCAL_MODEL` changes the model. The default is Qwen3-VL at 4B and 4-bit;
Qwen's own release notes for the family describe improved multi-target grounding over
Qwen2.5-VL, which is what a feature whose output is a mask rectangle needs most. Anything
mlx-vlm can load and that answers with boxes on a 0-1000 relative scale will work, but
note that the coordinate dialect is not a setting: see
[Using a different tool](#using-a-different-tool).

**What it costs while it is loaded.** The weights sit beside the image model's in the
same unified memory, and detections share the generation lock — so a detection waits
behind a running generation, and a generation waits behind a running detection. That is
deliberate: two MLX consumers interleaving on one device is not something this server
has ever been safe under. On a machine where the image model already fills memory, watch
`/health`'s `memory` block after switching this on, and see [Memory](#memory).

**The `worker` backend is how you reach a command line**, and is what to use for
`claude -p`, a cloud API behind a script of your own, or a local detector that isn't an
MLX model. There the server does no detection at all: it never loads a model, never
holds an API key and never makes an outbound call — it writes a file, holds one job in
memory, and copies a JSON array from one request to another.

```bash
MFLUXIBLE_VLM_BACKEND=worker MFLUXIBLE_VLM_DIR=~/.cache/mfluxible/vlm mfluxible-server
```

That split is deliberate rather than tidiness, and it is the reason the sidecar survives
having a default that doesn't need one: the worker runs an arbitrary configured command
with filesystem access, and an HTTP endpoint that spawned one would be by far the most
dangerous thing here. As a separate process it cannot be reached from the network at
all, and starting it is what consent to that command running looks like. None of that
argument reaches a local MLX model, which runs no command and spends nothing — which is
why that one is allowed to live in the server and this one is not.

A worker polling a server set to `local` is told so and exits, rather than long-polling
forever for jobs that will never come.

One variable both enables and configures because there is no useful "on, but nowhere to
put anything" state. Under `worker`, point the worker at the same directory: it is also
the working directory the detection command runs in, which matters twice on the default
— a stashed image inside it is already readable without widening the CLI's allowed
roots, and a directory with no `CLAUDE.md` keeps a one-shot detection from loading a
project's instructions on every call.

Stashed images are pruned after an hour, on the next detection.

#### Running the worker

Only under `MFLUXIBLE_VLM_BACKEND=worker`; a server left on the default answers its own
detections and hands a worker nothing.

`mfluxible/vlm_worker.py` runs a command line against the stashed image and posts the
regions back for the harness's open stream to deliver. It long-polls the server for a
pending detection, so it can be started and stopped at any time.

By default that command is [Claude Code](https://claude.com/claude-code) — the CLI on
`PATH` and already signed in — but nothing here is specific to it; see
[Using a different tool](#using-a-different-tool).

The worker is a **separate process you start yourself** — `MFLUXIBLE_VLM_DIR` on the
server opens the mailbox, it does not launch anything. Until the worker is running,
`/health` reports `regions.worker_attached: false` and the harness's **Find objects**
button says nothing is listening.

It needs no directory of its own — every job names the file to read, so there is nothing
to keep in step with the server:

```bash
mfluxible-vlm-worker
```

From a checkout, `uv run python -m mfluxible.vlm_worker`. Either way it has to be reached
as a module: running `mfluxible/vlm_worker.py` by path never imports the package around
it, so the [config file](#config-file) would not be applied and the settings would come
from a bare environment. It refuses that rather than starting with a configuration you
did not ask for.

Its four settings — `vlm_command`, `vlm_model`, `server_url` and `bearer_token` — are in
the [table above](#configuration) with the server's, and can live in the same config
file. `--url` overrides `server_url` and `--config` names a file explicitly. As with the
terminal clients, setting `MFLUXIBLE_BEARER_TOKEN` *and* putting credentials in the URL
is refused rather than resolved.

#### Using a different tool

What the worker actually requires is narrow: **a program that prints one JSON object to
stdout.** Anything that can do that from an image path works — another model's CLI, a
local detector, a script of your own.

```json
{
  "prompt": "a single red apple on a weathered oak table, soft window light, shallow depth of field",
  "regions": [{"label": "red apple", "box": [0.305, 0.344, 0.712, 0.736]}]
}
```

`prompt` is a text-to-image prompt that would plausibly regenerate the image — the whole
frame, written as a prompt rather than as a caption about a picture. `regions` lists what
could be masked, with `box` as `[x0, y0, x1, y1]` in fractions of the frame, `0,0` at the
top-left.

Both halves are optional and independent: a captioner that localizes nothing still fills
the prompt box, and a detector that only draws boxes still fills the chips. A **bare
array** is also accepted and read as regions with no prompt, so a tool written against
the earlier array-only contract keeps working.

Extra prose around the JSON is fine and a fenced code block is read too. Regions with a
malformed or out-of-range box are dropped rather than repaired, at most 8 are kept, and
the prompt is capped at 2000 characters.

The command is a template with two placeholders:

| Placeholder | |
| --- | --- |
| `{image}` | The image's filename. The working directory is the stash directory, so a bare name resolves. |
| `{prompt}` | The worker's standard detection prompt, already naming the image. Use it for a prompt-driven tool; leave it out entirely for a detector that doesn't take one. |

```bash
# a local detector that already speaks the format
MFLUXIBLE_VLM_COMMAND='detect-objects --format json {image}'

# a prompt-driven CLI that takes the image as an attachment
MFLUXIBLE_VLM_COMMAND='llm -m some-vision-model -a {image} {prompt}'

# your own wrapper, for a tool whose coordinates need converting
MFLUXIBLE_VLM_COMMAND='python3 ~/bin/boxes_to_fractions.py {image}'
```

That last one is the common case, and it is deliberate that it needs a wrapper: tools
disagree about coordinates — pixels, 0–1000, `xywh`, and Gemini even puts `y` first —
and a conversion setting here would turn a wrongly-scaled box into a plausible-looking
one. Keeping a single reader keeps a bad box visibly bad.

The `local` backend is the one exception, and it is not really one: it converts Qwen's
0-1000 relative coordinates to fractions itself, but by dividing by the fixed scale that
convention uses, not by a setting anyone configured. A backend that knows its own
model's convention is a driver; a menu of coordinate systems would be a dial. That is
also why pointing `MFLUXIBLE_VLM_LOCAL_MODEL` at a model that answers in some other
dialect (Qwen2.5-VL's absolute pixels, for instance) risks a plausible-looking wrong box
rather than a visibly dropped one — both dialects put values in a similar numeric range,
so the same failure a conversion *setting* would risk here is exactly what overriding to
a different dialect reintroduces.

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

Which is why a region is a starting point rather than a result. Selecting chips sets the
mask to their union and opens the editor, so the boxes can be nudged before they cost a
generation — the same reason the MCP tool has `preview_mask`.

While any chip is selected it *is* the mask: each toggle repaints from the whole
selection rather than adding to the canvas, because erasing one region's rectangle on
deselect would also cut into overlapping neighbours. The moment you pick up a brush the
chips let go, so a selected chip never describes something the canvas no longer shows —
and a hand edit is never silently discarded by a later toggle.

Detections are one at a time: starting a second supersedes the first, and the harness
says so rather than leaving the old one to time out.


## Running on a dedicated machine

The server doesn't have to run on the same machine as the clients. Point it at a spare Apple Silicon box (a Mac Mini, say) and bind it to the network instead of loopback:

```bash
mfluxible-server --host 0.0.0.0
```

Then everything else just points at that host instead of `127.0.0.1`, no code changes needed:

- `stream_client.py` / `stream_client.js`: `--url http://mac-mini.local:8420/mfluxible/v1/images/generations`
- `mfluxible-mcp`: set `MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations` when registering it, e.g. `claude mcp add mfluxible --scope user -e MFLUXIBLE_URL=http://mac-mini.local:8420/mfluxible/v1/images/generations -- uvx mfluxible-mcp` — or, in Claude Desktop's config, `"env": {"MFLUXIBLE_URL": "http://mac-mini.local:8420/mfluxible/v1/images/generations"}` alongside `command`/`args`. This is the case that package exists for: the MCP tool needs no mflux, so it installs on the laptop while the model stays on the Mac Mini

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
  mfluxible-server --host 0.0.0.0
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
- **`mfluxible-mcp`** — `MFLUXIBLE_BEARER_TOKEN` where the tool is registered; see [the MCP configuration table](mcp.md#configuration).
- **Open WebUI** — its API-key field, which sends this header already.
- **`curl`** — `-H "Authorization: Bearer $MFLUXIBLE_BEARER_TOKEN"`.

Two caveats worth stating plainly. Caddy's `header` matcher is a plain string comparison, not the constant-time one `mfluxible/auth.py` uses — not practically exploitable across a network with a high-entropy token, but a reason to use `openssl rand` rather than a passphrase. And a bearer token over plain HTTP is readable in transit exactly as Basic is, so the `tls internal` line in the example is doing real work; on a tailnet, `tls <host>.<tailnet>.ts.net` gets a publicly-trusted certificate and needs no trust step on any device already on the tailnet.

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
