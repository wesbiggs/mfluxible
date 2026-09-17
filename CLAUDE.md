# Project Notes

## Layout

`server/` (server.py, engine.py, models.py, schemas.py, chat_stub.py, schedulers.py, auth.py, requirements.txt) is the model + HTTP API. `clients/` (stream_client.py, stream_client.js, harness.html, mcp_server.py, requirements.txt, requirements-mcp.txt) is everything that talks to it over HTTP. They're independent dependency-wise -- installing one's requirements.txt doesn't pull in the other's. server.py/engine.py/models.py/schemas.py/chat_stub.py/auth.py import each other as flat sibling modules (`from engine import ...`), not a package, so `server/` must stay on `sys.path` when running (e.g. `uv run uvicorn server:app --app-dir server`) -- don't add an `__init__.py` or turn this into a `server.*` package without updating those imports and the run command together.

## uv, but with requirements.txt files -- deliberately not a uv project

Setup is `uv venv --python 3.11`, `uv pip install -r <one of the requirements files>`, and
`uv run <cmd>`; CI skips the venv entirely with `uv pip install --system`
(`.github/workflows/tests.yml`). There is intentionally **no `pyproject.toml` and no
`uv.lock`**, so `uv sync` is not the entry point here. Adding one would fold the four
separate dependency sets -- `server/requirements.txt`, `clients/requirements.txt`,
`clients/requirements-mcp.txt`, `requirements-dev.txt` -- into a single resolution, and
those files staying separate is exactly what the Layout note above is protecting: a
client-only or MCP-only install must never drag in mflux/PyTorch, and a machine running
only the terminal client should need nothing but `requests`.

`uv run` works fine outside a project: with no `pyproject.toml` it falls back to `./.venv`,
so `uv run pytest` and `uv run clients/stream_client.py ...` pick up whatever was installed
there without an `activate` step (verified against uv 0.12.9, not assumed). The MCP
registration examples in `docs/mcp.md` hardcode `/path/to/mfluxible/.venv/bin/python`
because Claude Desktop launches stdio servers with a minimal environment -- that path is
the venv `uv venv` creates, so it stays correct under uv.

## API changes

Always update `docs/api.md` when changing the API (endpoints, request/response shapes, SSE event
schema). `README.md` is only an overview + quickstart + index; the rest of the prose lives in four
pages under `docs/`, split by who the reader is rather than by topic: `server.md` (running the
server -- how it works, its env vars, the models, remote hosting, troubleshooting), `clients.md`
(the terminal scripts, the browser harness, OpenAI-compatible frontends), `mcp.md` (the MCP tool,
including its own env vars) and `api.md`. So a server env var goes in `server.md` and an MCP one
in `mcp.md`, even though both are "configuration".

The split that cuts across those four is `CONTRIBUTING.md`, at the top level: everything aimed at
someone *changing* the code rather than running it -- the test suite and CI, and the procedure for
adding a model (which fields of a `ModelSpec` have to be read out of mflux's source, and which
mflux models need engine changes rather than a table row). `docs/` is for users; `CONTRIBUTING.md`
is for contributors. Note the asymmetry with this file: `CONTRIBUTING.md` says what to do, while
the sections below say why the non-obvious parts are the way they are.

## No path from an exception to a response body, in either direction

`server/server.py` contains no `except` clause at all, and that is the point rather
than a coincidence. Validation is `MfluxEngine.request_problem`, which *returns* the
reason a request can't be honoured; a finished generation is `_collect_final_image`,
which *returns* the `image` or `error` event; a bad OpenAI `size` is
`_parse_openai_size` returning `None`. Each of those used to raise and be caught as
`str(exc)` into the response, which is what CodeQL flagged as `py/stack-trace-exposure`
(four alerts, one per sink).

The reason to keep it structural instead of sanitizing at each sink: the messages
themselves are *fine* -- they were all written here, for a client to read, and
`request_problem`'s are load-bearing API surface ("Z-Image-Turbo ignores guidance;
omit the field"). What is not fine is that an exception is a channel anything upstream
can also write to. `generate_image()` raising out of mflux, MLX, HF-hub or Pillow lands
in the same `except`, and those messages quote absolute cache paths -- a missing weight
file names `~/.cache/huggingface/...` in full, so the response body hands out the host's
account name. Keeping the curated values as *return* values is what makes "nothing from
upstream can reach a client" checkable by reading one module rather than by auditing
every message that might ever flow through a shared catch.

So: `check_request` still exists, wrapping `request_problem` for in-process callers
(`generate_stream`, and the tests) that have no response to put a string in -- but if
you find yourself adding `except ... : str(exc)` back into server.py to "simplify" the
pairs of returns, that is the leak coming back. The engine end of it is
`generate_stream`'s `except Exception`, which logs the traceback to
`mfluxible.engine` (stderr, so the terminal running uvicorn) and emits a fixed
`generation failed -- see the server log for the reason.` instead; the interrupt
message beside it stays verbatim, because "interrupted at step 5" describes the request
rather than the host. `_input_image_problem` splits off `_decode_input_image` for the
same reason -- it used to interpolate Pillow's exception (which includes a repr of the
in-memory buffer, address included) into its own message.

Covered by `test_engine_stream.py::test_generation_failure_reports_without_quoting_the_exception`
and the four `_LEAKY_MESSAGE` tests in `test_server_api.py`, which assert against a
fake `/Users/someone/.cache/...` rather than against the current wording.

The browser-side counterpart is in `clients/harness.html`: `previewMimeType` matches a
dropped file's `.type` against a fixed list and interpolates *the list's* copy into the
`data:` URL, because `.type` is the browser's guess from a filename, not a value this
page chose (CodeQL: `js/xss-through-dom`). Anything unlisted falls back to `image/png`
and still previews -- an `<img>` picks its decoder from the bytes' magic number, not
from the URL's declared type (verified, not assumed: a real JPEG relabelled
`image/png` decodes and reports its true `naturalWidth`). SVG is the one format that
does need its real type and is deliberately absent, since Pillow can't open one anyway.

## Basic auth: CORS must wrap it, and it must not buffer the stream

`server/auth.py` is off unless **both** `MFLUXIBLE_BASIC_AUTH_USERNAME` and
`MFLUXIBLE_BASIC_AUTH_PASSWORD` are set. Half-configured means off, not
half-open: setting one of the two is far likelier to be a half-finished
deployment than a deliberately blank username, and the failure worth avoiding is
silently serving unauthenticated while someone believes otherwise.

**The two `add_middleware` calls in server.py are in the order they are on
purpose, and it reads backwards.** Starlette's `add_middleware` inserts at the
*front* of the stack, so the middleware added **last** runs **first**. Auth is
added first precisely so `CORSMiddleware` ends up outside it. That ordering is
load-bearing: a browser preflight carries no credentials by spec, so if auth ran
first every cross-origin call would 401 at the preflight with no way for the page
to recover. `test_cors_preflight_is_answered_without_credentials` fails if the
two calls are swapped (verified by actually swapping them, not assumed), and it
is the only thing standing between a tidy-up and a subtly broken browser client.

Note the asymmetry that falls out of this and is *correct*: CORSMiddleware only
claims an OPTIONS request that carries `Access-Control-Request-Method`. A **bare**
OPTIONS is an ordinary request and is gated -- which is exactly SillyTavern's
reachability probe (see the shim section above), so **turning auth on blocks the
SillyTavern integration entirely**. Its `sdcpp` source sends no credentials on any
of its three calls, unlike its AUTOMATIC1111 source which has a Basic-auth field,
and credentials in the URL don't help either because Node's `fetch` rejects a URL
containing them. That is documented in `docs/clients.md` rather than worked around:
exempting those three endpoints from auth to keep one client working would be a
worse answer than a visible incompatibility. Both halves are pinned by
`test_the_sillytavern_ping_is_not_a_preflight_and_is_gated`.

**Pure ASGI middleware, not `BaseHTTPMiddleware`.** The latter runs the response
through an anyio task group and a memory stream, which is the wrong shape for a
server where nearly every interesting endpoint returns a `StreamingResponse` that
has to reach the client event by event -- the SSE `thinking` events carry per-step
previews a client renders live. A pure ASGI middleware forwards send/receive
untouched. `test_a_generation_still_streams_under_auth` asserts more than one
event arrives, and a live run confirmed five discrete events rather than one
buffered blob.

**The `except binascii.Error` in `_decoded_credentials` is compatible with the
invariant two sections above, not an exception to it.** `base64.b64decode` raises
on a malformed header, and that except *discards* the exception and returns
`None`, so a garbage header produces the byte-identical 401 a wrong password
does. Nothing from the exception reaches a response body, which is the property
being protected -- and the 401 deliberately says nothing about which half was
wrong, since differing bodies would enumerate valid usernames. Both are tested
against the response text, not the wording.

`resolve_bind_host()` is best-effort and says so: the app is handed to uvicorn
rather than starting it, so there is no socket to interrogate. It mirrors how
uvicorn itself resolves the value -- explicit `--host`, then `UVICORN_HOST` (its
CLI is a click command with `auto_envvar_prefix="UVICORN"`), then the `127.0.0.1`
default, verified against uvicorn 0.52.4 -- and only decides whether to print a
warning, so being wrong costs a missing or spurious warning, never a wrong bind.
An unparseable hostname counts as non-loopback, because the useful direction to
be wrong in is warning about a safe bind rather than staying quiet about an
exposed one. The warning is logged before `engine.load()` so an exposed server
says so immediately instead of after a multi-gigabyte download, and it reaches
stderr through `logging.lastResort` under uvicorn's own `LOGGING_CONFIG` (which
configures only the `uvicorn*` loggers and leaves root without a handler) -- the
same mechanism `engine.py` already depends on.

## Two auth schemes, and the browser is what decides between them

`MFLUXIBLE_BASIC_AUTH_*` (server/auth.py) and `Caddyfile.example` are **alternatives, not
layers**. The thing that makes them genuinely different -- rather than two spellings of
one idea -- is that a browser can satisfy exactly one of them unaided.

`clients/harness.html` is served at `GET /` and its Server URL defaults to the relative
`/mfluxible/v1/images/generations`, so its `fetch` is same-origin. Under Basic that is
all it needs: the browser prompts on the 401, caches the credential against the realm,
and reattaches it to the page's own requests with no code in the page at all. There is no
equivalent for a bearer token -- no prompt, no cache, no automatic header -- which is why
the **API token** field exists in the Advanced section and why it stores to
`localStorage`. That field is not a convenience; without it, a bearer-gated deployment
has no working browser client.

The corollary is the shape of the Caddyfile: it leaves `/`, `/docs`, `/openapi.json` and
`/health` open and gates everything else by exclusion. Serving those openly costs nothing
(the harness's source is in the public repo, `/health` returns model metadata and MLX
counters and *no filesystem paths* -- checked, given how carefully engine.py avoids
leaking cache paths elsewhere). Leaving `/health` open is load-bearing rather than lazy:
`refreshModel()` hides the guidance and negative-prompt fields based on that response, so
gating it would show an unauthenticated visitor a page offering fields the loaded model
rejects. Open, the page renders correctly and simply can't generate -- which is the
correct end state, because the GPU is protected by the API gate, not by hiding the page.

Gating by exclusion is also why `tests/test_proxy_config.py` exists. The dangerous
direction is one-way: a new endpoint is protected the day it lands, but a path *added* to
the open list silently becomes an ungated generation endpoint. That test parses the
`@public` matcher and fails if anything accepting a `POST` matches it.

**The app must stay on 127.0.0.1 under a proxy.** Anything that reaches it another way
reaches it unauthenticated, sitting on the network beside the proxy rather than behind
it -- which is exactly what `startup_warning` fires on, correctly. This is the specific
reason Docker is a poor fit here on macOS: MLX needs Metal and Docker Desktop's Linux VM
has no GPU passthrough, so only the proxy could be containerized, and a container
reaching the host generally means binding past loopback. Running Caddy natively has no
such tension.

Two Caddy details worth not rediscovering. It detects `text/event-stream` and streams it
through unbuffered with no `flush_interval` set, which this server depends on for its
per-step preview events -- nginx in the same position needs an explicit
`proxy_buffering off;`, and oauth2-proxy's non-zero default `--flush-interval` would
clump them. And `Caddyfile.example` deliberately carries **no global `{ ... }` block**, so
it pastes into an existing Caddyfile; a second global block is a parse error, not a merge,
and an `auto_https off` in one would disable `tls internal` for every other site in the
file.

## Client credentials: one URL, two libraries, and a header that used to vanish

`stream_client.js` parsed `http://user:pass@host/...` and silently dropped it. `URL`
populates `.username`/`.password`, but `http.request` only emits the header if handed an
`auth` option, and the options object didn't have one -- so a command line the docs said
would work produced a bare 401 with nothing to point at. Verified by recording the
`Authorization` header at a stub upstream: `requests` sent `Basic Ym9iOnBAc3M=` for the
same URL and Node sent nothing. Both now produce that identical header, including the
percent-decoding (`decodeURIComponent` here, `unquote` inside requests'
`get_auth_from_url`) -- so one URL means one thing to both clients.

The `MFLUXIBLE_` prefix is deliberate even though the server never holds this value.
The prefix names the *destination*, the way `HF_TOKEN` does -- an unprefixed generic name
like `HTTP_BEARER_TOKEN` reads more honestly but risks a token exported for some other
service being attached to requests aimed at this one, which is a credential sent to the
wrong host. `BEARER` rather than plain `TOKEN` is what keeps it distinguishable from the
server-side `MFLUXIBLE_BASIC_AUTH_*` pair, which the server genuinely *is* configured
with; `MFLUXIBLE_TOKEN` alone read like a third member of that family.

`MFLUXIBLE_BEARER_TOKEN` is environment-only in all three clients, not a `--token` flag: a flag
puts the secret in shell history and in `ps` output for the length of a generation, and
`mcp_server.py` has no command line to take one on anyway (an MCP host launches it as a
stdio subprocess).

Setting `MFLUXIBLE_BEARER_TOKEN` *and* putting credentials in the URL is **refused** rather than
resolved, in both terminal clients. It isn't defensiveness: requests derives Basic auth
from userinfo inside `prepare_auth`, which runs *after* `prepare_headers`, so the URL
would silently overwrite the bearer header rather than losing to it. Picking a winner
quietly is the failure worth avoiding; the same check is mirrored in the JS client so one
documented rule covers both.

## mflux is a fast-moving dependency

`server/engine.py` reaches into mflux internals that aren't public API: `model.callbacks.before_loop/in_loop/interrupt` are mutated directly, since `CallbackRegistry` has no `unregister()` as of mflux 0.19.1. The VAE-decode branching in `_decode_preview_b64` mirrors mflux's own `StepwiseHandler` on purpose, with one deliberate deviation: the non-packed branch goes through `VAEUtil.decode` rather than calling `vae.decode()` directly the way `StepwiseHandler` does. Qwen-Image's VAE is a 3D (video) decoder returning `(B, C, 1, H, W)` and `ImageUtil.to_image` wants 4D — `VAEUtil.decode` is what drops the singleton frame axis, and it's the same call each variant's own final decode makes, so previews and final images stay on identical handling. Calling `vae.decode()` bare here works for Z-Image and FLUX and breaks only on Qwen previews. If `uv pip install -U mflux` breaks this file, check `mflux/callbacks/callback_registry.py` and `mflux/callbacks/instances/stepwise_handler.py` in the installed package first — that's where this was reverse-engineered from (mflux ships no public docs for the callback system).

## server/schedulers.py depends on mflux conditioning the model on sigmas, not on the step index

`fractional_start` works by moving `sigmas[init_time_step]` off the grid, so the input
image is noised to a level that lies *between* two schedule rungs while the loop still
starts on a whole step. That is only coherent because all three variants derive the
transformer's time conditioning from `sigmas[t]` itself -- `z_image.py` computes
`timestep = 1 - sigmas[t]` inline, and `flux_transformer/transformer.py` and
`qwen_transformer.py` both index `config.scheduler.sigmas[...]`. If a future mflux
conditioned on the step index (or on `LinearScheduler._get_timesteps`, which returns a
bare `arange` nothing currently reads), the latents and the model's idea of where they
are would silently desync, and the failure would look like bad images rather than an
error. The module's docstring carries the rest: why `LinearScheduler` is the right base
for the linear-schedule models in `models.py`, and why the interpolation is done on the
request's own shifted schedule rather than by re-deriving mflux's shift math.

**Which models can take a fractional start is a per-model fact, now recorded as
`ModelSpec.default_scheduler`.** Handing mflux `SCHEDULER_PATH` *replaces* the scheduler
the variant would have chosen, so it is only a fractional start on a model that would
have run linear anyway. Elsewhere it is a sampler swap with no error — Z-Image base and
every FLUX.2 Klein default to `flow_match_euler_discrete`, and the failure looks like bad
images — or a `ValueError` thrown inside `generate_image()` on the worker thread with the
SSE headers already on the wire (`Krea2._resolve_scheduler` maps `"linear"` onto
`"er_sde"` and rejects every other name). `request_problem` rejects `fractional_start` up
front wherever `default_scheduler != "linear"`, for the same reason it rejects guidance:
a 400 before `StreamingResponse` starts, not a torn body. Extending it to a flow-match
model means a sibling scheduler subclassing mflux's flow-match class, not relaxing that
check.

The scheduler is selected by handing mflux a dotted import path
(`Config` -> `try_import_external_scheduler`), which resolves only because `server/` is
on `sys.path` as flat modules. That path must never be built from request data -- it is
an arbitrary module import in the server process -- which is why the API takes a bool
and `SCHEDULER_PATH` is a constant.

## Inpainting rides mflux's scheduler slot, because callbacks cannot change anything

Regenerating part of an image means writing the untouched region back over the latents
once per denoising step. The callback registry `engine.py` already hooks **cannot** do
that: a variant's loop is `latents = config.scheduler.step(...)` and only *then*
`ctx.in_loop(t, latents)`, and MLX arrays are immutable, so a subscriber can observe the
trajectory and never influence it. `scheduler.step()` is the only call whose return value
becomes the next step's latents, which is why `MaskedBlendLinearScheduler` lives in
`schedulers.py` beside the fractional-start one rather than anywhere near the callbacks.

What gets written back is exact, not an approximation, and that is a property of these
models rather than a design choice: all three variants noise an image by plain
interpolation (`LatentCreator.add_noise_by_interpolation` is
`(1 - sigma) * clean + sigma * noise`), so "the original at the level this step reached"
is that one line evaluated at `sigmas[timestep + 1]`, with the same noise sample the run
started from. The last sigma is 0, so the kept region lands on the encoded original
exactly -- which is what makes "outside the mask is unchanged" true rather than close.

**Two scheduler classes, not one with a flag.** Without a mask, the signal that says "no
fractional start" is engine.py *not passing a scheduler at all*. A mask forces one to be
passed on every request, which destroys that signal -- so a single class would have to
infer the fractional start from `config.image_strength` and would silently switch it on
for anyone whose strength happened to fall between two rungs. `scheduler_path(masked=,
fractional=)` is a fixed four-entry table for the same reason `SCHEDULER_PATH` is a
constant: the value is a dotted path mflux imports, i.e. an arbitrary module import in
the server process, and must never be assembled from request data.

**The job is module-level state, which the fractional-start scheduler deliberately
avoids.** There is no Config route for it: mflux resolves the scheduler from the Config
alone, and a Config carries no mask, no encoded latents and no seed. It is safe only
because of two properties of this server -- `MfluxEngine` serializes generations behind
an `asyncio.Lock`, and all MLX work runs on one dedicated worker thread -- so at most one
job can ever be live. `run()` sets nothing and clears it in its `finally`; the *set* is
in `_StreamCallback.call_before_loop`. If either invariant goes away this has to become
per-Config state rather than a wider lock.

**Why before_loop rather than before generate_image().** The job has to be built at the
dimensions mflux is really generating at, and `Config` floors width and height to a
multiple of 16 -- encoding at the requested size instead yields latents a different shape
to the ones the loop carries. `before_loop` is the earliest hook that knows the floored
values, and by then mflux has already resolved `config.scheduler` to build the starting
latents. So the scheduler necessarily exists before its job does, and `_MaskedBlend`
checks for a missing job in `step` rather than `__init__`. Missing raises: an unmasked
image would look completely fine and quietly ignore the mask, where a raise lands in
`generate_stream`'s existing `except` as a logged traceback and a failed generation.

**The mask reaches the latents through the model's own `pack_latents`, which is what
keeps engine.py free of per-model branching.** Packing is a pure spatial rearrangement,
so replicating one mask value across the channel axis and packing it puts each element's
mask on that element -- whether the variant folds 2x2 patches into channels (FLUX, Qwen)
or only shuffles axes (Z-Image). The latent grid's size comes from the encoder's own
output shape rather than a hardcoded /8, and the downsample is `Image.BOX`: a windowed
filter would ring past [0, 1] at a hard mask edge, and overshoot there means "more than
the original" or "less than nothing" in the blend.

**`image_strength` near 0 is what inpainting wants, and that surprises everyone -- so it
is the default under a mask rather than advice about one.** With a mask the field no longer
governs how much of the *frame* survives -- the mask does -- only how much of the original
content inside the mask does. mflux's convention is already the inverse of other tools', so
the failure mode is that 0.4 (mflux's CLI default, and what `DEFAULT_IMAGE_STRENGTH` still
means for an unmasked request) leaves the old object standing in the region just painted,
which reads as the mask having been ignored.

The reason that is a `DEFAULT_MASKED_IMAGE_STRENGTH = 0.0` in engine.py rather than a
sentence in `docs/api.md` is the *shape* of the failure: a 400 would be fine, but this one
is well-formed. The stream is correct, the image arrives, the generation reports success,
and the only evidence anything went wrong is that the picture looks like the input.
Measured on a 768x768 Z-Image-Turbo run with identical seed, box, feather and prompt: 5.1/255
of change inside the mask at 0.4 against 41.6 at 0.0 -- the cup that was supposed to become
a dinosaur came back a cup. And it was the *default* path, since omitting an optional field
is the ordinary thing to do. An explicit value still wins, because raising it to restyle
what is inside the region rather than replace it is a real request.

`harness.html` also moves its own field to 0 when the box is ticked, which is now belt and
braces -- it always sends an explicit `image_strength`, so the server's default never fires
for it. Keep it: the point there is that the user *sees* the value change.

**Above 0, the useful range is 0.1-0.2 and it is a composition anchor rather than a strength
dial -- which is also where a formula in `docs/api.md` was wrong.** `Config.init_time_step`
gates on `image_strength > 0.0` *before* applying `max(1, int(steps * strength))`, so 0.0
returns 0 and any non-zero strength returns at least 1. The docs stated only the `max()`,
which implies 0.0 -> 1 and makes 0.0 and 0.1 look adjacent; they are a whole rung apart, and
that rung is the entire effect. At 0.0 the masked region starts from pure noise and the
prompt alone decides where the new content lands inside the box. At rung 1 it starts from the
encoded original, so the new content inherits its position, scale and outline.

That is what makes a non-zero strength worth having, and it is not "restyling": it is the
only way to hold a replacement where the original sat when the prompt cannot reproduce the
original composition. A photograph is the case that forces it -- "a man reading a book" will
not be framed the way the photo was, so without an anchor the replacement is correctly
painted and in the wrong place. The cost is bleed-through and it rises with the anchor:
replacing an S chest emblem with an F, 0.0 gave a clean F at a prompt-chosen size, while 0.2
with fractional_start held the shield exactly in place and let the old S contaminate the
letterform. Anchor when the new content should share the old one's geometry, not otherwise;
no single value does both, which is why this stays a documented dial rather than a smarter
default. Note also that 0.1 and 0.2 are the same rung at 9 steps -- `fractional_start` is
what makes them distinguishable. It also makes `fractional_start` worth
more than it is for plain img2img, since the useful range is narrow and still quantized.

**Measured, so it doesn't get re-litigated:** with the blend alone, outside-mask mean
absolute error against the input is ~1.45/255 (p99 4.3) on a 1024x1024 Z-Image-Turbo run
-- at or below the VAE encode/decode floor of ~1.73, i.e. the blend costs nothing and the
residual is the round-trip. `mask_composite` pastes the result back through the mask in
pixel space and takes it to exactly 0. The blend itself is free in time too: 131.3s vs
136.2s for the same 8 steps masked and unmasked.

**`mask_feather` is not a region blend, and a big one is a double exposure.** It looks
like a pixel-space feather and isn't: a grey mask value is a *per-step* blend weight, so
it means "hold this pixel partway to the original at every denoising step" and the model
never commits to anything in that band. Verified against the real model rather than
reasoned about -- radius 8 on a 1024x1024 run gives the clean result, radius 96 with
everything else identical brings the original mug back as a ghost with the new cactus
faded through it. So the field exists to hide the latent grid's 8px staircase along an
edge and wants to stay within a couple of cells of it; harness.html caps its input at 32.
This is the one place where "more of the smoothing parameter" makes the output
qualitatively wrong rather than merely softer.

**The kept region's tone is not a constraint on the new content.** Nothing forces the
model to match the exposure of what it is keeping, so a mask that boxes in a swathe of
flat background gets a rectangle: measured on the same run, the wall inside the mask came
back at luminance ~159 against the input's ~149. Worth knowing precisely because the
instinct is to blame `mask_composite` -- it isn't that; the identical gap is in the raw
decode with compositing off, and the blend's own accuracy outside the mask on that run was
mean 1.43 / p99 4.00, i.e. the VAE floor. Masking tightly around the subject is the
answer, not widening the feather (see above).

**Two further limits are inherent and are documented rather than worked around.** It
fills, it does not erase -- nothing tells the model the region is a hole to continue the background
across, so masking an object and prompting "an empty wooden table" produces a table-shaped
object in the hole (and on a guidance-distilled model there is no negative prompt to say
"nothing here" with). And the mask edge is a hard boundary: content that wants to cross it
is cut off at it, which is a mask-authoring problem, confirmed by re-running the same seed
and prompt with the mask extended and getting the uncut result.

**`POST /v1/images/edits` still rejects `mask`, now for a different reason.** OpenAI marks
the editable region with *transparency*; the native `mask` field marks it *white* and
ignores alpha. Reading one as the other doesn't fail, it inpaints the complement -- every
region the caller meant to keep -- so that endpoint refuses and names the native one.
Honouring OpenAI's convention is real work (alpha handling, plus deciding what a fully
opaque mask means), not an alias.

One harness landmine worth not rediscovering: the progress sweep across the top of the
canvas is `body:has(#submitBtn:disabled) .canvas::before`, which works only because
Generate is disabled for exactly the length of a run. Gating the button on "a mask has
been painted" would latch that bar on forever, so the empty-mask case is refused in the
submit handler with a log line instead. The mask controls' visibility is pure CSS off
`#imagePreview.hidden` and `#inpaint:not(:checked)`, scoped to `body` rather than
`#baseSection` because one of them lives on the result pane.

## server/chat_stub.py hardcodes a specific tool name, confirmed against one caller

`POST /v1/chat/completions` only ever emits a tool call for a tool literally named
`generate_image` (see `GENERATE_IMAGE_TOOL_NAME` in `server/chat_stub.py`) if the
caller's request actually offers one by that name in `tools` -- it never fabricates a
tool call for a name it wasn't handed. That name isn't part of any OpenAI spec (tool
names are caller-defined) and isn't necessarily Open WebUI's own coinage either --
`generate_image` is an obvious enough name for this that other tool-calling systems may
independently land on the same one. What's actually confirmed, by reading the source, is
that Open WebUI's builtin tool uses exactly this name with `prompt` as its only public
argument (`backend/open_webui/tools/builtin.py::generate_image`) -- the same
read-the-actual-source approach as the mflux and mcp SDK notes elsewhere in this file,
not a guess from behavior. If a caller's tool is named anything else, this stub falls
back to its "no generate_image tool was offered" response rather than silently doing
nothing (see `_NO_TOOL_OFFERED_TEXT`) -- so a different caller using a different name is
a visible failure, not a silent one, even though `GENERATE_IMAGE_TOOL_NAME` would still
need updating (or generalizing past a single hardcoded name) to actually work with it.

The rest of chat_stub.py -- the request/response shapes, the streaming chunk format --
is the actual OpenAI chat-completions function-calling spec, not caller-specific, so
that part isn't expected to need chasing the way the tool name might.

## The SillyTavern shim is A1111-shaped on purpose, and inverts the reject-don't-drop rule

SillyTavern's image-generation extension has **no OpenAI-compatible source** -- as of
1.18.0 (release and staging both), its `openai` source hardcodes
`https://api.openai.com/v1/images/generations` inside SillyTavern's *own* Node backend
(`src/endpoints/openai.js`) with no base-URL setting, and its model dropdown is a
hardcoded array. That is unlike the "Custom (OpenAI-compatible)" source on its chat
side, which is what makes the absence surprising enough to be worth writing down. Read
out of the source rather than inferred from behavior, same as the mflux and mcp notes
above. So `POST /v1/images/generations` is unreachable from it, and that is not
something a config change on either side fixes.

What it does have is a `sdcpp` source (for stable-diffusion.cpp's server) that is a
**hybrid**: OpenAI-shaped discovery, A1111-shaped generation. It pings
`OPTIONS <base>/v1/images/generations`, lists models from `GET <base>/v1/models`
(parsed as `data.data.map(m => ({value: m.id, text: m.name || m.id}))`), and generates
via `POST <base>/sdapi/v1/txt2img`, reading `images[0]` as base64 PNG. Two of those
three this server already answered, which is the whole reason this is ~150 lines rather
than an A1111 emulation: the `auto` source would have needed roughly ten endpoints,
including both a GET and a POST of `/sdapi/v1/options`. See
`public/scripts/extensions/stable-diffusion/index.js` (`generateSdcppImage`,
`loadSdcppModels`) and `src/endpoints/stable-diffusion.js` (the `sdcpp` router).

**`POST /sdapi/v1/txt2img` drops fields the loaded model can't act on, where every
other endpoint here returns a 400.** That inversion is deliberate and caller-specific,
not a lapse. All three of SillyTavern's sdcpp routes end in `response.sendStatus(500)`,
attaching the upstream body to an Error's `cause` that is only ever `console.error`'d
on its server -- so `request_problem`'s messages, the ones the "No path from an
exception to a response body" section above exists to keep curated and readable, land
nowhere a user will look. And this caller sends `cfg_scale` and `steps` on *every*
request from sliders it always shows, whose defaults (7 and 20) suit neither a
guidance-distilled model nor a 9-step one. Rejecting those would make the source
unusable rather than correcting anyone. The line drawn instead is whether the caller
could otherwise detect the difference: a dropped `cfg_scale` changes the image, but
nothing was promised to compare it against, while `batch_size: 4` answered with one
image is a concretely wrong result to a request that named a number -- so that one
stays a 400 (and SillyTavern, which hardcodes `batch_size: 1`, can never trigger it).
What actually took effect is reported in the response's `info` string, which is the one
channel that survives the trip.

The drops in `_a1111_to_generate_request` duplicate two of `request_problem`'s rules
rather than sharing code with them, so the endpoint re-runs `request_problem` on its
own output: a rule that grows a third clause this doesn't mirror becomes a visible 400
rather than a silently wrong image, and
`test_a1111_shim_never_builds_a_request_its_own_model_rejects` pins that across every
model in the table without loading any weights.

Two smaller traps, both found by reading rather than guessing:

- **`model` is ignored here, and validated on `/v1/images/generations`.** SillyTavern
  only auto-selects from a freshly loaded model list when its stored `sd.model` is
  *empty* (`if (!extension_settings.sd.model && models.length > 0)`), so a name left
  over from a previously configured source survives, gets sent here, and matches no
  option in the dropdown the user is looking at. Validating it would turn an invisible
  stale setting into an unexplained failure; `info.sd_model_name` reports what really
  ran instead.
- **The `OPTIONS` handler is not CORS and is not redundant with it.** Starlette routes
  a request to `CORSMiddleware` as a preflight only when it carries
  `Access-Control-Request-Method`; SillyTavern's ping is a server-side
  `fetch(url, {method: 'OPTIONS'})` with no such header, so without an explicit route it
  falls through to a 405 and the "Validate" button fails. The two coexist because of
  that same header check -- covered by
  `test_a1111_ping_route_does_not_shadow_a_real_cors_preflight`.

All of the above is one version's behavior and will move; the endpoint is written
against what 1.18.0 actually sends, and re-reading those two files is the way to check
it rather than inferring from a failing generation.

## Object detection is a mailbox, and the thing that spends money is a client

`server/regions.py` holds one pending job and copies a JSON array between two HTTP
requests. It loads no model, holds no key and makes no outbound call -- the harness
POSTs an image and waits on an SSE stream, `clients/region_worker.py` claims the job,
shells out to `claude -p`, and posts regions back.

**The worker is in `clients/` because of what it does, not to keep `server/` tidy.**
It spawns a subprocess with filesystem access that makes network calls and spends a
Claude account. Had that lived behind an endpoint, `POST /.../detect` would be
comfortably the most dangerous route in this repo and one line away from
`tests/test_proxy_config.py`'s worst case -- a path drifting into `@public` would stop
being an ungated GPU and start being remote code execution against someone's Claude
account. As a client it is unreachable from the network, it is optional (the server
reports `worker_attached: false` and the harness says so), and *starting it* is what
consent to that spending looks like. It also keeps the CLI out of the server's
dependency set, which matters given how many fast-moving externals this file already
tracks.

**One slot, and no identifier.** The obvious design hashes the image so the two sides
can agree which one they mean. That cannot work, and the reason is measured rather than
assumed: an image that reaches a Claude Code session is **re-encoded** on the way into
the session transcript. A 622,650-byte PNG read with the Read tool was stored as a
70,548-byte **JPEG** -- 11% of the size, a different format, so no hash of the original
can ever match. Dimensions survived (768x768), which is the half that matters, because
boxes are fractions and fractions are scale-invariant for the same reason `mask_boxes`
are in the MCP tool.

So the mailbox pairs a waiting stream with an arriving answer and needs no key at all.
That is only coherent because `MfluxEngine` already serializes generations behind one
lock in one process: a second detection has nowhere useful to queue, so it supersedes
the first and closes that stream with a reason instead of leaving it to time out. If
the engine ever runs more than one generation at a time, this needs a real key -- and
the re-encode above is why that key cannot be the image's bytes.

Two guards replace it, for two different failures. Dimensions catch the server and the
browser disagreeing about one image's *oriented* size -- both sides apply EXIF
orientation (`regions.py` imports engine.py's `_oriented` rather than copying it, since
mflux rotates before encoding and a box chosen in one frame applied in another is
transposed), so a phone photo lines up without either side converting. And
`baseImageEpoch` in harness.html catches the likelier case: the displayed image being
swapped while a detection is still open, which a replacement of identical size would
otherwise slip past.

**Putting an id inside the image -- EXIF or otherwise -- answers a question nobody is
asking, and does not survive to a reader.** The round trip is already identified: the
server issues a `job_id`, stashes the exact bytes, and hands the worker that same file,
so nothing between the two re-encodes anything. The only consumer past that point is a
vision model looking at pixels, which cannot read a tag. And the field it would want is
taken -- mflux embeds its generation metadata in EXIF `UserComment`, which is precisely
what harness.html parses to recover a prompt and seed, so writing there would collide
with the feature one section up. The identity that actually needs tracking never leaves
the page, which is why the counter lives there.

**`MFLUXIBLE_REGIONS_DIR` is one variable doing two jobs on purpose.** It enables the
feature and it is where images are stashed -- there is no useful "on but nowhere to put
anything" state. It is *also* the working directory the worker runs `claude` in, which
buys two things at once: a stashed image inside the cwd is readable without widening the
CLI's allowed roots, and a directory with no `CLAUDE.md` in it keeps a one-shot
detection from loading this (very long) file on every call.

**The worker defaults to opus, and the usual latency-for-quality trade does not apply.**
Measured on one 768x768 photograph with an identical prompt, opus was both better and
*faster*: 11.4s against sonnet's 66.6s. Sonnet put the apple at
`[0.28, 0.28, 0.68, 0.62]` -- roughly the right size in the wrong place, clipping the
fruit's bottom third and taking in a band of forearm, which is exactly the failure the
MCP mask-box section already records. Opus gave `[0.305, 0.344, 0.712, 0.736]` against
`[0.310, 0.344, 0.694, 0.729]` measured by hand, to three decimals rather than the round
two-decimal numbers that signal estimating on a grid. Sonnet had produced a good box on
an earlier run of the same image, so it is variance rather than a constant offset --
which is the worse failure, since nothing downstream can tell the two apart. Hence a
chip sets the mask and opens the editor rather than being treated as final.

**The harness draws the button off `/health`, and reads a missing block as off.** That
inverts the rule the `supports_*` keys follow, deliberately: those default to "unknown,
let the server decide" because offering a field costs nothing if the server accepts it,
while a Find objects button on a server without the mailbox can only ever 404.

## mcp SDK also moved fast: FastMCP -> MCPServer

`clients/mcp_server.py` targets `mcp` 2.x, where `mcp.server.fastmcp.FastMCP` (the commonly-documented v1 API) was renamed to `mcp.server.mcpserver.MCPServer`. Importing the old path raises a `ModuleNotFoundError` with a migration pointer, it doesn't just silently break — if that happens, you're looking at v1-flavored example code (`FastMCP(...)`) against a v2 install. `Context`, `Image`, and the `@server.tool()` decorator are all still there, just re-exported from `mcp.server.mcpserver` instead.

## Progress notifications do not reliably hold a host's tool-call timeout open

This is why `clients/mcp_server.py` runs generation in a background task and hands back a
`check_image` handle instead of just blocking. Measured 2026-08-31 against a live Claude
Code session: a `generate_image` call died at ~60s with the MCP SDK's default
`Request timed out` while the server was sending a progress notification every ~8s. The
notification stream was working -- verified separately by driving the server over real
stdio JSON-RPC with a `progressToken` and watching `notifications/progress` arrive -- the
host simply doesn't rearm on it.

Other hosts do. Reading the Claude Code CLI binary (Homebrew cask 2.1.236) shows a
per-call *hard* timeout (per-server `timeout` -> `MCP_TOOL_TIMEOUT` -> a ~1e8ms default)
that progress explicitly does not extend, plus a separate *idle* watchdog (stdio default
30 min) that every progress notification does rearm, plus auto-backgrounding of any call
still running at 120s. Note the interaction: setting `MCP_TOOL_TIMEOUT` lowers the idle
watchdog too, since idle is clamped to at most the hard timeout. The point isn't the
specific numbers -- they're one host's build and will move -- it's that they differ per
host and per version, so the server can't depend on any of them. Keep `WAIT_SECONDS`
under the *shortest* timeout you care about; blocking longer only converts a working
handle into a failed call.

## mcp 2.x strips the message off any exception that isn't ToolError

`raise RuntimeError("mflux said X")` inside a tool reaches the model as a bare
`Error executing tool generate_image` -- the SDK wraps anything unrecognized in
`UnexpectedToolError` and drops the text (`mcp/server/mcpserver/tools/base.py`). Raising
`mcp.server.mcpserver.exceptions.ToolError` instead keeps it:
`Error executing tool generate_image: ConnectError: All connection attempts failed`.
That distinction is the difference between the model being able to tell the user the
HTTP server isn't running and it being told nothing at all, so anything a caller could
act on -- upstream errors, unknown handles -- must go out as `ToolError`.

## The MCP tool's mask boxes are fractions because the model never saw the full-size image

`generate_image`'s `mask_boxes` takes fractions of the frame rather than pixels, and that
is not a taste call. `_fit_result` downscales the inline copy of any image past
`MAX_RESULT_BYTES` -- a 1024x1280 PNG comes back at roughly half size -- while the
full-resolution original goes to `SAVE_DIR` and is what `image_path` points back at on the
next call. So the frame the model *looked at* when it chose a region and the file the mask
is applied to are routinely different sizes, and nothing in the response states the factor
outright: the displayed dimensions appear only inside the caption's prose note, and only
when a downscale actually happened. Pixel coordinates would be wrong by that factor,
silently, in exactly the two-call flow -- generate, then replace part of the result -- that
the feature exists for. A fraction means the same thing in both frames.

Rasterizing client-side follows from the same fact: it needs the input image's real size,
which is a property of a file on the *client's* disk. It uses the **EXIF-oriented** size,
and that is the trap. `_oriented` mirrors engine.py's function of the same name because
mflux rotates an input image before encoding it and `_input_mask_problem` therefore
compares oriented sizes. A phone photo is where the two diverge -- 4032x3024 of bytes
carrying an Orientation tag, 3024x4032 as displayed -- so a mask rasterized at the raw
size is rejected for a mismatch the caller has no way to see, and one drawn by hand
against what the photo looks like is right.

**Two defaults here disagree with the endpoint this tool proxies to, on the same
reasoning: someone writing JSON has read the field's documentation, while the model
calling this tool has read a docstring and will mostly not pass the field at all.** A
docstring is advice; a default is not.

`mask_feather` is 8 against the API's 0. That asymmetry has a sharp edge -- `request_problem`
rejects a non-default `mask_feather` *without* a mask, so a maskless call must send 0 rather
than the default it was handed, which is why that field is conditional in the request body
rather than passed straight through.

`image_strength` is 0.0 when a mask is present. The server now defaults this too
(`DEFAULT_MASKED_IMAGE_STRENGTH`, see the inpainting section above, which carries the
measurements), so this is **deliberate duplication rather than the only copy**: the same rule
that has `mcp_server.py` treat a missing `/health` field as "unknown, let the server decide"
means it has to keep working against a server that predates the server-side default. The two
can't disagree -- both are 0.0 -- and the client sending an explicit 0.0 means the server's
own default never fires for an MCP call either way. It has to be a literal 0.0 rather than
None, since None is what routes to 0.4 on an older server.

**The prompt describes the whole finished frame, not the masked region, and how much that
matters scales with the mask.** The model composes for the entire image; the masked region
is a window onto that composition. So a prompt naming only the new object is composed at
full-frame scale and the window lands somewhere inside it -- which on a small mask returns a
giant cropped fragment of the thing that was asked for.

**The first measurement of this was misleading, and the conclusion drawn from it was wrong.**
Replacing a mug with a toy dinosaur across ~35% of the frame, a bare "a small green toy
dinosaur figurine" gave a perfectly good image: 44.5/255 of change inside the mask against
the full-scene version's 41.6, and 38.4 apart from it -- a different image rather than a
broken one. Generalizing from that single large-mask case to "a preference, not a rule" did
not survive the next real use. Replacing a chest emblem across ~6% of the frame, "a stylized
bold letter F emblem, heroic shield badge shape" produced a letterform several times too big
for the region and cut flat at the mask edge. The identical seed and box, prompted "a cartoon
superhero frog with a red cape ... a bold letter F emblem on its chest, pale cream
background", produced a properly scaled badge. A three-way ablation settles which term
dominates: correcting the box while keeping the region-only prompt was *still* bad, while
fixing the prompt alone -- leaving the misplaced box -- was already good. So the prompt is
the dominant cause and box placement the secondary one, and the lesson for anything measured
here once is that mask area is a variable, not a constant.

Box placement is that secondary term and is worth its own check: in the failing case the
model's box sat about 13% of the frame left of the emblem it was meant to replace and clipped
its right edge, which by itself cost a flat-cut badge. The error has a shape worth recording,
since it was not random -- the box was centred within 0.014 of the *torso's* centre and 0.075
off the *emblem's*, and 1.5x too large in both axes. It had grounded the semantic region
("where a chest logo goes") rather than the object's extent, and the emblem sat off-centre on
the torso because of the pose. All four coordinates were round two-decimal numbers with two
identical, which is the signature of estimating on a coarse grid rather than measuring.

`preview_mask` exists for exactly this and is why it is a separate tool rather than a flag:
it renders the selection over the image with no model, no GPU and no HTTP -- not even a
`/health` read, since a caller is most likely to be checking a box while composing a request,
which is when the server may not be up. Both tools go through `_resolve_mask`, and that
sharing is the whole point rather than deduplication: a preview built by its own path would
drift, and a drifted preview is worse than none, because it reassures the caller about a
selection the server never sees.

`mask_path` exists beside `mask_boxes` rather than instead of it because the two serve
different hosts. Authoring a mask PNG needs a filesystem the MCP host may not give the
model -- in a Claude Desktop setup where mfluxible is the only server connected, it has no
way to write one -- whereas naming a rectangle on an image it can see needs nothing but
the image. `mask_path` is for a mask something else already drew.

## One model per process, and every per-model difference lives in models.py

`server/models.py` is the whole multi-model story: which mflux variant class, which `ModelConfig`, which latent creator, the default step count, whether `guidance`/`negative_prompt` mean anything, and which scheduler the variant picks for itself. `engine.py` has no per-model branching and shouldn't grow any — mflux's ZImage, Flux1, QwenImage, Krea2, ErnieImage and Flux2Klein happen to share a constructor signature, a `generate_image()` signature, a `save_model(base_path)` and a `callbacks` registry, which is the only reason this works.

That shared surface is **not** universal across mflux, which is where the table currently stops. `FIBO`, `BooguImage` and `LensImage` take no `lora_paths`/`lora_scales`, so `_load_sync`'s fresh-load branch raises `TypeError` before any weight loads; `Ideogram4` accepts no `image_path`/`image_strength`/`scheduler`; `BooguImage` has no latent creator at all (mflux's own CLI passes `latent_creator=None` and states stepwise output is unsupported), so step previews are impossible; `LensImage` has no `save_model()`, so the quantized-weight cache has nothing to write and every startup re-quantizes. Each of those needs a capability flag here *plus* engine.py honouring it — a new row alone would fail at load or mid-stream, not gracefully.

Three per-model facts are load-bearing and easy to get wrong from the README alone, so they were read out of the installed mflux source rather than assumed:

- **Guidance support is per-checkpoint, not per-family.** The three non-`base` FLUX.2 Klein checkpoints hard-error on any guidance but 1.0 (`flux2_generate` CLI, judged by `"base" not in model_name`), while `flux2-klein-base-*` allow it; ERNIE-Image-Turbo likewise errors off 1.0 while ERNIE-Image defaults to 4.0. Z-Image is the sharpest case: base and Turbo share the `ZImage` class, and `supports_guidance` is what decides both whether an unconditional branch gets built *and* which scheduler runs.
- **`supports_negative_prompt` is necessary but not sufficient.** Every CFG model here encodes the unconditional prompt only above guidance 1.0 (`CFG_GUIDANCE_FLOOR`) — ZImage/ErnieImage at `<= 1.0`, Krea2 at `== 1.0`, Flux2Klein at `not > 1.0`. Krea-2's own default guidance is exactly 1.0 (mflux's `DEFAULT_GUIDANCE`), so `check_request` tests the *effective* guidance; otherwise a negative prompt there would be accepted, encoded and never consulted, which is the silent drop the whole flag scheme exists to prevent.
- **`z-image`/`zimage` name the base checkpoint, not Turbo.** They used to be aliases for Turbo here, which was harmless while Turbo was the only Z-Image and wrong the moment the base model was added. mflux's registry is the tiebreaker for every alias in this table.

Two things in that file are load-bearing rather than stylistic:

- **The mflux imports are deferred into each spec's `load()` function**, so naming four models costs nothing. mflux downloads weights inside the variant's constructor (`WeightLoader.load` → `PathResolution.resolve` → HF snapshot download), never at import, so a process only ever fetches/quantizes/caches the one model `MFLUXIBLE_MODEL` selected. Don't hoist those imports to module level "for tidiness" — it wouldn't download anything, but it would drag every model's module graph into every process and quietly make that guarantee depend on mflux never doing work at import time.
- **`model_config` must be passed on the cached-load branch too** (`_load_sync`). A saved directory holds weights and tokenizers, not the model's scheduler/sequence-length settings, and each variant's own default would otherwise win — `Flux1` defaults to *schnell*, so a `flux-dev` cache dir would silently load as schnell.

The clients deliberately send `steps` (and `guidance`) as null rather than a number of their own: whichever model is loaded decides, so none of them needs reconfiguring when the server switches models. Don't "fix" a missing default back into `stream_client.*`, `harness.html` or `mcp_server.py` — a step count that suits Z-Image-Turbo is four times too small for FLUX.1-dev. `harness.html` and `mcp_server.py` additionally read `/health` to learn what the loaded model accepts; both treat a failed or model-less `/health` as "unknown, let the server decide" rather than an error, so they keep working against a server that predates that field.

Guidance and negative prompts are rejected with a 400 on models that can't act on them rather than accepted and dropped, because mflux accepts both arguments on every variant and silently ignores them (its own CLIs print a warning instead). `request_problem` runs in the endpoint, *before* `StreamingResponse` starts: failing inside the generator would mean a 200 status line already on the wire and a torn body.

## Quantized-weight cache marker is a heuristic, not an integrity check

`MfluxEngine.load()` treats the existence of `<saved_dir>/transformer/model.safetensors.index.json` as "this quantization level was already saved, load it instead of re-quantizing." It doesn't verify the save actually completed or matches the current mflux version — an interrupted `save_model()` call (e.g. killed mid-write) would leave a directory that looks cached but loads incorrectly. If a cached load ever misbehaves, delete `~/.cache/mfluxible/<model>-q<bits>/` and let it re-save. That marker path is the same for all three variants — every weight definition names its transformer's subdir `transformer` — which is what lets one heuristic cover them all; check the model's `*WeightDefinition` before assuming it holds for a newly added model.

## LoRA baking and the quantized-weight cache must stay in lockstep

LoRA weights get permanently merged into the model at load time (mflux's own `bake_lora=True` default) — there is no "unbake" step. `_load_sync` only passes `lora_paths`/`lora_scales` to the variant constructor on a *fresh* load; the cached-load branch (`model_path=str(saved_dir)`) must never pass them, or a LoRA already baked into that saved checkpoint gets applied a second time on top of itself. This is only safe because `_saved_model_dir()` folds a hash of the exact LoRA config into the cache dir name (`_lora_cache_suffix()`), and the model key into the front of it — different LoRA setups, and different models, can never collide on the same "is this already saved?" marker file. If you ever change what goes into a saved checkpoint (e.g. new bake-time options), make sure it's reflected in that hash too, or a stale cache dir will silently serve the wrong weights.

## Single global model, single in-flight generation

`MfluxEngine` loads one model at startup and serializes generations behind an `asyncio.Lock`. Don't add concurrency here without checking whether MLX/Metal tolerates concurrent `generate_image()` calls against one model instance — it wasn't designed for that, and the shared callback-list mutation isn't thread-safe across overlapping generations (confirmed empirically, not just in theory — see the next point).

## A disconnected client does not stop generation, and cleanup must wait for it anyway

mflux has no way to interrupt `generate_image()` from outside the thread running it (its only interrupt path is a literal `KeyboardInterrupt` on the server process). So when a client disconnects mid-stream, `generate_stream`'s `GeneratorExit` handling runs, but the background thread keeps computing regardless. The `finally` block **must** `await task` (if not already done) before removing this generation's callback from `self.model.callbacks.*` and letting the `asyncio.Lock` release — those lists are shared, unsynchronized across requests, and letting a new request start while an abandoned generation is still iterating them lets the zombie thread invoke the new request's callback and leak bogus events into its stream. Don't "simplify" that `await task` away.

## The terminal clients use chunked MultipartFile, not a single File=... sequence -- don't "simplify" that back

`show_image` in both `stream_client.*` scripts sends the base64 payload as many small `FilePart=` sequences (~200 bytes each, `CHUNK_SIZE`), bracketed by a `MultipartFile=` header and a `FileEnd` footer, instead of one `File=...:<base64>` sequence with the whole image in it. This matches iTerm2's own `imgcat` reference tool's default behavior (its `--legacy` flag is what reverts to the single-sequence form) -- confirmed by reading `imgcat`'s actual source (`gnachman/iTerm2-shell-integration`), not secondhand.

This isn't a style choice: iTerm2's OSC parser caps accumulated sequence data at exactly 1,048,576 bytes (`VT100XtermParser.m`, also confirmed by reading iTerm2's actual source) and truncates past that rather than dropping the sequence cleanly -- a truncated inline-image sequence can corrupt what renders afterward too, not just the one image. An earlier version of these scripts downscaled and JPEG-encoded a *display* copy to stay under that cap, which worked but meant the terminal showed a different (smaller, lossy) image than what got saved. Chunking makes the limit irrelevant regardless of image size or content, so the full-resolution original can be rendered directly -- verified with a round-trip test that reconstructed a real 1.4MB generated PNG from the emitted `FilePart` sequences and diffed it byte-for-byte against the saved file. If someone "simplifies" this back to a single `File=` sequence, the OSC-overflow bug returns for any sufficiently large or detailed image.

## stream_client.js: showImage must be awaited, not fire-and-forget

`showImage` in `stream_client.js` returns a Promise that only resolves via `stdout.write()`'s own completion callback, and `handleEvent` (now `async`) awaits it; `postSSE` chains each `onEvent` call through a `pending` promise so `res.on("end", ...)` doesn't resolve -- and the process can't exit -- until the last write has actually finished, not just been handed to `write()`. This existed because a large `stdout.write()` isn't guaranteed to fully flush before the process exits right after: Node's docs call TTY and file writes "synchronous" on POSIX, but pipes are documented as genuinely asynchronous, and a fire-and-forget write there can race process exit and get silently truncated -- consistent with an intermittent "final image sometimes doesn't render" symptom reported after the MultipartFile fix above, which by itself only fixed data *correctness*, not this write-completion race. Don't strip the `await`s back out for "simpler" code -- verified via a real pipe (`node stream_client.js ... | cat > out.bin`, the specific case Node calls asynchronous) that the full multipart sequence still reconstructs byte-exact with this fix in place.

## CHUNK_SIZE is 500_000, not imgcat's 200 -- and this one is a mitigation, not a confirmed fix

After the MultipartFile fix and the Node write-await fix above, the same "final image sometimes doesn't render" symptom was still reported, now for *both* clients. Captured raw stdout bytes from one real failing run and one real succeeding run (both against the live server, ~1.5MB images) and diffed each against its own saved output file: **both reconstructed byte-for-byte correct**. That's conclusive that the bug is not in what either client writes -- the fully correct escape sequence was present in the failure capture too. So this is not a data-correctness bug like the OSC-cap one above; something in iTerm2's own handling of ~10,000 tiny back-to-back `FilePart` sequences (200 bytes each, copying `imgcat`'s default) is where it's actually going wrong, and that's opaque without live-debugging iTerm2 itself.

`imgcat`'s own comment for the 200-byte choice is "this helps it get through tmux" -- a constraint that doesn't apply here (neither script wraps for tmux). Raising `CHUNK_SIZE` to 500,000 (still ~2x under the real 1,048,576-byte single-sequence cap) cuts a ~1.5MB image from ~10,000 sequences down to ~4. This is a reasonable, well-motivated mitigation for "maybe iTerm2 chokes under that much volume/frequency," verified to preserve byte-exact correctness -- but unlike the other entries in this file, **it is not a confirmed root-cause fix**, because the byte-level capture proved the bug isn't in the data volume or correctness per se. If reports of this symptom continue after this change, the next step is investigating iTerm2 itself (or its interaction with this specific terminal/session), not re-tuning this constant further.

## Final-image metadata reuses mflux's own embedding pipeline via a temp-file round-trip

`_encode_final_png_with_metadata` in `engine.py` calls `image.save(tmp_path, overwrite=True)` on the `GeneratedImage` mflux itself returns from `generate_image()` -- not a hand-rolled metadata dict. That method is file-path-based (it re-opens and re-saves via `PIL.Image.open(path)` / `image.save(path, ...)` internally, see `mflux.utils.image_util.ImageUtil.save_image` and `mflux.utils.metadata_builder.MetadataBuilder.embed_metadata`), so there's no way to get the embedded bytes without an actual file on disk -- hence `tempfile.mkstemp` + read-back + `os.unlink` in a `finally`. `overwrite=True` matters: `mkstemp` already creates the file (empty) before `.save()` touches it, and mflux's own `resolve_output_path` would otherwise treat that as a name collision and silently write to a *different*, auto-incremented path instead of the one we're about to read back.

This depends on `ImageUtil.embed_metadata_enabled` defaulting to `True` (it's mflux's `--no-metadata` CLI opt-out, which nothing here ever sets) -- if a future mflux version changes that default, or renames the class attribute, metadata embedding would silently stop rather than error. Preview images are deliberately not touched by this -- they're decoded via `ImageUtil.to_image` directly (see `_decode_preview_b64`), not through a `GeneratedImage.save()`-capable wrapper, and it wouldn't be worth the temp-file overhead per step anyway.

## Step timings require an explicit `mx.eval` in the in-loop callback

MLX evaluates lazily, and mflux's denoising loop calls in-loop subscribers *before* its own `mx.eval(latents)` (see `mflux/models/z_image/variants/z_image.py` — `predict` → `scheduler.step` → `ctx.in_loop(t, latents)` → `mx.eval(latents)`). So `_StreamCallback.call_in_loop` runs with step `t`'s graph merely *built*, not computed. Reading the clock there without forcing evaluation attributed step `t-1`'s compute to step `t`: step 1 reported ~10ms and every later timestamp lagged a full step, which is exactly how it looked in practice (`step 1 (10ms)` / `step 2 (48195ms)` / `step 3 (66099ms)` on an 82.9s run). The `mx.eval(latents)` at the top of `call_in_loop` fixes that and costs nothing — mflux evaluates the same graph on its very next line — but it is load-bearing, not defensive: remove it and the timings silently go back to being off by one, with no error anywhere.

mflux's own tqdm bar never had this problem because it advances at the end of the iteration, after that eval. Note also that this only holds while mflux keeps evaluating inside the loop; if a future version moves or drops that `mx.eval`, this callback becomes the thing forcing per-step evaluation rather than merely anticipating it.

## MLX's buffer cache, not the model, is what makes this server slow down over time

Symptom: generations get dramatically slower the longer the server has been up — first run ~7s, third run ~105s for identical work — with the *first* denoising step absorbing almost all of it and the final VAE decode inflating too. It looks like a model or callback problem. It isn't: the process is page-faulting, not computing (`ps -o state` shows `U`, uninterruptible I/O wait).

MLX retains buffers it has freed for reuse. Uncapped on a 32GB M2 Pro, that cache grew to 8.6GB on top of 10.1GB of q8 weights — a 20GB footprint, all of it `IOAccelerator (graphics)` in `footprint -p <pid>`, none of it visible in `ps` RSS (which reported 0.03GB for a process actually holding 24GB). Past the machine's headroom, the weights get compressed out between generations and faulted back in on the next one's first `mx.eval`. Hence `MFLUXIBLE_MLX_CACHE_LIMIT_MB`, defaulting to 1024: footprint 12-13GB, and the same three runs take 5.2s / 5.3s / 4.9s.

Debugging notes for next time, since three plausible theories were wrong before the right one:
- **It is not `mx.compile`.** mflux calls `mx.compile(predict)` only when `AppleSiliconUtil.is_m1_or_m2()` is false; "Apple M2 Pro" matches that check (it excludes only Max/Ultra), so nothing is compiled on this hardware at all.
- **It is not text-encoder compute.** A 220-word prompt costs the same as a 2-word one — 110x the tokens, no change — so the fixed cost is I/O, not prefill.
- **It is not one-time warmup.** It recurs on every generation, so there is nothing a load-time `mx.eval` of the weights could pre-pay. `active_bytes` on `/health` stays flat across runs while wall time climbs, which is the tell.
- Measure with a fresh process. A long-running server is already deep in the pathology, and its numbers say nothing about the model's actual speed.
