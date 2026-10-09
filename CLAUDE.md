# Project Notes

This file is the short form: what the repo is made of, and the rules that are expensive to
rediscover. The *reasoning* behind most of them -- the measurements, the dead ends, what
was read out of which upstream source -- lives in [`design/`](design/), one file per area.
**Before changing code in an area, read its design file**; each rule below names it. The
rules are the conclusions; those files are what let you tell whether one still applies.

## Layout

`mfluxible/` (server.py + routes/ + runtime.py, engine.py, imaging.py, limits.py, models.py,
schemas.py, chat_stub.py, schedulers.py, auth.py, vlm.py, vlm_local.py, vlm_reply.py,
vlm_worker.py, config.py, cli.py) is the model + HTTP API, and is the package published as
**`mfluxible`**. `server.py` only assembles the app; settings and the engine live in
`runtime.py` (routes read them as `runtime.NAME` at call time, so tests patch one place), and
each API surface is a module under `routes/`. `imaging.py` is the Pillow-only half of what
the engine does with images, split out so nothing that only needs to orient an image imports
MLX.

`clients/` (stream_client.py, stream_client.js, harness.html, mfluxible_mcp/,
requirements.txt) is everything that *consumes* the API over HTTP. `vlm_worker.py` also talks
to the API over HTTP and is deliberately **not** there: it is a **sidecar** that supplies a
capability the server advertises, must sit on the same machine (it reads the stash directory
off disk), and takes its settings from the server's configuration. **Nothing else in the
package imports it and nothing should** -- an import would put an arbitrary subprocess back
inside the request path ([design/detection.md](design/detection.md)).

Three things inside `clients/` are not the same kind of thing:

- `mfluxible_mcp/` is a **second distribution** (`mfluxible-mcp`, its own
  `clients/pyproject.toml`). It consumes the API over HTTP and so must never install
  mflux/MLX/PyTorch -- do not turn it into an `mfluxible[mcp]` extra.
- `stream_client.py`/`.js` stay plain scripts with a `requirements.txt`; a single-file
  script run with `uv run` is the whole point of them.
- `harness.html` is a client by the rule above, but it is also the only client the *server*
  serves (`GET /`), so the server's wheel force-includes a copy at `mfluxible/harness.html`
  and `routes/misc._harness_path()` prefers that over the checkout's. Its source of truth is
  `clients/`.

**`harness.html` is one self-contained file, on purpose: keep it that way.** All CSS and JS
are inline and it loads nothing external, so it can be opened straight from disk, copied
somewhere, or served by any static file server with no build step. Do not split it into
separate CSS/JS files or add a bundler, a CDN script or a font file, even though it is large
(~2.5k lines). Sections inside it are fenced by comment rules; add structure there instead.
(Opened as a `file://` URL its `Origin` is `null`, which the server's default CORS config
rejects -- see [docs/clients.md](docs/clients.md) -- so that route needs a deliberate
server-side opt-in.) Harness-specific notes: [design/harness.md](design/harness.md).

These are real Python packages. Imports are absolute (`from mfluxible.engine import ...`),
the dotted scheduler paths handed to mflux are `mfluxible.schedulers.*`, and `pytest.ini`
puts the repo root on `sys.path`. The run command is `mfluxible-server`, or
`uvicorn mfluxible.server:app` for anything it doesn't expose.

## Docs

Always update `docs/api.md` when changing the API (endpoints, request/response shapes, SSE
event schema). `README.md` is only an overview + quickstart + index; the rest of the prose
lives in four pages under `docs/`, split by who the reader is: `server.md` (running the
server -- env vars, models, remote hosting, troubleshooting), `clients.md` (terminal scripts,
the harness, OpenAI-compatible frontends), `mcp.md` (the MCP tool and its own env vars) and
`api.md`. A server env var goes in `server.md` and an MCP one in `mcp.md`, even though both
are "configuration". `docs/` is for people *running* it, `CONTRIBUTING.md` says what to *do*
when changing the code (tests, CI, adding a model), and `design/` says *why* the non-obvious
parts are the way they are.

## Rules

Each of these is a conclusion; the file in brackets has the evidence.

### Errors and auth  [design/errors-and-auth.md]

- **No path from an exception to a response body, in either direction.** Validation
  *returns* a reason (`MfluxEngine.request_problem`, `routes/common.collect_final_image`);
  nothing in `server.py` or `routes/` binds a caught exception to a name
  (`test_no_route_module_can_carry_an_exception_into_a_response`). Do not add
  `except ...: str(exc)` to "simplify" a pair of returns. Exception text from mflux/MLX/Pillow
  quotes absolute cache paths; log it, send a fixed sentence.
- **Auth is off unless *both* `MFLUXIBLE_BASIC_AUTH_*` variables are set** -- half-configured
  means off, and the startup warning covers an exposed unauthenticated bind.
- **Middleware order is load-bearing and reads backwards**: CORS must wrap auth (a preflight
  carries no credentials), so auth is added *first*. Auth and the body limit are **pure
  ASGI**, never `BaseHTTPMiddleware`, which buffers the SSE stream.
- **Basic auth (in-app) and `Caddyfile.example` (bearer) are alternatives, not layers.**
  The Caddyfile gates by exclusion; a path *added* to its `@public` list silently becomes an
  ungated endpoint (`tests/test_proxy_config.py` guards this -- it reads the OpenAPI schema,
  not `app.routes`). Behind a proxy the app must stay bound to 127.0.0.1.
- **Turning auth on blocks the SillyTavern integration; that is documented, not worked
  around.** Do not exempt endpoints from auth to keep one client working.
- **`MFLUXIBLE_BEARER_TOKEN` is environment-only** in all clients (no `--token` flag), and
  setting it *and* putting credentials in the URL is refused rather than resolved.

### Configuration and packaging  [design/packaging-and-config.md]

- **Every setting is read from the environment at module import.** The config file adds no
  settings: a key *is* the variable with `MFLUXIBLE_` dropped and lowercased, and
  `config.apply()` runs in `mfluxible/__init__.py` so it precedes every read.
- **The real environment wins over the file; an unknown key is an error;** values pass
  through verbatim. A new setting goes in `config.KNOWN`, `mfluxible.example.toml` and
  `docs/server.md` -- tests fail on drift in either direction. Read the variable at the call
  site, spelled out in full, and use `config.as_int`/`as_float` to parse it so an error
  names the variable.
- **`[tool.uv] managed = false` in both `pyproject.toml` files is the load-bearing line.**
  This is not a uv project: no `uv.lock`, no `uv sync`. Without it `uv run` would resolve the
  server's dependency tree on a machine that only wants `clients/stream_client.py`.
- **The two distributions release together off one tag and share a version**
  (`__version__` in two files; tests pin it). `uv build` builds the wheel *from the sdist*,
  so the sdist's `include` list decides whether the wheel can still force-include
  `harness.html`.
- Both XDG variables are honoured (`config.cache_home()`, `user_config_path()`); don't
  "make the two config file names consistent".

### The engine and models  [design/engine.md]

- **One model per process, one generation at a time, all MLX work on one thread.** Load and
  every generate/detect call share `MfluxEngine._executor`; generations hold the
  `asyncio.Lock`. Don't add concurrency without checking MLX tolerates it.
- **`generate_stream`'s `finally` must `await task` before removing callbacks and releasing
  the lock** -- a disconnected client does not stop the generation thread, and the callback
  lists are shared and unsynchronized. Don't simplify that away.
- **`mx.eval(latents)` at the top of `_StreamCallback.call_in_loop` is required**; without it
  step timings are silently off by one.
- **Every per-model difference lives in `models.py`; `engine.py` has no per-model
  branching.** Keep the mflux imports deferred into each spec's `load()`. A new row is only
  enough when the model shares the common mflux surface -- read [CONTRIBUTING.md](CONTRIBUTING.md)
  and the field-by-field notes before adding one.
- **Guidance and negative prompts are rejected with a 400 where the model can't act on
  them,** and `request_problem` tests the *effective* guidance (`CFG_GUIDANCE_FLOOR`).
  Clients send `steps`/`guidance` as null; don't "fix" a missing default back into them.
- **Quantized-cache marker is a heuristic.** `model_config` must be passed on the cached-load
  branch; `lora_paths` must *never* be (they are already baked in), and the cache directory
  name hashes the LoRA config. Change what goes into a saved checkpoint -> change the hash.
- **mflux internals are reached into** (`model.callbacks.*` mutated directly, the
  `VAEUtil.decode` preview path). After `uv pip install -U mflux`, re-check them first.
- **`MFLUXIBLE_MLX_CACHE_LIMIT_MB` (default 1024) is what stops the server slowing down
  over time.** Measure performance in a fresh process.
- **Final-image metadata comes from mflux's own `.save()` via a temp file**; it depends on
  `ImageUtil.embed_metadata_enabled` staying `True`.

### Inpainting and schedulers  [design/inpainting.md]

- **A scheduler path handed to mflux is an arbitrary module import: it comes from a fixed
  table (`scheduler_path`) and is never built from request data.** The API takes bools.
- **`fractional_start` and `mask` are only valid where `ModelSpec.default_scheduler ==
  "linear"`;** `request_problem` rejects them elsewhere. Extending to another model means a
  sibling scheduler class, not relaxing the check.
- **Masked requests default `image_strength` to 0.0** (`DEFAULT_MASKED_IMAGE_STRENGTH`),
  because 0.4 makes a mask look ignored while reporting success. The MCP tool duplicates
  this on purpose.
- **The mask job is module-level state,** safe only because of the lock and the single
  thread. If either invariant goes, it must become per-Config state.
- **`mask_feather` is not a region blend** -- a large value produces a double exposure.
  `POST /v1/images/edits` rejects `mask` because OpenAI's convention is the inverse.

### Object detection  [design/detection.md]

- **Two backends, chosen by `MFLUXIBLE_VLM_BACKEND`, never inferred** from what happens to
  be attached. `local` (default) is lazy, runs under `MfluxEngine.run_exclusive`, and has two
  timeouts; `worker` is the sidecar.
- **One slot, no job identifier** (the image is re-encoded on its way into a Claude session,
  so no hash can match). Dimensions and the harness's `baseImageEpoch` are the guards.
- **The output contract is fixed and backends adapt to it:** boxes are fractions, validated
  once in `vlm_reply.py`. There is no coordinate-conversion setting, because a wrongly
  scaled box looks plausible.
- **`MFLUXIBLE_VLM_DIR` is one variable doing two jobs** (enable + stash); the mailbox
  prunes only files it wrote.
- The harness reads a missing `vlm` block in `/health` as *off*.

### Third-party frontends  [design/integrations.md]

- **The chat stub only emits a tool call for a tool actually offered by name**
  (`generate_image`, or `<server>__generate_image`); if not offered it says so visibly.
- **`/sdapi/v1/txt2img` drops fields the loaded model can't act on** (SillyTavern shows no
  error text), but a concretely wrong result (`batch_size` > 1) stays a 400. It re-runs
  `request_problem` on its own output so drift becomes a visible 400. Written against
  SillyTavern 1.18.0; re-read its source rather than inferring from a failing generation.

### The MCP tool  [design/mcp-tool.md]

- **Every failure a caller could act on must be a `ToolError`** -- mcp 2.x strips the text
  from anything else. Targets `mcp` 2.x (`MCPServer`, not `FastMCP`).
- **Generation runs in a background task behind a `check_image` handle;** progress
  notifications don't hold a host's timeout open. Keep `WAIT_SECONDS` under the shortest
  timeout you care about.
- **`mask_boxes` are fractions, not pixels,** because the model saw a downscaled copy; the
  mask is rasterized client-side at the EXIF-*oriented* size. `generate_image` and
  `preview_mask` share `_resolve_mask`. The prompt describes the whole finished frame, not
  the masked region.

### Terminal clients  [design/terminal-clients.md]

- **`show_image` sends chunked `MultipartFile`/`FilePart`/`FileEnd`, not one `File=`
  sequence** (iTerm2 truncates an OSC past 1,048,576 bytes). In `stream_client.js`,
  `showImage` must be awaited and chained through `postSSE`'s `pending`. `CHUNK_SIZE`
  500_000 is a mitigation, not a confirmed fix.
- One URL means one thing to both clients (credentials in the URL produce the same header).

### The harness  [design/harness.md]

- The progress sweep and shimmer key off `body:has(#submitBtn:disabled)`; Generate must be
  disabled for exactly the length of a run, so never gate it on other state.
- `.canvas` stacks its panes in rows: `syncBasePane()` derives the fallback pane from the
  other panes' `.hidden` classes rather than from separate state. "Use as Base Image"
  deliberately bypasses `showInputImage()`.
- `previewMimeType` interpolates a fixed list's value, never the browser-supplied `.type`
  (CodeQL `js/xss-through-dom`; [design/errors-and-auth.md](design/errors-and-auth.md)).
