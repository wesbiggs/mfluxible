# Project Notes

## Layout

`server/` (server.py, engine.py, models.py, schemas.py, chat_stub.py, requirements.txt) is the model + HTTP API. `client/` (stream_client.py, stream_client.js, harness.html, mcp_server.py, requirements.txt, requirements-mcp.txt) is everything that talks to it over HTTP. They're independent dependency-wise -- installing one's requirements.txt doesn't pull in the other's. server.py/engine.py/models.py/schemas.py/chat_stub.py import each other as flat sibling modules (`from engine import ...`), not a package, so `server/` must stay on `sys.path` when running (e.g. `uvicorn server:app --app-dir server`) -- don't add an `__init__.py` or turn this into a `server.*` package without updating those imports and the run command together.

## API changes

Always update `README.md` when changing the API (endpoints, request/response shapes, SSE event schema).

## mflux is a fast-moving dependency

`server/engine.py` reaches into mflux internals that aren't public API: `model.callbacks.before_loop/in_loop/interrupt` are mutated directly, since `CallbackRegistry` has no `unregister()` as of mflux 0.19.1. The VAE-decode branching in `_decode_preview_b64` mirrors mflux's own `StepwiseHandler` on purpose, with one deliberate deviation: the non-packed branch goes through `VAEUtil.decode` rather than calling `vae.decode()` directly the way `StepwiseHandler` does. Qwen-Image's VAE is a 3D (video) decoder returning `(B, C, 1, H, W)` and `ImageUtil.to_image` wants 4D — `VAEUtil.decode` is what drops the singleton frame axis, and it's the same call each variant's own final decode makes, so previews and final images stay on identical handling. Calling `vae.decode()` bare here works for Z-Image and FLUX and breaks only on Qwen previews. If `pip install -U mflux` breaks this file, check `mflux/callbacks/callback_registry.py` and `mflux/callbacks/instances/stepwise_handler.py` in the installed package first — that's where this was reverse-engineered from (mflux ships no public docs for the callback system).

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

## mcp SDK also moved fast: FastMCP -> MCPServer

`client/mcp_server.py` targets `mcp` 2.x, where `mcp.server.fastmcp.FastMCP` (the commonly-documented v1 API) was renamed to `mcp.server.mcpserver.MCPServer`. Importing the old path raises a `ModuleNotFoundError` with a migration pointer, it doesn't just silently break — if that happens, you're looking at v1-flavored example code (`FastMCP(...)`) against a v2 install. `Context`, `Image`, and the `@server.tool()` decorator are all still there, just re-exported from `mcp.server.mcpserver` instead.

## Progress notifications do not reliably hold a host's tool-call timeout open

This is why `client/mcp_server.py` runs generation in a background task and hands back a
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

## One model per process, and every per-model difference lives in models.py

`server/models.py` is the whole multi-model story: which mflux variant class, which `ModelConfig`, which latent creator, the default step count, and whether `guidance`/`negative_prompt` mean anything. `engine.py` has no per-model branching and shouldn't grow any — mflux's ZImage, Flux1 and QwenImage happen to share a constructor signature, a `generate_image()` signature, a `save_model(base_path)` and a `callbacks` registry, which is the only reason this works.

Two things in that file are load-bearing rather than stylistic:

- **The mflux imports are deferred into each spec's `load()` function**, so naming four models costs nothing. mflux downloads weights inside the variant's constructor (`WeightLoader.load` → `PathResolution.resolve` → HF snapshot download), never at import, so a process only ever fetches/quantizes/caches the one model `MFLUXIBLE_MODEL` selected. Don't hoist those imports to module level "for tidiness" — it wouldn't download anything, but it would drag every model's module graph into every process and quietly make that guarantee depend on mflux never doing work at import time.
- **`model_config` must be passed on the cached-load branch too** (`_load_sync`). A saved directory holds weights and tokenizers, not the model's scheduler/sequence-length settings, and each variant's own default would otherwise win — `Flux1` defaults to *schnell*, so a `flux-dev` cache dir would silently load as schnell.

The clients deliberately send `steps` (and `guidance`) as null rather than a number of their own: whichever model is loaded decides, so none of them needs reconfiguring when the server switches models. Don't "fix" a missing default back into `stream_client.*`, `harness.html` or `mcp_server.py` — a step count that suits Z-Image-Turbo is four times too small for FLUX.1-dev. `harness.html` and `mcp_server.py` additionally read `/health` to learn what the loaded model accepts; both treat a failed or model-less `/health` as "unknown, let the server decide" rather than an error, so they keep working against a server that predates that field.

Guidance and negative prompts are rejected with a 400 on models that can't act on them rather than accepted and dropped, because mflux accepts both arguments on every variant and silently ignores them (its own CLIs print a warning instead). `check_request` runs in the endpoint, *before* `StreamingResponse` starts: raising inside the generator would mean a 200 status line already on the wire and a torn body.

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
