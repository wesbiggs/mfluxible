# Error handling and authentication

The invariant that no exception text reaches a response body, and the two auth schemes (in-app Basic auth, or a reverse proxy with a bearer token) with the client-side credential handling. Read before touching `server.py`/`routes/`, `auth.py`, `Caddyfile.example`, or any client's credential code.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## No path from an exception to a response body, in either direction

`mfluxible/server.py` and everything under `mfluxible/routes/` bind no exception to a name
(the one `except` there, in `routes/vlm.py`, discards it -- the same shape as auth.py's), and that is the point rather
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
you find yourself adding `except ... : str(exc)` back into the routes to "simplify" the
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

`mfluxible/auth.py` is off unless **both** `MFLUXIBLE_BASIC_AUTH_USERNAME` and
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

`MFLUXIBLE_BASIC_AUTH_*` (mfluxible/auth.py) and `Caddyfile.example` are **alternatives, not
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
`mfluxible_mcp/server.py` has no command line to take one on anyway (an MCP host launches it as a
stdio subprocess).

Setting `MFLUXIBLE_BEARER_TOKEN` *and* putting credentials in the URL is **refused** rather than
resolved, in both terminal clients. It isn't defensiveness: requests derives Basic auth
from userinfo inside `prepare_auth`, which runs *after* `prepare_headers`, so the URL
would silently overwrite the bearer header rather than losing to it. Picking a winner
quietly is the failure worth avoiding; the same check is mirrored in the JS client so one
documented rule covers both.
