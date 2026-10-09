# Third-party frontends: the chat stub and the SillyTavern shim

What the OpenAI chat stub and the A1111-shaped `txt2img` shim are for, what was read out of the callers' source, and why the shim inverts the reject-don't-drop rule. Read before touching `chat_stub.py`, `routes/chat.py` or `routes/a1111.py`.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## mfluxible/chat_stub.py hardcodes a specific tool name, confirmed against one caller

`POST /v1/chat/completions` only ever emits a tool call for a tool literally named
`generate_image` (see `GENERATE_IMAGE_TOOL_NAME` in `mfluxible/chat_stub.py`) if the
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
