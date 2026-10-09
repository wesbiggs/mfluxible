# The MCP tool

Notes on `clients/mfluxible_mcp/`: SDK quirks, timeouts, and why mask boxes are fractions. Read before touching the MCP server.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## mcp SDK also moved fast: FastMCP -> MCPServer

`clients/mfluxible_mcp/server.py` targets `mcp` 2.x, where `mcp.server.fastmcp.FastMCP` (the commonly-documented v1 API) was renamed to `mcp.server.mcpserver.MCPServer`. Importing the old path raises a `ModuleNotFoundError` with a migration pointer, it doesn't just silently break — if that happens, you're looking at v1-flavored example code (`FastMCP(...)`) against a v2 install. `Context`, `Image`, and the `@server.tool()` decorator are all still there, just re-exported from `mcp.server.mcpserver` instead.

## Progress notifications do not reliably hold a host's tool-call timeout open

This is why `clients/mfluxible_mcp/server.py` runs generation in a background task and hands back a
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
that has `mfluxible_mcp/server.py` treat a missing `/health` field as "unknown, let the server decide"
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
