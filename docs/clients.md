# Clients

## Example clients

Fully usable example clients live in `clients/` and talk to the server over HTTP — neither loads the model itself, so the server from the [Quickstart](../README.md#quickstart) must already be running. The [MCP tool](mcp.md) is a third client in the same vein, documented on its own page.

All clients share the same Apache-2.0 license as the server and can be modified and/or used as the basis for other tools.

### Python and node terminal scripts

`stream_client.py` (`uv pip install -r clients/requirements.txt` first — just `requests`) and the dependency-free `stream_client.js` (Node 18+) render step progress and previews inline in the terminal as they stream, saving the final image to disk:

```bash
uv run clients/stream_client.py "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
# or
node clients/stream_client.js "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
```

Both take `--steps`, `--seed`, `--guidance` and `--negative-prompt`, and leave all four to the server when you don't pass them — so `--steps` is only worth setting to override the loaded model's own default. `--guidance` and `--negative-prompt` are refused (with a message naming the model) on models that can't act on them; see [Models](server.md#models).

If the server is behind [authentication](server.md#authentication), both scripts carry it the same way. For a bearer token (a proxy in front), `export MFLUXIBLE_BEARER_TOKEN=...` — environment rather than a flag, so the secret stays out of shell history and out of `ps` for the length of a generation. For the server's own HTTP Basic, put the credentials in the URL: `--url http://user:pass@host:8420/mfluxible/v1/images/generations`. Setting `MFLUXIBLE_BEARER_TOKEN` *and* passing a URL with credentials is refused with a message rather than silently resolved — the two would collide in the request, and which one wins isn't worth leaving to the HTTP library.

Both also take `--image PATH` for image-to-image (read from disk and base64-encoded, not a URL) and `--image-strength` (0.0–1.0, only valid alongside `--image`; the server's own default, 0.4, applies if you omit it) — see [Image-to-image](api.md#image-to-image) for what `image_strength` actually controls (mflux's convention is the inverse of some other tools'). `--fractional-start` (also only valid alongside `--image`) makes that strength continuous instead of quantized to `1/steps`, at no extra cost — see [Fractional start](api.md#fractional-start).

Both render the exact full-resolution bytes returned by the server — no downscaling, no recompression, nothing client-side touches the image data. They use the iTerm2 inline-image protocol's chunked `MultipartFile` variant (also works in WezTerm; in an unsupported terminal the escape codes are just ignored, and the saved file and progress text still work either way), the same variant iTerm2's own [`imgcat`](https://github.com/gnachman/iTerm2-shell-integration/blob/master/utilities/imgcat) reference tool uses by default: the base64 payload is split into `FilePart=` sequences behind a metadata-only header and a `FileEnd` marker, rather than one giant `File=...:<base64>` sequence.

This matters because iTerm2's own source caps how much data it'll accumulate for a *single* OSC escape sequence at 1,048,576 bytes ([`VT100XtermParser.m`](https://github.com/gnachman/iTerm2/blob/master/sources/VT100/VT100XtermParser.m)) — past that it truncates rather than cleanly dropping the sequence, which can corrupt what renders afterward too, not just fail to show the one image. Diffusion output is detailed/photographic content that a full-resolution PNG can realistically approach or cross that limit for. Chunking (500,000 bytes/chunk here — `imgcat`'s own 200-byte default exists specifically to survive tmux, which doesn't apply since neither script wraps for tmux) means no image, at any size or detail level, can hit that cap.

Images render at `width=auto` (height defaults to auto too) — the same default `imgcat` uses: native pixel dimensions divided by the display's backing scale factor (e.g. a 1024px image renders at 512pt on a 2x/Retina display), rather than a fixed cell-count width that would scale with the terminal's font size instead of the image's actual dimensions.

Not tmux-aware — iTerm2's protocol needs extra passthrough wrapping inside tmux that these scripts don't do.

### Browser

`clients/harness.html` is a dependency-free page (plain HTML/CSS/JS in one file, no build step, no CDN — nothing is fetched but the API itself) laid out as a two-pane app: a scrolling column of controls on the left with Generate and Reset pinned beneath it, and the image stage on the right with the run log along its bottom edge. It has a form for prompt/width/height/steps/seed/preview_every that calls the streaming endpoint directly from the browser via `fetch`, reads [`/health`](api.md#get-health) on load to show which model the server is running (leaving Steps blank uses that model's default, and Guidance / Negative prompt appear only if it accepts them), parsing the SSE stream the same way the terminal clients do, and renders previews and the final image as `<img>` elements (via `data:` URLs) plus a download link for the final PNG. The stage shows one pane at a time — the mask editor, the live preview, the result, or, with none of those to show, the base image itself. A live preview is drawn at the size the result will be rather than shrunk: the server sends every preview at the full requested resolution (see [`preview`](api.md#post-mfluxiblev1imagesgenerations)), so there is nothing to gain by throwing it away, and a preview framed like the result means nothing jumps when the real one lands. What marks it provisional is the **Live preview** tag and a slow shimmer that passes over it for as long as the run is in flight — it stops if the run fails, leaving the last preview sitting there readable. **Lock aspect ratio**, the checkbox under the Width and Height boxes, is off by default; ticking it takes the ratio those two have at that moment and then moves whichever field you aren't editing to keep it, rounded to the nearest multiple of 16 (see below for why 16). The ratio is held from when you ticked it rather than re-read from the boxes on every keystroke, so nudging a dimension repeatedly can't walk it a rounding step at a time, and a half-typed number on the way to the real one corrects itself. Under 900px wide the panes stack — stage first, controls beneath, Generate stuck to the bottom of the viewport — so the image doesn't end up below a column of inputs taller than the screen. It follows the OS light/dark setting, drawing both schemes from one set of CSS custom properties at the top of the file.

It also does image-to-image. Get a base image onto the page either by dragging an image file onto it from anywhere (drop targets are the whole page, not just the drop-zone box — that's just where the visual highlight and preview show up) or by clicking the drop zone to pick a file; or click **Use as Base Image**, which sits under the generated image alongside Download PNG, to feed the most recent result straight back in as the next input, for chaining edits without a round trip through disk. The trash icon in the corner of the **Base image** heading drops back to plain text-to-image. Loading a base image any of those ways also sets Width and Height to the image's own dimensions, rounded down to a multiple of 16 — mflux resizes the input to whatever `width`/`height` the request carries with a plain `resize()` and no aspect-ratio handling, so a portrait image against the 1024×1024 default would be stretched, not letterboxed, and that distorted image is what the generation is seeded from. 16 rather than 8 because that's what the server itself does with the number (mflux floors both dimensions to a multiple of 16), so the box says the size you'll actually get; the fields step by 16 to match. If the aspect-ratio lock is on, loading an image re-takes the ratio from that image rather than imposing the old one on it — the shape worth holding is the input's own, for the same stretching reason. That trash icon, along with Image strength and Fractional start, is shown only while a base image is actually loaded — with none, the section is just its drop zone, and controls that have nothing to act on stay out of the way. The Image strength field controls how strongly that input constrains the output (default 0.4) — see [Image-to-image](api.md#image-to-image) for what the number actually means; the (i) tip beside the field is a reminder that it's the inverse of some other tools' "denoising strength". The **Fractional start** checkbox next to it is [the same flag](api.md#fractional-start) the API takes, and is sent only while an image is loaded. Both fields carry a small (i) button that reveals an explanatory tip on hover, keyboard focus or tap, rather than spending sidebar height on prose that's only read once. If the chosen image is a PNG this server generated, its embedded prompt and seed (read straight out of the PNG's `eXIf` metadata, client-side, no server round trip) are loaded into the Prompt and Seed boxes automatically — a photo with no such metadata just leaves both alone. Width and Height still come from the image's own dimensions rather than from that metadata, so a generated image that's since been cropped or resized still gets the size it actually is.

Loading a base image also takes over the stage, which answers the obvious question of where a dropped image went: a result from some earlier run isn't a picture of the image now in the form, so it comes down and the new input takes its place — or the mask editor does, if **Inpaint a region** is ticked, since that's the same image with the region drawn over it. **Use as Base Image** deliberately doesn't do this: there the result on screen *is* the image being adopted, so it stays put, Download PNG and all. The base image holds the stage for the length of a run too, so a generation with previews off has something to look at other than an empty box.

A result generated *from* a base image carries a **compare handle** — a vertical divider that rests at the image's right edge, showing the result whole, and wipes it leftwards to reveal the input underneath. Drag it, click anywhere on the image to jump it there, or focus it and use the arrow keys (Home and End go to the extremes). What it reveals is the base image the *displayed* result was generated from, captured when that request went out: dropping a different image while it runs, or clearing one afterwards, doesn't retarget it. The input is stretched to the result's dimensions rather than letterboxed inside them, because that's what the server did with it as well — see the `resize()` note above — so what you're wiping back to is the image the model actually saw. A plain text-to-image run has nothing to compare against and shows no handle.

**Inpaint a region**, the checkbox under Image strength, turns the stage into a mask editor: the base image with a paintable overlay on top, and a toolbar underneath with Brush / Rect / Erase, a brush-size slider, Invert, Clear and a **Latent grid** toggle. Paint the part you want the model to change — red is "regenerate this", everything else is held to the original. It's shown only while a base image is loaded and only if [`/health`](api.md#get-health) says the loaded model accepts a mask (`supports_mask`); as with the other `supports_*` flags, a server that doesn't report it at all is treated as "unknown, let the server decide" rather than as a no.

Some specifics worth knowing, because they're not guesses the UI is making on your behalf:

- **The mask is painted at the base image's own pixel size**, whatever size the editor happens to be on screen, so a small window doesn't cost resolution. That's also the size the API requires it to be.
- **Coverage is reported in latent cells, not pixels** — "27% of 16,384 latent cells". The mask is downsampled by 8 before the model sees it, so cells are the unit that actually exists; the **Latent grid** toggle draws that grid over the image, which is the honest way to show why detail finer than 8px doesn't survive and why **Mask feather** steps in 8s. That field is capped at 32: feather hides the grid's staircase, and a wide one cross-fades the old content with the new instead (see [Inpainting a region](api.md#inpainting-a-region)).
- **Ticking the box moves Image strength to 0**, which is where inpainting wants it and not where image-to-image wants it. See [Inpainting a region](api.md#inpainting-a-region) — with a mask, that field governs only how much of the original survives *inside* the mask, and the usual 0.4 leaves the old object standing there.
- **Pixel-exact outside** (on by default) pastes the result back over your input through the mask, so everything outside it is byte-identical to what you sent rather than a VAE round-trip of it.
- **Back to mask**, beside Download PNG on the result, returns to the editor with the mask still on it. Loading a *different* base image of the same dimensions also keeps it — so "Use as Base Image" then Generate re-runs the same region on the previous result — while one of different dimensions necessarily drops it.

Two notes the editor itself carries, because they're the surprises: the mask edge is a hard boundary (anything the new content needs, shadow included, has to be inside it or it's cut off), and the region is *filled with your prompt* rather than erased — asking for "an empty table" gets you something table-shaped. Generate with inpainting on and nothing painted is refused client-side with a line in the log rather than sent, since the server would reject it anyway.

**Reset all**, next to Generate, puts the page back the way it loads: every field to its default, the input image and the remembered last result dropped, and the log, preview and result panes cleared. It also re-checks `/health`, since the Server URL is one of the fields it resets. It's disabled while a generation is running, so a reset can't clear a log that's still being written to.

The server itself serves this page at its root, `GET /` — just open `http://localhost:8420/` (or whichever host/port `server.py` is bound to) once it's up. The Server URL field defaults to the relative path `/mfluxible/v1/images/generations`, which resolves against whatever origin served the page, so no configuration is needed for this same-origin case.

**Advanced → API token** is for the case where a reverse proxy is gating the API with a bearer token (see [Authentication](server.md#authentication)); leave it empty otherwise and the page sends no `Authorization` header at all. It's kept in `localStorage` rather than re-typed each visit, which means it survives a reload and is deliberately *not* cleared by **Reset all** — it's how you reach the server, not a parameter of the image. Entering one re-checks `/health`, so a deployment that gates that endpoint too fills in its model fields as soon as the token lands. The server's own HTTP Basic needs nothing here: the browser prompts for it and attaches it to the page's same-origin requests itself.

If you'd rather host the page separately (e.g. to point one harness at multiple servers, or to exercise the CORS path), it still works opened from any static file server — just not as a `file://` URL, since the browser's `Origin` header for a local file is `null`, which the server's default CORS config won't match:

```bash
cd clients && python3 -m http.server 8000
# then open http://localhost:8000/harness.html and point Server URL at the API host
```

(CORS is on by default and reflects back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin, so this works with no server-side configuration — see [CORS](server.md#cors) if you need something different.)

## Third party clients

### OpenAI compatible frontends

The example clients above all speak mfluxible's own native API. But `POST /v1/images/generations` and `POST /v1/images/edits` (see the [API reference](api.md)) are genuine [OpenAI Images API](https://platform.openai.com/docs/api-reference/images/create)-compatible endpoints, so any tool built against that API can point at this server directly, with no code changes on its side.

#### Open WebUI

For example, [Open WebUI](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/openai/)'s Settings → Admin → Images panel takes an arbitrary `IMAGES_OPENAI_API_BASE_URL` and a free-text model name — set the base URL to `http://127.0.0.1:8420/v1` and the model name to whatever `model.name` reports on [`/health`](api.md#get-health) (e.g. `z-image-turbo`), and Open WebUI's own chat UI becomes a frontend for this server.

Open WebUI's *Native* (agentic) mode also needs an actual chat model behind the connection to decide when to call the image tool — normally a separate LLM. If you'd rather not run one just for that, point Open WebUI's chat connection at mfluxible's own `POST /v1/chat/completions` (same base URL) too — see [that endpoint](api.md#post-v1chatcompletions) for what it does and, importantly, doesn't do.

### Stable Diffusion (sdcpp) compatible frontends

mfluxible also provides a **stable-diffusion.cpp** image generation endpoint that can be used with compatible clients (with some caveats).

#### SillyTavern

[SillyTavern](https://github.com/SillyTavern/SillyTavern)'s image-generation extension has no OpenAI-compatible source — its `openai` one hardcodes `api.openai.com` inside SillyTavern's own backend, with no base-URL setting (unlike the "Custom (OpenAI-compatible)" source on its chat side). So the `/v1` endpoints above are no use here. Instead, this server answers the three calls its **stable-diffusion.cpp server** source makes, which is the one local-URL source whose surface is small enough to be worth implementing: a reachability probe on [`OPTIONS /v1/images/generations`](api.md#options-v1imagesgenerations), a model list from [`GET /v1/models`](api.md#get-v1models), and generation on [`POST /sdapi/v1/txt2img`](api.md#post-sdapiv1txt2img).

Setup, under Extensions → Image Generation:

1. **Source** → `stable-diffusion.cpp server`.
2. **stable-diffusion.cpp URL** → `http://127.0.0.1:8420` — the base URL only, no path. SillyTavern's server is what fetches this, not your browser, so it has to be reachable from wherever SillyTavern is running (and CORS doesn't enter into it).
3. Click **Validate**. The model dropdown then fills from `/v1/models` with the one model this process has loaded — select it, so SillyTavern stops sending whatever name it had stored before.
4. Set **Sampling steps** and **CFG scale** to suit that model. SillyTavern's defaults are 20 and 7; check `default_steps` and `default_guidance` on [`/health`](api.md#get-health) for what the loaded model actually wants (Z-Image-Turbo: 9 steps, no guidance at all).

Step 4 matters more than it looks. SillyTavern sends both values on every request, from sliders it always shows, and this server **honours `steps` and drops `cfg_scale` where the model can't use it** rather than rejecting either — so a mismatch is slow or ignored, never an error you'd see. [The API reference](api.md#why-this-endpoint-drops-what-the-native-api-rejects) has the full reasoning; the short version is that SillyTavern replaces any upstream error with a bare `500` and no body, so a 400 explaining itself would reach you as an unexplained failed generation.

**It also can't authenticate, under either scheme.** SillyTavern's `sdcpp` source sends no credentials on any of its three calls — unlike its AUTOMATIC1111 source, which has a Basic-auth field — so it 401s against the server's own [HTTP Basic](server.md#authentication) and against a bearer-token proxy alike, with no setting on either side to work around it. Credentials in the URL don't help either: Node's `fetch` rejects a URL containing them. If you need this integration, either keep the server unauthenticated on a trusted network, or accept the trade and open its three paths (`OPTIONS /v1/images/generations`, `GET /v1/models`, `POST /sdapi/v1/txt2img`) in your own proxy config — the last of those generates, so opening it means anyone who can reach the proxy can spend GPU time.

Two consequences worth knowing:

- **The sampler and scheduler dropdowns do nothing here.** They're populated from a hardcoded stable-diffusion.cpp list, and mflux picks its own sampler; the values are accepted and ignored.
- **Anything else that needs a real A1111 server is out of scope** — img2img, upscaling, and the progress bar all use endpoints this shim doesn't implement. Text-to-image is what works.
