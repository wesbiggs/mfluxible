# Clients

Both live in `clients/` and talk to the server over HTTP — neither loads the model itself, so the server from the [Quickstart](../README.md#quickstart) must already be running. The [MCP tool](mcp.md) is a third client in the same vein, on its own page.

## Terminal

`stream_client.py` (`uv pip install -r clients/requirements.txt` first — just `requests`) or the dependency-free `stream_client.js` (Node 18+) render step progress and previews inline in the terminal as they stream, saving the final image to disk:

```bash
uv run clients/stream_client.py "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
# or
node clients/stream_client.js "a puffin on a cliff at sunset" --preview-every 2 --out puffin.png
```

Both take `--steps`, `--seed`, `--guidance` and `--negative-prompt`, and leave all four to the server when you don't pass them — so `--steps` is only worth setting to override the loaded model's own default. `--guidance` and `--negative-prompt` are refused (with a message naming the model) on models that can't act on them; see [Models](server.md#models).

Both also take `--image PATH` for image-to-image (read from disk and base64-encoded, not a URL) and `--image-strength` (0.0–1.0, only valid alongside `--image`; the server's own default, 0.4, applies if you omit it) — see [Image-to-image](api.md#image-to-image) for what `image_strength` actually controls (mflux's convention is the inverse of some other tools'). `--fractional-start` (also only valid alongside `--image`) makes that strength continuous instead of quantized to `1/steps`, at no extra cost — see [Fractional start](api.md#fractional-start).

Both render the exact full-resolution bytes returned by the server — no downscaling, no recompression, nothing client-side touches the image data. They use the iTerm2 inline-image protocol's chunked `MultipartFile` variant (also works in WezTerm; in an unsupported terminal the escape codes are just ignored, and the saved file and progress text still work either way), the same variant iTerm2's own [`imgcat`](https://github.com/gnachman/iTerm2-shell-integration/blob/master/utilities/imgcat) reference tool uses by default: the base64 payload is split into `FilePart=` sequences behind a metadata-only header and a `FileEnd` marker, rather than one giant `File=...:<base64>` sequence.

This matters because iTerm2's own source caps how much data it'll accumulate for a *single* OSC escape sequence at 1,048,576 bytes ([`VT100XtermParser.m`](https://github.com/gnachman/iTerm2/blob/master/sources/VT100/VT100XtermParser.m)) — past that it truncates rather than cleanly dropping the sequence, which can corrupt what renders afterward too, not just fail to show the one image. Diffusion output is detailed/photographic content that a full-resolution PNG can realistically approach or cross that limit for. Chunking (500,000 bytes/chunk here — `imgcat`'s own 200-byte default exists specifically to survive tmux, which doesn't apply since neither script wraps for tmux) means no image, at any size or detail level, can hit that cap.

Images render at `width=auto` (height defaults to auto too) — the same default `imgcat` uses: native pixel dimensions divided by the display's backing scale factor (e.g. a 1024px image renders at 512pt on a 2x/Retina display), rather than a fixed cell-count width that would scale with the terminal's font size instead of the image's actual dimensions.

Not tmux-aware — iTerm2's protocol needs extra passthrough wrapping inside tmux that these scripts don't do.

## Browser

`clients/harness.html` is a small, dependency-free page (plain HTML/CSS/JS, no build step) with a form for prompt/width/height/steps/seed/preview_every that calls the streaming endpoint directly from the browser via `fetch`, reads [`/health`](api.md#get-health) on load to show which model the server is running (leaving Steps blank uses that model's default, and Guidance / Negative prompt appear only if it accepts them), parsing the SSE stream the same way the terminal clients do, and renders previews and the final image as `<img>` elements (via `data:` URLs) plus a download link for the final PNG.

It also does image-to-image. Get a base image onto the page either by dragging an image file onto it from anywhere (drop targets are the whole page, not just the drop-zone box — that's just where the visual highlight and preview show up) or by clicking the drop zone to pick a file; or click **Use last result** to feed the most recently generated image straight back in as the next input, for chaining edits without a round trip through disk. **Clear image** drops back to plain text-to-image. Loading a base image any of those ways also sets Width and Height to the image's own dimensions, rounded down to a multiple of 16 — mflux resizes the input to whatever `width`/`height` the request carries with a plain `resize()` and no aspect-ratio handling, so a portrait image against the 1024×1024 default would be stretched, not letterboxed, and that distorted image is what the generation is seeded from. 16 rather than 8 because that's what the server itself does with the number (mflux floors both dimensions to a multiple of 16), so the box says the size you'll actually get; the fields step by 16 to match. The Image strength field controls how strongly that input constrains the output (default 0.4) — see [Image-to-image](api.md#image-to-image) for what the number actually means; the page's own hint text is a reminder that it's the inverse of some other tools' "denoising strength". The **Fractional start** checkbox next to it is [the same flag](api.md#fractional-start) the API takes, and is sent only while an image is loaded. If the chosen image is a PNG this server generated, its embedded prompt (read straight out of the PNG's `eXIf` metadata, client-side, no server round trip) is loaded into the Prompt box automatically — a photo with no such metadata just leaves the box alone.

The server itself serves this page, at `GET /harness.html` — just open `http://localhost:8420/harness.html` (or whichever host/port `server.py` is bound to) once it's up. The Server URL field defaults to the relative path `/mfluxible/v1/images/generations`, which resolves against whatever origin served the page, so no configuration is needed for this same-origin case.

If you'd rather host the page separately (e.g. to point one harness at multiple servers, or to exercise the CORS path), it still works opened from any static file server — just not as a `file://` URL, since the browser's `Origin` header for a local file is `null`, which the server's default CORS config won't match:

```bash
cd clients && python3 -m http.server 8000
# then open http://localhost:8000/harness.html and point Server URL at the API host
```

(CORS is on by default and reflects back any `http(s)://localhost:<any port>` or `127.0.0.1:<any port>` origin, so this works with no server-side configuration — see [CORS](server.md#cors) if you need something different.)

## OpenAI-compatible frontends

Neither of those, nor the [MCP tool](mcp.md), is this — they're the bundled clients, and they all speak mfluxible's own native API. But `POST /v1/images/generations` and `POST /v1/images/edits` (see the [API reference](api.md)) are genuine [OpenAI Images API](https://platform.openai.com/docs/api-reference/images/create)-compatible endpoints, so any tool built against that API can point at this server directly, with no code changes on its side. For example, [Open WebUI](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/openai/)'s Settings → Admin → Images panel takes an arbitrary `IMAGES_OPENAI_API_BASE_URL` and a free-text model name — set the base URL to `http://127.0.0.1:8420/v1` and the model name to whatever `model.name` reports on [`/health`](api.md#get-health) (e.g. `z-image-turbo`), and Open WebUI's own chat UI becomes a frontend for this server.

Open WebUI's *Native* (agentic) mode also needs an actual chat model behind the connection to decide when to call the image tool — normally a separate LLM. If you'd rather not run one just for that, point Open WebUI's chat connection at mfluxible's own `POST /v1/chat/completions` (same base URL) too — see [that endpoint](api.md#post-v1chatcompletions) for what it does and, importantly, doesn't do.
