# mfluxible

A minimal streaming HTTP API for image generation on Apple Silicon, built on [mflux](https://github.com/filipstrand/mflux). Runs Z-Image-Turbo (the default), FLUX.1-schnell, FLUX.1-dev, or Qwen-Image — one model per server process, picked at startup (see [Models](docs/server.md#models)).

Rather than a full node-graph tool (ComfyUI) or a proprietary format (Draw Things), this exposes a small API in the same spirit as a chat-completions endpoint: each denoising step streams as a "thinking" event while generation happens, with optional in-progress preview images (Draw Things-style), followed by the final image.

## Layout

```
server/   the model + HTTP API (FastAPI)
clients/   everything that talks to it: terminal scripts, a browser harness, an MCP tool for Claude
```

Nothing in `clients/` needs `server/`'s dependencies (mflux, PyTorch, etc.) or vice versa — install only what you need for what you're doing.

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — `brew install uv`, or see its [installation docs](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv --python 3.11
uv pip install -r server/requirements.txt

uv run uvicorn server:app --app-dir server --host 127.0.0.1 --port 8420
```

`uv venv` creates `.venv/` in the repo root; `uv pip install` and `uv run` both find it there, so nothing needs activating. There's deliberately no `pyproject.toml` — each half of the repo keeps its own `requirements.txt` (see [Layout](#layout)) and you install only the one you need, which is why this is `uv pip`/`uv run` rather than `uv sync`.

The model loads on startup, before the server accepts any requests. On first run this downloads its weights from Hugging Face — expect a sizable one-time download — then quantizes them and caches the quantized copy (see [Model cache](docs/server.md#model-cache)); both only happen once.

The default model is Z-Image-Turbo ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)). To run something else, set `MFLUXIBLE_MODEL` — `flux-schnell`, `flux-dev`, or `qwen-image` — before starting the server; only the model you select is ever downloaded. See [Models](docs/server.md#models) for what differs between them.

Once it's running:

```bash
curl -N -X POST http://127.0.0.1:8420/mfluxible/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a puffin on a cliff at sunset", "preview_every": 2}'
```

or use one of the [clients](docs/clients.md) for something more visual.

## Documentation

- **[Clients](docs/clients.md)** — the bundled terminal scripts, the browser harness, and pointing an OpenAI-compatible frontend at the server.
- **[MCP tool](docs/mcp.md)** — generating images from within Claude Code or Claude Desktop: what the tool does, how to register it, and its own environment variables.
- **[API](docs/api.md)** — every endpoint: the native streaming endpoint and its SSE event schema, image-to-image and fractional start, and the OpenAI-compatible `/v1` endpoints.
- **[Server](docs/server.md)** — running it: how a synchronous mflux call is streamed out of an async server, environment variables (memory, model cache, LoRAs, CORS), the models it can run, binding to the network, and troubleshooting.
- **[Testing](docs/testing.md)** — the weight-free test suite and CI.

## License

[Apache 2.0](LICENSE).
