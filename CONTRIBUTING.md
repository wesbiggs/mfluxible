# Contributing

Contributions to this project are welcome! This file is a guide for developers wishing to extend or add features to mfluxible.

## Development

No additional setup beyond what is documented in [README.md](README.md) is needed.

### Adding a model

Usually a new entry in `server/models.py` and nothing else: six of mflux's variant classes (`ZImage`, `Flux1`, `QwenImage`, `Krea2`, `ErnieImage`, `Flux2Klein`) share a constructor signature, a `generate_image()` signature and a `save_model()`, which is what keeps `server/engine.py` free of per-model branching.

A `ModelSpec` is inert data — its mflux imports are deferred into a `load()` function that only runs at startup — so naming a model costs nothing until someone selects it. What the entry has to get right is every field where the model differs from the others, and those are worth reading out of the installed mflux source rather than inferred from a README:

- **`default_steps`** — mflux's own `MODEL_INFERENCE_STEPS` (in `mflux/cli/defaults/defaults.py`), keyed by the *canonical* registry name. Copy the value rather than importing it; that table lives under `mflux.cli`, which is CLI-internal.
- **`supports_guidance` / `default_guidance`** — per checkpoint, not per family. Check the model's own `mflux-generate-*` CLI: several hard-error on any guidance but 1.0, which is the tell for a distilled checkpoint whose value has nowhere to go.
- **`supports_negative_prompt`** — whether the variant's prompt encoder builds an unconditional branch at all. Note this is necessary but not sufficient; see `CFG_GUIDANCE_FLOOR` in `server/models.py`.
- **`default_scheduler`** — which scheduler the variant picks when `generate_image()` isn't told one. This gates `fractional_start`, which only makes sense on the linear schedule `server/schedulers.py` extends.

Then verify the whole table still resolves before downloading anything:

```bash
uv run python -c "import sys; sys.path.insert(0, 'server'); from models import MODELS; [print(s.key, s.load()[0].__name__) for s in MODELS]"
```

That imports every loader and every `ModelConfig` without fetching a single weight, so a typo in a config name or an import path fails in a second rather than after a multi-gigabyte download. Follow it with one real generation against the new model — the test suite cannot cover that (see [Testing](#testing) below).

#### Models that need more than a row

mflux's shared variant surface is not universal, and the rest of its text-to-image models each break it somewhere:

| Model | What's missing |
|---|---|
| `FIBO`, `BooguImage`, `LensImage` | no `lora_paths`/`lora_scales` in the constructor — `_load_sync`'s fresh-load branch raises `TypeError` before any weight loads, even with no LoRA configured |
| `Ideogram4` | no `image_path`/`image_strength`/`scheduler` — image-to-image would become a `TypeError` mid-generation |
| `BooguImage` | no latent creator at all, so step previews are impossible (mflux's own CLI passes `latent_creator=None` and says stepwise output is unsupported) |
| `LensImage` | no `save_model()`, so the quantized-weight cache has nothing to write and every startup re-quantizes |

Each of those needs a capability flag in `models.py` **and** `engine.py` honouring it — `supports_lora`, `supports_img2img`, `supports_previews` alongside the existing pair, routed through `request_problem` so an unusable field is a 400 before `StreamingResponse` starts rather than an exception thrown into a half-sent body. Adding the row alone fails at load or mid-stream, not gracefully.

## Testing

```bash
uv pip install -r requirements-dev.txt
uv run pytest
```

The suite runs entirely against a fake, weight-free model (`tests/doubles/toy_model.py`) that always renders a solid color instead of doing real diffusion — no download, no GPU work, and it's fast enough for every push. It's wired in by passing a `ModelSpec` instance straight to `MfluxEngine(model=...)`, which skips `models.py`'s registry entirely (see `MfluxEngine.__init__` in `server/engine.py`), so no production code has to know it exists.

Runs on GitHub Actions on every push/PR (`.github/workflows/tests.yml`). Since `mlx` (mflux's own dependency) has no Linux or Intel build, that workflow — and any other CI you point at this repo — has to run on an Apple Silicon macOS runner (`macos-14` on GitHub-hosted); `ubuntu-latest` will fail to install.

Two things in the suite aren't about the model at all. `tests/test_docs.py` checks that the sample responses in `docs/` still match what the code emits, and `tests/test_proxy_config.py` checks `Caddyfile.example` against the app's own route table — that every path it serves without a token is one the server actually has, and that no endpoint accepting a `POST` has landed in that open list. Both exist for the same reason: a wrong answer there is silent, and nothing else would notice.

What CI can't cover is the clients' own auth handling. `requirements-dev.txt` pulls in the server's dependencies and pytest, not `requests` or `mcp` — that separation is deliberate (see the dependency note in `CLAUDE.md`), so `clients/stream_client.py` and `clients/mcp_server.py` aren't importable on the CI runner and nothing there exercises what header they put on the wire. Changing how a client authenticates means checking it by hand against a real proxy; recording the `Authorization` header at a stub upstream is enough, and catches the failure mode that matters (a credential parsed and then silently dropped).

The corollary of a weight-free suite is what it can't tell you: it exercises the request validation, the SSE plumbing, the step-callback timing and the preview/final-image encoding, but never that a given mflux model actually produces a good image. Anything that touches a real checkpoint — a new model, a change to the preview decode path, a mflux upgrade — needs at least one live generation against the running server before you trust it.


## Documentation

Always update `docs/api.md` when changing the API (endpoints, request/response shapes, SSE event schema).

`README.md` is an overview, a quickstart and an index into `docs/`. The prose there is split by use case: `server.md` (running the server), `clients.md` (the terminal scripts, the browser harness, OpenAI-compatible frontends), `mcp.md` (the MCP tool) and `api.md`. So a server environment variable is documented in `server.md` and an MCP one in `mcp.md`, even though both are "configuration". Anything aimed at someone changing the code, rather than running it, belongs in this file.
