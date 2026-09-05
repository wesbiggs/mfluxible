# Testing

```bash
uv pip install -r requirements-dev.txt
uv run pytest
```

The suite runs entirely against a fake, weight-free model (`tests/doubles/toy_model.py`) that always renders a solid color instead of doing real diffusion — no download, no GPU work, and it's fast enough for every push. It's wired in by passing a `ModelSpec` instance straight to `MfluxEngine(model=...)`, which skips `models.py`'s registry entirely (see `MfluxEngine.__init__` in `server/engine.py`), so no production code has to know it exists.

Runs on GitHub Actions on every push/PR (`.github/workflows/tests.yml`). Since `mlx` (mflux's own dependency) has no Linux or Intel build, that workflow — and any other CI you point at this repo — has to run on an Apple Silicon macOS runner (`macos-14` on GitHub-hosted); `ubuntu-latest` will fail to install.
