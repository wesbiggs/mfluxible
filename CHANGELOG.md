# Changelog

Notable changes, newest first. Versions follow [semantic versioning](https://semver.org/),
with the usual pre-1.0 caveat: the API surface described in [docs/api.md](docs/api.md) can
still change in a minor release, and any such change is listed here.

The server (`mfluxible`) and the MCP client (`mfluxible-mcp`) are released together from
this repository and always share a version number.

## 0.9.1 — 2026-09-17

Both directories mfluxible keeps files in now follow the XDG Base Directory spec's
variables, not just its default paths.

- **`MFLUXIBLE_MODEL_DIR` defaults under `$XDG_CACHE_HOME`** (still `~/.cache/mfluxible`
  when that is unset). This is the one with consequences: `huggingface_hub` reads
  `XDG_CACHE_HOME` itself, and its cache holds the raw download of the same model whose
  quantized copy mfluxible caches. Pointing that variable at a larger disk previously
  moved one of the two and left the other behind — and anyone setting it is doing so
  because they are short of space. An explicit `MFLUXIBLE_MODEL_DIR` still wins.
- **The machine-wide config file is `$XDG_CONFIG_HOME/mfluxible/config.toml`**, still
  `~/.config/mfluxible/config.toml` by default. Setting `XDG_CONFIG_HOME` used to be
  ignored silently.
- **Renamed** from `~/.config/mfluxible/mfluxible.toml` to `config.toml`. The prefix was
  redundant inside a directory already called `mfluxible`, and `<dir>/config.toml` is the
  conventional shape. `./mfluxible.toml` in a working directory keeps its name, where a
  bare `config.toml` would say nothing about whose it is. **If you created a user-level
  config under 0.9.0, rename it.**
- A relative value in either variable is ignored, as the spec requires — otherwise
  `XDG_CACHE_HOME=cache` would scatter a directory of weights into whatever directory the
  server was started from.

`mfluxible-mcp` has no changes in this release; it shares a version with the server by
design, so one tag publishes both.

## 0.9.0 — 2026-09-17

First versioned release. The functionality below was built up over the project's
unversioned history; this entry records what 0.9.0 contains rather than what changed
within it.

### Packaging

- **Installable from PyPI.** `pip install mfluxible` replaces cloning the repository and
  running `uvicorn` against a checkout. The server modules moved from `server/` into a
  real `mfluxible` package, which is what makes that possible — `models.py`, `auth.py`
  and `server.py` could not have gone into `site-packages` under those names.
- **Two console scripts**: `mfluxible-server` runs the API, `mfluxible-vlm-worker` runs
  the object-detection sidecar. `uvicorn mfluxible.server:app` still works and is still
  the way to reach any uvicorn option the wrapper doesn't expose.
- **`mfluxible-mcp` is a separate distribution**, so installing the MCP tool no longer
  means installing mflux, MLX and PyTorch on a machine that only makes HTTP calls.
  `uvx mfluxible-mcp` replaces pointing an MCP host at an interpreter inside a checkout.
- The browser harness ships inside the wheel, so `GET /` works on an installed server.

### Configuration

- **A TOML config file** is now an alternative to environment variables:
  `./mfluxible.toml`, a file under `~/.config/mfluxible/`, or `--config PATH`. A key is
  its environment variable with `MFLUXIBLE_` dropped and the case lowered, so there is
  one set of names rather than two. Real environment variables win over the file, an
  unknown key is an error rather than a line that quietly does nothing, and `[env]`
  carries variables belonging to other libraries (`HF_TOKEN`). See
  [`mfluxible.example.toml`](mfluxible.example.toml).
- `MFLUXIBLE_HOST` / `MFLUXIBLE_PORT` set where `mfluxible-server` listens when no flag
  says otherwise, and `MFLUXIBLE_SERVER_URL` tells the worker where the server is.
- `GET /health` reports `version`.

### Notes for anyone tracking the repository

- `server/` no longer exists; its contents are `mfluxible/`. Imports within it are
  absolute (`from mfluxible.engine import ...`), and the scheduler paths handed to mflux
  are now `mfluxible.schedulers.*`.
- `clients/mcp_server.py` is now `clients/mfluxible_mcp/server.py`.
- `server/requirements.txt` and `clients/requirements-mcp.txt` are gone: each
  distribution declares its own dependencies in its `pyproject.toml`.
  `clients/requirements.txt` stays, since the terminal client is a script rather than a
  package. The four dependency sets are still separate, and installing any one of them
  still never pulls in the others.
- `uv run` behaves exactly as before. Both `pyproject.toml` files set
  `[tool.uv] managed = false`, which keeps uv out of project mode — no `uv.lock`, no
  implicit sync, and `uv run clients/stream_client.py` on a client-only machine still
  needs nothing but `requests`.
