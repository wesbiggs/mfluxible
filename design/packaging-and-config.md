# Packaging, versioning and the config file

Why the repo is packaged the way it is, how releases work, and how the TOML config file maps onto environment variables. Read before touching `pyproject.toml`, `config.py`, a release workflow, or adding a setting.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## uv, but with pyproject.toml as a packaging manifest -- still not a uv project

Setup is `uv venv --python 3.11`, `uv pip install -e .` (or `-r requirements-dev.txt`),
and `uv run <cmd>`; CI skips the venv entirely with `uv pip install --system`
(`.github/workflows/tests.yml`). There are now two `pyproject.toml` files -- the root one
and `clients/` -- and **both set `[tool.uv] managed = false`, which is the load-bearing
line.** Without it, the mere existence of a `pyproject.toml` switches `uv run` into
project mode: it writes a `uv.lock` and installs the project, mflux and PyTorch behind
it, into `.venv`. That would break the thing the Layout note protects -- a machine
running only `clients/stream_client.py` needs nothing but `requests`, and
`uv run clients/stream_client.py` there must not start resolving a server's dependency
tree. `managed = false` keeps uv's prior behaviour byte for byte (`uv run` falls back to
`./.venv`, no lock is written, nothing is synced) while leaving `uv build` and
`uv pip install -e .` working normally. Verified against uv 0.12.13, not assumed, and
pinned by `test_uv_is_told_this_is_not_a_project`.

There is still **no `uv.lock` and no `uv sync`**, and the dependency sets are still
separate -- they just live in different files now: the root `pyproject.toml`
(`dependencies`), `clients/pyproject.toml` (the MCP tool), `clients/requirements.txt`
(the terminal client) and `requirements-dev.txt` (a superset of all of them, for the test
suite only). `server/requirements.txt` and `clients/requirements-mcp.txt` are gone; each
distribution declares its own. A client-only or MCP-only install still never drags in
mflux/PyTorch, which is the property that mattered, not the file format.

`uv run` works fine outside a project: it falls back to `./.venv`, so `uv run pytest` and
`uv run clients/stream_client.py ...` pick up whatever was installed there without an
`activate` step. `docs/mcp.md` no longer hardcodes `/path/to/mfluxible/.venv/bin/python`
-- an MCP host is pointed at `uvx mfluxible-mcp` instead, which is the whole reason that
second distribution exists.

## Versioning and releases

`__version__` lives in `mfluxible/__init__.py` and `clients/mfluxible_mcp/__init__.py`;
each `pyproject.toml` reads its own via `[tool.hatch.version]`, so those two literals are
the only copies. The two distributions are **released together off one tag and always
share a version** -- pinned by `test_the_two_distributions_ship_the_same_version`,
because "which versions of these two am I running" should have one answer.
`.github/workflows/release.yml` fires on `v*`, refuses to proceed if the tag disagrees
with either `__version__`, and publishes via PyPI Trusted Publishing (no API token in
repo secrets). `/health` reports `version`, and `docs/api.md`'s sample is checked against
it by `test_docs.py` -- a literal version string in a doc is exactly the thing that rots.

The build detail worth not rediscovering: `uv build` builds each wheel **from the sdist**,
so the sdist's `include` list is what decides whether the wheel's `force-include` of
`clients/harness.html` can resolve a second time. Lose that and you get a package that
installs, starts, generates images and 500s on its own front page -- which no test in a
checkout would notice, since a checkout always has the file. Hence
`test_the_wheel_carries_the_harness_the_server_serves` and the CI build job.

## The config file adds no settings, and applies at package import

`mfluxible/config.py` reads an optional TOML file and puts it into `os.environ`. That is
the whole mechanism, and it is what makes it free of drift: a key **is** an environment
variable with `MFLUXIBLE_` dropped and lowercased (`model` -> `MFLUXIBLE_MODEL`), so
there is one set of names, one parser per setting (in the module that owns it), and
nothing for a file to mean differently from an export. Values are passed through
verbatim -- `~` is not expanded and `quantize = 8` becomes `"8"` -- for the same reason;
the only conversions are the ones TOML forces (bool -> `"true"`/`"false"`, list ->
comma-joined, which is what `lora_paths` already took).

**`apply()` runs in `mfluxible/__init__.py`, and it has to.** Every setting here is read
at module import (`MODEL` in runtime.py, `MLX_CACHE_LIMIT_BYTES` in engine.py), so a file
loaded any later would be read after the values it sets had already been decided. Putting
it in the package rather than in `cli.py` is what makes `mfluxible-server` and
`uvicorn mfluxible.server:app` behave identically. The cost is one real piece of
awkwardness: `--config` cannot wait for argparse, because the package is imported before
`main()` exists -- so `config.config_flag()` reads it out of `sys.argv`, the same
technique `auth.resolve_bind_host()` already uses for `--host`, for the same reason.
It also means `python mfluxible/vlm_worker.py` would silently skip the file (running a
file by path never imports the package around it), which its `__main__` guard refuses
rather than allowing.

Three rules, each chosen against a specific silent failure:
- **The real environment wins**, so a stale file in `~/.config` can never overrule a
  launchd plist, and `MFLUXIBLE_MODEL=... mfluxible-server` is the one-off it looks like.
- **An unknown key is an error**, because `mdoel = "flux-dev"` ignored key-by-key starts
  a server on the wrong model and reports success.
- **`config.KNOWN` is checked against the source**, by
  `test_known_lists_exactly_the_variables_the_package_reads` -- it greps the package for
  `environ.get("MFLUXIBLE_...")` and fails on either direction of drift. A new setting
  that is readable from the environment and rejected from the config file is precisely
  the half-wired state that list exists to prevent.

**Both directories this package writes to follow XDG's *variables*, not just its default
paths** (`config.cache_home()`, `config.user_config_path()`). That is one fix rather than
two conventions: `huggingface_hub` reads `XDG_CACHE_HOME` itself -- checked in
`huggingface_hub.constants`, not assumed -- and its cache sits beside
`MFLUXIBLE_MODEL_DIR`'s holding the raw download of the same weights the quantized copy
came from. Honouring the variable in one and not the other means someone moving a cache
off a full disk moves half of it. A relative value is ignored, as the spec requires,
because `XDG_CACHE_HOME=cache` would otherwise put gigabytes wherever the server happened
to be started from. The user-level file is `config.toml` while the working-directory one
is `mfluxible.toml`: in a directory already named `mfluxible` the prefix is noise, and in
a working directory a bare `config.toml` says nothing about whose it is. Making those two
names "consistent" is the tidy-up that would silently stop finding everyone's file, which
is why a test pins both.

The path *is not* on `/health`, deliberately: `Caddyfile.example` leaves that endpoint
open specifically because it discloses no filesystem paths (see "Two auth schemes"), so
it goes to stderr at startup instead. And note the sharp edge documented rather than
worked around: `vlm_command` names an arbitrary program, and `./mfluxible.toml` is
discovered without being asked for, so *where you start the worker* is now part of its
trust boundary. `MFLUXIBLE_CONFIG=none` turns discovery off.
