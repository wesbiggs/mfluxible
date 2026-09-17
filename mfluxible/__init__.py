"""mfluxible -- a streaming HTTP API for image generation on Apple Silicon.

This file is deliberately almost empty, and what little it does have is load-bearing
in a way worth stating up front: **importing this package applies the config file.**

`config.apply()` runs here, before any submodule is imported, because every setting in
this package is read from `os.environ` at module import time (`MODEL` in server.py,
`MLX_CACHE_LIMIT_BYTES` in engine.py, and so on -- see CLAUDE.md on why they are
constants rather than lookups). A config file that loaded any later would be read after
the values it is meant to set had already been decided, which is the kind of failure
that produces a working server running the wrong model.

Putting it here rather than in the console script is what makes the two ways of
starting this server behave identically: `mfluxible-server` and
`uvicorn mfluxible.server:app` both import this package first, so both honour the same
file. A console-script-only hook would have made the documented uvicorn invocation
quietly ignore a config the user could see sitting in their working directory.

The version is defined here rather than in pyproject.toml so there is one copy:
pyproject reads it back out of this file (`[tool.hatch.version]`), and `/health`
reports it.
"""

from mfluxible import config as _config

__version__ = "0.9.1"

# The path actually loaded, or None when no config file was found. Read by cli.py to
# report it on startup. Deliberately *not* on /health: that endpoint is left open by
# Caddyfile.example specifically because it discloses no filesystem paths, and this
# would be one (see CLAUDE.md, "Two auth schemes").
CONFIG_PATH = _config.apply()
