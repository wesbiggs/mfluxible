"""`mfluxible-server` -- start the API without spelling out a uvicorn command line.

This is a convenience, not a layer: it resolves a host and a port, reports what it is
about to do, and hands the same `mfluxible.server:app` to uvicorn that the documented
`uvicorn mfluxible.server:app` invocation does. Anything this cannot express is a reason
to run uvicorn directly, and that stays supported -- the config file is applied on
package import (see __init__.py) precisely so that both routes behave identically.

One thing here is load-bearing rather than cosmetic. `auth.resolve_bind_host()` decides
whether to warn about an unauthenticated server on a reachable address, and it has to
guess, because the app is handed to uvicorn rather than starting it. Going through this
entry point there is nothing to guess: UVICORN_HOST is set below to the host actually
being bound, which is the second step of that function's own documented precedence. So
a host that came from a config file -- invisible to an argv scan -- still produces the
right warning, and the "best-effort" caveat stops applying to this path.
"""

from __future__ import annotations

import argparse
import os
import sys

from mfluxible import CONFIG_PATH, __version__

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfluxible-server",
        description="Run the mfluxible image-generation API.",
        epilog=(
            "Every setting is an environment variable or a key in mfluxible.toml; "
            "see the Configuration section of docs/server.md. The flags below are the "
            "two that decide where to listen, plus the file to read the rest from."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MFLUXIBLE_HOST") or DEFAULT_HOST,
        help=f"address to bind (default: {DEFAULT_HOST}; anything else is reachable from "
        f"the network -- see docs/server.md on authentication)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MFLUXIBLE_PORT") or DEFAULT_PORT),
        help=f"port to bind (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to read instead of the discovered one. Declared here so it "
        "appears in --help, but it is read straight out of argv at import time -- see "
        "config.config_flag() for why it cannot wait for this parser.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart on source changes (development only; reloads reload the model too)",
    )
    parser.add_argument("--version", action="version", version=f"mfluxible {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # See the module docstring: this is what makes the startup warning exact rather
    # than a guess, including when --host was never typed.
    os.environ["UVICORN_HOST"] = args.host

    # stderr, not stdout, and not /health: which file configured this process is
    # useful to whoever started it and is a filesystem path, which is the one thing
    # /health is careful not to disclose (see CLAUDE.md, "Two auth schemes").
    if CONFIG_PATH is not None:
        print(f"mfluxible {__version__}: configuration from {CONFIG_PATH}", file=sys.stderr)
    else:
        print(f"mfluxible {__version__}: no config file found, using the environment", file=sys.stderr)

    # Imported here rather than at module scope so that --help and --version answer
    # instantly instead of after mflux, MLX and Torch have loaded.
    import uvicorn

    uvicorn.run("mfluxible.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
