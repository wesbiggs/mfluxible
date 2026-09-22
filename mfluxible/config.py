"""An optional TOML config file, as an alternative to setting environment variables.

**This adds no settings and no second set of names.** A key here *is* an environment
variable with `MFLUXIBLE_` dropped and the case lowered -- `model` is `MFLUXIBLE_MODEL`,
`mlx_cache_limit_mb` is `MFLUXIBLE_MLX_CACHE_LIMIT_MB` -- and the file is applied by
putting those variables into `os.environ` before anything reads them. So there is one
table of settings to document (docs/server.md), one place each value is parsed (the
module that owns it), and a config file can never drift from what the environment does,
because by the time any of this package's code sees it, it *is* the environment.

That mapping is mechanical rather than a lookup table on purpose: a table would be a
second copy of the variable names, and the failure it invites -- a setting that exists
in the environment but silently isn't wired up in the file -- is exactly the kind of
silent drop this codebase rejects elsewhere. `KNOWN` below is the one list, and
test_config.py checks it against the variables the source actually reads.

Three decisions worth stating:

**The real environment wins.** A value already in `os.environ` is left alone. A
deployment sets variables in a launchd plist or a systemd unit, and a stale file in
someone's home directory silently overriding that is a worse failure than the file
appearing not to work -- the second is visible the moment you look at /health, the
first is invisible until something behaves oddly. It also makes
`MFLUXIBLE_MODEL=flux-dev mfluxible-server` work as the one-off override it looks like.

**An unknown key is an error, not a shrug.** `mdoel = "flux-dev"` in a file that is
ignored key-by-key starts a server on the wrong model and reports success. There is no
forward-compatibility cost worth that: this file and this package ship as one version.

**Values are passed through verbatim, not interpreted.** `~` is not expanded here, and
`quantize = 8` becomes the string `"8"` -- because `~` expansion and int parsing already
live in the modules that own those settings (engine.py, server.py), and doing either
here would mean a config file could behave differently from the environment variable it
stands in for. The only conversions are the ones TOML forces: a bool becomes
`"true"`/`"false"`, and a list becomes the comma-separated string the env var already
takes, so `lora_paths = ["a.safetensors", "b.safetensors"]` reads naturally.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

# Every MFLUXIBLE_* variable this distribution reads, minus the prefix. Adding a
# setting means adding it here; test_config.py fails if the two get out of step, by
# scanning the package source for the variables it actually reads rather than trusting
# this list to have been maintained.
#
# MFLUXIBLE_CONFIG is deliberately absent: a config file naming the config file to load
# is circular, and the answer ("which one wins?") has no good version.
KNOWN = frozenset(
    {
        # server
        "model",
        "quantize",
        "model_dir",
        "lora_paths",
        "lora_scales",
        "mlx_cache_limit_mb",
        "mlx_wired_limit_mb",
        "basic_auth_username",
        "basic_auth_password",
        "cors_origin_regex",
        "cors_origins",
        "vlm_dir",
        "vlm_timeout",
        "vlm_backend",
        "vlm_load_timeout",
        "vlm_local_model",
        # how `mfluxible-server` binds, when no flag says otherwise
        "host",
        "port",
        # vlm_worker (the detection sidecar -- same file, since it is co-configured
        # with the server and runs on the same machine; see CLAUDE.md)
        "vlm_command",
        "vlm_model",
        "server_url",
        "bearer_token",
    }
)

# Where a file is looked for when MFLUXIBLE_CONFIG doesn't name one. First hit wins;
# they are not merged, because a value that could come from either of two files is a
# value you have to go looking for.
#
# The two names differ on purpose. In a working directory the file has to say whose it
# is, so it is `mfluxible.toml`; inside a directory already called `mfluxible` that
# prefix is noise, and `config.toml` is the shape the XDG convention uses everywhere.
LOCAL_NAME = "mfluxible.toml"
USER_NAME = "config.toml"


def _xdg_home(variable: str, default: str, environ: dict | None = None) -> Path:
    """An XDG base directory: what `variable` names, or the spec's default.

    The absolute-path check is the spec's, not defensiveness: it says a relative path in
    one of these variables is invalid and must be ignored. Honouring that matters most
    for the one people set by hand -- an `XDG_CACHE_HOME=cache` typed once would
    otherwise scatter a directory of multi-gigabyte weights into whatever directory the
    server happened to be started from, a different one each time.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(variable, "").strip()
    if raw and Path(raw).is_absolute():
        return Path(raw)
    return Path(default).expanduser()


def cache_home(environ: dict | None = None) -> Path:
    """The XDG cache directory -- what `~/.cache` means on this machine.

    Exists because mfluxible is not the only thing caching weights for a generation:
    huggingface_hub holds the raw download and reads XDG_CACHE_HOME itself
    (`huggingface_hub.constants`, checked rather than assumed). Both caches hold
    multi-gigabyte copies of the same model, so a machine that relocates one and not the
    other has moved half of what the person was trying to move -- and they were trying
    because they were out of disk, which is when a surprise is least welcome.
    """
    return _xdg_home("XDG_CACHE_HOME", "~/.cache", environ)


def user_config_path(environ: dict | None = None) -> Path:
    """The machine-wide config file's path, wherever XDG says it lives."""
    return _xdg_home("XDG_CONFIG_HOME", "~/.config", environ) / "mfluxible" / USER_NAME


# What MFLUXIBLE_CONFIG is set to in order to turn discovery off entirely -- for a
# deployment that configures everything through the environment and does not want a
# file in the working directory to have any say.
DISABLED = "none"


class ConfigError(Exception):
    """A config file that exists but cannot be honoured.

    Raised, unlike most failure signalling in this package, because there is no
    response to put a reason in: this happens at import, before a server exists. It
    reaches whoever started the process, on stderr, which is the audience for it --
    the same reasoning as engine.py's logged tracebacks, arrived at from the other end.
    """


def config_flag(argv: list[str] | None = None) -> str | None:
    """The value of a `--config` flag on the command line, or None.

    Read out of argv rather than from the parsed arguments because of an ordering
    problem there is no way around: importing this package applies the config file (see
    __init__.py), and that necessarily happens before `cli.main()` exists to parse
    anything. By the time argparse could tell us about `--config`, the file it names
    would already have lost to whatever discovery found first.

    auth.resolve_bind_host() reads `--host` out of argv for the same reason, and this
    is the same trade: it is worth knowing that the one case where this reads a flag
    that isn't ours is a program with its own `--config` that imports `mfluxible`.
    `MFLUXIBLE_CONFIG=none` turns the whole mechanism off for anyone in that position.
    """
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def discover(environ: dict | None = None, cwd: Path | None = None, argv: list[str] | None = None) -> Path | None:
    """The config file to load, or None if there is nothing to load.

    Order: `--config`, then MFLUXIBLE_CONFIG, then ./mfluxible.toml, then
    $XDG_CONFIG_HOME/mfluxible/config.toml (i.e. ~/.config/mfluxible/config.toml unless
    that variable says otherwise). First hit wins and they are never merged.

    A path named explicitly -- by either the flag or the variable -- that does not exist
    is an error, while a missing file at either default location is not. Naming a file
    is a statement that it is there; finding nothing where a file is merely *allowed* to
    be is the ordinary case for anyone configuring this through the environment.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd

    flag = config_flag(argv)
    named = flag if flag is not None else environ.get("MFLUXIBLE_CONFIG", "").strip()
    if named.lower() == DISABLED:
        return None
    if named:
        path = Path(named).expanduser()
        if not path.is_file():
            source = "--config" if flag is not None else "MFLUXIBLE_CONFIG"
            raise ConfigError(f"{source} names {path}, which is not a file")
        return path

    local = cwd / LOCAL_NAME
    if local.is_file():
        return local

    user = user_config_path(environ)
    if user.is_file():
        return user

    return None


def _as_env_value(key: str, value: object) -> str:
    """One TOML value as the string the matching environment variable would hold."""
    if isinstance(value, bool):
        # Before int: bool is a subclass of it, and "True" is not a value anything
        # here parses.
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise ConfigError(f"{key} is a list containing {type(item).__name__}; it may only hold strings or numbers")
            parts.append(str(item))
        return ",".join(parts)
    raise ConfigError(f"{key} is a {type(value).__name__}; it must be a string, number, boolean or list")


def to_env(data: dict) -> dict[str, str]:
    """A parsed config file as the environment variables it stands for.

    The `[env]` table is the escape hatch, and it is the only part of the file that is
    not about this package's own settings: keys there are passed through under their
    exact names, so a variable that belongs to something else -- `HF_TOKEN`, which
    huggingface_hub reads and gated models need -- can live in the same file instead of
    being the one thing you still have to export by hand.
    """
    env: dict[str, str] = {}

    raw = data.get("env", {})
    if not isinstance(raw, dict):
        raise ConfigError("[env] must be a table of NAME = \"value\" entries")
    for name, value in raw.items():
        env[name] = _as_env_value(f"env.{name}", value)

    for key, value in data.items():
        if key == "env":
            continue
        if isinstance(value, dict):
            raise ConfigError(
                f"[{key}] is a table; this file is flat apart from [env]. "
                f"Write `{key} = ...` at the top level, or move it into [env] if it is "
                f"not an mfluxible setting."
            )
        if key not in KNOWN:
            suggestion = _nearest(key)
            hint = f" Did you mean `{suggestion}`?" if suggestion else ""
            raise ConfigError(f"`{key}` is not an mfluxible setting.{hint}")
        env[f"MFLUXIBLE_{key.upper()}"] = _as_env_value(key, value)

    return env


def _nearest(key: str) -> str | None:
    """The known key a typo most likely meant, or None if nothing is close."""
    import difflib

    matches = difflib.get_close_matches(key, sorted(KNOWN), n=1, cutoff=0.7)
    return matches[0] if matches else None


def load(path: Path) -> dict[str, str]:
    """One config file as environment variables. Does not touch os.environ."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        # strerror rather than the exception: it names the file we already know about
        # and nothing else, where str(exc) on some platforms carries more.
        raise ConfigError(f"could not read {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        # This one *is* worth quoting: it is a parse error in a file the person
        # reading the message wrote, and it names a line in it.
        raise ConfigError(f"{path} is not valid TOML: {exc}") from None
    return load_data(data, path)


def load_data(data: dict, path: Path | None = None) -> dict[str, str]:
    """Parsed TOML as environment variables, with the file named in any error."""
    try:
        return to_env(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}" if path is not None else str(exc)) from None


def apply(environ: dict | None = None, cwd: Path | None = None, argv: list[str] | None = None) -> Path | None:
    """Put the discovered config file into the environment. Returns the path used.

    Only sets variables that are not already set -- see the module docstring on why the
    real environment wins.
    """
    environ = os.environ if environ is None else environ

    path = discover(environ, cwd, argv)
    if path is None:
        return None

    for name, value in load(path).items():
        environ.setdefault(name, value)
    return path
