"""Coverage of mfluxible/config.py, and of the packaging metadata around it.

The loader's functions are driven directly rather than through the import-time
`apply()` in mfluxible/__init__.py: that runs once per process, before any of this is
collected, so there is nothing a test could arrange by the time it gets a turn.
conftest.py sets MFLUXIBLE_CONFIG=none for exactly that reason -- see its docstring.
"""

import pathlib
import re
import tomllib

import pytest

from mfluxible import __version__
from mfluxible.config import (
    KNOWN,
    USER_NAME,
    ConfigError,
    apply,
    cache_home,
    config_flag,
    discover,
    load,
    to_env,
    user_config_path,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "mfluxible"


# -- the mapping ---------------------------------------------------------------------


def test_a_key_is_its_environment_variable_with_the_prefix_dropped():
    assert to_env({"model": "flux-dev"}) == {"MFLUXIBLE_MODEL": "flux-dev"}
    assert to_env({"mlx_cache_limit_mb": 2048}) == {"MFLUXIBLE_MLX_CACHE_LIMIT_MB": "2048"}


def test_a_list_becomes_the_comma_separated_string_the_variable_already_takes():
    # The point of accepting a list at all: MFLUXIBLE_LORA_PATHS is comma-separated, and
    # writing that as a TOML array is the natural thing to reach for.
    env = to_env({"lora_paths": ["/a.safetensors", "/b.safetensors"], "lora_scales": [0.8, 1.0]})
    assert env["MFLUXIBLE_LORA_PATHS"] == "/a.safetensors,/b.safetensors"
    assert env["MFLUXIBLE_LORA_SCALES"] == "0.8,1.0"


def test_a_boolean_is_lowercased_rather_than_pythons_repr():
    # str(True) is "True", which nothing in this package parses. Nothing takes a boolean
    # today; this pins the conversion before something does.
    assert to_env({"env": {"SOME_FLAG": True}}) == {"SOME_FLAG": "true"}
    assert to_env({"env": {"SOME_FLAG": False}}) == {"SOME_FLAG": "false"}


def test_values_are_not_interpreted_here():
    # `~` expansion and int parsing belong to the modules that own those settings, so a
    # config file behaves exactly like the environment variable it stands in for.
    assert to_env({"model_dir": "~/models"})["MFLUXIBLE_MODEL_DIR"] == "~/models"
    assert to_env({"quantize": 8})["MFLUXIBLE_QUANTIZE"] == "8"


def test_the_env_table_passes_names_through_untouched():
    # The escape hatch for variables that aren't ours: HF_TOKEN is read by
    # huggingface_hub and gated models need it, so it should be able to live in the
    # same file rather than being the one thing left to export by hand.
    assert to_env({"env": {"HF_TOKEN": "hf_x"}}) == {"HF_TOKEN": "hf_x"}


def test_an_mfluxible_setting_still_wins_over_the_same_name_written_into_env():
    # Both halves write into one dict, so the order they are applied in is a real
    # decision rather than an accident. The typed key wins: it is the documented way to
    # set it, and [env] is for names this package doesn't own.
    assert to_env({"model": "flux-dev", "env": {"MFLUXIBLE_MODEL": "qwen-image"}}) == {
        "MFLUXIBLE_MODEL": "flux-dev"
    }


# -- refusals ------------------------------------------------------------------------


def test_an_unknown_key_is_an_error_rather_than_ignored():
    # The failure this prevents: a typo'd key silently doing nothing, so the server
    # starts on the wrong model and reports success.
    with pytest.raises(ConfigError) as exc:
        to_env({"mdoel": "flux-dev"})
    assert "mdoel" in str(exc.value)


def test_a_near_miss_is_named_in_the_error():
    with pytest.raises(ConfigError, match="model"):
        to_env({"mdoel": "flux-dev"})


def test_a_table_other_than_env_says_the_file_is_flat():
    with pytest.raises(ConfigError, match="flat"):
        to_env({"server": {"model": "flux-dev"}})


def test_a_value_of_a_type_no_variable_could_hold_is_refused():
    import datetime

    with pytest.raises(ConfigError, match="quantize"):
        to_env({"quantize": datetime.date(2026, 1, 1)})


def test_a_list_of_lists_is_refused_rather_than_flattened():
    with pytest.raises(ConfigError, match="lora_paths"):
        to_env({"lora_paths": [["/a"], ["/b"]]})


def test_invalid_toml_names_the_file_and_the_parse_error(tmp_path):
    bad = tmp_path / "mfluxible.toml"
    bad.write_text('model = "unterminated\n')
    with pytest.raises(ConfigError) as exc:
        load(bad)
    assert str(bad) in str(exc.value)


# -- discovery -----------------------------------------------------------------------


def test_discovery_finds_a_file_in_the_working_directory(tmp_path):
    (tmp_path / "mfluxible.toml").write_text('model = "flux-dev"\n')
    assert discover({}, tmp_path, []) == tmp_path / "mfluxible.toml"


def test_discovery_returns_nothing_when_there_is_nothing(tmp_path):
    # XDG_CONFIG_HOME points the user-level lookup at an empty directory, so a developer
    # who happens to have a real ~/.config/mfluxible/config.toml doesn't see this fail.
    # Steering it through the variable rather than patching the module also means this
    # exercises the same code path a real machine would.
    assert discover({"XDG_CONFIG_HOME": str(tmp_path / "empty")}, tmp_path, []) is None


def test_an_explicit_variable_beats_the_working_directory(tmp_path):
    (tmp_path / "mfluxible.toml").write_text('model = "flux-dev"\n')
    named = tmp_path / "other.toml"
    named.write_text('model = "qwen-image"\n')
    assert discover({"MFLUXIBLE_CONFIG": str(named)}, tmp_path, []) == named


def test_the_flag_beats_the_variable(tmp_path):
    from_var = tmp_path / "var.toml"
    from_var.write_text('model = "flux-dev"\n')
    from_flag = tmp_path / "flag.toml"
    from_flag.write_text('model = "qwen-image"\n')
    argv = ["mfluxible-server", "--config", str(from_flag)]
    assert discover({"MFLUXIBLE_CONFIG": str(from_var)}, tmp_path, argv) == from_flag


def test_naming_a_file_that_is_not_there_is_an_error(tmp_path):
    # Asymmetric with a missing ./mfluxible.toml on purpose: naming a file asserts it
    # exists, whereas the default locations are places a file is merely allowed to be.
    with pytest.raises(ConfigError, match="MFLUXIBLE_CONFIG"):
        discover({"MFLUXIBLE_CONFIG": str(tmp_path / "absent.toml")}, tmp_path, [])
    with pytest.raises(ConfigError, match="--config"):
        discover({}, tmp_path, ["mfluxible-server", "--config", str(tmp_path / "absent.toml")])


def test_none_turns_discovery_off_even_with_a_file_sitting_there(tmp_path):
    (tmp_path / "mfluxible.toml").write_text('model = "flux-dev"\n')
    assert discover({"MFLUXIBLE_CONFIG": "none"}, tmp_path, []) is None


def test_the_flag_is_read_in_both_spellings():
    assert config_flag(["prog", "--config", "/a.toml"]) == "/a.toml"
    assert config_flag(["prog", "--config=/b.toml"]) == "/b.toml"
    assert config_flag(["prog", "--host", "0.0.0.0"]) is None


# -- XDG base directories -------------------------------------------------------------


def test_the_user_level_file_follows_xdg_config_home(tmp_path):
    cfg = tmp_path / "elsewhere" / "mfluxible"
    cfg.mkdir(parents=True)
    (cfg / USER_NAME).write_text('model = "flux-dev"\n')

    # cwd is empty, so this can only be found via the variable.
    found = discover({"XDG_CONFIG_HOME": str(tmp_path / "elsewhere")}, tmp_path, [])
    assert found == cfg / USER_NAME


def test_without_the_variable_the_user_level_file_is_under_dot_config():
    assert user_config_path({}) == pathlib.Path("~/.config").expanduser() / "mfluxible" / USER_NAME


def test_the_user_level_file_is_config_toml_not_mfluxible_toml():
    """The two locations are named differently, and it is not an oversight.

    In a working directory a file has to say whose it is; inside a directory already
    called `mfluxible` the prefix is noise. Pinned because "make them consistent" is an
    obvious-looking tidy-up that would silently stop finding everyone's existing file.
    """
    assert user_config_path({}).name == "config.toml"
    assert user_config_path({}).parent.name == "mfluxible"


def test_a_relative_xdg_path_is_ignored_as_the_spec_requires(tmp_path):
    """The spec says a relative value is invalid and must be ignored.

    Worth honouring rather than treating as pedantry: `XDG_CACHE_HOME=cache` would
    otherwise put gigabytes of weights in a different place for every directory the
    server was started from, and each one would look like a fresh download.
    """
    assert cache_home({"XDG_CACHE_HOME": "cache"}) == pathlib.Path("~/.cache").expanduser()
    assert cache_home({"XDG_CACHE_HOME": ""}) == pathlib.Path("~/.cache").expanduser()
    assert cache_home({"XDG_CACHE_HOME": "/somewhere/else"}) == pathlib.Path("/somewhere/else")


def test_the_model_cache_follows_xdg_cache_home():
    """The point of the whole exercise: it has to land beside huggingface_hub's cache.

    HF reads XDG_CACHE_HOME itself, and the two directories hold the raw and the
    quantized copy of one model. Relocating one without the other moves half of what
    the person was trying to move, and they are doing it because they are out of disk.
    """
    from mfluxible.engine import _default_model_cache_dir

    moved = _default_model_cache_dir({"XDG_CACHE_HOME": "/Volumes/big/cache"})
    assert moved == pathlib.Path("/Volumes/big/cache/mfluxible")
    assert _default_model_cache_dir({}) == pathlib.Path("~/.cache/mfluxible").expanduser()


def test_an_explicit_model_dir_still_beats_xdg():
    """Nothing moves for anyone who had already said where they wanted this."""
    from mfluxible.engine import _default_model_cache_dir

    environ = {"MFLUXIBLE_MODEL_DIR": "~/weights", "XDG_CACHE_HOME": "/Volumes/big/cache"}
    assert _default_model_cache_dir(environ) == pathlib.Path("~/weights").expanduser()


# -- applying ------------------------------------------------------------------------


def test_the_real_environment_wins_over_the_file(tmp_path):
    (tmp_path / "mfluxible.toml").write_text('model = "flux-dev"\nquantize = 4\n')
    environ = {"MFLUXIBLE_MODEL": "z-image-turbo"}

    apply(environ, tmp_path, [])

    # Set already: left alone, so `MFLUXIBLE_MODEL=... mfluxible-server` is the one-off
    # override it looks like. Unset: filled in from the file.
    assert environ["MFLUXIBLE_MODEL"] == "z-image-turbo"
    assert environ["MFLUXIBLE_QUANTIZE"] == "4"


def test_applying_nothing_reports_nothing(tmp_path):
    environ = {"XDG_CONFIG_HOME": str(tmp_path / "empty")}
    assert apply(environ, tmp_path, []) is None
    assert environ == {"XDG_CONFIG_HOME": str(tmp_path / "empty")}


# -- the list of settings, against what the code actually reads ----------------------


def _variables_read_by(package: pathlib.Path) -> set[str]:
    """Every MFLUXIBLE_* variable the source reads, by name, minus the prefix.

    Matches reads specifically (`environ.get("MFLUXIBLE_X")` / `environ["MFLUXIBLE_X"]`)
    rather than any occurrence of the string, because this file and the docs mention
    plenty of these names in prose.
    """
    pattern = re.compile(r"""environ(?:\.get)?\(?\[?["']MFLUXIBLE_([A-Z0-9_]+)["']""")
    found = set()
    for source in package.glob("*.py"):
        found |= {name.lower() for name in pattern.findall(source.read_text())}
    return found


def test_known_lists_exactly_the_variables_the_package_reads():
    """The one thing keeping config.py's KNOWN honest.

    A new setting that isn't added here would be readable from the environment and
    rejected from the config file -- a plausible-looking file that quietly does nothing
    for one of its keys. A stale entry is the milder opposite: a key accepted here and
    read by nothing. Both are drift, and neither shows up in any other test.
    """
    read = _variables_read_by(PACKAGE)

    # MFLUXIBLE_CONFIG is the one variable deliberately not settable from a config file,
    # since a file naming the file to load is circular.
    assert read - {"config"} == set(KNOWN), (
        "config.KNOWN and the variables the package reads have drifted: "
        f"read but not in KNOWN = {sorted(read - {'config'} - set(KNOWN))}, "
        f"in KNOWN but read by nothing = {sorted(set(KNOWN) - read - {'config'})}"
    )


def test_the_example_config_only_names_real_settings():
    """mfluxible.example.toml is the file people copy and uncomment.

    Every line in it is commented out, which means nothing ever parses it and a typo'd
    key would sit there indefinitely -- until someone uncomments it and gets a startup
    error pointing at a file they were told to copy. So this reads the keys out of the
    comments and checks them directly, rather than trying to reconstruct a live file
    (uncommenting by pattern would quietly mis-scope anything under `[env]`).
    """
    text = (ROOT / "mfluxible.example.toml").read_text()
    documented = set(re.findall(r"^# ([a-z_]+) = ", text, flags=re.M))

    assert documented, "no commented-out settings found -- has the file's shape changed?"
    assert documented <= set(KNOWN), f"not real settings: {sorted(documented - set(KNOWN))}"

    # And the other direction, which is the one that rots: a setting added to the code
    # and never shown in the file people copy.
    assert set(KNOWN) <= documented, f"settings missing from the example: {sorted(set(KNOWN) - documented)}"


# -- packaging -----------------------------------------------------------------------


def _pyproject(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_the_wheel_carries_the_harness_the_server_serves():
    """GET / serves harness.html, and installed there is no clients/ to serve it from.

    This asserts the build config still says so. Losing the force-include would produce
    a package that imports, starts, generates images and 500s on its own front page --
    and no other test here would notice, since a checkout always has the file.
    """
    build = _pyproject(ROOT / "pyproject.toml")["tool"]["hatch"]["build"]
    assert build["targets"]["wheel"]["force-include"] == {"clients/harness.html": "mfluxible/harness.html"}
    # ...and the sdist has to keep it at that same path, because `uv build` builds the
    # wheel from the sdist, where the force-include above is resolved a second time.
    assert "/clients/harness.html" in build["targets"]["sdist"]["include"]
    assert (ROOT / "clients" / "harness.html").is_file()


def test_uv_is_told_this_is_not_a_project():
    """`managed = false` is what stops a pyproject.toml changing what `uv run` does.

    Without it, uv treats the repo as a project: `uv run clients/stream_client.py` on a
    client-only machine would resolve and install the *server's* dependencies, mflux and
    PyTorch included. Both files, since `cd clients` finds the other one.
    """
    assert _pyproject(ROOT / "pyproject.toml")["tool"]["uv"]["managed"] is False
    assert _pyproject(ROOT / "clients" / "pyproject.toml")["tool"]["uv"]["managed"] is False


def test_both_distributions_report_the_version_they_are_built_from():
    """Each pyproject reads its version out of the package rather than restating it."""
    for pyproject, module in (
        (ROOT / "pyproject.toml", "mfluxible/__init__.py"),
        (ROOT / "clients" / "pyproject.toml", "mfluxible_mcp/__init__.py"),
    ):
        data = _pyproject(pyproject)
        assert data["project"]["dynamic"] == ["version"]
        assert data["tool"]["hatch"]["version"]["path"] == module


def test_the_two_distributions_ship_the_same_version():
    """They are released together off one tag, so a drift is a packaging bug.

    Cheap to keep true and expensive to discover in the wild: the MCP client and the
    server negotiate over /health's `supports_*` fields, and "which versions of these
    two am I running" stops having one answer the moment they can differ.
    """
    import mfluxible_mcp

    assert mfluxible_mcp.__version__ == __version__
