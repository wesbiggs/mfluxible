#!/usr/bin/env python3
"""Answers the server's object-detection jobs by shelling out to a detection command.

The harness can see an image and drive the GPU but has no vision model. This process
is the piece that supplies one: it long-polls mfluxible for a pending detection, runs
a command against the stashed image, and posts the regions back for the harness's open
stream to deliver.

The command is `claude -p` by default and is one env var away from being anything else
-- another model's CLI, a local detector, a script of your own. What this worker
actually depends on is narrow: a program that prints a JSON object carrying a prompt and
labelled boxes to stdout. See MFLUXIBLE_VLM_COMMAND below.

**A sidecar, not a client, and not part of the server process either.** It doesn't
consume the API the way everything in `clients/` does -- it supplies a capability the
server advertises, has to sit on the same machine (it reads the stash directory off
disk), takes its directory from the server's own configuration, and is the thing
`/health`'s `worker_attached` reports on. Run it alone and nothing happens.

That it is a separate *process* is the part carrying the safety argument, and it
survives the reclassification unchanged: this runs an arbitrary configured command with
filesystem access, which on the default setting also makes network calls and spends a
Claude account. Behind an endpoint, that would be comfortably the most dangerous thing
in this repo and one path away from `tests/test_proxy_config.py`'s worst case. Out here
it can't be reached from the network at all, it's optional (the server runs fine with
nothing listening, and says so on /health), and starting it is what consent to that
command running looks like.

**Nothing else in this package imports this, and nothing should.** Living in the same
package as server.py makes it importable from inside the server process, and that is a
consequence of where it has to sit -- beside the server, on the machine holding the
stash directory -- not an interface. It is a script with a `main()`, spawned by a
person; an import from server.py would put an arbitrary subprocess back inside the
request path, which is the whole thing this placement avoids.

Run it alongside the server:

    mfluxible-vlm-worker

or, from a checkout, `uv run python -m mfluxible.vlm_worker`. It has to be reached as a
module either way: running this file by path skips the package's __init__, which is what
applies the config file, so the settings below would be read from a bare environment.
The `__main__` guard at the bottom refuses that rather than letting it happen quietly.

**It takes no MFLUXIBLE_VLM_DIR.** The server's copy of that setting is the only
one: every job names the file it wants read, so the worker never invents a path and has
nothing to keep in step. It runs the detection command in that file's own directory,
which matters twice for the default -- an image inside the cwd needs no --add-dir to be
readable, and a directory with no CLAUDE.md in it keeps a one-shot detection from
loading this project's instructions on every call.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from mfluxible import __version__
from mfluxible.vlm_reply import clean_payload, extract_json

# Environment-only, like the other clients: a --token flag would put the secret in
# shell history and in `ps` output for as long as the worker runs, which here is
# indefinitely. See CLAUDE.md for why the name is prefixed and why it says BEARER.
TOKEN = os.environ.get("MFLUXIBLE_BEARER_TOKEN", "")

# Where the server is. A setting rather than only a flag so that a worker started by
# launchd -- which has a config file and no convenient command line -- can be pointed
# at a non-default port without one. An explicit --url still wins.
DEFAULT_SERVER_URL = os.environ.get("MFLUXIBLE_SERVER_URL", "").strip() or "http://127.0.0.1:8420"

# Which model the *default* command uses. Ignored once MFLUXIBLE_VLM_COMMAND is
# set, since that names the whole command line.
#
# Opus, and the usual latency-for-quality trade turns out not to apply -- measured on
# the same 768x768 photograph, same prompt, it was both better and *faster*: 11.4s
# against sonnet's 66.6s.
#
# The quality gap is the reason this is not tunable-by-taste. Sonnet put the apple at
# [0.28, 0.28, 0.68, 0.62] -- roughly the right size in the wrong place, clipping the
# fruit's bottom third and taking in a band of forearm, which is exactly the failure
# docs/mcp.md records for a hand-estimated box. Opus put it at
# [0.305, 0.344, 0.712, 0.736] against [0.310, 0.344, 0.694, 0.729] measured by hand:
# within two percent on every edge, and quoted to three decimals rather than the round
# two-decimal numbers that signal estimating on a grid. Sonnet had also produced a good
# box on an earlier run of the same image, so the problem is variance rather than a
# constant offset -- which is worse for this, since a client can't tell the two apart.
MODEL = os.environ.get("MFLUXIBLE_VLM_MODEL", "opus")

# What actually gets run. Claude Code is the default because it needs no extra install
# on a machine that already has it, but nothing here is specific to it: this worker
# spawns a command, reads stdout and parses one JSON array. Anything that can do that
# from an image path can take its place -- another model's CLI, a local detector, a
# script of your own.
#
# `{image}` is the image's filename (the working directory is the stash directory, so
# a bare name resolves) and `{prompt}` is PROMPT below, already carrying the filename.
# A tool driven by a prompt uses both; a dedicated detector uses only `{image}` and the
# prompt is simply never substituted anywhere.
#
# Split with shlex and run without a shell: quoting works as it does in a shell, while
# `;` and `|` are argv characters rather than syntax. That matters less than it looks
# -- the template is set by whoever starts the worker and the filename is a uuid this
# server generated -- but there is no reason to spawn a shell to run one program.
#
# What *is* fixed is the output contract, and it has to be: stdout must contain a JSON
# array of {"label": str, "box": [x0, y0, x1, y1]}, boxes as fractions of the frame
# with 0,0 top-left. Tools disagree about coordinates (pixels, 0-1000, xywh), so
# anything that doesn't already speak this wants a few lines of wrapper rather than a
# conversion setting here -- one reader, in one place, is what keeps a bad box a
# visible bad box instead of a silently rescaled one.
DEFAULT_COMMAND = f"claude -p {{prompt}} --allowedTools Read --model {MODEL}"
COMMAND = os.environ.get("MFLUXIBLE_VLM_COMMAND", "").strip() or DEFAULT_COMMAND

# Long-polls hang for ~25s server-side; this has to outlast that or every poll looks
# like a timeout to requests.
HTTP_TIMEOUT = 60

# A detection is one image in, one answer out. If it hasn't finished by now something
# is wrong -- an interactive prompt nothing can answer, a wedged subprocess -- and the
# server's own wait is 180s, so failing first leaves room to report why.
DETECT_TIMEOUT = 150

PROMPT = """Read the image at {path} and reply with ONLY a JSON object of this shape:

{{"prompt": string, "regions": [{{"label": string, "box": [x0, y0, x1, y1]}}]}}

"prompt" is a text-to-image prompt that would plausibly regenerate this picture. Describe the whole frame the way a prompt is written -- subject, pose, composition, setting, lighting, colour, style and medium -- not the way a caption describes a photograph. Do not open with "an image of" or "a picture showing". One paragraph, no line breaks.

"regions" lists the distinct objects a person might want to replace using an inpainting model. Coordinates are fractions of the frame from 0 to 1, with 0,0 at the TOP-LEFT: x0/x1 are the left and right edges, y0/y1 the top and bottom.

Measure each box against the object's actual extent. Do not round to a coarse grid, and do not box the semantic region an object sits in -- box the object itself, tightly. A box centred on the wrong thing, or half again too large, is the failure to avoid. Prefer whole objects someone might swap out over parts of them, most prominent first, at most 8.

Example: {{"prompt": "a single red apple resting on a weathered oak table, soft window light from the left, shallow depth of field, photographic", "regions": [{{"label": "red apple", "box": [0.31, 0.37, 0.68, 0.72]}}]}}
"""


def build_command(template: str, *, image: str, prompt: str) -> list[str]:
    """The command template as argv, with the placeholders filled in.

    Split *first*, substitute *second*, and that order is the whole point: doing it the
    other way round would let the prompt's own punctuation reshape the command. The
    prompt contains spaces, newlines, quotes and braces, and a quote inside it would
    silently split one argument into two, or swallow the rest of the line.

    Substituting per token rather than across the joined string is what makes
    `--image={image}` work as well as a bare `{image}`, while keeping a filled-in
    placeholder exactly one argument whatever it holds.

    Returns [] for a template that isn't parseable, which the caller reports.
    """
    try:
        tokens = shlex.split(template)
    except ValueError:
        return []
    return [token.replace("{image}", image).replace("{prompt}", prompt) for token in tokens]


def _headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


# The reply's shape lives in mfluxible/vlm_reply.py, not here, because there are two
# backends now and they have to agree on it exactly -- see that module's docstring.
# This one still holds the *command* end of the contract: what gets run, and how its
# stdout becomes one of those replies.


def detect(image_path: str) -> dict:
    """Run one detection. Always returns a body for the server, never raises."""
    path = Path(image_path)
    # The job's own directory, which is the one the server stashed into. Used as cwd so
    # the file is already inside an allowed root and no CLAUDE.md is picked up.
    workdir = path.parent
    if not path.is_file():
        return {"error": "the image for this job is no longer on disk."}

    cmd = build_command(COMMAND, image=path.name, prompt=PROMPT.format(path=path.name))
    if not cmd:
        # Empty or unquotable -- both mean there is nothing runnable to report about.
        return {"error": "the detection command is empty or has unbalanced quotes."}

    try:
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=DETECT_TIMEOUT
        )
    except FileNotFoundError:
        return {"error": f"{cmd[0]} is not on PATH -- check MFLUXIBLE_VLM_COMMAND."}
    except OSError as exc:
        # A template naming something unrunnable (a directory, a file without +x).
        return {"error": f"could not run {cmd[0]}: {exc.strerror}."}
    except subprocess.TimeoutExpired:
        return {"error": f"the detection did not finish within {DETECT_TIMEOUT}s."}

    if proc.returncode != 0:
        # stderr is the tool's own message to its operator, so it's safe to pass on --
        # and it is the only thing that distinguishes "not logged in" from "no such
        # model", which the person reading the harness needs to know.
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1][:200] if detail else f"exit code {proc.returncode}"
        return {"error": f"{cmd[0]} failed: {tail}"}

    parsed = extract_json(proc.stdout or "")
    if parsed is None:
        return {"error": "the reply held no JSON object."}
    return clean_payload(parsed)


def run(base: str) -> None:
    next_url = f"{base}/mfluxible/v1/vlm/next"
    session = requests.Session()
    idle_note = True

    while True:
        try:
            resp = session.get(next_url, headers=_headers(), timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            # The server being down is the normal way this loop spends its time before
            # anyone starts it, so back off quietly rather than spinning.
            print(f"waiting for the server ({exc.__class__.__name__})", file=sys.stderr)
            time.sleep(5)
            continue

        if resp.status_code == 404:
            sys.exit(
                "the server has object detection switched off -- restart it with "
                "MFLUXIBLE_VLM_DIR set to a directory it can write to."
            )
        if resp.status_code == 401:
            sys.exit("the server rejected the credentials (set MFLUXIBLE_BEARER_TOKEN).")
        if resp.status_code == 204:
            if idle_note:
                print("attached, waiting for detections", file=sys.stderr)
                idle_note = False
            continue
        if not resp.ok:
            print(f"unexpected {resp.status_code} from the server", file=sys.stderr)
            time.sleep(5)
            continue

        idle_note = True
        job = resp.json()
        started = time.monotonic()
        print(f"detecting ({job['width']}x{job['height']})...", file=sys.stderr)
        body = detect(job["image_path"])
        body["width"], body["height"] = job["width"], job["height"]

        found = len(body.get("regions") or [])
        took = time.monotonic() - started
        if body.get("error"):
            summary = body["error"]
        else:
            summary = f"{found} regions" + (", prompt" if body.get("prompt") else ", no prompt")
        print(f"  {summary} in {took:.1f}s", file=sys.stderr)

        try:
            session.post(
                f"{base}/mfluxible/v1/vlm/{job['job_id']}",
                json=body,
                headers=_headers(),
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"  could not deliver the result ({exc.__class__.__name__})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mfluxible-vlm-worker",
        description="Answer mfluxible's object-detection jobs by running MFLUXIBLE_VLM_COMMAND.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER_URL,
        help=f"mfluxible's base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to read instead of the discovered one. Declared here so it "
        "appears in --help; it is read straight out of argv at import time, because "
        "the settings above are resolved before this parser exists.",
    )
    parser.add_argument("--version", action="version", version=f"mfluxible {__version__}")
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if TOKEN and (parsed.username or parsed.password):
        sys.exit(
            "error: MFLUXIBLE_BEARER_TOKEN is set and --url carries credentials; use one or the other"
        )

    try:
        run(args.url.rstrip("/"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # `python mfluxible/vlm_worker.py` would run, and would silently ignore the config
    # file: executing a file by path never imports the package around it, so
    # mfluxible/__init__.py -- which is what applies mfluxible.toml -- never runs, and
    # the settings above would be read from a bare environment. A worker quietly using
    # the default detection command because its config was skipped is exactly the kind
    # of silent wrong answer this codebase refuses elsewhere, so it is a refusal here
    # too rather than a warning.
    if not __package__:
        sys.exit(
            "error: run this as `python -m mfluxible.vlm_worker` (or the installed "
            "`mfluxible-vlm-worker`), not as a file path -- otherwise the config file "
            "is not applied."
        )
    main()
