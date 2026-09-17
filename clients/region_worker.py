#!/usr/bin/env python3
"""Answers the server's object-detection jobs by shelling out to the Claude Code CLI.

The harness can see an image and drive the GPU but has no vision model; Claude has
vision and can read a file off disk but has no UI. This process is the piece that
joins them: it long-polls mfluxible for a pending detection, runs `claude -p` against
the stashed image, and posts the regions back for the harness's open stream to
deliver.

**This is a client, and that placement is the design rather than filing.** It shells
out to a binary that makes network calls and spends a Claude account, so putting it in
`server/` would have meant an HTTP endpoint that spawns a subprocess with filesystem
access -- comfortably the most dangerous thing in this repo, and one path away from
`tests/test_proxy_config.py`'s worst case. Here it can't be reached from the network
at all, it's optional (the server runs fine with nothing listening, and says so on
/health), and starting it is what consent to spending that account looks like.

Run it alongside the server:

    MFLUXIBLE_REGIONS_DIR=~/.cache/mfluxible/regions uv run clients/region_worker.py

The same directory the server was given: the worker never invents a path, it reads
the one each job names, and that directory is also the working directory `claude` runs
in. Both halves of that matter -- an image inside the cwd needs no --add-dir to be
readable, and a directory with no CLAUDE.md in it keeps a one-shot detection from
loading this project's instructions on every call.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

# Environment-only, like the other clients: a --token flag would put the secret in
# shell history and in `ps` output for as long as the worker runs, which here is
# indefinitely. See CLAUDE.md for why the name is prefixed and why it says BEARER.
TOKEN = os.environ.get("MFLUXIBLE_BEARER_TOKEN", "")

CLAUDE_BIN = os.environ.get("MFLUXIBLE_CLAUDE_BIN", "claude")

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
#
# Left configurable for a machine where opus isn't available, not as a tuning knob.
MODEL = os.environ.get("MFLUXIBLE_REGIONS_MODEL", "opus")

# Long-polls hang for ~25s server-side; this has to outlast that or every poll looks
# like a timeout to requests.
HTTP_TIMEOUT = 60

# A detection is one Read and one reply. If it hasn't finished by now something is
# wrong -- an interactive prompt nothing can answer, a wedged subprocess -- and the
# server's own wait is 180s, so failing first leaves room to report why.
CLAUDE_TIMEOUT = 150

PROMPT = """Read the image at {path} and list the distinct objects a person might \
want to replace using an inpainting model.

Reply with ONLY a JSON array of objects, each {{"label": string, "box": [x0, y0, x1, y1]}}.
Coordinates are fractions of the frame from 0 to 1, with 0,0 at the TOP-LEFT: x0/x1 are \
the left and right edges, y0/y1 the top and bottom.

Measure each box against the object's actual extent. Do not round to a coarse grid, and \
do not box the semantic region an object sits in -- box the object itself, tightly. A box \
centred on the wrong thing, or half again too large, is the failure to avoid. Prefer \
whole objects someone might swap out over parts of them, most prominent first, at most 8.

Example: [{{"label": "red apple", "box": [0.31, 0.37, 0.68, 0.72]}}]
"""


def _headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def _extract_json_array(text: str) -> list | None:
    """The array out of a reply, read the way an eye would. Returns None if there
    isn't one -- the caller turns that into a message for the harness rather than a
    traceback, since a model answering in prose is an ordinary outcome, not a crash."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    for candidate in ([fence.group(1)] if fence else []) + [text]:
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _clean(parsed: list) -> list[dict]:
    """Only the entries that are usable as a mask rectangle.

    Dropped rather than repaired, and silently: a box outside [0,1] or inverted means
    the reply wasn't measured against the frame it was asked about, and a coerced
    version of it would be a rectangle nobody chose sitting in a list the user is about
    to click. An empty result reads as "nothing found", which is honest.
    """
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            continue
        label = str(item.get("label") or "object").strip()[:80]
        out.append({"label": label or "object", "box": [x0, y0, x1, y1]})
    return out[:8]


def detect(image_path: str) -> dict:
    """Run one detection. Always returns a body for the server, never raises."""
    path = Path(image_path)
    # The job's own directory, which is the one the server stashed into. Used as cwd so
    # the file is already inside an allowed root and no CLAUDE.md is picked up.
    workdir = path.parent
    if not path.is_file():
        return {"error": "the image for this job is no longer on disk."}

    cmd = [
        CLAUDE_BIN,
        "-p",
        PROMPT.format(path=path.name),
        "--allowedTools",
        "Read",
        "--model",
        MODEL,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT
        )
    except FileNotFoundError:
        return {"error": f"{CLAUDE_BIN} is not on PATH -- set MFLUXIBLE_CLAUDE_BIN, or install Claude Code."}
    except subprocess.TimeoutExpired:
        return {"error": f"the detection did not finish within {CLAUDE_TIMEOUT}s."}

    if proc.returncode != 0:
        # stderr is the CLI's own message to its operator, so it's safe to pass on --
        # and it is the only thing that distinguishes "not logged in" from "no such
        # model", which the person reading the harness needs to know.
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1][:200] if detail else f"exit code {proc.returncode}"
        return {"error": f"claude failed: {tail}"}

    parsed = _extract_json_array(proc.stdout or "")
    if parsed is None:
        return {"error": "the reply held no JSON array of regions."}
    return {"regions": _clean(parsed)}


def run(base: str) -> None:
    next_url = f"{base}/mfluxible/v1/regions/next"
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
                "the server has object detection switched off -- start it with "
                "MFLUXIBLE_REGIONS_DIR set to the same directory as this worker."
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
        print(
            f"  {body['error'] if body.get('error') else f'{found} regions'} in {took:.1f}s",
            file=sys.stderr,
        )

        try:
            session.post(
                f"{base}/mfluxible/v1/regions/{job['job_id']}",
                json=body,
                headers=_headers(),
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"  could not deliver the result ({exc.__class__.__name__})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer mfluxible's object-detection jobs using the Claude Code CLI."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8420", help="mfluxible's base URL")
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
    main()
