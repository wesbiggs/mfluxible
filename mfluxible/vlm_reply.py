"""What a detection reply looks like, for whichever backend produced it.

There are two now -- the in-process model in `vlm_local.py` and the subprocess in
`vlm_worker.py` -- and they have to agree on the answer's shape down to the last
field, because the harness cannot tell which one served it and neither can a test.
So the reading lives here rather than in either of them.

It is a module of its own rather than a few functions in `vlm.py` for one reason:
`vlm.py` imports `engine.py` for `_oriented`, which imports MLX. The worker is a
subprocess whose whole point is that it needs none of that, and making it import a
GPU framework to parse a JSON array would be a silent cost paid by the backend
that was supposed to be the cheap one.

**The output contract is fixed, and it is the thing a backend adapts *to*.** Boxes
are fractions of the frame from 0 to 1 with 0,0 at the top-left, x first. A backend
whose model speaks something else (Qwen emits absolute pixels; other tools use
0-1000, or xywh) converts on its own side before calling in here -- see the note in
CLAUDE.md about why that conversion is a driver and never a dial.
"""

from __future__ import annotations

import json
import re

# At most this many regions reach the harness. They become chips in a row, and a
# reply listing thirty things is a tool that has described the picture rather than
# answered the question.
MAX_REGIONS = 8

# A label lands in a chip and in a JSON body; nothing useful survives past this.
MAX_LABEL_CHARS = 80

# The prompt lands in a textarea. Same reasoning, two orders of magnitude up.
MAX_PROMPT_CHARS = 2000


def extract_json(text: str) -> dict | list | None:
    """The JSON value out of a reply, read the way an eye would. Returns None if there
    isn't one -- the caller turns that into a message for the harness rather than a
    traceback, since a model answering in prose is an ordinary outcome, not a crash.

    An object `{prompt, regions}` is the shape asked for; a bare array is accepted as
    regions with no prompt, because that was this contract's earlier shape and a wrapper
    written against it should keep working rather than start returning nothing.
    """
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    for candidate in ([fence.group(1)] if fence else []) + [text]:
        # Whichever bracket opens *first* is the outer container, and only its matching
        # closer can end it. Trying "{...}" before "[...]" unconditionally would read an
        # array of objects as its own first element -- the last "}" sits inside the array,
        # so the slice parses cleanly and silently returns one region instead of all of
        # them. Picking by position is what makes the two shapes unambiguous.
        openers = [(candidate.find(o), o, c) for o, c in (("{", "}"), ("[", "]"))]
        openers = sorted((pos, o, c) for pos, o, c in openers if pos != -1)
        for start, _opener, closer in openers:
            end = candidate.rfind(closer)
            if end <= start:
                continue
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
    return None


def clean_payload(parsed: dict | list) -> dict:
    """The reply reduced to what the server accepts: a prompt and a list of regions.

    A bare array means regions only. A missing or blank prompt stays None rather than
    becoming "": the harness distinguishes "no prompt offered" from "an empty one".
    """
    if isinstance(parsed, list):
        return {"prompt": None, "regions": clean_regions(parsed)}
    raw_prompt = parsed.get("prompt")
    prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    regions = parsed.get("regions")
    return {
        "prompt": prompt[:MAX_PROMPT_CHARS] or None,
        "regions": clean_regions(regions if isinstance(regions, list) else []),
    }


def clean_regions(parsed: list) -> list[dict]:
    """Only the entries that are usable as a mask rectangle.

    Dropped rather than repaired, and silently: a box outside [0,1] or inverted means
    the reply wasn't measured against the frame it was asked about, and a coerced
    version of it would be a rectangle nobody chose sitting in a list the user is about
    to click. An empty result reads as "nothing found", which is honest.

    This is also the range check that a converting backend leans on. `vlm_local.py`
    divides Qwen's pixel coordinates by the frame it handed the model; if that frame
    were ever wrong, the quotients land outside [0,1] and arrive here to be dropped.
    That is the whole reason the conversion is allowed to be arithmetic with no
    validation of its own -- the validation is here, once, for every backend.
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
        label = str(item.get("label") or "object").strip()[:MAX_LABEL_CHARS]
        out.append({"label": label or "object", "box": [x0, y0, x1, y1]})
    return out[:MAX_REGIONS]
