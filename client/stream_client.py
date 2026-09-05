#!/usr/bin/env python3
"""Generate an image via mfluxible, rendering step progress and previews inline in
the terminal as they stream in, then save the final image to disk.

Uses the iTerm2 inline-image protocol, supported by iTerm2, WezTerm, and some
other modern terminals. In an unsupported terminal the escape codes are just
ignored (or print as stray characters) -- the text progress lines and the saved
file still work either way. Not tmux-aware: iTerm2's protocol needs extra
passthrough wrapping inside tmux, which this script doesn't do.

Renders using the protocol's chunked MultipartFile variant -- the same one
iTerm2's own `imgcat` reference tool uses by default -- rather than one giant
File=...:<base64> sequence. iTerm2's own source caps how much data it'll
accumulate for a single OSC sequence at 1,048,576 bytes (VT100XtermParser.m),
truncating (not just dropping) whatever comes after, which can corrupt what
renders afterward too. Detailed/photographic diffusion output compresses poorly
and unpredictably as PNG, so a full-resolution image can realistically cross
that limit. Splitting the base64 payload into many small FilePart sequences (no
single one over ~200 bytes) means no image, at any size, can ever hit that cap
-- so unlike an earlier version of this script, there's no need to downscale or
recompress anything: the terminal renders the exact bytes that get saved.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

# imgcat (iTerm2's own reference tool) uses 200-byte chunks, but its own
# comment says that's specifically "to help it get through tmux" -- we're not
# tmux-wrapping at all (see the module docstring above), so we don't need
# chunks that small. A ~1.5MB image at 200 bytes/chunk means ~10,000 separate
# escape sequences in one burst; using a much larger chunk size (still safely
# under iTerm2's real 1,048,576-byte single-sequence cap, with margin) cuts
# that by ~2500x, which is worth trying against intermittent-failure reports
# that aren't explained by anything in the data itself (verified byte-for-byte
# correct in both a failing and a succeeding capture).
CHUNK_SIZE = 500_000


def show_image(png_bytes: bytes, width: str = "auto") -> None:
    # width="auto" (with height defaulting to auto too) renders at the image's
    # native pixel size divided by the display's backing scale factor -- the
    # same sizing `imgcat` uses by default. A fixed cell-count width instead
    # scales with the terminal's font/cell size, unrelated to the image's
    # actual dimensions, and renders inconsistently with `imgcat`'s output.
    b64 = base64.b64encode(png_bytes).decode("ascii")
    parts = [f"\033]1337;MultipartFile=inline=1;size={len(png_bytes)};width={width};preserveAspectRatio=1\a"]
    for i in range(0, len(b64), CHUNK_SIZE):
        parts.append(f"\033]1337;FilePart={b64[i:i + CHUNK_SIZE]}\a")
    parts.append("\033]1337;FileEnd\a\n")
    sys.stdout.write("".join(parts))
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image via mfluxible, showing progress inline.")
    parser.add_argument("prompt")
    parser.add_argument("--url", default="http://127.0.0.1:8420/mfluxible/v1/images/generations")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # Left unset, steps/guidance are the server's decision: it knows which model it
    # loaded and what that model's sensible default is (a step count tuned for
    # Z-Image-Turbo is four times too small for FLUX.1-dev). Sending null asks for
    # that default rather than overriding it with this script's guess.
    parser.add_argument("--steps", type=int, default=None, help="default: the server model's own")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None, help="models that use guidance only")
    parser.add_argument("--negative-prompt", default=None, help="Qwen-Image only")
    parser.add_argument("--preview-every", type=int, default=0, help="0 disables in-progress previews")
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--image", default=None, help="input image path, for image-to-image")
    parser.add_argument(
        "--image-strength",
        type=float,
        default=None,
        help=(
            "0.0-1.0, only with --image; mflux's own convention (not the inverse used by some "
            "other img2img tools): higher means the image constrains the output MORE, not less. "
            "Server default (0.4) applies if --image is given without this."
        ),
    )
    args = parser.parse_args()

    if args.image_strength is not None and args.image is None:
        print("error: --image-strength requires --image", file=sys.stderr)
        sys.exit(1)

    body = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "guidance": args.guidance,
        "negative_prompt": args.negative_prompt,
        "preview_every": args.preview_every,
        "stream": True,
        "image": base64.b64encode(Path(args.image).read_bytes()).decode("ascii") if args.image else None,
        "image_strength": args.image_strength,
    }

    with requests.post(args.url, json=body, stream=True) as resp:
        if resp.status_code != 200:
            # Worth unwrapping rather than raise_for_status()'ing: the server rejects
            # --guidance/--negative-prompt on a model that can't act on them, and its
            # message says which, where a bare "400 Client Error" says nothing.
            try:
                detail = resp.json().get("message") or resp.text
            except ValueError:
                detail = resp.text
            print(f"error: {resp.status_code} {detail}", file=sys.stderr)
            sys.exit(1)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            etype = event["type"]

            if etype == "start":
                print(f"seed={event['seed']} steps={event['total_steps']}")
            elif etype == "thinking":
                print(
                    f"  step {event['step']}/{event['total_steps']} "
                    f"({event['step_ms']}ms, {event['elapsed_ms']}ms total)"
                )
                if "preview" in event:
                    show_image(base64.b64decode(event["preview"]))
            elif etype == "image":
                png_bytes = base64.b64decode(event["data"])
                with open(args.out, "wb") as f:
                    f.write(png_bytes)
                print(f"saved {args.out} (seed={event['seed']}, {event['generation_time']:.1f}s)")
                show_image(png_bytes)
            elif etype == "error":
                print(f"error: {event['message']}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
