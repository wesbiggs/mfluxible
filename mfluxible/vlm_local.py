"""Object detection by a vision model loaded into the server process.

This is the default backend. The other one -- `vlm_worker.py`, a sidecar running an
arbitrary command -- is still here and still a setting away, because the two are
answering different needs rather than competing: see MFLUXIBLE_VLM_BACKEND in
docs/server.md, and the long note in CLAUDE.md about which argument applies to which.

The short version of that note, because it is the thing most likely to be
re-litigated: the sidecar exists for `claude -p`, which runs an arbitrary command
with filesystem access, makes network calls and spends an account -- none of which
may sit behind an HTTP endpoint. A local MLX model does none of those things, so
that argument simply does not reach it, and the cost of a separate process (a second
thing to start, and a feature that silently does nothing when you forget) buys
nothing. What a local model *does* cost is memory, which is the subject of most of
the rest of this file.

**Lazy, because the cost is real and most runs never pay it.** Loading at startup
would put ~3GB of weights beside the image model's ~10-12GB on every server, for a
button many users never press, and the first thing this repo learned the hard way is
what happens when MLX's footprint passes the machine's headroom (see CLAUDE.md on the
buffer cache: identical work went from 7s to 105s, page-faulting rather than
computing). So nothing is loaded until the first detection asks for it, and a server
whose users never detect anything is byte-for-byte the server it was before.

**Everything here runs on MfluxEngine's one MLX thread, under its one lock.** Not a
convenience -- `engine.py`'s module docstring explains why all MLX work has to share
that thread, and a second framework allocating Metal buffers from another thread
while a generation is mid-loop is exactly the unsynchronized overlap that invariant
exists to forbid. `MfluxEngine.run_exclusive` is the door in. The visible consequence
is that a detection waits behind a running generation and vice versa, which is the
trade this backend is: convenience bought with the one resource this project has
already been bitten by.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path

from PIL import Image

from mfluxible.engine import _oriented
from mfluxible.vlm import BACKEND_LOCAL, backend_from_env
from mfluxible.vlm_reply import MAX_REGIONS, clean_payload, extract_json

log = logging.getLogger("mfluxible.vlm_local")

# Qwen2.5-VL at 3B and 4-bit: ~2.9GB on disk, and the size the community consistently
# reports as the one whose *grounding* holds up -- the 7B is the better describer and
# the worse localizer, which is the wrong way round for a feature whose output is a
# mask rectangle. A box in the wrong place is this feature's defining failure (see the
# measurements in CLAUDE.md's MCP section), so localization is what the default
# optimizes for. Override with MFLUXIBLE_VLM_LOCAL_MODEL; anything mlx-vlm can load
# and that answers in the dialect below will work.
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
MODEL = os.environ.get("MFLUXIBLE_VLM_LOCAL_MODEL", "").strip() or DEFAULT_MODEL

# Enough for a one-paragraph prompt plus eight boxes, and no more. This is the only
# bound on how long a detection can hold the generation lock: unlike the sidecar there
# is no subprocess to time out, because there is no way to interrupt work already
# running on the MLX thread -- the same property CLAUDE.md records for generate_image().
# A token cap is the one lever that decides when this returns.
MAX_TOKENS = 1024

# Qwen's own defaults, used only when the loaded processor doesn't carry its own (see
# _frame_for). `factor` is patch_size * merge_size; the pixel figures are areas, not
# edges, which the "shortest_edge"/"longest_edge" spelling in some processor configs
# actively disguises.
DEFAULT_FACTOR = 28
DEFAULT_MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 1280 * 28 * 28

# The dialect this backend adapts from. Qwen2.5-VL is trained to emit absolute pixel
# coordinates under this key -- which is itself a change from Qwen2-VL's 0-1000
# normalized scheme, i.e. one model family broke its own convention between versions.
# That is the entire case for the output contract in vlm_reply.py being fixed.
BOX_KEY = "bbox_2d"

PROMPT = """Reply with ONLY a JSON object of this shape:

{{"prompt": string, "regions": [{{"label": string, "bbox_2d": [x1, y1, x2, y2]}}]}}

"prompt" is a text-to-image prompt that would plausibly regenerate this picture. Describe the whole frame the way a prompt is written -- subject, pose, composition, setting, lighting, colour, style and medium -- not the way a caption describes a photograph. Do not open with "an image of" or "a picture showing". One paragraph, no line breaks.

"regions" lists the distinct objects a person might want to replace using an inpainting model. Each bbox_2d is [left, top, right, bottom] in absolute pixels of this image, with 0,0 at the TOP-LEFT.

Measure each box against the object's actual extent. Do not round to a coarse grid, and do not box the semantic region an object sits in -- box the object itself, tightly. A box centred on the wrong thing, or half again too large, is the failure to avoid. Prefer whole objects someone might swap out over parts of them, most prominent first, at most {max_regions}."""


def smart_resize(
    height: int,
    width: int,
    factor: int = DEFAULT_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen's own image-sizing algorithm, reimplemented so this side can predict it.

    Returns (height, width), both multiples of `factor`, with the aspect ratio held as
    closely as that allows and the area inside [min_pixels, max_pixels].

    **Why this is here rather than left to the processor.** The model answers in pixels
    of the frame it was shown, and the processor -- not the caller -- decides what that
    frame is. Asking it afterwards would mean reaching into a transformers internal that
    is free to move. Instead `detect_sync` resizes the image to this function's output
    *before* handing it over, which makes the processor's own call a no-op: an image
    already a multiple of `factor` and already inside the budget is its own fixed point.
    The frame the model sees is then a number this module computed, and dividing by it
    is arithmetic rather than a guess.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _budget(processor) -> tuple[int, int, int]:
    """(factor, min_pixels, max_pixels) read off the loaded processor, not assumed.

    This is the one place a wrong number would be *silently* wrong rather than visibly
    wrong. If the frame this module computes is larger than the processor's own budget,
    the processor shrinks the image again, the model answers in that smaller frame, and
    dividing by the larger one yields fractions that are too small -- but still inside
    [0,1], so vlm_reply's range check passes them through. Every box would come back
    correctly shaped and systematically too close to the top-left corner.

    Hence reading the model's own configuration rather than hardcoding Qwen's published
    defaults: a quantized repack that lowered max_pixels to fit a smaller machine is an
    ordinary thing to publish, and it would produce exactly that failure.
    """
    image_processor = getattr(processor, "image_processor", processor)

    patch = getattr(image_processor, "patch_size", None)
    merge = getattr(image_processor, "merge_size", None)
    factor = patch * merge if isinstance(patch, int) and isinstance(merge, int) else DEFAULT_FACTOR

    min_pixels = getattr(image_processor, "min_pixels", None)
    max_pixels = getattr(image_processor, "max_pixels", None)
    # Newer transformers moves the same two areas into a `size` dict, under edge names
    # they are not: both values stay areas in pixels.
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        min_pixels = size.get("shortest_edge", min_pixels)
        max_pixels = size.get("longest_edge", max_pixels)

    if not isinstance(min_pixels, int) or min_pixels <= 0:
        min_pixels = DEFAULT_MIN_PIXELS
    if not isinstance(max_pixels, int) or max_pixels <= 0:
        max_pixels = DEFAULT_MAX_PIXELS
    return factor, min_pixels, max_pixels


def to_fractions(parsed: dict | list, width: int, height: int) -> dict | list:
    """Qwen's pixel boxes rewritten as the fractions vlm_reply.py accepts.

    A driver, not a dial: `width`/`height` are the frame this module resized the image
    to and handed the model, not a setting anyone chose. CLAUDE.md's rule forbids a
    *configurable* conversion -- the kind where an operator picks "0-1000" from a menu
    and a wrongly-scaled box comes back looking plausible. A backend that knows its own
    model's convention and divides by a frame it computed itself is the opposite of
    that, and it is why this lives beside the prompt that asks for those pixels rather
    than anywhere a second backend could reach it.

    Entries are rewritten in place of `bbox_2d`, under the `box` key the shared cleaner
    reads. Anything unparseable is left alone rather than repaired, so it fails that
    cleaner's range check and is dropped there -- one rejection point, not two.
    """
    if isinstance(parsed, list):
        regions = parsed
    elif isinstance(parsed, dict):
        regions = parsed.get("regions")
        if not isinstance(regions, list):
            return parsed
    else:
        return parsed

    for item in regions:
        if not isinstance(item, dict):
            continue
        box = item.get(BOX_KEY, item.get("box"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        item["box"] = [x0 / width, y0 / height, x1 / width, y1 / height]
    return parsed


def _generated_text(output) -> str:
    """mlx-vlm's `generate` has returned a bare string and, later, an object carrying
    one. Both are read here rather than pinning a version, for the reason CLAUDE.md
    gives about mflux: this is a fast-moving dependency reached through a surface it
    does not document as stable, and the failure worth avoiding is a server that stops
    detecting after a routine `uv pip install -U`."""
    text = getattr(output, "text", output)
    return text if isinstance(text, str) else str(text)


class LocalDetector:
    """The model, loaded at most once, and the one detection it can run at a time.

    Neither method may be called from the event loop: both do MLX work and belong on
    MfluxEngine's worker thread, which `server.py` arranges through `run_exclusive`.
    """

    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._config = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load_sync(self) -> str | None:
        """Load the weights if they aren't already. Returns a reason on failure, or
        None on success -- *returned*, like every other failure path in this package,
        so nothing an import or a downloader says about a path reaches a response body.
        """
        if self._model is not None:
            return None
        try:
            # Imported here and not at module scope, for the same reason models.py
            # defers its mflux imports: naming this backend must not cost anything to a
            # server that never uses it, and this import pulls transformers, opencv and
            # a good deal else behind it.
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError:
            return (
                "the local vision model needs mlx-vlm, which isn't installed -- "
                "`uv pip install 'mfluxible[vlm]'`, or set MFLUXIBLE_VLM_BACKEND=worker "
                "to use a detection sidecar instead."
            )

        log.info("loading the local vision model (%s) -- the first run downloads it", self.model_name)
        try:
            self._model, self._processor = load(self.model_name)
            self._config = load_config(self.model_name)
        except Exception:
            # Logged in full to stderr, reported as one curated sentence: the messages
            # underneath quote absolute cache paths (`~/.cache/huggingface/...`), which
            # is the disclosure the "no path from an exception to a response body"
            # invariant exists to stop. Same shape as generate_stream's handler.
            log.exception("could not load the local vision model")
            self._model = self._processor = self._config = None
            return "could not load the local vision model -- see the server log for the reason."
        log.info("local vision model ready")
        return None

    def detect_sync(self, image_path: Path) -> dict:
        """Run one detection. Always returns a body for the server, never raises."""
        problem = self.load_sync()
        if problem is not None:
            return {"error": problem}

        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:  # pragma: no cover -- load_sync already proved these import
            return {"error": "mlx-vlm is not installed."}

        if not Path(image_path).is_file():
            return {"error": "the image for this job is no longer on disk."}

        try:
            raw = Path(image_path).read_bytes()
            # The same orientation the mask will be drawn against. vlm.py measures the
            # job's dimensions through engine.py's _oriented for this reason, and a box
            # chosen in the unrotated frame and applied in the rotated one is transposed.
            image = _oriented(raw).convert("RGB")
        except (OSError, ValueError):
            return {"error": "that image could not be read."}

        factor, min_pixels, max_pixels = _budget(self._processor)
        height, width = smart_resize(
            image.height, image.width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        # BOX rather than a windowed filter, matching the mask downsample in engine.py:
        # this image is about to be measured, and ringing at a high-contrast edge moves
        # the edge the model is being asked to find.
        framed = image.resize((width, height), Image.BOX)

        # A file rather than the PIL object, because a path is the input shape mlx-vlm
        # documents and the one least likely to move under it. Written beside the stash
        # so it shares that directory's permissions, and removed on every path out --
        # the mailbox's own pruning is an hourly sweep, not a cleanup for this.
        handle, tmp_path = tempfile.mkstemp(suffix=".png", dir=str(Path(image_path).parent))
        os.close(handle)
        try:
            framed.save(tmp_path, format="PNG")
            prompt = apply_chat_template(
                self._processor, self._config, PROMPT.format(max_regions=MAX_REGIONS), num_images=1
            )
            output = generate(
                self._model,
                self._processor,
                prompt,
                [tmp_path],
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
        except Exception:
            log.exception("the local vision model failed mid-detection")
            return {"error": "the detection failed -- see the server log for the reason."}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover -- best effort, as in engine.py
                pass

        parsed = extract_json(_generated_text(output))
        if parsed is None:
            return {"error": "the model's reply held no JSON object."}
        # Convert first, validate second. The range check in clean_payload is what
        # catches a frame this module got wrong, so it has to run on the fractions
        # rather than on the pixels -- see _budget for the failure it is guarding.
        return clean_payload(to_fractions(parsed, width=width, height=height))


def detector_from_env() -> LocalDetector | None:
    """A detector when MFLUXIBLE_VLM_BACKEND selects the in-process model, else None.

    Nothing is loaded here. Constructing one is free -- it is a name and three Nones --
    which is what lets server.py decide the backend at import and still defer the
    weights to the first request that needs them.
    """
    return LocalDetector() if backend_from_env() == BACKEND_LOCAL else None
