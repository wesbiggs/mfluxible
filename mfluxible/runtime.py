"""Process-wide state: the settings read at import, and the engine they configure.

Split out of server.py so the route modules under `mfluxible/routes/` can share it without
importing the app that includes them. They read these as `runtime.NAME` *at call time*, never
`from mfluxible.runtime import NAME`, and that is the whole contract: a module-level name
copied into another module's namespace at import is a snapshot, and tests (and anything
else that swaps a setting) patch it here, once.

Everything below is read at import, like every other setting in the package -- see
config.py and CLAUDE.md on why that is also why the config file is applied by
`mfluxible/__init__.py`.
"""

import os
from pathlib import Path

from mfluxible.auth import credentials_from_env
from mfluxible.config import as_float, as_int
from mfluxible.engine import MfluxEngine
from mfluxible.limits import max_body_bytes_from_env
from mfluxible.vlm import backend_from_env, mailbox_from_env
from mfluxible.vlm_local import detector_from_env

MODEL = os.environ.get("MFLUXIBLE_MODEL", "z-image-turbo")

_raw_quantize = os.environ.get("MFLUXIBLE_QUANTIZE", "8")
QUANTIZE = None if _raw_quantize.strip().lower() == "none" else as_int("MFLUXIBLE_QUANTIZE", _raw_quantize)

_raw_lora_paths = os.environ.get("MFLUXIBLE_LORA_PATHS", "").strip()
LORA_PATHS = [p.strip() for p in _raw_lora_paths.split(",") if p.strip()] or None

_raw_lora_scales = os.environ.get("MFLUXIBLE_LORA_SCALES", "").strip()
if LORA_PATHS and _raw_lora_scales:
    LORA_SCALES = [as_float("MFLUXIBLE_LORA_SCALES", s.strip()) for s in _raw_lora_scales.split(",")]
    if len(LORA_SCALES) != len(LORA_PATHS):
        raise ValueError(
            f"MFLUXIBLE_LORA_SCALES has {len(LORA_SCALES)} entries but "
            f"MFLUXIBLE_LORA_PATHS has {len(LORA_PATHS)} -- they must match."
        )
else:
    LORA_SCALES = None

MAX_BODY_BYTES = max_body_bytes_from_env()

# (username, password) or None. Read once at import like every other setting here;
# the middleware reads it back through a callable so the tests can swap it.
BASIC_AUTH = credentials_from_env()

# None unless MFLUXIBLE_VLM_DIR names a directory -- see vlm.py for why one
# variable both enables and configures. Off by default because a detection spends the
# Claude account of whoever runs the worker.
MAILBOX = mailbox_from_env()

# Which side runs a detection -- see vlm.py. Raises here, at import, on a value that is
# neither backend, rather than quietly defaulting: the whole point of naming one is that
# the operator knows which model answers.
VLM_BACKEND = backend_from_env()

# The in-process model when that is the backend, else None. Constructing it loads
# nothing; the weights arrive on the first detection that needs them.
DETECTOR = detector_from_env()

# How long the harness's stream waits for an answer. Generous: a cold `claude -p` start
# plus the model's own thinking measured ~11s against a 768x768 image, and the local
# backend has to wait out any generation holding the engine lock before it even begins.
VLM_TIMEOUT_S = as_float("MFLUXIBLE_VLM_TIMEOUT", os.environ.get("MFLUXIBLE_VLM_TIMEOUT", "180"))

# What the first local detection waits instead, because it includes downloading a few
# gigabytes of weights. Splitting the two rather than raising VLM_TIMEOUT for everyone
# is the point: a warm detection that has stopped responding should still fail in three
# minutes, and a cold one that is merely on a slow connection should not fail at all.
# The harness is told which of the two it is waiting on (see the `pending` event), so a
# long first run reads as a download rather than as a hang.
VLM_LOAD_TIMEOUT_S = as_float("MFLUXIBLE_VLM_LOAD_TIMEOUT", os.environ.get("MFLUXIBLE_VLM_LOAD_TIMEOUT", "900"))

# How long a worker's long-poll hangs before returning empty. Short enough that
# stopping the worker doesn't leave a request pinned for minutes, long enough that
# the loop isn't really polling.
VLM_POLL_S = 25.0

# An unknown MFLUXIBLE_MODEL raises here, at import, listing the valid names -- before
# lifespan starts a multi-gigabyte download for something that was never going to run.
engine = MfluxEngine(model=MODEL, quantize=QUANTIZE, lora_paths=LORA_PATHS, lora_scales=LORA_SCALES)


# Set once lifespan's engine.load() finishes -- used as the `created` timestamp on
# GET /v1/models, since OpenAI's schema requires one and "when this process actually
# started serving this model" is the only real value there is to give it (there's no
# meaningful weight-publish date to report instead).
MODEL_LOADED_AT: int = 0


# Where finished images live, served read-only at /mfluxible/v1/output/<name>. The default
# is the MCP tool's own save directory (MFLUXIBLE_MCP_SAVE_DIR in clients/mfluxible_mcp),
# which is a separate setting in a separate distribution -- point the two at the same
# place if you move either. This server writes nothing there; it only serves what is in it.
OUTPUT_DIR = Path(os.environ.get("MFLUXIBLE_OUTPUT_DIR", "~/Pictures/mfluxible")).expanduser()
OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
