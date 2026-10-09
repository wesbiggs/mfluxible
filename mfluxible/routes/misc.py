"""The browser harness at `GET /`, and `GET /health`."""

from pathlib import Path

import mlx.core as mx
from fastapi import APIRouter
from fastapi.responses import FileResponse

from mfluxible import __version__
from mfluxible import runtime as rt
from mfluxible.models import MODELS

router = APIRouter()


def _harness_path() -> Path:
    """Where harness.html is, which differs between an install and a checkout.

    Its source of truth is `clients/harness.html`, because it *is* a client -- it
    consumes this API over HTTP like everything else in that directory (see CLAUDE.md's
    Layout note). But it is also the one client the server itself serves, at `GET /`, so
    the wheel has to carry a copy: pyproject.toml force-includes it next to this module,
    and installed that is the only copy there is.

    Hence two candidates rather than one, in this order. The installed location is
    checked first so that a checkout sitting beside an installed copy still serves its
    own file -- the direction you want to be wrong in while editing the harness.

    Nothing here falls back to "no harness": if neither exists the FileResponse raises,
    which surfaces as a 500 naming the path on the server's own log. That is a broken
    install rather than a request problem, and a stub page saying so would be a worse
    answer than a loud failure.
    """
    package_dir = Path(__file__).resolve().parents[1]  # mfluxible/, not mfluxible/routes/
    packaged = package_dir / "harness.html"
    if packaged.is_file():
        return packaged
    return package_dir.parent / "clients" / "harness.html"


# Resolved once at import, like every other setting here. The file does not move while
# a server runs, and a per-request stat to discover that it hasn't would be two syscalls
# on the hottest path this server has.
HARNESS_PATH = _harness_path()


@router.get("/")
async def harness():
    return FileResponse(HARNESS_PATH, media_type="text/html")


def _vlm_health() -> dict:
    """The `vlm` block on /health.

    **`backend` is absent when the feature is off**, rather than reporting whichever
    value the setting happens to hold. With no directory named, nothing answers a
    detection and the routes 404, so naming the backend that *would* have answered is a
    fact about a code path this server will never take -- and `enabled: false` beside a
    backend name reads like a half-configured feature, which is the one thing it is not.
    Its absence means "no backend is in play" rather than "an older server", because the
    key it sits beside already answers that: a client reads `enabled` first.

    `worker_attached` and `model_loaded` stay in both cases, because `false` is what
    they actually are -- nothing has polled, nothing is loaded -- rather than an answer
    about a road not taken.
    """
    return {
        "enabled": rt.MAILBOX is not None,
        # Which backend answers, so a client knows whether `worker_attached` is a fact
        # about this server or a fact about a process that isn't involved.
        **({"backend": rt.VLM_BACKEND} if rt.MAILBOX is not None else {}),
        # Left literal under the local backend, where it is simply False: reporting True
        # because "something is attached" would make the one field whose name says
        # "worker" stop meaning it. A client that predates `backend` shows a spurious
        # "start the worker" note and detects successfully anyway, which is the harmless
        # direction for this to be wrong in.
        "worker_attached": rt.MAILBOX is not None and rt.MAILBOX.worker_attached(),
        # Whether the first detection will include a download. Only meaningful for the
        # local backend; False and static for the worker.
        "model_loaded": rt.DETECTOR is not None and rt.DETECTOR.loaded,
    }


@router.get("/health")
async def health():
    # MLX's own accounting, in bytes. `active` is memory currently backing live
    # arrays (the model's weights, once something has forced them to
    # materialize); `cache` is buffers MLX has freed but holds onto for reuse,
    # which is reclaimable and counts toward the process footprint all the same;
    # `peak` is the high-water mark of active since startup or the last reset.
    # These are plain counters -- no graph work -- so they are safe to read from
    # the event loop rather than the MLX worker thread.
    spec = rt.engine.spec
    return {
        "status": "ok",
        # Which release a client is talking to. Safe to leave open alongside the rest
        # of this response for the same reason: it names no path and no account.
        "version": __version__,
        "model_loaded": rt.engine.model is not None,
        # What this process is running and which request fields it will accept, so a
        # client can pick sane defaults without knowing how the server was configured.
        # `available` is the whole table; only `model.name` was ever downloaded.
        "available": [m.key for m in MODELS],
        "model": {
            "name": spec.key,
            "label": spec.label,
            "repo": spec.repo,
            "quantize": rt.engine.quantize,
            "default_steps": spec.default_steps,
            "supports_guidance": spec.supports_guidance,
            "default_guidance": spec.default_guidance,
            "supports_negative_prompt": spec.supports_negative_prompt,
            # False on models whose own default scheduler isn't the linear one
            # mfluxible/schedulers.py extends -- see ModelSpec.default_scheduler. A client
            # that predates this field should treat a missing value as "unknown, let
            # the server decide", the same way it treats a missing `model`.
            "supports_fractional_start": spec.supports_fractional_start,
            # Same "missing means unknown" rule as the field above. A client that
            # shows a mask editor should hide it when this is False: the request
            # would come back a 400, since holding the unmasked region in place
            # means replacing the variant's scheduler (see ModelSpec.supports_mask).
            "supports_mask": spec.supports_mask,
        },
        # Whether the VLM mailbox is configured, and whether anything is currently
        # listening on it. The harness draws its "Find objects" button off `enabled` and
        # warns off `worker_attached`, so an unconfigured server shows no control that
        # could only fail -- the same "ask before you offer it" rule the model block
        # above exists for. No path here: the directory is the server's business, and
        # /health is the one endpoint the bundled Caddyfile leaves open.
        "vlm": _vlm_health(),
        "memory": {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
    }
