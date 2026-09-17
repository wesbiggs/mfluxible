"""FastAPI server exposing a streaming image-generation endpoint over mflux.

One process runs one model, chosen at startup with MFLUXIBLE_MODEL (see models.py
for the table). Weights are only fetched for the model actually selected.
"""

import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import chat_stub
from auth import BasicAuthMiddleware, credentials_from_env, resolve_bind_host, startup_warning
from engine import MfluxEngine
from models import CFG_GUIDANCE_FLOOR, MODELS
from schemas import (
    A1111Txt2ImgRequest,
    ChatCompletionRequest,
    GenerateRequest,
    OpenAIImageGenerationRequest,
    VlmRequest,
    VlmResult,
)
from vlm import mailbox_from_env

# Nothing configures logging here, so this lands on stderr via logging.lastResort --
# the terminal running uvicorn -- the same way engine.py's does.
log = logging.getLogger("mfluxible.server")

MODEL = os.environ.get("MFLUXIBLE_MODEL", "z-image-turbo")

_raw_quantize = os.environ.get("MFLUXIBLE_QUANTIZE", "8")
QUANTIZE = None if _raw_quantize.strip().lower() == "none" else int(_raw_quantize)

_raw_lora_paths = os.environ.get("MFLUXIBLE_LORA_PATHS", "").strip()
LORA_PATHS = [p.strip() for p in _raw_lora_paths.split(",") if p.strip()] or None

_raw_lora_scales = os.environ.get("MFLUXIBLE_LORA_SCALES", "").strip()
if LORA_PATHS and _raw_lora_scales:
    LORA_SCALES = [float(s.strip()) for s in _raw_lora_scales.split(",")]
    if len(LORA_SCALES) != len(LORA_PATHS):
        raise ValueError(
            f"MFLUXIBLE_LORA_SCALES has {len(LORA_SCALES)} entries but "
            f"MFLUXIBLE_LORA_PATHS has {len(LORA_PATHS)} -- they must match."
        )
else:
    LORA_SCALES = None

# (username, password) or None. Read once at import like every other setting here;
# the middleware reads it back through a callable so the tests can swap it.
BASIC_AUTH = credentials_from_env()

# None unless MFLUXIBLE_VLM_DIR names a directory -- see vlm.py for why one
# variable both enables and configures. Off by default because a detection spends the
# Claude account of whoever runs the worker.
MAILBOX = mailbox_from_env()

# How long the harness's stream waits for a worker's answer. Generous: a cold
# `claude -p` start plus the model's own thinking measured ~11s against a 768x768
# image, but the worker is a subprocess on a machine that may also be mid-generation.
VLM_TIMEOUT_S = float(os.environ.get("MFLUXIBLE_VLM_TIMEOUT", "180"))

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED_AT
    # Before the weights, so an exposed-and-unauthenticated server says so
    # immediately rather than after a multi-gigabyte download.
    warning = startup_warning(resolve_bind_host(), BASIC_AUTH is not None)
    if warning is not None:
        log.warning(warning)
    # Before the weights too: a bad MFLUXIBLE_VLM_DIR should fail now rather than
    # after a multi-gigabyte download, for the same reason the warning above is here.
    if MAILBOX is not None:
        MAILBOX.prepare()
    await engine.load()
    MODEL_LOADED_AT = int(time.time())
    yield
    engine.shutdown()


app = FastAPI(title="mfluxible", lifespan=lifespan)

# Default: reflect back any http(s)://localhost:<any port> or 127.0.0.1:<any
# port> origin (Starlette's allow_origin_regex does a fullmatch against the
# Origin header and, on match, echoes that exact origin in
# Access-Control-Allow-Origin rather than "*" -- so a local harness page
# served from either hostname on any port just works). Add other specific
# origins via MFLUXIBLE_CORS_ORIGINS (comma-separated) if needed.
CORS_ORIGIN_REGEX = os.environ.get("MFLUXIBLE_CORS_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?")
_raw_cors_origins = os.environ.get("MFLUXIBLE_CORS_ORIGINS", "").strip()
CORS_ORIGINS = [o.strip() for o in _raw_cors_origins.split(",") if o.strip()]

# Order matters, and it is the reverse of how it reads. Starlette's add_middleware
# inserts at the front of the stack, so the middleware added *last* runs *first* --
# meaning CORS below wraps auth here, not the other way round. That is the order
# required: CORSMiddleware answers a browser preflight itself and returns before
# auth ever sees it, which it has to, because a preflight carries no credentials
# by spec and a 401 there would make every cross-origin call fail with no way for
# the page to send them. Pinned by test_cors_preflight_is_answered_without_credentials.
app.add_middleware(BasicAuthMiddleware, credentials=lambda: BASIC_AUTH)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX or None,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# server/ and clients/ live in the same repo checkout but stay dependency-independent
# (see CLAUDE.md) -- this reaches across that boundary only to serve a static file, not
# to import anything, so it doesn't compromise that separation.
HARNESS_PATH = Path(__file__).resolve().parent.parent / "clients" / "harness.html"


@app.get("/")
async def harness():
    return FileResponse(HARNESS_PATH, media_type="text/html")


@app.get("/health")
async def health():
    # MLX's own accounting, in bytes. `active` is memory currently backing live
    # arrays (the model's weights, once something has forced them to
    # materialize); `cache` is buffers MLX has freed but holds onto for reuse,
    # which is reclaimable and counts toward the process footprint all the same;
    # `peak` is the high-water mark of active since startup or the last reset.
    # These are plain counters -- no graph work -- so they are safe to read from
    # the event loop rather than the MLX worker thread.
    spec = engine.spec
    return {
        "status": "ok",
        "model_loaded": engine.model is not None,
        # What this process is running and which request fields it will accept, so a
        # client can pick sane defaults without knowing how the server was configured.
        # `available` is the whole table; only `model.name` was ever downloaded.
        "available": [m.key for m in MODELS],
        "model": {
            "name": spec.key,
            "label": spec.label,
            "repo": spec.repo,
            "quantize": engine.quantize,
            "default_steps": spec.default_steps,
            "supports_guidance": spec.supports_guidance,
            "default_guidance": spec.default_guidance,
            "supports_negative_prompt": spec.supports_negative_prompt,
            # False on models whose own default scheduler isn't the linear one
            # server/schedulers.py extends -- see ModelSpec.default_scheduler. A client
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
        "vlm": {
            "enabled": MAILBOX is not None,
            "worker_attached": MAILBOX is not None and MAILBOX.worker_attached(),
        },
        "memory": {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
    }


async def _sse(req: GenerateRequest):
    async for event in engine.generate_stream(req):
        yield f"data: {json.dumps(event)}\n\n"


async def _collect_final_image(req: GenerateRequest) -> dict:
    """Runs a generation to completion and returns its final event -- `image`, or
    `error` if it ended in one instead. Callers check which; both endpoints below turn
    an `error` into their own error shape rather than sharing a response type, since
    the native and OpenAI-compat contracts disagree on what an error body looks like.

    Handed back rather than raised, for the reason in MfluxEngine.request_problem's
    docstring: an exception travelling out to a response body is how text from inside
    mflux would reach a client, so this module has no `except ... : str(exc)` in it at
    all and the messages below are only ever ones engine.py wrote to be read.
    """
    async for event in engine.generate_stream(req):
        if event["type"] in ("image", "error"):
            return event
    return {"type": "error", "message": "generation ended without an image or error event"}


# This is mfluxible's own API -- streaming `thinking` events with step timings and
# previews, guidance/negative_prompt, etc. -- moved off /v1/images/generations so that
# path can be a genuine OpenAI-compatible endpoint instead (see below). Every bundled
# client (harness.html, stream_client.py/js, mcp_server.py) targets this path.
@app.post("/mfluxible/v1/images/generations")
async def generate(req: GenerateRequest):
    # Checked before the response begins: for a stream, failing once StreamingResponse
    # has started would mean a torn body with a 200 already on the wire.
    problem = engine.request_problem(req)
    if problem is not None:
        return JSONResponse(status_code=400, content={"type": "error", "message": problem})

    if req.stream:
        return StreamingResponse(_sse(req), media_type="text/event-stream")

    final = await _collect_final_image(req)
    if final["type"] == "error":
        return JSONResponse(status_code=500, content={"type": "error", "message": final["message"]})
    return final


# ---------------------------------------------------------------------------
# Object detection mailbox
# ---------------------------------------------------------------------------
#
# Three endpoints, two clients: the harness submits an image and holds a stream,
# server/vlm_worker.py claims the job and posts regions back. The server does no
# detection -- see vlm.py's docstring for why that stays out of here.

_VLM_OFF = {"type": "error", "message": "object detection is not enabled on this server."}


def _vlm_disabled() -> JSONResponse:
    # 404 rather than 501: with MFLUXIBLE_VLM_DIR unset these routes are not a
    # feature this server has, and /health already says so before anything calls them.
    return JSONResponse(status_code=404, content=_VLM_OFF)


def _decoded_image(b64_data: str) -> bytes | None:
    """The bytes, or None if that string wasn't base64. Returned, not raised: the
    caller turns None into its own curated message, so binascii's never reaches a
    response body (see the invariant in CLAUDE.md)."""
    try:
        return base64.b64decode(b64_data, validate=True)
    except (ValueError, TypeError):
        return None


async def _vlm_sse(job):
    # Sent before anything blocks, so the harness can render "waiting" with the real
    # dimensions and say up front whether a worker is even listening.
    yield "data: " + json.dumps({
        "type": "pending",
        "job_id": job.id,
        "width": job.width,
        "height": job.height,
        "worker_attached": MAILBOX.worker_attached(),
    }) + "\n\n"

    result = await MAILBOX.wait(job, VLM_TIMEOUT_S)
    if result.get("error"):
        yield "data: " + json.dumps({"type": "error", "message": result["error"]}) + "\n\n"
        return
    # One event carrying everything the tool said. Named `result` rather than `regions`
    # because it no longer only carries those: `prompt` is a text-to-image prompt for
    # the whole frame, and a tool may return either half on its own.
    yield "data: " + json.dumps({
        "type": "result",
        "prompt": result.get("prompt"),
        "regions": result.get("regions") or [],
        # What the worker measured, for the harness to compare against its own oriented
        # size -- the guard against regions arriving for a different image.
        "width": result.get("width"),
        "height": result.get("height"),
    }) + "\n\n"


@app.post("/mfluxible/v1/vlm/describe")
async def vlm_describe(req: VlmRequest):
    if MAILBOX is None:
        return _vlm_disabled()

    raw = _decoded_image(req.image)
    if raw is None:
        return JSONResponse(
            status_code=400, content={"type": "error", "message": "image is not valid base64."}
        )
    problem = MAILBOX.submit_problem(raw)
    if problem is not None:
        return JSONResponse(status_code=400, content={"type": "error", "message": problem})

    # Same shape as a generation: validate fully, *then* start the stream, so a bad
    # request is a 400 rather than a 200 with an error torn into the body.
    job = MAILBOX.submit(raw)
    return StreamingResponse(_vlm_sse(job), media_type="text/event-stream")


@app.get("/mfluxible/v1/vlm/next")
async def vlm_next():
    """A worker's long-poll. Returns the pending job, or 204 when none arrives.

    This is the one endpoint here that deliberately returns a filesystem path, which
    is the opposite of what the rest of this server does with them. It isn't a leak of
    the kind CLAUDE.md's invariant guards against -- that one is about text from inside
    an imaging library reaching an arbitrary caller through an exception. Here the path
    is the entire payload: the worker's job is to read that file. It stays safe because
    the path only ever names a file this server wrote inside its own regions directory,
    and because this route is gated -- it is not in Caddyfile.example's `@public`
    matcher, which tests/test_proxy_config.py exists to keep true.
    """
    if MAILBOX is None:
        return _vlm_disabled()

    job = await MAILBOX.claim(VLM_POLL_S)
    if job is None:
        return Response(status_code=204)
    return {
        "job_id": job.id,
        "image_path": str(job.path),
        "width": job.width,
        "height": job.height,
    }


@app.post("/mfluxible/v1/vlm/{job_id}")
async def vlm_complete(job_id: str, result: VlmResult):
    if MAILBOX is None:
        return _vlm_disabled()

    delivered = MAILBOX.complete(job_id, result.model_dump())
    if not delivered:
        # Not an error worth a 4xx on the worker's side: the usual cause is the user
        # clicking the button again, which supersedes the job this worker was running.
        return {"delivered": False, "reason": "that job is no longer the pending one."}
    return {"delivered": True}


def _openai_error(
    status_code: int, message: str, error_type: str = "invalid_request_error", code: str | None = None
) -> JSONResponse:
    # https://platform.openai.com/docs/guides/error-codes -- an OpenAI client's error
    # handling reads .error.message, not the mfluxible native shape's top-level .message.
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )


# (width, height), or None if `size` isn't a shape this understands -- returned rather
# than raised so the caller composes the 400 itself, keeping this module free of the
# exception-to-response-body paths _collect_final_image's docstring explains.
def _parse_openai_size(size: str) -> tuple[int, int] | None:
    if size.strip().lower() == "auto":
        return 1024, 1024
    parts = size.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def _bad_size_error(size: str) -> JSONResponse:
    return _openai_error(400, f"size must look like '1024x1024', got {size!r}")


def _preview_every_from_partial_images(partial_images: int) -> int:
    # OpenAI's partial_images is a total count, not a stride -- approximated by
    # spacing previews evenly across the model's default step count, since neither
    # OpenAI request shape below has a `steps` field for a client to have overridden it
    # with (see OpenAIImageGenerationRequest's docstring; `steps` is left unset for the
    # same reason mfluxible's own bundled clients leave it null -- whichever model is
    # loaded picks its own default, see CLAUDE.md).
    return max(1, engine.spec.default_steps // partial_images) if partial_images else 0


def _openai_to_generate_request(req: OpenAIImageGenerationRequest, width: int, height: int) -> GenerateRequest:
    return GenerateRequest(
        prompt=req.prompt,
        width=width,
        height=height,
        preview_every=_preview_every_from_partial_images(req.partial_images),
        stream=req.stream,
    )


async def _openai_sse(req: GenerateRequest, created: int):
    partial_index = 0
    async for event in engine.generate_stream(req):
        if event["type"] == "thinking" and "preview" in event:
            yield f"data: {json.dumps({'type': 'image_generation.partial_image', 'b64_json': event['preview'], 'partial_image_index': partial_index, 'created_at': created})}\n\n"
            partial_index += 1
        elif event["type"] == "image":
            yield f"data: {json.dumps({'type': 'image_generation.completed', 'b64_json': event['data'], 'created_at': created})}\n\n"
        elif event["type"] == "error":
            yield f"data: {json.dumps({'error': {'message': event['message'], 'type': 'api_error', 'param': None, 'code': None}})}\n\n"
    # Deliberately no trailing [DONE] sentinel: unlike chat completions streaming,
    # OpenAI's own image-generation stream ending on `image_generation.completed`
    # without one isn't independently confirmed here (their public docs don't show
    # the raw wire format) -- don't add one on a guess.


def _openai_model_object() -> dict:
    return {
        "id": engine.spec.key,
        "object": "model",
        "created": MODEL_LOADED_AT,
        "owned_by": "mfluxible",
    }


@app.get("/v1/models")
async def openai_list_models():
    # https://platform.openai.com/docs/api-reference/models/list -- always exactly one
    # entry, since one model runs per process (see the module docstring). `owned_by`
    # has no real mfluxible equivalent to OpenAI's org-id convention; "mfluxible" names
    # what's actually serving it rather than leaving it blank or fabricating an org.
    return {"object": "list", "data": [_openai_model_object()]}


@app.get("/v1/models/{model_id}")
async def openai_retrieve_model(model_id: str):
    # https://platform.openai.com/docs/api-reference/models/retrieve -- only the one
    # ID that GET /v1/models just listed resolves; anything else 404s the same way
    # OpenAI's own API does for an unknown model, "model_not_found" code included,
    # rather than a generic 404 a client's error handling might not recognize.
    if model_id != engine.spec.key:
        return _openai_error(
            404,
            f"The model '{model_id}' does not exist -- this server is running "
            f"{engine.spec.key!r} ({engine.spec.label}); see GET /v1/models.",
            code="model_not_found",
        )
    return _openai_model_object()


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: ChatCompletionRequest):
    # No model to validate `req.model` against the way the other endpoints do -- this
    # isn't the diffusion model responding, so there's nothing to check compatibility
    # with; see chat_stub.py for what this is actually for and why it's deliberately
    # not a real chat model.
    if req.stream:
        return StreamingResponse(chat_stub.stream_response_lines(req), media_type="text/event-stream")
    return chat_stub.non_streaming_response(req)


# Not CORS, and not reachable through the CORS middleware above: Starlette only
# treats an OPTIONS request as a preflight when it carries Access-Control-Request-
# Method, and the caller this exists for is a server-side fetch that sends no such
# header, so without a route here it falls through to a 405. SillyTavern's
# image-generation extension uses exactly this -- a bare `OPTIONS <base>/v1/images/
# generations`, checking only the status -- as the reachability probe behind its
# "Validate" button (`src/endpoints/stable-diffusion.js`, the sdcpp router's /ping),
# so answering it is what makes this server selectable there at all. See
# POST /sdapi/v1/txt2img at the bottom of this file for the rest of that story.
#
# Deliberately *not* gated on engine.model being loaded: this answers "does this
# endpoint exist here", which is true from the moment the process is serving. Whether
# the weights have finished downloading is /health's `model_loaded` to report.
@app.options("/v1/images/generations")
async def openai_generate_options() -> Response:
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})


# A genuine OpenAI Images-API-compatible endpoint (see
# https://platform.openai.com/docs/api-reference/images/create), for pointing an
# existing OpenAI-client-based tool (e.g. Open WebUI's "OpenAI" image engine, which
# takes an arbitrary base URL) at this server without modifying it. Deliberately a
# strict subset: fields mflux has no equivalent for (quality, style, background,
# output_format, output_compression, moderation, user) are accepted and ignored
# rather than faked, and requests this can't honestly satisfy (model mismatch, n > 1,
# a `url` response_format this server can't host) are rejected with a 400 rather than
# silently approximated.
@app.post("/v1/images/generations")
async def openai_generate(req: OpenAIImageGenerationRequest):
    if req.model != engine.spec.key:
        return _openai_error(
            400,
            f"model {req.model!r} is not loaded -- this server is running "
            f"{engine.spec.key!r} ({engine.spec.label}); one model runs per process (see /health).",
        )
    if req.n != 1:
        return _openai_error(400, "n must be 1 -- mfluxible generates one image per request.")
    if req.response_format != "b64_json":
        return _openai_error(
            400,
            f"response_format {req.response_format!r} is not supported -- only 'b64_json' is "
            "(mfluxible does not host images for a 'url' response).",
        )

    parsed_size = _parse_openai_size(req.size)
    if parsed_size is None:
        return _bad_size_error(req.size)
    gen_req = _openai_to_generate_request(req, *parsed_size)

    problem = engine.request_problem(gen_req)
    if problem is not None:
        return _openai_error(400, problem)

    created = int(time.time())
    if gen_req.stream:
        return StreamingResponse(_openai_sse(gen_req, created), media_type="text/event-stream")

    final = await _collect_final_image(gen_req)
    if final["type"] == "error":
        return _openai_error(500, final["message"], error_type="api_error")

    return {"created": created, "data": [{"b64_json": final["data"]}]}


# A genuine OpenAI Images-Edit-API-compatible endpoint (see
# https://platform.openai.com/docs/api-reference/images/createEdit), for image-to-image
# via an existing OpenAI-client-based tool. Multipart form data, like OpenAI's own --
# not JSON, hence plain Form()/File() parameters here rather than a pydantic request
# model the way every other endpoint uses one.
#
# This maps onto mflux's strength-based img2img (see README's "Image-to-image" section
# on /mfluxible/v1/images/generations), and `mask` is still a 400 here even though the
# native endpoint now inpaints -- for a different reason than it used to be. The two
# APIs disagree about what a mask *is*: OpenAI's is read from the alpha channel and
# marks the editable region by being **transparent** there, while the native `mask`
# field is read as luminance and marks it **white** (see engine._decode_mask). Feeding
# one to the other doesn't fail, it inpaints the complement -- every region the caller
# meant to keep -- which is the worst shape a failure can take. Honouring OpenAI's
# convention is a real piece of work (alpha handling, and the flatten-to-opaque choice
# for a fully opaque mask), not an alias, so until it exists this refuses and names the
# endpoint that does support one. `image_strength` isn't part of OpenAI's request shape
# at all, so it's accepted as a non-standard extension the same way `partial_images`
# already is on `POST /v1/images/generations` above -- mfluxible's own default (0.4)
# applies if it's left out.
@app.post("/v1/images/edits")
async def openai_edit_image(
    prompt: str = Form(...),
    model: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
    n: int = Form(default=1),
    size: str = Form(default="1024x1024"),
    response_format: str = Form(default="b64_json"),
    stream: bool = Form(default=False),
    partial_images: int = Form(default=0, ge=0, le=3),
    image_strength: float | None = Form(default=None),
):
    if model != engine.spec.key:
        return _openai_error(
            400,
            f"model {model!r} is not loaded -- this server is running "
            f"{engine.spec.key!r} ({engine.spec.label}); one model runs per process (see /health).",
        )
    if n != 1:
        return _openai_error(400, "n must be 1 -- mfluxible generates one image per request.")
    if response_format != "b64_json":
        return _openai_error(
            400,
            f"response_format {response_format!r} is not supported -- only 'b64_json' is "
            "(mfluxible does not host images for a 'url' response).",
        )
    if mask is not None:
        return _openai_error(
            400,
            "mask is not supported on this endpoint -- OpenAI marks the editable region "
            "with transparency, and reading it the way mfluxible's own mask field is read "
            "(white = regenerate) would inpaint exactly the region you meant to keep. Use "
            "POST /mfluxible/v1/images/generations, which takes `mask` as a base64 image, "
            "for masked inpainting; this endpoint does whole-image edits only.",
        )

    parsed_size = _parse_openai_size(size)
    if parsed_size is None:
        return _bad_size_error(size)
    width, height = parsed_size

    image_bytes = await image.read()
    gen_req = GenerateRequest(
        prompt=prompt,
        width=width,
        height=height,
        preview_every=_preview_every_from_partial_images(partial_images),
        stream=stream,
        image=base64.b64encode(image_bytes).decode("ascii"),
        image_strength=image_strength,
    )

    problem = engine.request_problem(gen_req)
    if problem is not None:
        return _openai_error(400, problem)

    created = int(time.time())
    if gen_req.stream:
        return StreamingResponse(_openai_sse(gen_req, created), media_type="text/event-stream")

    final = await _collect_final_image(gen_req)
    if final["type"] == "error":
        return _openai_error(500, final["message"], error_type="api_error")

    return {"created": created, "data": [{"b64_json": final["data"]}]}


def _a1111_error(message: str, status_code: int = 400) -> JSONResponse:
    # A1111's own handled-error envelope shape (a short `error` plus a `detail`),
    # minus the fields it fills from a traceback -- which this server would have
    # nothing to put in anyway, that being the point of _collect_final_image's
    # docstring. Returned, never raised, for the same reason as everything else here.
    code = "invalid_request" if status_code < 500 else "generation_failed"
    return JSONResponse(status_code=status_code, content={"error": code, "detail": message})


def _a1111_to_generate_request(req: A1111Txt2ImgRequest) -> GenerateRequest:
    """The A1111 payload narrowed to what the loaded model can actually act on.

    Mirrors the two rules request_problem enforces -- guidance needs
    `supports_guidance`; a negative prompt needs both a negative branch and
    classifier-free guidance actually switched on above CFG_GUIDANCE_FLOOR -- as
    *drops* rather than rejections, for the reason in the endpoint's comment below.

    That duplication is deliberate but not unchecked: the endpoint still runs
    request_problem on what this returns, so if those rules ever gain a third clause,
    this drifts into a visible 400 rather than a silently wrong image.
    `test_a1111_shim_never_builds_a_request_its_own_model_rejects` pins that across
    every model in the table.
    """
    spec = engine.spec
    guidance = req.cfg_scale if spec.supports_guidance else None

    negative_prompt = (req.negative_prompt or "").strip()
    if negative_prompt and not spec.supports_negative_prompt:
        negative_prompt = ""
    if negative_prompt:
        # Having a negative branch isn't sufficient on its own -- CFG has to be on for
        # the unconditional prompt to be encoded at all, so at or below the floor this
        # would be accepted and never consulted. Same check as request_problem's, and
        # the reason it tests the *effective* guidance rather than the requested one.
        effective = guidance if guidance is not None else spec.default_guidance
        if effective is not None and effective <= CFG_GUIDANCE_FLOOR:
            negative_prompt = ""

    return GenerateRequest(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        # A1111 spells "pick one for me" as -1, GenerateRequest spells it None; passing
        # -1 through would be a literal seed, and a reproducible one at that.
        seed=req.seed if req.seed is not None and req.seed >= 0 else None,
        guidance=guidance,
        negative_prompt=negative_prompt or None,
        stream=False,
    )


# An AUTOMATIC1111-shaped txt2img, narrow on purpose. It exists for SillyTavern's
# image-generation extension, which as of 1.18.0 has no OpenAI-compatible image source
# at all -- its `openai` source hardcodes api.openai.com inside SillyTavern's own
# backend with no base-URL setting, unlike the "Custom (OpenAI-compatible)" source on
# its chat side. Its `sdcpp` source (for stable-diffusion.cpp's server) is the one
# local-URL source that fits, and it is a hybrid: OpenAI-shaped discovery, A1111-shaped
# generation. It probes `OPTIONS /v1/images/generations` (see that handler above),
# reads the model list from `GET /v1/models` -- already OpenAI-shaped here, and parsed
# there as `data.data.map(m => ({value: m.id, text: m.name || m.id}))` -- and then
# posts *this* shape, reading `images[0]` as a base64 PNG back out. Read out of
# SillyTavern's own source rather than inferred from behavior, the same way the mflux
# and mcp notes in CLAUDE.md were: `public/scripts/extensions/stable-diffusion/
# index.js` (generateSdcppImage) and `src/endpoints/stable-diffusion.js` (sdcpp router).
#
# **Why this endpoint ignores what it cannot do, where the native API rejects it.**
# All three of SillyTavern's sdcpp routes end in `response.sendStatus(500)`, attaching
# the upstream body to an Error's `cause` that is only ever console.error'd on its
# server. So a 400 from here reaches the user as a bare failed generation with no
# reason attached: request_problem's carefully-written messages ("Z-Image-Turbo ignores
# guidance; omit the field") land nowhere a user will look. And this caller sends
# cfg_scale and steps on *every* request, from its own sliders, whose defaults (7 and
# 20) suit neither a guidance-distilled model nor a 9-step one. Rejecting those would
# make the source unusable rather than correcting anyone. Hence: drop what the loaded
# model can't act on, and report what actually took effect in `info`, which is the one
# channel that survives the trip.
#
# The line between ignoring and rejecting is whether the caller could otherwise detect
# the difference. A dropped cfg_scale changes the image, but the caller named no
# specific image and has nothing to compare against; batch_size=4 answered with one
# image is a concretely wrong result to a request that named a number. So the first is
# ignored and the second is a 400 -- one SillyTavern can never trigger, since it
# hardcodes batch_size: 1.
#
# `model` is ignored rather than validated, which is the one place this diverges from
# `POST /v1/images/generations` above. One model runs per process, so there is nothing
# to switch to either way; the difference is that SillyTavern only auto-selects from a
# freshly loaded model list when its stored `sd.model` is *empty* (`index.js`: `if
# (!extension_settings.sd.model && models.length > 0)`), so a value left over from a
# previously configured source survives, is sent here, and matches no option in the
# dropdown the user is looking at. Validating it would turn that invisible stale
# setting into an unexplained failure. What actually ran is reported as
# `sd_model_name` in `info` instead, and `GET /v1/models` remains the honest answer to
# what this server has loaded.
#
# Not implemented, and not needed by this caller: /sdapi/v1/progress and
# /sdapi/v1/interrupt (SillyTavern's `auto` source polls and posts those; its `sdcpp`
# source does neither), along with the rest of A1111's endpoint surface. Adding the
# `auto` source instead would mean roughly ten endpoints including a GET and a POST of
# /sdapi/v1/options -- see docs/clients.md.
@app.post("/sdapi/v1/txt2img")
async def a1111_txt2img(req: A1111Txt2ImgRequest):
    if req.batch_size != 1 or req.n_iter != 1:
        return _a1111_error(
            "mfluxible generates one image per request; batch_size and n_iter must both be 1."
        )

    gen_req = _a1111_to_generate_request(req)

    problem = engine.request_problem(gen_req)
    if problem is not None:
        # Only reachable if the drops above have drifted out of sync with
        # request_problem's rules -- see _a1111_to_generate_request's docstring.
        return _a1111_error(problem)

    final = await _collect_final_image(gen_req)
    if final["type"] == "error":
        return _a1111_error(final["message"], status_code=500)

    return {
        "images": [final["data"]],
        "parameters": req.model_dump(),
        # A1111 puts a JSON *string* here, and it is the only part of the response that
        # says what happened rather than what was asked. So it reports the effective
        # values, not an echo of the request: the seed that actually ran, the steps
        # that actually ran, and guidance/negative_prompt as they survived the drops
        # above -- `cfg_scale: null` against a request that sent 7 is how a caller can
        # see that this model ignored it. `sd_model_name` is what `model` would have
        # been validated against if this endpoint validated it.
        "info": json.dumps(
            {
                "prompt": gen_req.prompt,
                "negative_prompt": gen_req.negative_prompt or "",
                "seed": final["seed"],
                "all_seeds": [final["seed"]],
                "width": gen_req.width,
                "height": gen_req.height,
                "steps": gen_req.steps if gen_req.steps is not None else engine.spec.default_steps,
                "cfg_scale": gen_req.guidance,
                "sd_model_name": engine.spec.key,
            }
        ),
    }
