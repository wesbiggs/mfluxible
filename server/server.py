"""FastAPI server exposing a streaming image-generation endpoint over mflux.

One process runs one model, chosen at startup with MFLUXIBLE_MODEL (see models.py
for the table). Weights are only fetched for the model actually selected.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import chat_stub
from engine import MfluxEngine
from models import MODELS
from schemas import ChatCompletionRequest, GenerateRequest, OpenAIImageGenerationRequest

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

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX or None,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# server/ and client/ live in the same repo checkout but stay dependency-independent
# (see CLAUDE.md) -- this reaches across that boundary only to serve a static file, not
# to import anything, so it doesn't compromise that separation.
HARNESS_PATH = Path(__file__).resolve().parent.parent / "client" / "harness.html"


@app.get("/harness.html")
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
    """Runs a generation to completion and returns the final `image` event. Raises
    RuntimeError (carrying the engine's own message) if generation ends in an `error`
    event instead -- both endpoints below turn that into their own error shape rather
    than sharing a response type, since the native and OpenAI-compat contracts
    disagree on what an error body looks like."""
    async for event in engine.generate_stream(req):
        if event["type"] == "image":
            return event
        if event["type"] == "error":
            raise RuntimeError(event["message"])
    raise RuntimeError("generation ended without an image or error event")


# This is mfluxible's own API -- streaming `thinking` events with step timings and
# previews, guidance/negative_prompt, etc. -- moved off /v1/images/generations so that
# path can be a genuine OpenAI-compatible endpoint instead (see below). Every bundled
# client (harness.html, stream_client.py/js, mcp_server.py) targets this path.
@app.post("/mfluxible/v1/images/generations")
async def generate(req: GenerateRequest):
    # Checked before the response begins: for a stream, raising once StreamingResponse
    # has started would mean a torn body with a 200 already on the wire.
    try:
        engine.check_request(req)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"type": "error", "message": str(exc)})

    if req.stream:
        return StreamingResponse(_sse(req), media_type="text/event-stream")

    try:
        return await _collect_final_image(req)
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"type": "error", "message": str(exc)})


def _openai_error(
    status_code: int, message: str, error_type: str = "invalid_request_error", code: str | None = None
) -> JSONResponse:
    # https://platform.openai.com/docs/guides/error-codes -- an OpenAI client's error
    # handling reads .error.message, not the mfluxible native shape's top-level .message.
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )


def _parse_openai_size(size: str) -> tuple[int, int]:
    if size.strip().lower() == "auto":
        return 1024, 1024
    parts = size.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"size must look like '1024x1024', got {size!r}")
    return int(parts[0]), int(parts[1])


def _openai_to_generate_request(req: OpenAIImageGenerationRequest) -> GenerateRequest:
    width, height = _parse_openai_size(req.size)
    # OpenAI's partial_images is a total count, not a stride -- approximated by
    # spacing previews evenly across the model's default step count, since the OpenAI
    # request shape has no `steps` field for a client to have overridden it with (see
    # OpenAIImageGenerationRequest's docstring; `steps` is left unset below for the
    # same reason mfluxible's own bundled clients leave it null -- whichever model is
    # loaded picks its own default, see CLAUDE.md).
    preview_every = max(1, engine.spec.default_steps // req.partial_images) if req.partial_images else 0
    return GenerateRequest(prompt=req.prompt, width=width, height=height, preview_every=preview_every, stream=req.stream)


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

    try:
        gen_req = _openai_to_generate_request(req)
    except ValueError as exc:
        return _openai_error(400, str(exc))

    try:
        engine.check_request(gen_req)
    except ValueError as exc:
        return _openai_error(400, str(exc))

    created = int(time.time())
    if gen_req.stream:
        return StreamingResponse(_openai_sse(gen_req, created), media_type="text/event-stream")

    try:
        final = await _collect_final_image(gen_req)
    except RuntimeError as exc:
        return _openai_error(500, str(exc), error_type="api_error")

    return {"created": created, "data": [{"b64_json": final["data"]}]}
