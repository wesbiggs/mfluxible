"""FastAPI server exposing a streaming image-generation endpoint over mflux.

One process runs one model, chosen at startup with MFLUXIBLE_MODEL (see models.py
for the table). Weights are only fetched for the model actually selected.
"""

import json
import os
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from engine import MfluxEngine
from models import MODELS
from schemas import GenerateRequest

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.load()
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


@app.post("/v1/images/generations")
async def generate(req: GenerateRequest):
    # Checked before the response begins: for a stream, raising once StreamingResponse
    # has started would mean a torn body with a 200 already on the wire.
    try:
        engine.check_request(req)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"type": "error", "message": str(exc)})

    if req.stream:
        return StreamingResponse(_sse(req), media_type="text/event-stream")

    final = None
    async for event in engine.generate_stream(req):
        if event["type"] == "image":
            final = event
        elif event["type"] == "error":
            return JSONResponse(status_code=500, content=event)
    return final
