"""FastAPI server exposing a streaming image-generation endpoint over mflux's Z-Image-Turbo."""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from engine import ZImageEngine
from schemas import GenerateRequest

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

engine = ZImageEngine(quantize=QUANTIZE, lora_paths=LORA_PATHS, lora_scales=LORA_SCALES)


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
    return {"status": "ok", "model_loaded": engine.model is not None}


async def _sse(req: GenerateRequest):
    async for event in engine.generate_stream(req):
        yield f"data: {json.dumps(event)}\n\n"


@app.post("/v1/images/generations")
async def generate(req: GenerateRequest):
    if req.stream:
        return StreamingResponse(_sse(req), media_type="text/event-stream")

    final = None
    async for event in engine.generate_stream(req):
        if event["type"] == "image":
            final = event
        elif event["type"] == "error":
            return JSONResponse(status_code=500, content=event)
    return final
