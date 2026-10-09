"""`/v1/chat/completions` (the one-turn stub) and the output files it links."""

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from mfluxible import chat_stub, runtime as rt
from mfluxible.schemas import ChatCompletionRequest

router = APIRouter()


def _output_file(name: str) -> Path | None:
    """`name` inside rt.OUTPUT_DIR if it is an image file directly in it, else None.

    Names only, never paths: anything with a separator, a leading dot or a non-image
    suffix is refused outright, and the resolved file must still sit in the directory
    (a symlink out of it is not served). This is an unauthenticated-by-itself read of
    the host's disk, so what it is allowed to name is kept as small as possible."""
    if not name or name != Path(name).name or name.startswith(".") or Path(name).suffix.lower() not in rt.OUTPUT_SUFFIXES:
        return None
    path = (rt.OUTPUT_DIR / name).resolve()
    if path.parent != rt.OUTPUT_DIR.resolve() or not path.is_file():
        return None
    return path


@router.get("/mfluxible/v1/output/{name}")
async def output_image(name: str):
    path = _output_file(name)
    if path is None:
        raise HTTPException(status_code=404, detail="no such image")
    return FileResponse(path)


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: ChatCompletionRequest, request: Request):
    # No model to validate `req.model` against the way the other endpoints do -- this
    # isn't the diffusion model responding, so there's nothing to check compatibility
    # with; see chat_stub.py for what this is actually for and why it's deliberately
    # not a real chat model.
    base = str(request.base_url).rstrip("/")

    def image_url(name: str) -> str | None:
        return f"{base}/mfluxible/v1/output/{quote(name)}" if _output_file(name) else None

    if req.stream:
        return StreamingResponse(chat_stub.stream_response_lines(req, image_url), media_type="text/event-stream")
    return chat_stub.non_streaming_response(req, image_url)
