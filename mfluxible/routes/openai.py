"""The OpenAI-compatible surface: models, image generation and image edits."""

import base64
import json
import time

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from mfluxible import runtime as rt
from mfluxible.routes.common import collect_final_image
from mfluxible.schemas import MAX_SIDE, MIN_SIDE, GenerateRequest, OpenAIImageGenerationRequest

router = APIRouter()


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
# exception-to-response-body paths collect_final_image's docstring explains.
def _parse_openai_size(size: str) -> tuple[int, int] | None:
    if size.strip().lower() == "auto":
        return 1024, 1024
    parts = size.lower().split("x")
    if len(parts) != 2 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    width, height = int(parts[0]), int(parts[1])
    # Out of range is "not a shape this understands" too: GenerateRequest would refuse it,
    # and constructing one from an unchecked size would raise instead of returning a 400.
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        return None
    return width, height


def _bad_size_error(size: str) -> JSONResponse:
    return _openai_error(
        400, f"size must look like '1024x1024' with each side {MIN_SIDE}-{MAX_SIDE}, got {size!r}"
    )


def _preview_every_from_partial_images(partial_images: int) -> int:
    # OpenAI's partial_images is a total count, not a stride -- approximated by
    # spacing previews evenly across the model's default step count, since neither
    # OpenAI request shape below has a `steps` field for a client to have overridden it
    # with (see OpenAIImageGenerationRequest's docstring; `steps` is left unset for the
    # same reason mfluxible's own bundled clients leave it null -- whichever model is
    # loaded picks its own default, see CLAUDE.md).
    return max(1, rt.engine.spec.default_steps // partial_images) if partial_images else 0


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
    async for event in rt.engine.generate_stream(req):
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


def _unsupported_request(model: str, n: int, response_format: str) -> JSONResponse | None:
    """The 400 for a request shape this server can't honour, or None. The three checks the
    generation and edit endpoints share: one model runs per process, one image comes back
    per call, and only base64 is hosted (see the comment above `openai_generate`)."""
    if model != rt.engine.spec.key:
        return _openai_error(
            400,
            f"model {model!r} is not loaded -- this server is running "
            f"{rt.engine.spec.key!r} ({rt.engine.spec.label}); one model runs per process (see /health).",
        )
    if n != 1:
        return _openai_error(400, "n must be 1 -- mfluxible generates one image per request.")
    if response_format != "b64_json":
        return _openai_error(
            400,
            f"response_format {response_format!r} is not supported -- only 'b64_json' is "
            "(mfluxible does not host images for a 'url' response).",
        )
    return None


async def _generate_and_respond(gen_req: GenerateRequest):
    """Validate, then answer in OpenAI's shapes: a stream of partial/completed events, or
    one JSON body. Shared by the generation and edit endpoints, which differ only in how
    they build `gen_req`."""
    problem = rt.engine.request_problem(gen_req)
    if problem is not None:
        return _openai_error(400, problem)

    created = int(time.time())
    if gen_req.stream:
        return StreamingResponse(_openai_sse(gen_req, created), media_type="text/event-stream")

    final = await collect_final_image(gen_req)
    if final["type"] == "error":
        return _openai_error(500, final["message"], error_type="api_error")

    return {"created": created, "data": [{"b64_json": final["data"]}]}


def _openai_model_object() -> dict:
    return {
        "id": rt.engine.spec.key,
        "object": "model",
        "created": rt.MODEL_LOADED_AT,
        "owned_by": "mfluxible",
    }


@router.get("/v1/models")
async def openai_list_models():
    # https://platform.openai.com/docs/api-reference/models/list -- always exactly one
    # entry, since one model runs per process (see the module docstring). `owned_by`
    # has no real mfluxible equivalent to OpenAI's org-id convention; "mfluxible" names
    # what's actually serving it rather than leaving it blank or fabricating an org.
    return {"object": "list", "data": [_openai_model_object()]}


@router.get("/v1/models/{model_id}")
async def openai_retrieve_model(model_id: str):
    # https://platform.openai.com/docs/api-reference/models/retrieve -- only the one
    # ID that GET /v1/models just listed resolves; anything else 404s the same way
    # OpenAI's own API does for an unknown model, "model_not_found" code included,
    # rather than a generic 404 a client's error handling might not recognize.
    if model_id != rt.engine.spec.key:
        return _openai_error(
            404,
            f"The model '{model_id}' does not exist -- this server is running "
            f"{rt.engine.spec.key!r} ({rt.engine.spec.label}); see GET /v1/models.",
            code="model_not_found",
        )
    return _openai_model_object()


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
@router.options("/v1/images/generations")
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
@router.post("/v1/images/generations")
async def openai_generate(req: OpenAIImageGenerationRequest):
    refusal = _unsupported_request(req.model, req.n, req.response_format)
    if refusal is not None:
        return refusal

    parsed_size = _parse_openai_size(req.size)
    if parsed_size is None:
        return _bad_size_error(req.size)
    gen_req = _openai_to_generate_request(req, *parsed_size)

    return await _generate_and_respond(gen_req)


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
@router.post("/v1/images/edits")
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
    refusal = _unsupported_request(model, n, response_format)
    if refusal is not None:
        return refusal
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

    return await _generate_and_respond(gen_req)
