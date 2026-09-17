from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int | None = Field(
        default=None,
        description="Denoising steps. Defaults to the configured model's own default (see /health).",
    )
    seed: int | None = None
    guidance: float | None = Field(
        default=None,
        description=(
            "Classifier-free guidance scale. Only accepted by models that use guidance "
            "(see /health); rejected outright by guidance-distilled ones rather than "
            "silently ignored."
        ),
    )
    negative_prompt: str | None = Field(
        default=None,
        description=(
            "What to steer away from. Only accepted by models with a negative branch "
            "(see /health); rejected outright by the others."
        ),
    )
    preview_every: int = Field(
        default=0,
        description="Decode and include an in-progress preview image every N steps. 0 disables previews.",
    )
    stream: bool = True
    image: str | None = Field(
        default=None,
        description=(
            "Base64-encoded input image (no data: URI prefix), for image-to-image. Loaded, "
            "scaled to width/height, and blended with noise per image_strength before the "
            "first denoising step -- every model this server can run accepts it, since mflux's "
            "ZImage/Flux1/QwenImage generate_image() all share this parameter. Must be paired "
            "with image_strength; rejected with a 400 if it isn't valid base64 or isn't a "
            "decodable image."
        ),
    )
    image_strength: float | None = Field(
        default=None,
        description=(
            "How strongly the input image constrains the output, in [0.0, 1.0] -- mflux's own "
            "convention, which is the *inverse* of some other img2img tools' 'denoising "
            "strength': 0.0 means the image has no influence (equivalent to plain text-to-image); "
            "1.0 means maximum influence, which can mean very few or even zero denoising steps "
            "actually run, so the output stays close to the input. Only meaningful, and only "
            "accepted, alongside image. Omitted, it defaults to 0.4 (mflux's own CLI default) "
            "-- or to 0.0 when a mask is also set, because there the field governs only how "
            "much of the old content inside the masked region survives, and 0.4 brings the "
            "thing that was meant to be replaced back very nearly intact."
        ),
    )

    mask: str | None = Field(
        default=None,
        description=(
            "Base64-encoded mask image (no data: URI prefix) selecting the region to "
            "regenerate: white where the model is free to change the image, black where "
            "the input must be preserved, greys crossfading between the two. Must be the "
            "same pixel dimensions as `image`, and only accepted alongside it. Everything "
            "black is held to the input image at every denoising step, so it comes back "
            "unchanged -- but the model still sees it, which is what lets the new content "
            "match the original's lighting and perspective. Only accepted on models "
            "running mflux's linear schedule (`supports_mask` on /health)."
        ),
    )
    mask_feather: int = Field(
        default=0,
        ge=0,
        description=(
            "Gaussian blur radius, in input-image pixels, applied to `mask` before it is "
            "used. Its job is to hide the latent grid's 8-pixel staircase along a mask "
            "edge, and it wants to stay within a few cells of that: 8-16 is the useful "
            "range and 0 disables it. It is NOT a way to blend two regions together. A "
            "grey mask value means 'hold this pixel partway to the original at every "
            "denoising step', so a wide feather pins a wide band to a ghost of the input "
            "and the result cross-fades the old content with the new rather than "
            "replacing it -- at radius 96 on a 1024px image, visibly a double exposure. "
            "Only accepted alongside `mask`."
        ),
    )
    mask_composite: bool = Field(
        default=True,
        description=(
            "Paste the generated image back over the input through `mask`, so the pixels "
            "outside it are byte-identical to what was sent. The masked blend already "
            "holds that region during denoising, but it still passes through the VAE, "
            "which shifts it by roughly 1.5/255 on average -- invisible on a photograph "
            "and not on text or a logo. Set false to get the model's own decode of the "
            "whole frame. Only accepted alongside `mask`."
        ),
    )

    fractional_start: bool = Field(
        default=False,
        description=(
            "Start image-to-image between two steps of the sigma schedule instead of "
            "flooring to one. image_strength normally reaches the model only as an int "
            "(mflux's init_time_step), quantizing it to 1/steps -- at 9 steps, 0.35 and "
            "0.4 are the same image. With this set, the noise level the input is blended "
            "to is placed at the exact position the strength names, while the loop still "
            "starts on -- and runs -- the same whole steps, so finer strength control "
            "costs no extra compute. Only meaningful, and only accepted, alongside image. "
            "Off by default: it changes the pixels a given strength produces, so an "
            "existing seed/strength pair keeps reproducing its old image unless asked."
        ),
    )


class OpenAIImageGenerationRequest(BaseModel):
    """Request body for the OpenAI-compatible `POST /v1/images/generations` endpoint
    (mirrors https://platform.openai.com/docs/api-reference/images/create). Only
    fields mflux can actually act on are modeled; everything else OpenAI's schema
    defines (quality, style, background, output_format, output_compression,
    moderation, user) is accepted and silently dropped -- pydantic ignores unknown
    fields by default, the same way an older OpenAI-compatible server would ignore a
    newer client's extra fields.

    `model` and `n` exist only to be validated against, not acted on: one model runs
    per process (see server.py's module docstring), so `model` must name the model
    actually loaded (see /health) rather than silently substituting, and `n` must be
    1 since mflux generates one image per call.
    """

    prompt: str
    model: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    stream: bool = False
    partial_images: int = Field(
        default=0,
        ge=0,
        le=3,
        description=(
            "How many in-progress previews to emit while streaming (OpenAI's semantics: "
            "a total count, not a stride). Translated to mflux's preview_every by dividing "
            "it into the model's default step count -- an approximation, since OpenAI's "
            "own image models don't expose a step count to divide by either."
        ),
    )


class ChatMessage(BaseModel):
    """One message in an OpenAI-compatible `POST /v1/chat/completions` request. Loose
    on purpose: `content` covers both a plain string and OpenAI's multimodal
    list-of-parts form (only the text parts of which chat_stub.py reads), and a tool
    result message's own fields (`tool_call_id`, `name`) are accepted without being
    acted on -- chat_stub.py only needs to know *that* the last message is a tool
    result, not which tool or what it returned."""

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for `POST /v1/chat/completions`. See chat_stub.py -- this isn't a
    real chat model; the whole point is a stub that plays exactly one Native-mode
    function-calling turn, for a client like Open WebUI where the "chat model" only
    exists to trigger mfluxible's own image generation as a tool call. `tools` is
    modeled because chat_stub.py's response depends on whether the caller actually
    offered a `generate_image` tool; everything else OpenAI's schema defines
    (temperature, top_p, tool_choice, ...) is accepted and ignored."""

    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    stream: bool = False


class A1111Txt2ImgRequest(BaseModel):
    """Request body for the AUTOMATIC1111-shaped `POST /sdapi/v1/txt2img` shim -- see
    server.py for what that endpoint exists for (SillyTavern's image-generation
    extension, whose `sdcpp` source posts this shape).

    Every field a caller can send that mflux has no equivalent for -- `sampler_name`,
    `scheduler`, `clip_skip`, and the rest of A1111's much larger payload -- is
    accepted and ignored rather than rejected, which is the opposite of how
    `GenerateRequest` treats a field the loaded model can't act on. The reason is in
    server.py's comment on the endpoint: this caller never shows the user a 400's
    message, so rejecting a field it sends on every request would be an unexplained
    failure rather than the useful correction it is on the native API.

    Defaults are A1111's own where mflux has a matching concept and `None` where it
    doesn't: `steps` and `cfg_scale` default to None rather than A1111's 50 and 7 so
    an omitted field means "whatever the loaded model picks" (as on GenerateRequest),
    not a number this schema invented.
    """

    prompt: str = ""
    negative_prompt: str | None = None
    width: int = 512
    height: int = 512
    steps: int | None = None
    cfg_scale: float | None = None
    # A1111 spells "random" as -1 here, where GenerateRequest spells it None; the
    # endpoint maps between them rather than passing -1 through as a literal seed.
    seed: int | None = None
    batch_size: int = 1
    n_iter: int = 1
    # Accepted, never acted on -- one model runs per process, so there is nothing to
    # switch to. See the endpoint's comment for why this one is ignored rather than
    # validated the way OpenAIImageGenerationRequest.model is.
    model: str | None = None


class RegionDetectRequest(BaseModel):
    """Request body for `POST /mfluxible/v1/regions/detect` -- the harness handing over
    the image it already has loaded, so a worker can be told where to find it.

    Just the image: the server measures its own dimensions (with EXIF orientation
    applied, since that is the frame a mask is built in) rather than trusting a
    caller's idea of them, and there is nothing else to configure per detection."""

    image: str = Field(description="Base64-encoded image, no data: URI prefix.")


class Region(BaseModel):
    """One named thing a detection found, with its extent as fractions of the frame.

    Fractions rather than pixels, for the reason the MCP tool's mask_boxes are: the
    worker may be looking at a different-sized copy than the one a mask ends up being
    built against, and a fraction means the same thing in both frames. Order is
    (x0, y0, x1, y1) with 0,0 at the top-left, matching mask_boxes exactly, so a
    region can be handed straight to a generation without conversion."""

    label: str
    box: tuple[float, float, float, float]


class RegionsResult(BaseModel):
    """What `server/region_worker.py` posts back for a claimed job.

    Exactly one of `regions` or `error` is meaningful. An error is carried rather
    than signalled with a status code because the worker failing (no `claude` on
    PATH, a reply that held no JSON) is a normal outcome the harness should show,
    not a transport failure -- and the message is written by the worker for a person
    to read, the same way request_problem's are."""

    regions: list[Region] | None = None
    error: str | None = None
    # What the worker measured. The harness compares this to its own oriented size
    # and warns on a mismatch: cheap insurance against regions arriving for a
    # different image than the one on screen, which byte-equality can't catch since
    # nothing guarantees the worker read the same encoding.
    width: int | None = None
    height: int | None = None
