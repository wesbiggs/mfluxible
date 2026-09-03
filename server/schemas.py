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
