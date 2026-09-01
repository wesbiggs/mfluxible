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
