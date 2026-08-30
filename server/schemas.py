from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 9
    seed: int | None = None
    preview_every: int = Field(
        default=0,
        description="Decode and include an in-progress preview image every N steps. 0 disables previews.",
    )
    stream: bool = True
