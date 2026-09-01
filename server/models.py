"""The table of models this server can run, and everything that differs between them.

mflux's three text-to-image variants -- ZImage, Flux1, QwenImage -- happen to share
an identical surface: the same constructor keywords (`quantize`, `model_path`,
`lora_paths`, `lora_scales`, `model_config`), the same `generate_image()` signature,
the same `save_model(base_path)`, and the same `callbacks` registry. That's what lets
engine.py stay model-agnostic and confines the differences to this file.

Every mflux import here is deferred into the loader functions on purpose. A spec is
inert data until `load()` is called on it, so naming all four models costs nothing at
import time, and mflux only downloads weights inside the variant's constructor
(`WeightLoader.load` -> `PathResolution.resolve` -> a HF snapshot download). One
server process therefore fetches, quantizes and caches exactly the one model it was
configured with; the other specs are a few dozen bytes of strings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    aliases: tuple[str, ...]
    repo: str  # Hugging Face repo the raw weights come from, for /health and docs
    label: str
    # Returns (variant_class, model_config, latent_creator). Called once, on the MLX
    # worker thread, at load time -- never at import time.
    load: Callable[[], tuple[type, Any, Any]]
    # mflux's own per-model default from mflux/cli/defaults/defaults.py
    # (MODEL_INFERENCE_STEPS). Copied rather than imported: that table lives under
    # mflux.cli, which is CLI-internal and not something to depend on from a server.
    default_steps: int
    supports_guidance: bool
    default_guidance: float | None = None
    supports_negative_prompt: bool = False


def _load_z_image_turbo():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.latent_creator import ZImageLatentCreator
    from mflux.models.z_image.variants.z_image import ZImage

    return ZImage, ModelConfig.z_image_turbo(), ZImageLatentCreator


def _load_flux(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1, getattr(ModelConfig, model_config_name)(), FluxLatentCreator

    return loader


def _load_qwen_image():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    return QwenImage, ModelConfig.qwen_image(), QwenLatentCreator


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="z-image-turbo",
        aliases=("z-image", "zimage", "zimage-turbo"),
        repo="Tongyi-MAI/Z-Image-Turbo",
        label="Z-Image-Turbo",
        load=_load_z_image_turbo,
        default_steps=9,
        # Guidance-distilled: mflux forces guidance to 0.0 and, with CFG off, never
        # encodes a negative prompt (see IGNORED_OPTIONS in mflux's
        # z_image_turbo_generate CLI). Sending either would silently do nothing.
        supports_guidance=False,
    ),
    ModelSpec(
        key="flux-schnell",
        aliases=("schnell", "flux.1-schnell", "flux-1-schnell"),
        repo="black-forest-labs/FLUX.1-schnell",
        label="FLUX.1-schnell",
        load=_load_flux("schnell"),
        default_steps=4,
        # schnell builds no guidance embedder at all, so a guidance value has no path
        # to reach the output; FLUX has no negative branch in either variant.
        supports_guidance=False,
    ),
    ModelSpec(
        key="flux-dev",
        aliases=("dev", "flux.1-dev", "flux-1-dev"),
        repo="black-forest-labs/FLUX.1-dev",
        label="FLUX.1-dev",
        load=_load_flux("dev"),
        default_steps=25,
        supports_guidance=True,
        default_guidance=3.5,
    ),
    ModelSpec(
        key="qwen-image",
        aliases=("qwen", "qwen-image-2512", "qwen-2512"),
        repo="Qwen/Qwen-Image-2512",
        label="Qwen-Image",
        load=_load_qwen_image,
        default_steps=20,
        # True CFG: the transformer runs twice per step (conditional + unconditional)
        # and the two are blended by `guidance`, so this is the one model here where a
        # negative prompt does something. Note the ModelConfig entry says
        # supports_guidance=None -- that flag gates FLUX's *distilled* guidance
        # embedder and isn't what drives Qwen's CFG loop, so it isn't consulted here.
        supports_guidance=True,
        default_guidance=3.5,
        supports_negative_prompt=True,
    ),
)

_BY_NAME = {name: spec for spec in MODELS for name in (spec.key, *spec.aliases)}


def resolve(name: str) -> ModelSpec:
    spec = _BY_NAME.get(name.strip().lower())
    if spec is None:
        known = ", ".join(spec.key for spec in MODELS)
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return spec
