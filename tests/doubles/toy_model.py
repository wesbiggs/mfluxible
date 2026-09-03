"""A weight-free stand-in for an mflux text-to-image variant.

server/engine.py is written against the surface mflux's ZImage/Flux1/QwenImage classes
all share -- constructor kwargs, generate_image(), a callbacks registry, save_model()
(see server/models.py's module docstring). ToyModel implements that same surface
without downloading or running any real model: every "denoising step" is a no-op, and
the final/preview image is always a flat block of color derived deterministically from
the seed. That makes it possible to exercise engine.py's SSE streaming, step-callback
timing, preview decoding, and final-image plumbing in CI, where no GPU/weights are
available and none should be needed.

This reuses mflux's own CallbackRegistry, Config, ModelConfig, VAEUtil and ImageUtil --
all pure data/glue with no weights attached -- rather than reimplementing the parts of
the real pipeline engine.py actually calls directly (see _decode_preview_b64 in
engine.py, which calls VAEUtil.decode/ImageUtil.to_image itself). Only the VAE and the
transformer's "prediction" are faked; everything downstream of them is the real code.

Wiring: pass TOY_MODEL_SPEC straight to MfluxEngine(model=TOY_MODEL_SPEC, ...) instead
of a model-name string. MfluxEngine only calls models.resolve() when `model` is a str
(see server/engine.py), so handing it a ModelSpec instance directly bypasses models.py's
registry entirely -- nothing in server/ needs to change to "register" this model.
"""

from __future__ import annotations

import random

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.utils.image_util import ImageUtil

from models import ModelSpec

# A fictitious ModelConfig for the toy model. supports_guidance/requires_sigma_shift
# are both False so nothing here ever needs a real scheduler (Config.scheduler is
# never touched by engine.py or by ToyModel itself).
TOY_MODEL_CONFIG = ModelConfig(
    priority=0,
    aliases=["toy"],
    model_name="mfluxible-toy-solid-color",
    base_model=None,
    controlnet_model=None,
    custom_transformer_model=None,
    num_train_steps=None,
    max_sequence_length=None,
    supports_guidance=False,
    requires_sigma_shift=False,
)

_SPATIAL_SCALE = 8  # stand-in for a real VAE's encode/decode downsample factor


def _seed_color(seed: int) -> tuple[float, float, float]:
    """Deterministic RGB in [-1, 1] (VAE-output range) from a seed, so the same seed
    always renders the same solid color and different seeds render different ones."""
    rng = random.Random(seed)
    return tuple(rng.uniform(-1.0, 1.0) for _ in range(3))


class ToyLatentCreator:
    """Identity unpacking -- ToyModel never packs latents in the first place, so
    there's nothing for a real LatentCreator's unpack step to undo. Exists only
    because engine.py's _decode_preview_b64 calls self._latent_creator.unpack_latents
    unconditionally."""

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents


class _ToyVAE:
    """decode() ignores the transformer's "prediction" entirely (there isn't one) and
    just paints the constant color already baked into every pixel of `latent` across
    the full requested resolution. Deliberately has no decode_packed_latents:
    engine.py's hasattr() check is what routes models with and without one correctly,
    so leaving it off exercises the same branch Z-Image/FLUX take."""

    latent_channels = 3

    def decode(self, latent: mx.array) -> mx.array:
        h = latent.shape[-2] * _SPATIAL_SCALE
        w = latent.shape[-1] * _SPATIAL_SCALE
        pixel = latent[:, :, :1, :1]
        return mx.broadcast_to(pixel, (1, 3, h, w))


class ToyModel:
    """Drop-in for mflux's ZImage/Flux1/QwenImage in tests. See module docstring."""

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        self.model_config = model_config or TOY_MODEL_CONFIG
        self.bits = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.vae = _ToyVAE()
        self.callbacks = CallbackRegistry()

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 2,
        width: int = 64,
        height: int = 64,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **_ignored,
    ):
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance or 0.0,
        )

        r, g, b = _seed_color(seed)
        color = mx.array([r, g, b], dtype=mx.float32).reshape(1, 3, 1, 1)
        lat_h = max(1, height // _SPATIAL_SCALE)
        lat_w = max(1, width // _SPATIAL_SCALE)
        latents = mx.broadcast_to(color, (1, 3, lat_h, lat_w))

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        # A plain range, passed explicitly, so GenerationContext.in_loop doesn't fall
        # back to config.time_steps (a real tqdm progress bar) -- nothing here reads
        # it, and skipping it keeps test output quiet.
        steps = range(num_inference_steps)
        for t in steps:
            # No real denoising: the "answer" is already in `latents`, so each step
            # just gives engine.py's in-loop callback something to fire against.
            ctx.in_loop(t, latents, time_steps=steps)
            mx.eval(latents)
        ctx.after_loop(latents)

        unpacked = ToyLatentCreator.unpack_latents(latents, height, width)
        decoded = VAEUtil.decode(vae=self.vae, latent=unpacked)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=0.0,
        )

    def save_model(self, base_path: str) -> None:
        # Real variants persist quantized weights here for MfluxEngine's on-disk cache
        # (see server/engine.py's _load_sync). ToyModel has no weights, and tests
        # always construct MfluxEngine with model_cache_dir=None so this branch is
        # never reached -- if it is, that's a test setup bug, not a missing feature.
        raise NotImplementedError("ToyModel has no weights to save; use MfluxEngine(..., model_cache_dir=None)")


TOY_MODEL_SPEC = ModelSpec(
    key="toy-solid-color",
    aliases=("toy",),
    repo="n/a -- test double, no weights",
    label="Toy Solid-Color Model",
    load=lambda: (ToyModel, TOY_MODEL_CONFIG, ToyLatentCreator),
    default_steps=2,
    supports_guidance=False,
)
