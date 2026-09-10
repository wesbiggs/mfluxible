"""The table of models this server can run, and everything that differs between them.

Six of mflux's text-to-image variant classes -- ZImage, Flux1, QwenImage, Krea2,
ErnieImage, Flux2Klein -- happen to share an identical surface: the same constructor
keywords (`quantize`, `model_path`, `lora_paths`, `lora_scales`, `model_config`), the
same `generate_image()` signature, the same `save_model(base_path)`, and the same
`callbacks` registry. That's what lets engine.py stay model-agnostic and confines the
differences to this file.

Not every mflux variant does, which is why this table stops where it does. `FIBO`,
`BooguImage` and `LensImage` drop `lora_paths`/`lora_scales` from the constructor, so
`_load_sync`'s fresh-load branch would raise TypeError before a single weight loaded;
`Ideogram4` takes no `image_path`/`image_strength`/`scheduler` at all; `BooguImage` has
no latent creator, so step previews are impossible (mflux's own CLI passes
`latent_creator=None` and says stepwise output is unsupported); and `LensImage` has no
`save_model()`, so there is nothing for the quantized-weight cache to write. Adding any
of those means capability flags here *and* engine.py learning to honour them -- not a
new row.

Every mflux import here is deferred into the loader functions on purpose. A spec is
inert data until `load()` is called on it, so naming fifteen models costs nothing at
import time, and mflux only downloads weights inside the variant's constructor
(`WeightLoader.load` -> `PathResolution.resolve` -> a HF snapshot download). One
server process therefore fetches, quantizes and caches exactly the one model it was
configured with; the other specs are a few dozen bytes of strings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Classifier-free guidance is what makes a negative prompt do anything, and every
# CFG-capable model here builds the unconditional branch only *above* guidance 1.0:
# ZImage and ErnieImage skip it at `guidance <= 1.0`, Krea2 at `guidance == 1.0`, and
# Flux2Klein at `not (guidance > 1.0)`. So 1.0 is the shared floor rather than a value
# picked here. Krea-2 is the one model whose own default sits exactly on it, which is
# why engine.check_request tests the *effective* guidance rather than trusting the flag
# alone -- a negative prompt at guidance 1.0 would otherwise be accepted, encoded and
# then never consulted.
CFG_GUIDANCE_FLOOR = 1.0


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
    # Which scheduler the variant picks for itself when generate_image() isn't told
    # one -- and the only value `fractional_start` is safe on is "linear".
    #
    # server/schedulers.py's FractionalStartLinearScheduler subclasses mflux's
    # LinearScheduler and reaches the model by being passed as `scheduler=`, replacing
    # whatever the variant would have chosen. On a linear-by-default model that's the
    # same schedule with one rung moved, which is the whole design. On any other model
    # it is a *different sampler*: Flux2Klein and Z-Image base default to
    # "flow_match_euler_discrete", and Krea2 maps "linear" onto "er_sde" and raises
    # ValueError on any name but "er_sde"/"euler" -- from inside the worker thread,
    # after the SSE headers are already out. check_request rejects fractional_start up
    # front wherever this isn't "linear"; see schedulers.py's module docstring for why
    # subclassing LinearScheduler is what ties the two together.
    default_scheduler: str = "linear"

    @property
    def supports_fractional_start(self) -> bool:
        return self.default_scheduler == "linear"


def _load_z_image(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image.latent_creator import ZImageLatentCreator
        from mflux.models.z_image.variants.z_image import ZImage

        return ZImage, getattr(ModelConfig, model_config_name)(), ZImageLatentCreator

    return loader


def _load_flux(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1, getattr(ModelConfig, model_config_name)(), FluxLatentCreator

    return loader


def _load_flux2(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        return Flux2Klein, getattr(ModelConfig, model_config_name)(), Flux2LatentCreator

    return loader


def _load_krea2(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.krea2.latent_creator import Krea2LatentCreator
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2

        return Krea2, getattr(ModelConfig, model_config_name)(), Krea2LatentCreator

    return loader


def _load_ernie(model_config_name: str):
    def loader():
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

        return ErnieImage, getattr(ModelConfig, model_config_name)(), ErnieLatentCreator

    return loader


def _load_qwen_image():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    return QwenImage, ModelConfig.qwen_image(), QwenLatentCreator


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="z-image-turbo",
        # Note what is *not* here: "z-image" and "zimage" both name the base model in
        # mflux's own registry, so they belong to that entry below rather than to this
        # one. Keeping mflux's spelling is the point of carrying aliases at all.
        aliases=("zimage-turbo",),
        repo="Tongyi-MAI/Z-Image-Turbo",
        label="Z-Image-Turbo",
        load=_load_z_image("z_image_turbo"),
        default_steps=9,
        # Guidance-distilled: mflux forces guidance to 0.0 and, with CFG off, never
        # encodes a negative prompt (see IGNORED_OPTIONS in mflux's
        # z_image_turbo_generate CLI). Sending either would silently do nothing.
        supports_guidance=False,
    ),
    ModelSpec(
        key="z-image",
        aliases=("zimage",),
        repo="Tongyi-MAI/Z-Image",
        label="Z-Image",
        load=_load_z_image("z_image"),
        default_steps=50,
        # The base checkpoint, and the first Z-Image entry that isn't distilled: with
        # supports_guidance true, ZImage._encode_prompts builds a real unconditional
        # branch and _predict blends it, so guidance and negative_prompt both bite.
        supports_guidance=True,
        # mflux's z_image README uses `--guidance 4` / `guidance=4.0`. Its *CLI* default
        # is 0.0, which disables CFG outright -- following the README instead keeps the
        # negative prompt working at the default, and keeps this entry consistent with
        # every other CFG model in the table.
        default_guidance=4.0,
        supports_negative_prompt=True,
        # ZImage picks its scheduler from supports_guidance: "linear" when distilled,
        # "flow_match_euler_discrete" otherwise. So the base model and the Turbo above
        # differ here even though they share a class.
        default_scheduler="flow_match_euler_discrete",
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
        key="krea-dev",
        aliases=("dev-krea", "flux.1-krea-dev", "flux-1-krea-dev"),
        repo="black-forest-labs/FLUX.1-Krea-dev",
        label="FLUX.1-Krea-dev",
        # A FLUX.1-dev derivative, so it rides the same Flux1 class and the same
        # linear-by-default schedule -- only the weights and the ModelConfig differ.
        load=_load_flux("krea_dev"),
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
    ModelSpec(
        key="krea-2",
        aliases=("krea2", "krea-2-turbo"),
        repo="krea/Krea-2-Turbo",
        label="Krea-2-Turbo",
        load=_load_krea2("krea2"),
        default_steps=8,
        # Distilled, but unlike the other distilled models here guidance is not pinned:
        # mflux's krea2 CLI defaults it to 1.0 and lets any value through, and Krea2's
        # own _predict does real CFG (`v_neg + guidance * (v - v_neg)`) above 1.0.
        supports_guidance=True,
        # mflux's own DEFAULT_GUIDANCE for this model. It sits exactly on
        # CFG_GUIDANCE_FLOOR, so at the default there is no unconditional branch and a
        # negative prompt would be encoded and then ignored -- check_request rejects
        # that combination rather than raising this above what mflux recommends.
        default_guidance=1.0,
        supports_negative_prompt=True,
        # Krea2._resolve_scheduler maps "linear" (and None) onto "er_sde" and raises
        # ValueError on anything else -- including a dotted path -- so this model never
        # runs a linear schedule and can never take a fractional start.
        default_scheduler="er_sde",
    ),
    ModelSpec(
        key="krea-2-raw",
        aliases=("krea2-raw",),
        repo="krea/Krea-2-Raw",
        label="Krea-2-Raw",
        load=_load_krea2("krea2_raw"),
        # mflux's MODEL_INFERENCE_STEPS has no entry for this model, so its own CLI
        # falls back to DEFAULT_INFERENCE_STEPS (25); that fallback is what this copies.
        # Krea recommends Raw as the base for finetuning and Turbo for inference, so a
        # non-distilled step count is the right shape even though it isn't tabulated.
        default_steps=25,
        supports_guidance=True,
        default_guidance=1.0,
        supports_negative_prompt=True,
        default_scheduler="er_sde",
    ),
    ModelSpec(
        key="ernie-image-turbo",
        aliases=("ernie-turbo",),
        repo="baidu/ERNIE-Image-Turbo",
        label="ERNIE-Image-Turbo",
        load=_load_ernie("ernie_image_turbo"),
        default_steps=8,
        # Guidance-distilled: mflux's ernie_image_turbo_generate CLI hard-errors on any
        # --guidance but 1.0, and at 1.0 ErnieImage never encodes the negative prompt.
        supports_guidance=False,
    ),
    ModelSpec(
        key="ernie-image",
        aliases=("ernie",),
        repo="baidu/ERNIE-Image",
        label="ERNIE-Image",
        load=_load_ernie("ernie_image"),
        default_steps=50,
        supports_guidance=True,
        # mflux's ernie_image_generate CLI: `set_defaults(guidance=4.0)`.
        default_guidance=4.0,
        supports_negative_prompt=True,
    ),
    # FLUX.2 Klein. The three distilled checkpoints require guidance 1.0 -- mflux's
    # flux2_generate CLI errors on any other value for a builtin name without "base" in
    # it -- so only the two base checkpoints expose guidance here. None of the five
    # takes a negative_prompt at all: Flux2Klein.generate_image has no such parameter
    # and hardcodes a blank unconditional prompt internally.
    ModelSpec(
        key="flux2-klein-4b",
        aliases=("flux2-klein", "klein-4b"),
        repo="black-forest-labs/FLUX.2-klein-4B",
        label="FLUX.2-klein-4B",
        load=_load_flux2("flux2_klein_4b"),
        default_steps=4,
        supports_guidance=False,
        default_scheduler="flow_match_euler_discrete",
    ),
    ModelSpec(
        key="flux2-klein-9b",
        aliases=("klein-9b",),
        repo="black-forest-labs/FLUX.2-klein-9B",
        label="FLUX.2-klein-9B",
        load=_load_flux2("flux2_klein_9b"),
        default_steps=4,
        supports_guidance=False,
        default_scheduler="flow_match_euler_discrete",
    ),
    ModelSpec(
        key="flux2-klein-9b-kv",
        aliases=("klein-9b-kv",),
        repo="black-forest-labs/FLUX.2-klein-9b-kv",
        label="FLUX.2-klein-9B-kv",
        load=_load_flux2("flux2_klein_9b_kv"),
        default_steps=4,
        supports_guidance=False,
        default_scheduler="flow_match_euler_discrete",
    ),
    ModelSpec(
        key="flux2-klein-base-4b",
        aliases=("flux2-base-4b", "klein-base-4b"),
        repo="black-forest-labs/FLUX.2-klein-base-4B",
        label="FLUX.2-klein-base-4B",
        load=_load_flux2("flux2_klein_base_4b"),
        default_steps=50,
        supports_guidance=True,
        # mflux's flux2 README: "Base variants support guidance values above 1.0", and
        # its own base example passes `--guidance 1.5`.
        default_guidance=1.5,
        default_scheduler="flow_match_euler_discrete",
    ),
    ModelSpec(
        key="flux2-klein-base-9b",
        aliases=("flux2-base-9b", "klein-base-9b"),
        repo="black-forest-labs/FLUX.2-klein-base-9B",
        label="FLUX.2-klein-base-9B",
        load=_load_flux2("flux2_klein_base_9b"),
        default_steps=50,
        supports_guidance=True,
        default_guidance=1.5,
        default_scheduler="flow_match_euler_discrete",
    ),
)

_BY_NAME = {name: spec for spec in MODELS for name in (spec.key, *spec.aliases)}


def resolve(name: str) -> ModelSpec:
    spec = _BY_NAME.get(name.strip().lower())
    if spec is None:
        known = ", ".join(spec.key for spec in MODELS)
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return spec
