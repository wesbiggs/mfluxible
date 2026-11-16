"""The table of models this server can run, and everything that differs between them.

Six of mflux's text-to-image variant classes -- ZImage, Flux1, QwenImage, Krea2,
ErnieImage, Flux2Klein -- happen to share an identical surface: the same constructor
keywords (`quantize`, `model_path`, `lora_paths`, `lora_scales`, `model_config`), the
same `generate_image()` signature, the same `save_model(base_path)`, and the same
`callbacks` registry. That's what lets engine.py stay model-agnostic and confines the
differences to this file.

Not every mflux variant does, which is why this table stops where it does. Where a variant
drops `lora_paths`/`lora_scales` from its constructor, `supports_lora` is False and
engine.py leaves the keywords out of the fresh load (FIBO and Ming-Image are in the table
on that basis; Iris and `BooguImage` are not yet). `Ideogram4` takes no
`image_path`/`image_strength`/`scheduler` at all, which `supports_image` covers for the
text-to-image-only rows. `BooguImage` has no latent creator, so step previews are
impossible (mflux's own CLI passes `latent_creator=None` and says stepwise output is
unsupported); `Iris` has no VAE, which breaks the preview decode, so `supports_preview` is off
for it and for Boogu; and `LensImage` has no `save_model()`, so there is nothing for the
quantized-weight cache to write. Adding any of
those means a capability flag here *and* engine.py learning to honour it -- not a new row.

Every mflux import here is deferred into the loader functions on purpose. A spec is
inert data until `load()` is called on it, so naming twenty-eight models costs nothing at
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
# ZImage and ErnieImage skip it at `guidance <= 1.0`, Krea2 at `guidance == 1.0`,
# Flux2Klein at `not (guidance > 1.0)`, and Qwen-Image-2.1 at `guidance > 1.0 and
# negative_prompt` (its `do_true_cfg`). So 1.0 is the shared floor rather than a value
# picked here. Krea-2 and Qwen-Image-2.1 are the models whose own default sits exactly on
# it, which is why engine.request_problem tests the *effective* guidance rather than
# trusting the flag alone -- a negative prompt at guidance 1.0 would otherwise be
# accepted, encoded and then never consulted.
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
    # mfluxible/schedulers.py's FractionalStartLinearScheduler subclasses mflux's
    # LinearScheduler and reaches the model by being passed as `scheduler=`, replacing
    # whatever the variant would have chosen. On a linear-by-default model that's the
    # same schedule with one rung moved, which is the whole design. On any other model
    # it is a *different sampler*: Flux2Klein and Z-Image base default to
    # "flow_match_euler_discrete", and Krea2 maps "linear" onto "er_sde" and raises
    # ValueError on any name but "er_sde"/"euler" -- from inside the worker thread,
    # after the SSE headers are already out. request_problem rejects fractional_start up
    # front wherever this isn't "linear"; see schedulers.py's module docstring for why
    # subclassing LinearScheduler is what ties the two together.
    default_scheduler: str = "linear"
    # Whether the variant's generate_image() takes an input image (`image_path`). False
    # for the text-to-image-only variants added in mflux 0.24 (Anima, FIBO-2, FLUX.3):
    # sending `image` to one of them is a 400 from engine.request_problem, rather than a
    # TypeError from generate_image() after the SSE headers are out. Defaults to True
    # because every row that predates the flag takes one.
    supports_image: bool = True
    # Whether the variant's constructor takes lora_paths/lora_scales. False for FIBO,
    # Ming-Image and Iris, which mflux builds without them: engine.py leaves the kwargs
    # out of the fresh load, and MfluxEngine refuses a configured LoRA at construction
    # rather than letting it be silently ignored.
    supports_lora: bool = True
    # Whether a step preview can be decoded. False where the model has no VAE to decode
    # latents through (Iris, which is pixel-space) or no latent creator at all (Boogu).
    # engine.request_problem rejects preview_every on those, so no endpoint reaches the
    # decode path -- and the harness hides the control, via /health's supports_preview.
    supports_preview: bool = True

    @property
    def supports_fractional_start(self) -> bool:
        return self.default_scheduler == "linear"

    @property
    def supports_mask(self) -> bool:
        # Two capabilities, one precondition: masked inpainting also reaches the model
        # by replacing its scheduler (mfluxible/schedulers.py's MaskedBlend* classes
        # subclass the same LinearScheduler), so it is safe in exactly the same place
        # a fractional start is. Kept as its own property rather than an alias because
        # a client asks two different questions of /health -- and because the two
        # could diverge: a flow-match sibling of MaskedBlendLinearScheduler would make
        # this true on models where the fractional rung move still isn't.
        return self.default_scheduler == "linear"


def _loader(variant: str, latent_creator: str):
    """A `ModelSpec.load` for a variant class and latent creator named by dotted path
    (`module.path:Name`), plus the `ModelConfig` classmethod called on it.

    Returns a function taking the ModelConfig name, so each family is spelled once:
    `_load_z_image = _loader(...)`, then `_load_z_image("z_image_turbo")`. The imports are
    done by name *inside* the returned loader rather than at module scope, for the reason
    in the module docstring -- naming a model must not import mflux's model for it.
    """

    def with_config(model_config_name: str):
        def loader():
            from importlib import import_module

            from mflux.models.common.config.model_config import ModelConfig

            def resolve_name(spec: str):
                module, name = spec.split(":")
                return getattr(import_module(module), name)

            return (
                resolve_name(variant),
                getattr(ModelConfig, model_config_name)(),
                resolve_name(latent_creator),
            )

        return loader

    return with_config


_load_z_image = _loader(
    "mflux.models.z_image.variants.z_image:ZImage", "mflux.models.z_image.latent_creator:ZImageLatentCreator"
)
_load_flux = _loader(
    "mflux.models.flux.variants.txt2img.flux:Flux1",
    "mflux.models.flux.latent_creator.flux_latent_creator:FluxLatentCreator",
)
_load_flux2 = _loader(
    "mflux.models.flux2.variants.txt2img.flux2_klein:Flux2Klein",
    "mflux.models.flux2.latent_creator.flux2_latent_creator:Flux2LatentCreator",
)
_load_krea2 = _loader(
    "mflux.models.krea2.variants.txt2img.krea2:Krea2", "mflux.models.krea2.latent_creator:Krea2LatentCreator"
)
_load_ernie = _loader(
    "mflux.models.ernie_image.variants.txt2img.ernie_image:ErnieImage",
    "mflux.models.ernie_image.latent_creator:ErnieLatentCreator",
)


def _load_qwen_image():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    return QwenImage, ModelConfig.qwen_image(), QwenLatentCreator


_load_qwen_image_21 = _loader(
    "mflux.models.qwen21.variants.txt2img.qwen_image_21:QwenImage21",
    "mflux.models.qwen21.latent_creator.qwen21_latent_creator:Qwen21LatentCreator",
)
_load_fibo = _loader(
    "mflux.models.fibo.variants.txt2img.fibo:FIBO",
    "mflux.models.fibo.latent_creator.fibo_latent_creator:FiboLatentCreator",
)
_load_ming = _loader(
    "mflux.models.ming_image.variants.ming_image:MingImage",
    "mflux.models.ming_image.latent_creator.ming_latent_creator:MingLatentCreator",
)
_load_iris = _loader(
    "mflux.models.iris.variants.txt2img.iris:Iris",
    "mflux.models.iris.latent_creator.iris_latent_creator:IrisLatentCreator",
)


def _load_boogu():
    # Boogu has no latent creator in mflux (its own CLI passes latent_creator=None), so the
    # third element is None and the preview path must never be reached for this model.
    from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage
    from mflux.models.common.config.model_config import ModelConfig

    return BooguImage, ModelConfig.boogu_image_turbo(), None


_load_anima = _loader(
    "mflux.models.anima.variants.txt2img.anima:Anima",
    "mflux.models.anima.latent_creator.anima_latent_creator:AnimaLatentCreator",
)
_load_fibo2 = _loader(
    "mflux.models.fibo2.variants.txt2img.fibo2:Fibo2",
    "mflux.models.fibo2.latent_creator.fibo2_latent_creator:Fibo2LatentCreator",
)
_load_flux3 = _loader(
    "mflux.models.flux3.variants.txt2img.flux3_klein:Flux3Klein",
    "mflux.models.flux3.latent_creator.flux3_latent_creator:Flux3LatentCreator",
)


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
    # Qwen-Image-2.1 (mflux 0.24). Same QwenImage-style constructor and generate_image()
    # surface as the 2512 entry above, so it rides on the same engine path; the two
    # checkpoints differ in CFG and in schedule, below.
    ModelSpec(
        key="qwen-image-2.1",
        aliases=("qwen-2.1", "qwen-image-21"),
        repo="Qwen/Qwen-Image-2.1",
        label="Qwen-Image-2.1",
        load=_load_qwen_image_21("qwen_image_21"),
        default_steps=40,
        # mflux's qwen21_generate CLI defaults guidance to 1.0 and its README recommends
        # 40 steps *without* guidance. Above 1.0 the transformer runs a true-CFG pass
        # (`do_true_cfg = guidance > 1.0 and bool(negative_prompt)`), so guidance and a
        # negative prompt both do something once the caller asks for them.
        supports_guidance=True,
        default_guidance=1.0,
        supports_negative_prompt=True,
        default_scheduler="linear",
    ),
    ModelSpec(
        key="qwen-image-2.1-turbo",
        aliases=("qwen-2.1-turbo", "qwen-image-21-turbo"),
        repo="Qwen/Qwen-Image-2.1-Turbo",
        label="Qwen-Image-2.1-Turbo",
        load=_load_qwen_image_21("qwen_image_21_turbo"),
        default_steps=8,
        # Distilled to run without CFG: Qwen21TurboScheduler.check_generate rejects any
        # guidance other than 1 and any scheduler other than "linear" outright, so both
        # fields are a 400 here rather than something mflux would quietly ignore.
        supports_guidance=False,
        # Its own fixed 8-node sigma schedule (from model_index.json), not mflux's linear
        # one. A fractional start would replace that schedule with the linear
        # subclass, which Qwen21TurboScheduler refuses -- so this must not read "linear".
        default_scheduler="turbo-fixed",
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
        # CFG_GUIDANCE_FLOOR (as does qwen-image-2.1's, see there), so at the default there is no unconditional branch and a
        # negative prompt would be encoded and then ignored -- request_problem rejects
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
    # The three rows below are text-to-image only (supports_image=False): mflux 0.24's
    # generate_image() on each takes no image_path, so an `image` in a request is a 400.
    # Every one of them accepts LoRA in its constructor and bakes it by default, so the
    # quantized-cache rule in engine.py holds for them. MOSS-Image is deliberately absent:
    # its constructor defaults bake_lora to False, so save_model() would write a cache
    # without the LoRA and the next startup would load it silently unmodified.
    ModelSpec(
        key="anima-turbo",
        aliases=("anima",),
        repo="circlestone-labs/Anima",
        label="Anima-Turbo",
        load=_load_anima("anima_turbo"),
        default_steps=10,
        # mflux's Anima.DEFAULT_GUIDANCE is 1.0 for Turbo, so the negative branch is off
        # at the default and request_problem's CFG-floor check applies, as for Krea-2.
        supports_guidance=True,
        default_guidance=1.0,
        supports_negative_prompt=True,
        # Anima never runs a linear schedule: its sampler is "er_sde" (or "euler"), so a
        # fractional start would replace it.
        default_scheduler="er_sde",
        supports_image=False,
    ),
    ModelSpec(
        key="anima-aesthetic",
        aliases=(),
        repo="circlestone-labs/Anima",
        label="Anima-Aesthetic",
        load=_load_anima("anima_aesthetic"),
        default_steps=30,
        supports_guidance=True,
        default_guidance=4.0,
        supports_negative_prompt=True,
        default_scheduler="er_sde",
        supports_image=False,
    ),
    ModelSpec(
        key="anima-base",
        aliases=(),
        repo="circlestone-labs/Anima",
        label="Anima-Base",
        load=_load_anima("anima_base"),
        default_steps=30,
        supports_guidance=True,
        default_guidance=4.0,
        supports_negative_prompt=True,
        default_scheduler="er_sde",
        supports_image=False,
    ),
    ModelSpec(
        key="fibo2",
        aliases=("fibo-2.0-dev",),
        repo="briaai/Fibo-2.0-dev",
        label="FIBO-2",
        load=_load_fibo2("fibo2"),
        default_steps=50,
        # Dev runs with guidance 5.0 (Fibo2.DEFAULT_GUIDANCE); the negative prompt is a
        # second pass that only runs above 1.0, and the default clears that floor.
        supports_guidance=True,
        default_guidance=5.0,
        supports_negative_prompt=True,
        # Hard-wired to flow-match in fibo2_core.py, whatever scheduler is asked for.
        default_scheduler="flow_match_euler_discrete",
        supports_image=False,
    ),
    # FLUX.3 Klein. Only the text-to-image path is wired here: the same checkpoints also
    # do editing with up to ten reference images, which is a different request surface.
    ModelSpec(
        key="flux3-image-klein",
        aliases=("flux3-klein", "flux3", "flux-3-image-klein"),
        repo="black-forest-labs/flux-3-image-klein",
        label="FLUX.3-Klein",
        load=_load_flux3("flux3_image_klein"),
        default_steps=16,
        # Distilled: no guidance embedder, and Flux3Klein.generate_image has no
        # negative_prompt parameter at all.
        supports_guidance=False,
        default_scheduler="flow_map",
        supports_image=False,
    ),
    ModelSpec(
        key="flux3-image-klein-base",
        aliases=("flux3-klein-base", "flux3-base", "flux-3-image-klein-base"),
        repo="black-forest-labs/flux-3-image-klein-base",
        label="FLUX.3-Klein-Base",
        load=_load_flux3("flux3_image_klein_base"),
        default_steps=50,
        # Base takes guidance with a default of 3.5 (mflux's GUIDANCE_SCALE), but still
        # no negative_prompt parameter.
        supports_guidance=True,
        default_guidance=3.5,
        default_scheduler="flow_map",
        supports_image=False,
    ),
    # No LoRA in mflux's constructor (supports_lora=False). FIBO is the legacy
    # predecessor of FIBO-2 above and keeps the image path; Ming-Image is text-only.
    ModelSpec(
        key="fibo",
        aliases=(),
        repo="briaai/FIBO",
        label="FIBO",
        load=_load_fibo("fibo"),
        default_steps=50,
        # mflux's FIBO.generate_image defaults guidance to 4.0 and runs the CFG blend
        # above 1.0. It accepts negative_prompt but never encodes it (the value only
        # reaches the run's Config), so the flag stays off rather than accepting it and
        # dropping it silently.
        supports_guidance=True,
        default_guidance=4.0,
        supports_negative_prompt=False,
        default_scheduler="flow_match_euler_discrete",
        supports_lora=False,
    ),
    ModelSpec(
        key="ming-image-design",
        aliases=("ming-image", "ming-design", "ming"),
        repo="inclusionAI/Ming-Image-0.1-Design",
        label="Ming-Image-Design",
        load=_load_ming("ming_image_design"),
        default_steps=12,
        # Above guidance 1.0 the transformer blends a pass with both text streams zeroed
        # (MingImage._predict). There is no negative_prompt parameter at all.
        supports_guidance=True,
        default_guidance=1.0,
        supports_negative_prompt=False,
        # The generate call takes no scheduler and runs its own flow schedule, so it is
        # never mflux's linear one.
        default_scheduler="ming_flow",
        supports_image=False,
        supports_lora=False,
    ),
    # Iris (pixel-space, no VAE): no LoRA, no image input, no step previews. The turbo
    # checkpoint is distilled to a fixed 4-step schedule with no guidance; the base one
    # takes guidance and a negative prompt on mflux's own DPM-solver schedule.
    ModelSpec(
        key="iris-3b",
        aliases=("iris-3b-turbo", "iris-turbo", "iris"),
        repo="speridlabs/iris-3b",
        label="Iris-3B-Turbo",
        load=_load_iris("iris_3b"),
        default_steps=4,
        supports_guidance=False,
        default_scheduler="iris_turbo",
        supports_image=False,
        supports_lora=False,
        supports_preview=False,
    ),
    ModelSpec(
        key="iris-3b-base",
        aliases=("iris-base",),
        repo="speridlabs/iris-3b",
        label="Iris-3B-Base",
        load=_load_iris("iris_3b_base"),
        default_steps=100,
        # mflux's Iris.DEFAULT_GUIDANCE is 3.0. The negative branch is built whenever
        # guidance is not exactly 1.0; the floor check in request_problem still rejects a
        # negative prompt at or below 1.0, which is the conservative reading of that rule.
        supports_guidance=True,
        default_guidance=3.0,
        supports_negative_prompt=True,
        default_scheduler="iris_dpm",
        supports_image=False,
        supports_lora=False,
        supports_preview=False,
    ),
    # Boogu (4-step DMD, no CFG, no scheduler, no latent creator): text-to-image only, no
    # LoRA in its constructor, no guidance or negative prompt, and no step previews.
    ModelSpec(
        key="boogu-image-turbo",
        aliases=("boogu-turbo", "boogu-image", "boogu"),
        repo="Boogu/Boogu-Image-0.1-Turbo",
        label="Boogu-Image-Turbo",
        load=_load_boogu,
        default_steps=4,
        supports_guidance=False,
        default_scheduler="boogu_dmd",
        supports_image=False,
        supports_lora=False,
        supports_preview=False,
    ),
)

_BY_NAME = {name: spec for spec in MODELS for name in (spec.key, *spec.aliases)}


def resolve(name: str) -> ModelSpec:
    spec = _BY_NAME.get(name.strip().lower())
    if spec is None:
        known = ", ".join(spec.key for spec in MODELS)
        raise ValueError(f"Unknown model {name!r}. Known models: {known}")
    return spec
