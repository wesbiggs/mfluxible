"""Wraps an mflux text-to-image model (Z-Image-Turbo, FLUX.1, Qwen-Image) behind an
async, streaming-friendly interface.

mflux's generate_image() is synchronous: it runs the whole denoising loop in the
calling thread and invokes any registered callbacks in-loop. To turn that into an
async SSE stream, generation runs in a worker thread while an InLoopCallback bridges
each step back to the event loop via call_soon_threadsafe.

Model loading and every generation must run on the SAME single worker thread, not
just "some worker thread" (e.g. via asyncio.to_thread, which uses an ambient pool
that can pick a different thread per call). mflux builds its weight-quantization
graph lazily and never evaluates it during load -- the first real evaluation happens
inside generate_image()'s denoising loop. MLX ties a lazy graph's evaluation to the
thread its stream was registered on, so evaluating a graph built on one thread from
a different thread fails with "There is no Stream(gpu, 0) in current thread." Hence
the single dedicated executor below, used for both load() and every generate call.

Which model this engine runs is a startup choice, one per process -- see models.py.
Everything below is written against the surface all three mflux variants share, so
nothing here branches on the model itself; the per-model differences live in the
ModelSpec.
"""

import asyncio
import base64
import concurrent.futures
import contextlib
import hashlib
import io
import logging
import os
import random
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.utils.image_util import ImageUtil

from models import CFG_GUIDANCE_FLOOR, ModelSpec, resolve
from schedulers import MaskJob, clear_mask_job, scheduler_path, set_mask_job, start_fraction
from schemas import GenerateRequest

# Where anything a *client* must not see goes instead: exception text out of mflux,
# MLX, HF-hub or Pillow, all of which is written for whoever is running the server.
# Nothing configures logging here, so these land on stderr via logging.lastResort --
# i.e. in the terminal running uvicorn -- at WARNING and above, which is every level
# used below.
log = logging.getLogger("mfluxible.engine")

_DONE = object()

DEFAULT_MODEL_CACHE_DIR = Path(os.environ.get("MFLUXIBLE_MODEL_DIR", "~/.cache/mfluxible")).expanduser()

# mflux's own CLI default (mflux.cli.defaults.defaults.IMAGE_STRENGTH), applied here when
# a client sends `image` without `image_strength` -- not copied from mflux.cli itself,
# same reasoning as ModelSpec.default_steps in models.py: that module is CLI-internal.
DEFAULT_IMAGE_STRENGTH = 0.4

# ...and what a *masked* request gets instead when it omits the field. The two differ
# because with a mask the field means something else: without one it decides how much of
# the frame survives, and 0.4 is a reasonable img2img middle ground; with one, the mask
# decides that and image_strength governs only how much of the old content *inside the
# region* survives. At 0.4 that region starts 40% along the schedule and the object meant
# to be replaced comes back very nearly intact -- measured on a 768x768 Z-Image-Turbo run,
# identical seed and mask, 5.1/255 of change inside the mask against 41.6 at 0.0. That is
# a no-op that still costs a full generation and still reports success, which is the worst
# shape a wrong default can have. Omitting an optional field is the ordinary case rather
# than the exception, so this is a default and not a paragraph in docs/api.md.
DEFAULT_MASKED_IMAGE_STRENGTH = 0.0

# MLX holds on to buffers it has freed so it can reuse them instead of asking
# Metal for new ones. That cache is reclaimable, but it still counts toward the
# process's memory footprint, and on a machine where the model already fills most
# of RAM the extra headroom is what tips the system into swapping -- at which
# point every generation pays to fault its weights back in.
#
# Measured, q8 at 512x512 on a 32GB M2 Pro: uncapped, the cache grew to 8.6GB on
# top of 10.1GB of weights (19-20GB footprint) and generations went 6.9s -> 80.6s
# -> 104.6s as the machine started thrashing. Capped at 1GB the footprint sits at
# 12-13GB and the same three runs took 5.2s / 5.3s / 4.9s. The cap cost nothing
# measurable in exchange -- 1GB is ample for reuse within a single generation.
# Set to "none" for MLX's own (effectively uncapped) default.
_raw_mlx_cache_limit = os.environ.get("MFLUXIBLE_MLX_CACHE_LIMIT_MB", "1024").strip()
MLX_CACHE_LIMIT_BYTES = None if _raw_mlx_cache_limit.lower() == "none" else int(_raw_mlx_cache_limit) * 1024 * 1024

# Wiring memory tells the OS it may not page these buffers out at all, which is a
# stronger guarantee than merely fitting: it protects the weights from being
# evicted by pressure from *other* processes between generations. Keep it above
# the model's resident size but well under the GPU's recommended working set
# (mx.device_info()["max_recommended_working_set_size"]) -- wiring too much
# starves the rest of the system. Unset leaves the OS default in place.
_raw_mlx_wired_limit = os.environ.get("MFLUXIBLE_MLX_WIRED_LIMIT_MB", "").strip()
MLX_WIRED_LIMIT_BYTES = int(_raw_mlx_wired_limit) * 1024 * 1024 if _raw_mlx_wired_limit else None

# The server always sends full-resolution images -- both step previews and the
# final image -- at their true requested width/height. Deciding whether/how to
# downscale for display (e.g. to stay under a terminal's max OSC escape-sequence
# length) is a client concern; see stream_client.py / stream_client.js.


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _input_image_problem(b64_data: str) -> str | None:
    """Why a client-supplied base64 input image can't be used, or None if it can.

    Called from request_problem so a garbled `image` field gets the same clean 400 a bad
    guidance/negative_prompt request gets rather than surfacing as an opaque error
    mid-stream.

    Returns the reason rather than raising it, and returns a sentence written here
    rather than the underlying exception's own text -- both for the reasons in
    MfluxEngine.request_problem's docstring. The real exception goes to the log instead,
    where it is worth having and harmless: Pillow's carries a repr of the in-memory
    buffer (address included) and base64's describes this decode call, so neither told a
    client anything it could act on anyway.
    """
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        log.warning("rejecting input image: not valid base64", exc_info=True)
        return "image is not valid base64."
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        log.warning("rejecting input image: not a decodable image", exc_info=True)
        return "image could not be decoded as an image."
    return None


def _decode_input_image(b64_data: str) -> bytes:
    """The actual bytes to write to disk (see mflux's own image_path parameter -- it
    wants a file path, not bytes or a PIL.Image, so there's no way to hand it a decoded
    image directly).

    No validation of its own: generate_stream, its only caller, runs check_request first
    -- so _input_image_problem has already vetted this exact string, and a second round
    of Pillow work per generation would buy nothing.
    """
    return base64.b64decode(b64_data, validate=True)


def _oriented(raw: bytes) -> Image.Image:
    """An image decoded with its EXIF Orientation applied, the way mflux's own
    ImageUtil.load_image does it.

    Used for the mask as well as for measuring the input, and that is the point: mflux
    rotates the input image before encoding it, so a mask compared or composited
    against the unrotated bytes would be measured in one frame and applied in another.
    A mask straight out of a canvas carries no EXIF and this is a no-op for it.
    """
    with Image.open(io.BytesIO(raw)) as img:
        return ImageOps.exif_transpose(img)


def _decode_mask(b64_data: str, feather: int = 0) -> Image.Image:
    """The client's mask as a single-channel image: white where the model may change
    the image, black where the input must survive.

    Read as luminance, so an RGB mask works and an alpha channel is ignored. That is
    worth stating because OpenAI's own edits endpoint uses the opposite convention --
    transparent means "edit here" -- so a mask drawn for that API would come out
    inverted here rather than failing. See docs/api.md.

    Feathering is applied at the mask's own resolution, before any resampling, which
    is why mask_feather is documented in input-image pixels: request_problem has
    already established that the mask and the input image are the same size.
    """
    mask = _oriented(base64.b64decode(b64_data, validate=True)).convert("L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def _composite_through_mask(generated: Image.Image, image_path: str, mask_b64: str, feather: int) -> Image.Image:
    """The generated image pasted back over the input, through the mask.

    The masked blend already holds the kept region during denoising, but the whole
    frame still goes through the VAE on the way out -- measured at roughly 1.5/255
    mean absolute error against the input, which is the encode/decode floor rather
    than anything the blend did wrong. Invisible on a photograph and not on text, a
    logo or a flat colour, so this exists to make "unchanged" mean unchanged.

    Sized off the *generated* image rather than the request: Config floors width and
    height to multiples of 16, so a request for 1000px produces a 992px image, and
    that is the frame both the original and the mask have to land in. The input is
    re-read from the same file mflux encoded, through mflux's own loader, so the two
    agree on orientation and on the resampling filter used to scale it.
    """
    width, height = generated.size
    original = ImageUtil.scale_to_dimensions(
        image=ImageUtil.load_image(image_path).convert("RGB"), target_width=width, target_height=height
    )
    mask = _decode_mask(mask_b64, feather).resize((width, height), Image.LANCZOS)
    return Image.composite(generated.convert("RGB"), original, mask)


def _input_mask_problem(mask_b64: str, image_b64: str) -> str | None:
    """Why a client-supplied mask can't be used, or None if it can. Same contract as
    _input_image_problem: a reason written here, never an exception's own text.

    The size check is strict on purpose. mflux scales the *input image* to
    width/height with a plain resize and no aspect-ratio handling, and a mask that
    doesn't match would be stretched the same way -- silently selecting a different
    region than the one the user painted. Every other failure mode here is loud, so
    this one is too.
    """
    try:
        raw = base64.b64decode(mask_b64, validate=True)
    except Exception:
        log.warning("rejecting mask: not valid base64", exc_info=True)
        return "mask is not valid base64."
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        log.warning("rejecting mask: not a decodable image", exc_info=True)
        return "mask could not be decoded as an image."

    mask_img = _oriented(raw)
    mask_size = mask_img.size
    # Before feathering, which can only spread white that is already there: an
    # all-black mask asks for nothing to change, which is a request nobody means to
    # make and which would otherwise spend a full generation reproducing the input.
    empty = mask_img.convert("L").getextrema()[1] == 0
    image_size = _oriented(base64.b64decode(image_b64, validate=True)).size
    if mask_size != image_size:
        return (
            f"mask is {mask_size[0]}x{mask_size[1]} but image is {image_size[0]}x{image_size[1]}; "
            "they must be the same size, or the mask selects a different region than it looks like."
        )
    if empty:
        return "mask is entirely black, which selects no region to regenerate; omit it or paint some white."
    return None


def _encode_final_png_with_metadata(image) -> bytes:
    # `image` (mflux's GeneratedImage, not the bare PIL image) already carries
    # everything -- prompt, seed, steps, model, quantize, LoRA config,
    # generation time -- that its own .save() embeds as EXIF UserComment, XMP,
    # and IPTC (see mflux.utils.image_util.ImageUtil.save_image and
    # mflux.utils.metadata_builder.MetadataBuilder). Reused as-is rather than
    # hand-rolling a second metadata format; .save() is file-path-based (it
    # re-opens and re-saves), so this round-trips through a temp file.
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(tmp_path, overwrite=True)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


class _StreamCallback:
    """One-shot InLoopCallback: registered for a single generation, then discarded."""

    def __init__(
        self,
        engine: "MfluxEngine",
        preview_every: int,
        emit,
        fractional_start: bool = False,
        mask: tuple[str, int] | None = None,
    ):
        self.engine = engine
        self.preview_every = preview_every
        self.emit = emit
        self.fractional_start = fractional_start
        # (base64 mask, feather radius), or None. Held rather than pre-encoded
        # because building the job has to happen in call_before_loop -- see there.
        self.mask = mask
        self.start_ts = 0.0
        self.last_ts = 0.0

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        if self.mask is not None:
            # The first and only hook that has all three things the job needs at once:
            # the MLX worker thread (it encodes through the VAE), the floored
            # width/height mflux is really generating at (Config rounds both down to a
            # multiple of 16 -- encoding at the requested size instead would produce
            # latents a different shape to the ones the loop carries), and the seed
            # mflux noised the input with. The matching clear_mask_job() is in
            # generate_stream's run().
            mask_b64, feather = self.mask
            set_mask_job(
                self.engine._build_mask_job(
                    mask_b64=mask_b64,
                    feather=feather,
                    image_path=config.image_path,
                    seed=seed,
                    width=config.width,
                    height=config.height,
                )
            )
        self.start_ts = self.last_ts = time.monotonic()
        # Image-to-image doesn't start the denoising loop at step 0: mflux noises the
        # input image to the sigma partway down the schedule and starts there, so the
        # first `thinking` event is step start_step + 1 and only total_steps -
        # start_step steps ever run (mflux.models.common.config.config.Config --
        # init_time_step, and time_steps = range(init_time_step, num_inference_steps)).
        # Without this the client has no way to tell a skipped step from a slow one,
        # and progress against total_steps alone starts partway along.
        start_step = config.init_time_step
        fraction = (
            start_fraction(config.num_inference_steps, config.image_strength, start_step)
            if self.fractional_start
            else 0.0
        )
        self.emit(
            {
                "type": "start",
                "seed": seed,
                "total_steps": config.num_inference_steps,
                "start_step": start_step,
                # init_time_step is an int, so image_strength is normally quantized to
                # 1/steps before it reaches the model: at 9 steps, 0.35 and 0.4 both
                # floor to 3 and generate identical pixels for the same seed. Reporting
                # the bucket the request actually landed in makes that visible instead
                # of leaving a nudged slider looking like it was ignored. This is the
                # bucket's lower edge (start_step / total_steps), which is exact as a
                # fraction but a hair below it as a float -- feeding it back verbatim
                # can floor to the next bucket down, so it's for display, not for
                # round-tripping. With fractional_start the quantization is gone and
                # this reports the strength that actually took effect, which is the
                # requested one except where schedulers.start_fraction documents a
                # clamp -- computed through that same helper so the two can't disagree.
                "effective_image_strength": (
                    (start_step + fraction) / config.num_inference_steps
                    if config.image_path is not None
                    else None
                ),
            }
        )

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        # MLX is lazy, so this callback fires *before* step t has actually been
        # computed: mflux builds the step's graph, calls in-loop subscribers, and
        # only then runs its own mx.eval(latents) (see the denoising loop in each
        # variant, e.g. mflux.models.z_image.variants.z_image). Timing without
        # forcing evaluation therefore attributes step t-1's compute to step t --
        # step 1 reports ~10ms and every later timestamp lags a full step. This
        # eval costs nothing, since mflux evaluates the same graph on its very
        # next line; it just moves the wait to before the clock is read. Don't
        # remove it, or the timings silently go back to being off by one.
        mx.eval(latents)
        now = time.monotonic()
        step = t + 1
        event = {
            "type": "thinking",
            "step": step,
            "total_steps": config.num_inference_steps,
            "step_ms": int((now - self.last_ts) * 1000),
            "elapsed_ms": int((now - self.start_ts) * 1000),
        }
        # Re-baselined before the preview decode, not after, so a preview's cost
        # lands in the next step's step_ms rather than vanishing -- that keeps the
        # step_ms values summing to elapsed_ms.
        self.last_ts = now
        if self.preview_every and step % self.preview_every == 0:
            event["preview"] = self.engine._decode_preview_b64(latents, config, seed, prompt)
        self.emit(event)

    def call_interrupt(self, t, seed, prompt, latents, config, time_steps):
        self.emit({"type": "error", "message": f"generation interrupted at step {t + 1}"})


class MfluxEngine:
    def __init__(
        self,
        model: str | ModelSpec = "z-image-turbo",
        quantize: int | None = 8,
        model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # Resolved eagerly so an unknown MFLUXIBLE_MODEL fails at startup with the
        # list of valid names, rather than after the first weight download.
        self.spec = model if isinstance(model, ModelSpec) else resolve(model)
        self.quantize = quantize
        self.model_cache_dir = model_cache_dir
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model = None
        self._latent_creator = None
        self._lock = asyncio.Lock()
        # Single worker: all MLX work (load + every generate call) must run on this
        # same OS thread -- see the module docstring for why.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")

    def _lora_cache_suffix(self) -> str:
        # No LoRA configured -> no suffix, so existing caches from before this
        # feature keep working unchanged. Different LoRA path/scale combos must
        # get distinct cache dirs (baking is permanent -- see _load_sync), so the
        # suffix is a hash of the exact config, not just "a LoRA was used".
        if not self.lora_paths:
            return ""
        scales = self.lora_scales or [1.0] * len(self.lora_paths)
        key = "|".join(f"{p}:{s}" for p, s in zip(self.lora_paths, scales))
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        return f"-lora-{digest}"

    def _saved_model_dir(self) -> Path | None:
        # The model key leads the directory name for the same reason the LoRA hash
        # trails it: the marker file below only answers "was *something* saved
        # here", so anything that changes what the saved weights contain has to
        # change the path. (z-image-turbo keeps its original name, so caches
        # written before this server knew about other models still load.)
        if self.model_cache_dir is None:
            return None
        bits_label = str(self.quantize) if self.quantize is not None else "full"
        return self.model_cache_dir / f"{self.spec.key}-q{bits_label}{self._lora_cache_suffix()}"

    def _load_sync(self) -> None:
        if MLX_CACHE_LIMIT_BYTES is not None:
            mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        if MLX_WIRED_LIMIT_BYTES is not None:
            mx.set_wired_limit(MLX_WIRED_LIMIT_BYTES)

        # First point at which anything mflux-model-specific is imported, and the
        # only point at which weights are downloaded -- so the models this process
        # was not configured for cost nothing beyond their entry in models.py.
        model_cls, model_config, self._latent_creator = self.spec.load()

        saved_dir = self._saved_model_dir()
        marker = saved_dir / "transformer" / "model.safetensors.index.json" if saved_dir else None

        if marker is not None and marker.exists():
            # Already quantized, and already LoRA-baked if any LoRA was configured
            # for this cache dir -- lora_paths must NOT be passed again here, or
            # it would apply LoRA a second time on top of the already-baked
            # weights. See _lora_cache_suffix: a different LoRA config gets its
            # own cache dir, so "this dir exists" already implies "this exact
            # LoRA config, if any, is what's baked into it."
            #
            # model_config still has to be passed: a saved directory carries
            # weights and tokenizers, not the model's scheduler/sequence-length
            # settings, and each variant's own default would otherwise win (Flux1
            # defaults to schnell, so a flux-dev cache would load as schnell).
            self.model = model_cls(model_path=str(saved_dir), quantize=self.quantize, model_config=model_config)
        else:
            self.model = model_cls(
                quantize=self.quantize,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
                model_config=model_config,
            )
            if saved_dir is not None:
                saved_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_model(str(saved_dir))

    async def load(self) -> None:
        """Loads the configured model on the dedicated worker thread, quantizing it
        once and caching the quantized weights to disk on first run (see mflux's
        `mflux-save` / `save_model`) so later startups skip re-quantizing -- only the
        raw weights need downloading once. Called once at server startup.
        """
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_sync)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _decode_preview_b64(self, latents, config, seed, prompt) -> str:
        # Mirrors mflux.callbacks.instances.stepwise_handler.StepwiseHandler._save_image,
        # which is likewise written once for every model and told which latent
        # creator to use (mflux's CLIs pass Flux/Qwen/ZImageLatentCreator the same way).
        model = self.model
        unpacked = self._latent_creator.unpack_latents(latents=latents, height=config.height, width=config.width)
        vae_latent_channels = getattr(model.vae, "latent_channels", 32)
        if hasattr(model.vae, "decode_packed_latents") and unpacked.shape[1] > vae_latent_channels:
            decoded = model.vae.decode_packed_latents(unpacked)
        else:
            # VAEUtil rather than vae.decode() directly, which is where StepwiseHandler
            # stops: Qwen-Image's VAE is a 3D (video) decoder and returns
            # (B, C, 1, H, W), and ImageUtil.to_image wants 4D. VAEUtil.decode is what
            # drops that singleton frame axis -- it's the same call each variant's own
            # final decode makes, so the preview and the final image go through
            # identical handling.
            tiling = getattr(model, "tiling_config", None)
            decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=tiling)
        wrapped = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,
            generation_time=0,
        )
        return _pil_to_b64_png(wrapped.image)

    def _build_mask_job(
        self, *, mask_b64: str, feather: int, image_path: str, seed: int, width: int, height: int
    ) -> MaskJob:
        """The three packed arrays schedulers.MaskJob blends between.

        Runs on the MLX worker thread -- it encodes through the VAE, and this engine's
        whole threading rule (see the module docstring) is that every MLX graph is
        built and evaluated on the one thread the model was loaded on.

        No per-model branching, for the same reason the rest of this file has none.
        `clean` goes through mflux's own LatentCreator.encode_image, which is exactly
        what the img2img path already called to build the starting latents, so the two
        agree by construction rather than by matching implementations. `noise` is
        regenerated from the same seed mflux used, so the kept region is re-noised
        with the sample the trajectory actually started from. And the mask reaches the
        latents' own layout through the model's own pack_latents -- packing is a pure
        spatial rearrangement, so replicating one mask value across the channel axis
        and packing it lands each element's mask on that element, whether the variant
        packs 2x2 patches into channels (FLUX, Qwen) or only shuffles axes (Z-Image).
        """
        model = self.model
        tiling = getattr(model, "tiling_config", None)
        encoded = LatentCreator.encode_image(
            vae=model.vae, image_path=image_path, width=width, height=height, tiling_config=tiling
        )
        clean = self._latent_creator.pack_latents(encoded, height, width)
        noise = self._latent_creator.create_noise(seed, height, width)

        # The encoder's own output shape, rather than a hardcoded /8: it is the one
        # place that knows this VAE's downsample factor and channel count.
        _, channels, lat_h, lat_w = encoded.shape
        # BOX, not LANCZOS: this is an area average down to the latent grid, and a
        # windowed filter would ring past [0, 1] at a hard mask edge -- overshoot that
        # becomes "more than the original" or "less than nothing" in the blend.
        mask = _decode_mask(mask_b64, feather).resize((lat_w, lat_h), Image.BOX)
        keep = 1.0 - np.asarray(mask, dtype=np.float32) / 255.0
        keep = mx.repeat(mx.array(keep).reshape(1, 1, lat_h, lat_w), channels, axis=1)
        keep = self._latent_creator.pack_latents(keep, height, width)
        return MaskJob(keep=keep.astype(clean.dtype), clean=clean, noise=noise)

    def request_problem(self, req: GenerateRequest) -> str | None:
        """Why this model cannot honour `req`, or None if it can.

        Called by the endpoint before the response starts, so a bad request is a 400
        rather than an exception thrown mid-stream once the SSE headers are already
        out; generate_stream calls check_request below, so a direct caller gets the
        same guarantee.

        A return value rather than an exception because that is what this is -- a
        verdict on the request, not a failure -- and because keeping it one is what
        makes the next part checkable at a glance: every string that leaves here is
        written in this function, for a client to read. Text from inside mflux, MLX or
        Pillow reaches a client only by riding an exception object out to a response
        body, so no path from an exception to a response is the invariant, and
        server.py holds the other end of it (nothing there does str(exc) either).
        """
        if req.guidance is not None and not self.spec.supports_guidance:
            return f"{self.spec.label} ignores guidance (it is guidance-distilled); omit the field."
        if req.negative_prompt is not None:
            if not self.spec.supports_negative_prompt:
                return f"{self.spec.label} has no negative-prompt branch; omit the field."
            # Having a negative branch isn't enough -- it has to be switched on. Every
            # CFG model here encodes the unconditional prompt only above
            # CFG_GUIDANCE_FLOOR, so at or below it the negative prompt would be
            # accepted and then never consulted, which is exactly the silent drop the
            # checks around it exist to prevent. This bites on Krea-2, whose own default
            # guidance is 1.0; on every other CFG model the default already clears the
            # floor and this only fires if a request lowers it.
            guidance = req.guidance if req.guidance is not None else self.spec.default_guidance
            if guidance is not None and guidance <= CFG_GUIDANCE_FLOOR:
                return (
                    f"{self.spec.label} only encodes a negative prompt above guidance "
                    f"{CFG_GUIDANCE_FLOOR} (classifier-free guidance is off at or below it, and "
                    f"this request's guidance is {guidance}); raise guidance or omit negative_prompt."
                )
        if req.image_strength is not None and req.image is None:
            return "image_strength requires image to also be set."
        if req.image is not None:
            if req.image_strength is not None and not (0.0 <= req.image_strength <= 1.0):
                return "image_strength must be between 0.0 and 1.0."
            problem = _input_image_problem(req.image)
            if problem is not None:
                return problem
            if req.mask is not None:
                # After the image, not before: the size check needs an image it can
                # open, and running it on one already known to be undecodable would
                # report the wrong field.
                problem = _input_mask_problem(req.mask, req.image)
                if problem is not None:
                    return problem
        else:
            if req.fractional_start:
                return "fractional_start requires image to also be set."
            if req.mask is not None:
                return "mask requires image to also be set -- there is nothing to mask a region of."
        if req.mask is None:
            # Both fields only describe how a mask is applied, so sending one without
            # a mask is a request that can't be honoured rather than a harmless extra
            # -- the same rule image_strength follows above. Only a *non-default*
            # value is a request: the defaults are what every maskless call sends.
            if req.mask_feather:
                return "mask_feather requires mask to also be set."
            if not req.mask_composite:
                return "mask_composite requires mask to also be set."
        elif not self.spec.supports_mask:
            # Same root cause as the fractional_start check below: holding the
            # unmasked region in place means replacing the variant's scheduler, and
            # replacing it is only free on a model that would have run linear anyway.
            # See ModelSpec.supports_mask and schedulers.py's masked-inpainting note.
            return (
                f"{self.spec.label} runs mflux's {self.spec.default_scheduler!r} scheduler, and "
                "mask needs the linear schedule server/schedulers.py extends; omit the field "
                "(image alone still works, regenerating the whole frame)."
            )
        if req.fractional_start and not self.spec.supports_fractional_start:
            # SCHEDULER_PATH *replaces* whatever scheduler the variant would have picked
            # for itself, so it is only a fractional start on a model that would have
            # run a linear schedule anyway. Elsewhere it either swaps the sampler
            # silently (flow-match models: wrong images, no error) or raises inside
            # generate_image() on the worker thread with the SSE headers already out
            # (Krea2). See ModelSpec.default_scheduler.
            return (
                f"{self.spec.label} runs mflux's {self.spec.default_scheduler!r} scheduler, and "
                "fractional_start only applies to models on the linear schedule it extends; "
                "omit the field (image_strength still works, quantized to 1/steps)."
            )
        return None

    def check_request(self, req: GenerateRequest) -> None:
        """request_problem, raised instead of returned, for in-process callers that
        have no response to put it in (generate_stream, and the tests)."""
        problem = self.request_problem(req)
        if problem is not None:
            raise ValueError(problem)

    def _generation_kwargs(self, req: GenerateRequest) -> dict:
        """Per-request knobs that only some models can act on.

        Guidance and negative prompts are passed only where they do something. Every
        variant *accepts* both arguments, but on a guidance-distilled model (Z-Image
        Turbo, FLUX.1-schnell) the value has no path to the output, and mflux's own
        CLIs warn rather than pretend otherwise -- so rather than accept a value and
        silently drop it, request_problem above rejects the field outright.
        """
        kwargs = {}
        if self.spec.supports_guidance:
            kwargs["guidance"] = req.guidance if req.guidance is not None else self.spec.default_guidance
        if self.spec.supports_negative_prompt and req.negative_prompt is not None:
            kwargs["negative_prompt"] = req.negative_prompt
        return kwargs

    async def generate_stream(self, req: GenerateRequest):
        """Yields event dicts: {"type": "start"|"thinking"|"image"|"error", ...}."""
        if self.model is None:
            raise RuntimeError("model not loaded")

        self.check_request(req)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        steps = req.steps if req.steps is not None else self.spec.default_steps
        kwargs = self._generation_kwargs(req)
        callback = _StreamCallback(
            self,
            req.preview_every,
            emit,
            fractional_start=req.fractional_start,
            mask=(req.mask, req.mask_feather) if req.mask is not None else None,
        )

        # mflux's image_path wants an actual file on disk, not bytes or a PIL.Image (see
        # _decode_input_image) -- written up front, outside the lock, since encoding a
        # few hundred KB doesn't need it. Removed in the outer finally below, which only
        # runs once the generation thread is confirmed done with it (see that finally's
        # own comment on why it must wait for `task`).
        tmp_image_path: str | None = None
        if req.image is not None:
            image_bytes = _decode_input_image(req.image)
            fd, tmp_image_path = tempfile.mkstemp(suffix=".input-image")
            os.close(fd)
            with open(tmp_image_path, "wb") as f:
                f.write(image_bytes)
            kwargs["image_path"] = tmp_image_path
            if req.image_strength is not None:
                kwargs["image_strength"] = req.image_strength
            else:
                # Which default applies is the mask's call rather than the field's --
                # see DEFAULT_MASKED_IMAGE_STRENGTH for why they differ.
                kwargs["image_strength"] = (
                    DEFAULT_MASKED_IMAGE_STRENGTH if req.mask is not None else DEFAULT_IMAGE_STRENGTH
                )
            # A dotted path mflux imports, chosen from a fixed table and never built
            # from request data -- see SCHEDULER_PATH's comment. None for an ordinary
            # img2img request, which leaves the variant whatever scheduler it picks
            # for itself.
            path = scheduler_path(masked=req.mask is not None, fractional=req.fractional_start)
            if path is not None:
                kwargs["scheduler"] = path

        try:
            # Only one generation at a time: MLX/Metal + the shared callback list on
            # self.model aren't set up for concurrent generate_image() calls.
            async with self._lock:
                self.model.callbacks.before_loop.append(callback)
                self.model.callbacks.in_loop.append(callback)
                self.model.callbacks.interrupt.append(callback)

                def run():
                    try:
                        return self.model.generate_image(
                            seed=seed,
                            prompt=req.prompt,
                            num_inference_steps=steps,
                            width=req.width,
                            height=req.height,
                            **kwargs,
                        )
                    finally:
                        # Set in the before_loop callback, cleared here rather than
                        # there: an exception anywhere in the loop skips after_loop
                        # entirely, and a job left behind would be picked up by the
                        # next generation's scheduler. This finally is the one place
                        # that runs on every path out of generate_image().
                        clear_mask_job()
                        emit({"type": _DONE})

                # run_in_executor submits to the executor immediately and returns a Future
                # (not a coroutine) -- no create_task() wrapper needed or valid here.
                task = loop.run_in_executor(self._executor, run)
                try:
                    while True:
                        item = await queue.get()
                        if item.get("type") is _DONE:
                            break
                        yield item

                    image = await task
                    if req.mask is not None and req.mask_composite:
                        # On the event loop rather than the worker thread on purpose:
                        # this is Pillow, not MLX, so it is not bound by this engine's
                        # single-thread rule, and it runs after the generation has
                        # released its hold on the model.
                        image.image = _composite_through_mask(
                            image.image, tmp_image_path, req.mask, req.mask_feather
                        )
                    yield {
                        "type": "image",
                        "mime_type": "image/png",
                        "data": base64.b64encode(_encode_final_png_with_metadata(image)).decode("ascii"),
                        "seed": seed,
                        "generation_time": image.generation_time,
                    }
                except Exception:
                    # Everything mflux, MLX, HF-hub or Pillow can throw arrives here,
                    # and none of it is written for a client: an HF-hub miss or a
                    # failed weight load quotes absolute cache paths, which on the
                    # default ~/.cache layout means handing out the operator's
                    # username and disk layout. The whole traceback goes to the log
                    # (the terminal running uvicorn, for a local server) and the
                    # client is told that it failed and where to look.
                    log.exception("generation failed")
                    yield {
                        "type": "error",
                        "message": "generation failed -- see the server log for the reason.",
                    }
                finally:
                    # mflux has no way to interrupt generate_image() from outside the
                    # thread it's running on (its only interrupt path is a literal
                    # KeyboardInterrupt on the server process, not a client hangup) --
                    # so if we're getting here early (e.g. GeneratorExit from a
                    # disconnected client), the background thread is still running
                    # regardless. We must not unregister this callback, or let the
                    # lock above release, until that thread genuinely finishes:
                    # self.model.callbacks.in_loop is a single shared, unsynchronized
                    # list, and letting a new request start while an abandoned
                    # generation is still iterating it lets that zombie thread invoke
                    # the NEW request's callback and leak bogus events into its
                    # stream -- reproduced empirically, not just theoretical.
                    if not task.done():
                        with contextlib.suppress(Exception):
                            await task
                    self.model.callbacks.before_loop.remove(callback)
                    self.model.callbacks.in_loop.remove(callback)
                    self.model.callbacks.interrupt.remove(callback)
        finally:
            if tmp_image_path is not None:
                os.unlink(tmp_image_path)
