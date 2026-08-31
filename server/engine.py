"""Wraps mflux's ZImage (Z-Image-Turbo) model behind an async, streaming-friendly interface.

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
"""

import asyncio
import base64
import concurrent.futures
import contextlib
import hashlib
import io
import os
import random
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.image_util import ImageUtil

from schemas import GenerateRequest

_DONE = object()

DEFAULT_MODEL_CACHE_DIR = Path(os.environ.get("MFLUXIBLE_MODEL_DIR", "~/.cache/mfluxible")).expanduser()

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

    def __init__(self, engine: "ZImageEngine", preview_every: int, emit):
        self.engine = engine
        self.preview_every = preview_every
        self.emit = emit
        self.start_ts = 0.0
        self.last_ts = 0.0

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        self.start_ts = self.last_ts = time.monotonic()
        self.emit({"type": "start", "seed": seed, "total_steps": config.num_inference_steps})

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        # MLX is lazy, so this callback fires *before* step t has actually been
        # computed: mflux builds the step's graph, calls in-loop subscribers, and
        # only then runs its own mx.eval(latents) (see the denoising loop in
        # mflux.models.z_image.variants.z_image). Timing without forcing
        # evaluation first therefore attributes step t-1's compute to step t --
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


class ZImageEngine:
    def __init__(
        self,
        quantize: int | None = 8,
        model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        self.quantize = quantize
        self.model_cache_dir = model_cache_dir
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model: ZImage | None = None
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
        if self.model_cache_dir is None:
            return None
        bits_label = str(self.quantize) if self.quantize is not None else "full"
        return self.model_cache_dir / f"z-image-turbo-q{bits_label}{self._lora_cache_suffix()}"

    def _load_sync(self) -> None:
        if MLX_CACHE_LIMIT_BYTES is not None:
            mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        if MLX_WIRED_LIMIT_BYTES is not None:
            mx.set_wired_limit(MLX_WIRED_LIMIT_BYTES)
        saved_dir = self._saved_model_dir()
        marker = saved_dir / "transformer" / "model.safetensors.index.json" if saved_dir else None

        if marker is not None and marker.exists():
            # Already quantized, and already LoRA-baked if any LoRA was configured
            # for this cache dir -- lora_paths must NOT be passed again here, or
            # it would apply LoRA a second time on top of the already-baked
            # weights. See _lora_cache_suffix: a different LoRA config gets its
            # own cache dir, so "this dir exists" already implies "this exact
            # LoRA config, if any, is what's baked into it."
            self.model = ZImage(model_path=str(saved_dir), quantize=self.quantize)
        else:
            self.model = ZImage(
                quantize=self.quantize,
                lora_paths=self.lora_paths,
                lora_scales=self.lora_scales,
            )
            if saved_dir is not None:
                saved_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_model(str(saved_dir))

    async def load(self) -> None:
        """Loads the model on the dedicated worker thread, quantizing it once and
        caching the quantized weights to disk on first run (see mflux's `mflux-save`
        / `ZImage.save_model`) so later startups skip re-quantizing -- only the raw
        weights need downloading once. Called once at server startup.
        """
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_sync)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _decode_preview_b64(self, latents, config, seed, prompt) -> str:
        # Mirrors mflux.callbacks.instances.stepwise_handler.StepwiseHandler._save_image
        model = self.model
        unpacked = ZImageLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        vae_latent_channels = getattr(model.vae, "latent_channels", 32)
        if hasattr(model.vae, "decode_packed_latents") and unpacked.shape[1] > vae_latent_channels:
            decoded = model.vae.decode_packed_latents(unpacked)
        else:
            decoded = model.vae.decode(unpacked)
        wrapped = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,
            generation_time=0,
        )
        return _pil_to_b64_png(wrapped.image)

    async def generate_stream(self, req: GenerateRequest):
        """Yields event dicts: {"type": "start"|"thinking"|"image"|"error", ...}."""
        if self.model is None:
            raise RuntimeError("model not loaded")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        callback = _StreamCallback(self, req.preview_every, emit)

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
                        num_inference_steps=req.steps,
                        width=req.width,
                        height=req.height,
                    )
                finally:
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
                yield {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": base64.b64encode(_encode_final_png_with_metadata(image)).decode("ascii"),
                    "seed": seed,
                    "generation_time": image.generation_time,
                }
            except Exception as exc:
                yield {"type": "error", "message": str(exc)}
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
