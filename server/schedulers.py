"""Two things mflux's scheduler slot is used for here, and one of them is not a schedule.

The first is below: a scheduler that lets image-to-image start *between* two rungs of the
sigma schedule. The second is at the bottom of the file: masked inpainting, which needs a
place to write the untouched region back once per step and finds it in `step()` -- the only
call whose return value becomes the next step's latents. They share this module because
they share the slot: mflux takes exactly one `scheduler=`, so the two can't be handed over
separately and the combinations are spelled out as classes (see `scheduler_path`).

mflux turns `image_strength` into a single integer -- `init_time_step = max(1, int(steps
* image_strength))` (`mflux.models.common.config.config.Config`) -- and then uses that
one integer for two different jobs:

1. the step the denoising loop starts at (`range(init_time_step, num_inference_steps)`),
2. the index of the noise level blended into the input image, `sigmas[init_time_step]`
   (`mflux.models.common.latent_creator.latent_creator.LatentCreator`).

Job 1 genuinely has to be an integer: each step integrates between adjacent grid points
(`dt = sigmas[t+1] - sigmas[t]`), so there is no such thing as starting halfway through
one. Job 2 doesn't -- it's an array lookup, and nothing requires the noise level to land
on a grid point. Sharing the integer is what quantizes `image_strength` to `1/steps`: at
9 steps there are ten reachable settings, and 0.35 and 0.4 give byte-identical pixels.

This scheduler separates the two by moving the rung instead of the index. It keeps
whatever schedule the request's own step count produces and replaces `sigmas[init]` with
a point interpolated toward its neighbour, at the exact position `image_strength` asked
for. The loop still starts at `init` and still runs `steps - init` steps -- only its
first step is shorter (or longer). So the extra granularity costs nothing to run, and
sweeping a strength range no longer changes the step count underneath you.

**Why moving the rung is safe rather than a latents/conditioning mismatch:** all three
variants condition the transformer on `sigmas[t]` itself, not on the step index -- see
`z_image.py`'s `sigma_t = config.scheduler.sigmas[t]; timestep = 1 - sigma_t`, and
`flux_transformer/transformer.py:153` / `qwen_transformer.py:101`, both of which read
`config.scheduler.sigmas[...]`. Moving the rung therefore moves the model's conditioning
with it, and the latents it receives are noised to exactly the level it is told they
are. If a future mflux ever conditioned on the index instead, this would silently desync
the two -- that's the thing to re-check if generated images start coming out wrong here.

**Why `LinearScheduler` is the base class:** every model this server runs resolves to
mflux's `"linear"` scheduler by default -- `Flux1` and `QwenImage` default the parameter
itself, and `ZImage` picks `"flow_match_euler_discrete"` only when `supports_guidance`
is true, which Z-Image-Turbo is not. Subclassing keeps the schedule identical to the one
that would otherwise have been used, with a single entry moved. A model whose default is
*not* linear would have its sampler silently swapped by asking for a fractional start,
so check that before adding one to `models.py`.

**The interpolation is on the shifted sigmas**, i.e. on the schedule the request itself
would have used, rather than re-deriving mflux's sigma-shift math for a finer grid. That
keeps zero copies of mflux internals here, at the cost of a small deviation from "the
rung you'd get by running more steps": for Z-Image-Turbo at 1024x1024, a half-step
interpolation lands within 0.001 of the true 20-step rung across the low-sigma-index
region img2img actually uses, growing to 0.013 at the very tail of the schedule. Note
also that "more steps gives you the same rungs plus extra ones" is itself only true for
models with no `sigma_shift_terminal` (Z-Image-Turbo, FLUX): Qwen-Image sets it to 0.02,
and its terminal stretch is scaled by the last raw sigma, `1/steps`, so its grids at
different step counts don't nest at all. Interpolating the request's own schedule is
well-defined either way, which is the other reason to do it this way round.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler

# What mflux resolves back to this class. Config accepts a scheduler as a dotted path
# and imports it (`try_import_external_scheduler`), so this string is the whole wiring.
# It resolves because server/ is on sys.path as flat modules -- the same reason
# `from engine import ...` works (see CLAUDE.md). Never build this string from request
# data: an arbitrary dotted path is an arbitrary module import in the server process,
# which is why the API exposes a bool and picks the path itself.
SCHEDULER_PATH = "schedulers.FractionalStartLinearScheduler"


def start_fraction(num_inference_steps: int, image_strength: float | None, init_time_step: int) -> float:
    """How far past `init_time_step` the requested `image_strength` actually falls, in [0, 1).

    0.0 means the strength landed exactly on a rung (or on one of the edge cases below),
    so a fractional start is a no-op and the schedule is left alone. Shared by the
    scheduler and by engine.py's `start` event, so the strength the client is told took
    effect and the one that did are the same number by construction.

    Two edge cases collapse to 0.0 deliberately, both because mflux's own clamps have
    already moved the start off the position the strength names:

    - `init_time_step >= num_inference_steps` (strength 1.0): there is no `init + 1`
      rung to interpolate toward, and the schedule's last sigma is 0 -- no denoising
      steps run at all, so there is nothing to place between.
    - a strength below `1/steps`, which `max(1, ...)` floors *up* to rung 1. Honouring
      the fraction there would mean interpolating back toward `sigmas[0]` (pure noise),
      i.e. extrapolating past the earliest start mflux considers img2img at all.
    """
    if not image_strength or init_time_step >= num_inference_steps:
        return 0.0
    position = min(max(float(image_strength), 0.0), 1.0) * num_inference_steps
    return min(max(position - init_time_step, 0.0), 1.0)


class FractionalStartLinearScheduler(LinearScheduler):
    """mflux's linear schedule with the img2img starting rung moved to a fractional position.

    Reads everything it needs off the `Config` it is constructed with, so there is no
    state to hand it and no way for the fraction it applies to disagree with the one the
    request asked for.
    """

    def __init__(self, config):
        # Computed before super().__init__, which calls _get_sigmas() below. Uses
        # config.init_time_step rather than recomputing max(1, int(...)) so mflux stays
        # the single authority on where a given strength starts.
        self.start_fraction = start_fraction(
            config.num_inference_steps, config.image_strength, config.init_time_step
        )
        self.init_time_step = config.init_time_step
        super().__init__(config)

    def _get_sigmas(self) -> mx.array:
        sigmas = super()._get_sigmas()
        if self.start_fraction <= 0.0:
            return sigmas
        i = self.init_time_step
        moved = sigmas[i] * (1.0 - self.start_fraction) + sigmas[i + 1] * self.start_fraction
        return mx.concatenate([sigmas[:i], moved.reshape(1), sigmas[i + 1 :]])


# ---------------------------------------------------------------------------
# Masked inpainting
# ---------------------------------------------------------------------------
#
# Regenerating only part of an image needs one thing mflux does not expose: a place
# to put the untouched region back, once per step, *inside* the denoising loop. The
# callback registry can't do it -- a variant's loop reads
# `latents = config.scheduler.step(...)` and only then calls `ctx.in_loop(t, latents)`,
# and MLX arrays are immutable, so a subscriber can observe the trajectory but never
# influence it. `scheduler.step()` is the only call whose return value becomes the
# next step's latents, which is why the blend lives here rather than in engine.py.
#
# What gets blended back is exact rather than approximate, because all three variants
# noise an image by plain interpolation -- mflux's own
# `LatentCreator.add_noise_by_interpolation` is `(1 - sigma) * clean + sigma * noise`.
# So "the original, noised to the level this step just reached" is that one line at
# `sigmas[timestep + 1]`, using the same noise sample the run started from. On the
# last step sigma is 0 and the kept region lands on the encoded original exactly.


@dataclass(frozen=True)
class MaskJob:
    """The three packed arrays a masked generation blends between, all shaped exactly
    like the latents the loop carries.

    `keep` is 1.0 where the original must survive and 0.0 where the model is free --
    the inverse of the API's mask, which is white where the user wants a change.
    Fractional values in between are what a feathered mask edge becomes, and they
    cross-fade rather than picking a side.
    """

    keep: mx.array
    clean: mx.array
    noise: mx.array

    def blend(self, latents: mx.array, sigma: mx.array) -> mx.array:
        known = (1.0 - sigma) * self.clean + sigma * self.noise
        keep = self.keep.astype(latents.dtype)
        return (1.0 - keep) * latents.astype(self.clean.dtype) + keep * known


# The job for the generation currently on the MLX worker thread, or None.
#
# Out-of-band state, which the fractional-start scheduler above deliberately avoids --
# it reads everything off the Config it is handed. There is no Config route for this:
# mflux constructs the scheduler inside Config.__init__, from the config alone, and a
# Config carries no mask, no encoded latents and no seed. Building them needs the VAE
# and therefore the MLX worker thread, which is exactly where `Config(...)` is running.
#
# It is safe here for reasons that are properties of this server rather than of this
# module, so they are worth naming: MfluxEngine serializes every generation behind an
# asyncio.Lock and runs all MLX work on one dedicated worker thread, so at most one job
# can ever be live. engine.py sets it inside the same `run()` that calls
# generate_image() and clears it in that function's `finally`, so the window is one
# generation wide and an exception cannot leave it set. If either of those two
# invariants goes away, this has to become per-Config state, not a wider lock.
_active_job: MaskJob | None = None


def set_mask_job(job: MaskJob) -> None:
    global _active_job
    _active_job = job


def clear_mask_job() -> None:
    global _active_job
    _active_job = None


def active_mask_job() -> MaskJob | None:
    return _active_job


class _MaskedBlend:
    """Mixin: hold the unmasked region to the input image after every step.

    The job is read per step rather than captured at construction, and the check for
    a missing one lives here rather than in __init__, because of *when* the job can
    exist. Building it needs the floored width/height mflux actually generates at
    (`Config` rounds both down to a multiple of 16), and the earliest that is knowable
    is the `before_loop` callback -- by which point mflux has already resolved
    `config.scheduler` to build the starting latents. So the scheduler necessarily
    exists before its job does; only `step` can insist on one.

    Missing means raise, not run unmasked. The raise lands inside generate_image() on
    the worker thread, where engine.py's own `except Exception` turns it into a logged
    traceback and a fixed client-facing message -- a visibly failed generation, where
    the alternative is an image that looks fine and quietly ignored the mask.
    """

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        latents = super().step(noise=noise, timestep=timestep, latents=latents, **kwargs)
        job = active_mask_job()
        if job is None:
            raise RuntimeError(
                f"{type(self).__name__} ran with no mask job set; "
                "engine.py must call schedulers.set_mask_job before the loop starts."
            )
        # self.sigmas, not the config's -- with FractionalStartLinearScheduler in the
        # MRO this is the schedule with the starting rung moved, and the blend has to
        # re-noise the kept region to the level the trajectory is actually on.
        return job.blend(latents, self.sigmas[timestep + 1])


class MaskedBlendLinearScheduler(_MaskedBlend, LinearScheduler):
    """Masked inpainting on the stock linear schedule."""


class MaskedFractionalStartScheduler(_MaskedBlend, FractionalStartLinearScheduler):
    """Masked inpainting with the fractional starting rung as well.

    Two classes rather than one that reads a flag, because mflux is told which
    scheduler to use by a dotted path and nothing else: without a mask, *not* passing
    SCHEDULER_PATH is what says "no fractional start". Once a mask forces a scheduler
    to be passed on every request, that signal is gone, and a single class would have
    to infer the fractional start from `config.image_strength` -- silently turning it
    on for anyone whose strength happens to fall between two rungs, which is a
    different image than they asked for.
    """


# The only four values engine.py may pass to mflux as `scheduler=`. A dotted path is
# an arbitrary module import in the server process, so this is a fixed table keyed by
# two bools rather than anything assembled from a request -- see SCHEDULER_PATH above.
def scheduler_path(*, masked: bool, fractional: bool) -> str | None:
    if masked and fractional:
        return "schedulers.MaskedFractionalStartScheduler"
    if masked:
        return "schedulers.MaskedBlendLinearScheduler"
    if fractional:
        return SCHEDULER_PATH
    return None
