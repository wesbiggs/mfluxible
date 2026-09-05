"""A scheduler that lets image-to-image start *between* two rungs of the sigma schedule.

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
