"""Coverage of server/schedulers.py: the fractional-start schedule itself.

Weight-free like the rest of the suite, but deliberately *not* against ToyModel -- what
matters here is the sigma array a real mflux Config produces, so these run against real
ModelConfigs (no weights are loaded by building a Config or a scheduler).
"""

import mlx.core as mx
import pytest
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig

from schedulers import (
    SCHEDULER_PATH,
    FractionalStartLinearScheduler,
    MaskJob,
    _MaskedBlend,
    clear_mask_job,
    scheduler_path,
    set_mask_job,
    start_fraction,
)

STEPS = 10


def _config(strength, scheduler="linear", steps=STEPS, model="z-image-turbo", size=1024):
    # image_path only has to be non-None for Config to treat this as img2img -- nothing
    # in the scheduler path opens it (the VAE encode that does never runs here).
    return Config(
        model_config=ModelConfig.from_name(model),
        num_inference_steps=steps,
        width=size,
        height=size,
        guidance=0.0,
        image_path="unread.png",
        image_strength=strength,
        scheduler=scheduler,
    )


def _sigmas(strength, **kw):
    return _config(strength, scheduler="schedulers.FractionalStartLinearScheduler", **kw).scheduler.sigmas.tolist()


def _stock_sigmas(strength, **kw):
    return _config(strength, **kw).scheduler.sigmas.tolist()


def test_mflux_resolves_the_scheduler_by_its_dotted_path():
    # The whole wiring is a string: engine.py hands mflux SCHEDULER_PATH and mflux
    # imports it (try_import_external_scheduler). If server/ ever stops being importable
    # as flat modules, this is what breaks, and it breaks here rather than mid-request.
    assert isinstance(_config(0.25, "schedulers.FractionalStartLinearScheduler").scheduler,
                      FractionalStartLinearScheduler)


def test_a_strength_between_two_rungs_moves_only_that_rung():
    # 0.25 at 10 steps: init_time_step is still 2, but the noise level the input is
    # blended to now sits halfway between rung 2 and rung 3 instead of on rung 2.
    stock = _stock_sigmas(0.25)
    moved = _sigmas(0.25)

    assert moved[2] == pytest.approx((stock[2] + stock[3]) / 2)
    assert moved[:2] == stock[:2]
    assert moved[3:] == stock[3:]


def test_strengths_that_land_on_a_rung_leave_the_schedule_alone():
    # 0.2 and 0.3 are exactly rungs 2 and 3 at 10 steps, so there is nothing to move --
    # fractional_start must be a no-op there, not a source of drift.
    for strength in (0.2, 0.3):
        assert _sigmas(strength) == _stock_sigmas(strength)


def test_strengths_that_used_to_collapse_now_differ():
    # The point of the feature: at 10 steps every strength in [0.2, 0.3) floors to rung
    # 2 and produces one identical image. Here each gets its own noise level, ordered.
    rungs = [_sigmas(s)[2] for s in (0.20, 0.22, 0.25, 0.28)]
    assert rungs == sorted(rungs, reverse=True)  # more strength -> less noise
    assert len(set(rungs)) == 4


def test_no_interpolation_past_the_clamps_mflux_applies():
    # Both cases are documented in start_fraction: a strength under 1/steps is floored
    # *up* to rung 1 by mflux, and 1.0 starts past the last rung with no steps to run.
    # Neither has a meaningful neighbour to interpolate toward, and 1.0 would index off
    # the end of the array, so both must fall back to the stock schedule.
    assert _sigmas(0.05) == _stock_sigmas(0.05)
    assert _sigmas(1.0) == _stock_sigmas(1.0)


def test_a_half_step_lands_near_the_rung_twice_the_steps_would_give():
    # The interpolation is on the request's own (already shifted) schedule rather than a
    # re-derivation of mflux's shift math, so it only approximates the true rung from a
    # finer grid. Bounding that here keeps the approximation honest: if a future mflux
    # changes the schedule's shape enough for this to drift, this fails rather than
    # silently making `fractional_start` mean something else.
    moved = _sigmas(0.25)[2]
    true_finer_grid = _stock_sigmas(0.25, steps=2 * STEPS)[5]
    assert moved == pytest.approx(true_finer_grid, abs=0.002)


def test_qwen_image_is_interpolated_on_its_own_schedule():
    # Qwen-Image sets sigma_shift_terminal, whose stretch is scaled by the last raw
    # sigma (1/steps) -- so its grids at different step counts do NOT nest, and "the
    # rung more steps would give" isn't even well defined. Interpolating the request's
    # own schedule still is, which is the other reason it's done that way round.
    stock = _stock_sigmas(0.25, model="qwen-image")
    moved = _sigmas(0.25, model="qwen-image")
    assert moved[2] == pytest.approx((stock[2] + stock[3]) / 2)


def test_start_fraction_matches_the_position_the_strength_names():
    assert start_fraction(10, 0.25, 2) == pytest.approx(0.5)
    assert start_fraction(10, 0.28, 2) == pytest.approx(0.8)
    assert start_fraction(10, 0.3, 3) == 0.0
    assert start_fraction(10, None, 0) == 0.0
    assert start_fraction(10, 0.05, 1) == 0.0  # floored up to rung 1 by mflux
    assert start_fraction(10, 1.0, 10) == 0.0  # no rung past the end to interpolate to


# --- Masked inpainting --------------------------------------------------------------
#
# The blend itself, at the one place it happens: scheduler.step(). ToyModel does step
# through these classes (see tests/doubles/toy_model.py), which proves the wiring
# resolves, but its VAE throws away spatial structure -- so what a mask actually does
# to a latent is only visible here.

MASKED = "schedulers.MaskedBlendLinearScheduler"
MASKED_FRACTIONAL = "schedulers.MaskedFractionalStartScheduler"


@pytest.fixture
def job():
    """Three elements, one per case: fully kept, fully free, and half-feathered."""
    j = MaskJob(
        keep=mx.array([1.0, 0.0, 0.5]),
        clean=mx.array([10.0, 10.0, 10.0]),
        noise=mx.array([2.0, 2.0, 2.0]),
    )
    set_mask_job(j)
    yield j
    clear_mask_job()


def _stepped(scheduler, timestep, latents=(7.0, 7.0, 7.0)):
    # A zero "prediction" so LinearScheduler.step returns the latents untouched and
    # everything the assertion sees is the blend, not the denoising.
    return scheduler.step(
        noise=mx.zeros(3), timestep=timestep, latents=mx.array(list(latents))
    ).tolist()


@pytest.mark.parametrize("path", [MASKED, MASKED_FRACTIONAL])
def test_mflux_resolves_the_masked_schedulers_by_their_dotted_paths(path):
    assert isinstance(_config(0.25, path).scheduler, _MaskedBlend)


def test_the_kept_region_is_re_noised_to_the_step_s_own_sigma(job):
    scheduler = _config(0.25, MASKED).scheduler
    sigma = scheduler.sigmas[3].item()  # what step 2 lands on
    kept, free, feathered = _stepped(scheduler, 2)

    known = (1 - sigma) * 10.0 + sigma * 2.0
    assert kept == pytest.approx(known)
    assert free == pytest.approx(7.0)  # untouched by the mask
    # A grey mask value is a crossfade, not a rounded-off decision -- which is the
    # whole point of mask_feather.
    assert feathered == pytest.approx(0.5 * 7.0 + 0.5 * known)


def test_the_last_step_lands_the_kept_region_on_the_encoded_original(job):
    # sigmas ends at 0, so the final blend is (1-0)*clean + 0*noise. That exactness is
    # what makes "outside the mask comes back unchanged" true rather than approximate.
    scheduler = _config(0.25, MASKED).scheduler
    assert scheduler.sigmas[STEPS].item() == 0.0
    kept, _, _ = _stepped(scheduler, STEPS - 1)
    assert kept == pytest.approx(10.0)


def test_the_plain_masked_scheduler_never_moves_the_starting_rung():
    # The trap two classes exist to avoid. Once a mask forces a scheduler to be passed
    # on every request, "no scheduler passed" no longer means "no fractional start" --
    # so a single class inferring it from image_strength would silently hand a
    # different image to anyone whose strength happened to fall between two rungs.
    assert _config(0.25, MASKED).scheduler.sigmas.tolist() == _stock_sigmas(0.25)


def test_the_masked_fractional_scheduler_moves_it_exactly_as_the_unmasked_one_does():
    assert _config(0.25, MASKED_FRACTIONAL).scheduler.sigmas.tolist() == _sigmas(0.25)


def test_the_blend_follows_the_moved_rung_rather_than_the_stock_one(job):
    # The fractional variant re-noises to self.sigmas, so on the moved rung the kept
    # region is noised to the level the trajectory is really on. Stepping *onto* the
    # moved rung (index 2) is the only step where the two schedules disagree.
    moved = _config(0.25, MASKED_FRACTIONAL).scheduler
    stock = _config(0.25, MASKED).scheduler
    assert moved.sigmas[2].item() != pytest.approx(stock.sigmas[2].item())
    assert _stepped(moved, 1)[0] != pytest.approx(_stepped(stock, 1)[0])


def test_running_without_a_job_raises_rather_than_generating_unmasked():
    # An unmasked image would look completely fine and quietly ignore the mask. The
    # raise lands on the worker thread inside generate_image(), where engine.py turns
    # it into a logged traceback and a failed generation -- see _MaskedBlend.
    clear_mask_job()
    scheduler = _config(0.25, MASKED).scheduler
    with pytest.raises(RuntimeError, match="no mask job set"):
        _stepped(scheduler, 2)


def test_scheduler_path_is_a_fixed_table_of_four():
    # Never assembled from request data: a dotted path is an arbitrary module import in
    # the server process. Two bools in, one of four constants out.
    assert scheduler_path(masked=False, fractional=False) is None
    assert scheduler_path(masked=False, fractional=True) == SCHEDULER_PATH
    assert scheduler_path(masked=True, fractional=False) == MASKED
    assert scheduler_path(masked=True, fractional=True) == MASKED_FRACTIONAL
