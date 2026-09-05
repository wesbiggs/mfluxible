"""Coverage of server/schedulers.py: the fractional-start schedule itself.

Weight-free like the rest of the suite, but deliberately *not* against ToyModel -- what
matters here is the sigma array a real mflux Config produces, so these run against real
ModelConfigs (no weights are loaded by building a Config or a scheduler).
"""

import pytest
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig

from schedulers import FractionalStartLinearScheduler, start_fraction

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
