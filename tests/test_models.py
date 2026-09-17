import pytest

from mfluxible.models import CFG_GUIDANCE_FLOOR, MODELS, resolve


def test_resolve_by_key():
    assert resolve("z-image-turbo").key == "z-image-turbo"


@pytest.mark.parametrize("alias", ["schnell", "flux.1-schnell", "flux-1-schnell"])
def test_resolve_by_alias(alias):
    assert resolve(alias).key == "flux-schnell"


def test_resolve_is_case_and_whitespace_insensitive():
    assert resolve("  Flux-Dev  ").key == "flux-dev"


def test_resolve_unknown_model_lists_every_known_key():
    with pytest.raises(ValueError) as exc_info:
        resolve("not-a-real-model")
    message = str(exc_info.value)
    for spec in MODELS:
        assert spec.key in message


def test_every_model_key_and_alias_is_unique():
    # A duplicate would silently shadow an earlier entry in models.py's _BY_NAME
    # lookup table -- resolve() would still succeed, just for the wrong model.
    seen = set()
    for spec in MODELS:
        for name in (spec.key, *spec.aliases):
            assert name not in seen, f"{name!r} is registered more than once"
            seen.add(name)


def test_flux_schnell_and_dev_both_reject_negative_prompt():
    # FLUX has no negative branch in either variant (see models.py's comment on
    # flux-dev) -- guard against that regressing silently if a future model gets it
    # right and this one gets copy-pasted without updating the flag.
    assert resolve("flux-schnell").supports_negative_prompt is False
    assert resolve("flux-dev").supports_negative_prompt is False


def test_z_image_aliases_name_the_base_model_not_turbo():
    # mflux's own registry gives "z-image"/"zimage" to the base checkpoint and reaches
    # Turbo only as "z-image-turbo". This table used to point both spellings at Turbo,
    # which was fine while Turbo was the only Z-Image here and actively wrong once the
    # base model joined it. Carrying mflux's spelling is the whole point of aliases.
    assert resolve("z-image").key == "z-image"
    assert resolve("zimage").key == "z-image"
    assert resolve("z-image-turbo").key == "z-image-turbo"
    assert resolve("zimage-turbo").key == "z-image-turbo"


def test_z_image_base_and_turbo_differ_in_every_way_that_matters():
    # Two entries sharing one variant class, which is exactly where a copy-paste slip
    # would go unnoticed: ZImage reads supports_guidance to decide whether to build an
    # unconditional branch *and* which scheduler to run, so these move together.
    turbo, base = resolve("z-image-turbo"), resolve("z-image")
    assert (turbo.supports_guidance, base.supports_guidance) == (False, True)
    assert (turbo.supports_negative_prompt, base.supports_negative_prompt) == (False, True)
    assert turbo.default_scheduler == "linear"
    assert base.default_scheduler == "flow_match_euler_discrete"


@pytest.mark.parametrize("spec", MODELS, ids=lambda s: s.key)
def test_guidance_default_is_present_exactly_when_guidance_is_supported(spec):
    # _generation_kwargs sends `default_guidance` whenever supports_guidance is set, so
    # a True flag with no default would put None on the wire, and a default on a
    # distilled model is a value that can never be reached.
    assert (spec.default_guidance is not None) is spec.supports_guidance


@pytest.mark.parametrize("spec", MODELS, ids=lambda s: s.key)
def test_a_negative_prompt_implies_a_guidance_dial_to_switch_it_on(spec):
    # CFG is what gives a negative prompt an effect, so there is no coherent model with
    # a negative branch and no way to raise guidance above CFG_GUIDANCE_FLOOR --
    # check_request would reject every negative prompt such a model ever received.
    if spec.supports_negative_prompt:
        assert spec.supports_guidance, f"{spec.key} offers a negative prompt it can never encode"


@pytest.mark.parametrize("spec", MODELS, ids=lambda s: s.key)
def test_fractional_start_is_offered_only_on_the_linear_schedule(spec):
    # The property is the single place engine.py and /health both read, so pin it to
    # the underlying fact rather than letting the two drift.
    assert spec.supports_fractional_start is (spec.default_scheduler == "linear")


def test_krea_2_is_the_documented_exception_on_the_cfg_floor():
    # Every other CFG model here defaults above the floor. Krea-2 sits exactly on it
    # because that is mflux's own DEFAULT_GUIDANCE, and models.py deliberately keeps
    # mflux's recommendation rather than inventing a higher one -- so a negative prompt
    # there needs guidance raised with it. If this ever stops being the only exception,
    # the comment in models.py explaining why needs updating too.
    on_the_floor = {
        spec.key
        for spec in MODELS
        if spec.supports_negative_prompt and spec.default_guidance <= CFG_GUIDANCE_FLOOR
    }
    assert on_the_floor == {"krea-2", "krea-2-raw"}
