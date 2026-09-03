import pytest

from models import MODELS, resolve


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
