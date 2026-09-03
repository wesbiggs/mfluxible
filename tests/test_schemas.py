import pytest
from pydantic import ValidationError

from schemas import GenerateRequest


def test_defaults():
    req = GenerateRequest(prompt="a cat")
    assert req.width == 1024
    assert req.height == 1024
    assert req.steps is None
    assert req.seed is None
    assert req.guidance is None
    assert req.negative_prompt is None
    assert req.preview_every == 0
    assert req.stream is True


def test_prompt_is_required():
    with pytest.raises(ValidationError):
        GenerateRequest()


def test_rejects_wrong_types():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="a cat", width="not a number")
