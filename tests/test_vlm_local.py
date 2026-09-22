"""The in-process detection backend, minus the model.

Nothing here loads weights -- CI would be downloading gigabytes per run for a
non-determinism it could not assert on anyway. What it does cover is the part that is
arithmetic rather than inference: the frame `smart_resize` predicts before handing an
image to the processor, and the division that turns Qwen3-VL's 0-1000 boxes into the
fractions everything downstream assumes.
"""

import pytest
from PIL import Image

from mfluxible.vlm_local import (
    DEFAULT_FACTOR,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    PROMPT,
    LocalDetector,
    _budget,
    _generated_text,
    smart_resize,
    to_fractions,
)
from mfluxible.vlm_reply import MAX_REGIONS, clean_payload


# --- the prompt template -------------------------------------------------------------


def test_the_json_example_survives_formatting_intact():
    """PROMPT is filled with `.format(max_regions=...)`, and it also carries a literal
    JSON example (`{"prompt": ..., "regions": [...]}`) for the model to imitate. Every
    brace in that example has to survive as a literal brace rather than being read as a
    format field -- doubled to `{{`/`}}` in the source -- or `.format` raises KeyError on
    the first one it meets (regressed once: `KeyError: '"prompt"'`, from the unescaped
    example above). Only `{max_regions}` is a real placeholder.
    """
    filled = PROMPT.format(max_regions=MAX_REGIONS)
    assert '{"prompt": string, "regions": [{"label": string, "bbox_2d": [x1, y1, x2, y2]}]}' in filled
    assert f"at most {MAX_REGIONS}." in filled
    assert "{" not in filled.split("at most")[-1]


# --- the frame ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "height,width",
    [(1024, 1024), (768, 1024), (4032, 3024), (3024, 4032), (100, 3000), (40, 40), (1, 1)],
)
def test_the_resized_frame_is_its_own_fixed_point(height, width):
    """The whole backend rests on this.

    `detect_sync` resizes to smart_resize's output and hands *that* over, so the
    processor's own call to the same algorithm has to be a no-op -- otherwise the model
    answers in a frame this module didn't compute and the division below is against the
    wrong number. Idempotence is what makes "the frame the model saw" a value we know
    rather than one we hope for.
    """
    once = smart_resize(height, width)
    assert smart_resize(*once) == once


@pytest.mark.parametrize(
    "height,width", [(1024, 1024), (768, 1024), (4032, 3024), (100, 3000), (1, 1)]
)
def test_the_frame_stays_inside_the_budget_and_on_the_grid(height, width):
    h, w = smart_resize(height, width)
    assert h % DEFAULT_FACTOR == 0 and w % DEFAULT_FACTOR == 0
    assert h >= DEFAULT_FACTOR and w >= DEFAULT_FACTOR
    assert h * w <= DEFAULT_MAX_PIXELS


def test_a_huge_image_is_shrunk_and_a_tiny_one_grown():
    tall = smart_resize(8000, 6000)
    assert tall[0] * tall[1] <= DEFAULT_MAX_PIXELS

    tiny = smart_resize(8, 8)
    assert tiny[0] * tiny[1] >= DEFAULT_MIN_PIXELS


@pytest.mark.parametrize("height,width", [(3000, 4000), (768, 1024), (4032, 3024)])
def test_the_aspect_ratio_is_held_loosely_and_that_is_enough(height, width):
    """Qwen floors each axis onto its 32-grid independently, so the ratio drifts by a
    couple of percent on an ordinary photograph and further on an extreme panorama.
    That is upstream's algorithm, not a rounding bug here.

    It does not cost coordinate accuracy, which is the thing worth being clear about:
    the resize maps the whole width onto the whole width, so an object covering half the
    frame covers half of it in either size. The drift restretches shapes slightly -- a
    circle arrives a shade elliptical -- which is a question about how well the model
    sees, not about where its answer lands.
    """
    h, w = smart_resize(height, width)
    assert abs((w / h) / (width / height) - 1) < 0.05


def test_the_conversion_needs_no_information_about_the_image_at_all():
    """Unlike Qwen2.5-VL's absolute pixels -- which needed dividing by the exact frame
    `smart_resize` computed for that request -- Qwen3-VL's 0-1000 values are already
    relative to the whole image (see BOX_KEY's comment in vlm_local.py). So the same
    reply converts identically no matter what size the image was or what frame it got
    resized to before generation; `to_fractions` doesn't even take one any more.
    """
    reply = [{"label": "x", "bbox_2d": [250, 500, 750, 900]}]
    assert to_fractions(reply)[0]["box"] == [0.25, 0.5, 0.75, 0.9]


# --- the budget is read off the model, not assumed ---------------------------------


class _ImageProcessor:
    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class _Processor:
    def __init__(self, image_processor):
        self.image_processor = image_processor


def test_the_budget_comes_from_the_loaded_processor():
    """A repack that lowered max_pixels to fit a smaller machine is an ordinary thing to
    publish, and it is the one variation that would be silently wrong: the processor
    would shrink the image again, the model would answer in that smaller frame, and the
    quotients would still land inside [0,1] -- correctly shaped boxes, all of them too
    close to the top-left. So the numbers are read rather than hardcoded."""
    processor = _Processor(
        _ImageProcessor(patch_size=14, merge_size=2, min_pixels=3136, max_pixels=200704)
    )
    assert _budget(processor) == (28, 3136, 200704)

    h, w = smart_resize(2048, 2048, *_budget(processor))
    assert h * w <= 200704


def test_the_budget_reads_the_size_dict_spelling_too():
    """Newer transformers moves the same two areas into `size`, under edge names that
    they are not -- both values stay areas in pixels."""
    processor = _Processor(
        _ImageProcessor(
            patch_size=14, merge_size=2, size={"shortest_edge": 3136, "longest_edge": 401408}
        )
    )
    assert _budget(processor) == (28, 3136, 401408)


def test_the_budget_falls_back_when_the_processor_says_nothing():
    assert _budget(_Processor(_ImageProcessor())) == (
        DEFAULT_FACTOR,
        DEFAULT_MIN_PIXELS,
        DEFAULT_MAX_PIXELS,
    )


def test_a_nonsense_budget_is_ignored_rather_than_used():
    processor = _Processor(_ImageProcessor(min_pixels=0, max_pixels=-1, patch_size="x"))
    assert _budget(processor) == (DEFAULT_FACTOR, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)


# --- relative coordinates to fractions ----------------------------------------------


def test_relative_boxes_become_fractions_of_the_whole_image():
    parsed = to_fractions(
        {"prompt": "a mug", "regions": [{"label": "mug", "bbox_2d": [100, 200, 400, 500]}]}
    )
    assert parsed["regions"][0]["box"] == [0.1, 0.2, 0.4, 0.5]


def test_x_and_y_are_not_transposed():
    """Both axes divide by the same FRAME_SPAN now, so there is no width/height mix-up
    to guard against -- but a transposition bug reading x1/y1 as y1/x1 would still
    produce a plausible-looking wrong box, so it is worth pinning that x-values land in
    x-slots and y-values in y-slots."""
    parsed = to_fractions([{"label": "x", "bbox_2d": [100, 300, 200, 600]}])
    assert parsed[0]["box"] == [0.1, 0.3, 0.2, 0.6]


def test_a_bare_array_converts_the_same_way_an_object_does():
    """vlm_reply accepts both shapes, so this has to as well -- otherwise a model
    answering in the older array form would reach the cleaner still holding raw values."""
    parsed = to_fractions([{"label": "a", "bbox_2d": [0, 0, 500, 250]}])
    assert parsed[0]["box"] == [0.0, 0.0, 0.5, 0.25]


def test_a_reply_outside_the_0_1000_scale_is_dropped_not_rescaled():
    """The conversion deliberately does no validation of its own. A reply measured in
    some other scale -- absolute pixels, say -- produces quotients outside [0,1], and
    vlm_reply's range check drops them there -- one rejection point for every backend,
    rather than two."""
    payload = clean_payload(to_fractions([{"label": "x", "bbox_2d": [0, 0, 2000, 2000]}]))
    assert payload["regions"] == []


def test_entries_that_are_not_boxes_are_left_for_the_cleaner():
    parsed = to_fractions(
        [
            {"label": "short", "bbox_2d": [1, 2]},
            {"label": "words", "bbox_2d": ["a", "b", "c", "d"]},
            "not an object",
            {"label": "fine", "bbox_2d": [0, 0, 100, 100]},
        ]
    )
    assert clean_payload(parsed)["regions"] == [{"label": "fine", "box": [0.0, 0.0, 0.1, 0.1]}]


def test_a_reply_already_speaking_box_is_converted_too():
    """Some replies come back under the generic key rather than Qwen's. They are still
    on the same 0-1000 scale -- this prompt asks for that scale -- so the key is what
    varies, not the unit."""
    parsed = to_fractions([{"label": "a", "box": [0, 0, 250, 500]}])
    assert parsed[0]["box"] == [0.0, 0.0, 0.25, 0.5]


def test_something_that_is_not_a_reply_passes_through_untouched():
    assert to_fractions("nonsense") == "nonsense"
    assert to_fractions({"prompt": "no regions here"}) == {"prompt": "no regions here"}


# --- the generate() return shape ---------------------------------------------------


def test_both_of_mlx_vlms_return_shapes_are_read():
    """It has returned a bare string and, later, an object carrying one. Pinning a
    version would trade a working server for a tidier line -- see the mflux note in
    CLAUDE.md for the same argument about the same class of dependency."""

    class _Result:
        text = "boxed"

    assert _generated_text("plain") == "plain"
    assert _generated_text(_Result()) == "boxed"


# --- a reply that never resolves to JSON --------------------------------------------


def test_an_unparseable_reply_is_logged_for_diagnosis(monkeypatch, tmp_path, caplog):
    """`detect_sync` hands the harness only the curated message -- the raw reply is the
    one piece of evidence that would tell an operator whether the model refused, added
    a preamble, or simply ran on past MAX_TOKENS, so it has to reach the server log even
    though it can never reach a response body. Before this, the failure was a black box:
    reproducing it required patching `_generated_text` by hand to see what the model
    actually said.

    mlx-vlm is faked into `sys.modules` rather than imported for real, the same reason
    the module docstring gives for loading no weights here: CI never installs the `vlm`
    extra (it drags in transformers), so a bare `import mlx_vlm` would only ever pass
    where a developer happened to have it installed already -- which is exactly how this
    test's first version passed locally and failed in CI.
    """
    import sys
    import types

    image_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(image_path)

    detector = LocalDetector()
    detector._model = object()
    detector._processor = object()
    detector._config = {"model_type": "qwen2_5_vl"}

    class _Result:
        text = "I'm not able to help with that request."

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    fake_mlx_vlm.generate = lambda *a, **k: _Result()
    fake_prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    fake_prompt_utils.apply_chat_template = lambda *a, **k: "prompt"
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", fake_prompt_utils)

    with caplog.at_level("WARNING", logger="mfluxible.vlm_local"):
        result = detector.detect_sync(image_path)

    assert result == {"error": "the model's reply held no JSON object."}
    assert _Result.text in caplog.text


# --- failing without mlx-vlm installed ---------------------------------------------


def test_a_missing_mlx_vlm_is_a_sentence_rather_than_a_traceback(monkeypatch):
    """The likeliest first-run failure by a distance: the extra is optional, the backend
    is the default, and the two meet on the first press of the button. It has to name
    both ways out -- install it, or switch backends -- because either is a real answer.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("mlx_vlm"):
            raise ImportError("no mlx_vlm here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    detector = LocalDetector()
    problem = detector.load_sync()
    assert problem is not None
    assert "mlx-vlm" in problem and "mfluxible[vlm]" in problem
    assert "MFLUXIBLE_VLM_BACKEND=worker" in problem
    assert detector.loaded is False


def test_a_detection_without_the_model_reports_rather_than_raises(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("mlx_vlm"):
            raise ImportError("no mlx_vlm here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    result = LocalDetector().detect_sync(tmp_path / "nothing.png")
    assert "error" in result and "mlx-vlm" in result["error"]


# --- which backend is selected -----------------------------------------------------


def test_the_local_backend_is_the_default(monkeypatch):
    from mfluxible.vlm import BACKEND_LOCAL, backend_from_env
    from mfluxible.vlm_local import detector_from_env

    monkeypatch.delenv("MFLUXIBLE_VLM_BACKEND", raising=False)
    assert backend_from_env() == BACKEND_LOCAL
    assert isinstance(detector_from_env(), LocalDetector)


def test_naming_the_worker_backend_builds_no_detector(monkeypatch):
    from mfluxible.vlm import BACKEND_WORKER, backend_from_env
    from mfluxible.vlm_local import detector_from_env

    monkeypatch.setenv("MFLUXIBLE_VLM_BACKEND", "worker")
    assert backend_from_env() == BACKEND_WORKER
    assert detector_from_env() is None


def test_an_unknown_backend_is_refused_rather_than_defaulted(monkeypatch):
    """`MFLUXIBLE_VLM_BACKEND=sidecar` is a typo for something real. Shrugging and using
    the default would answer detections with a model the operator believes isn't
    running -- the same reason an unknown MFLUXIBLE_MODEL raises at import."""
    from mfluxible.vlm import backend_from_env

    monkeypatch.setenv("MFLUXIBLE_VLM_BACKEND", "sidecar")
    with pytest.raises(ValueError, match="sidecar"):
        backend_from_env()


def test_the_backend_name_is_read_case_insensitively(monkeypatch):
    from mfluxible.vlm import BACKEND_WORKER, backend_from_env

    monkeypatch.setenv("MFLUXIBLE_VLM_BACKEND", "  Worker  ")
    assert backend_from_env() == BACKEND_WORKER
