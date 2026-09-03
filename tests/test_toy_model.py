"""Sanity checks on the test double itself, independent of engine.py.

If these fail, failures in test_engine_stream.py / test_server_api.py are about the
fixture, not about server/ -- worth ruling out first.
"""

from tests.doubles.toy_model import ToyModel


def test_generate_image_is_a_single_solid_color():
    model = ToyModel()
    image = model.generate_image(seed=42, prompt="anything", num_inference_steps=1, width=32, height=32)
    pixels = set(image.image.get_flattened_data())
    assert len(pixels) == 1, f"expected one uniform color, got {len(pixels)} distinct pixels"


def test_same_seed_is_deterministic_regardless_of_prompt():
    model = ToyModel()
    first = model.generate_image(seed=7, prompt="a", num_inference_steps=1, width=16, height=16)
    second = model.generate_image(seed=7, prompt="a completely different prompt", num_inference_steps=1, width=16, height=16)
    assert list(first.image.get_flattened_data())[0] == list(second.image.get_flattened_data())[0]


def test_different_seeds_render_different_colors():
    model = ToyModel()
    a = model.generate_image(seed=1, prompt="x", num_inference_steps=1, width=16, height=16)
    b = model.generate_image(seed=2, prompt="x", num_inference_steps=1, width=16, height=16)
    assert list(a.image.get_flattened_data())[0] != list(b.image.get_flattened_data())[0]


def test_output_size_matches_the_request():
    model = ToyModel()
    image = model.generate_image(seed=1, prompt="x", num_inference_steps=1, width=48, height=32)
    assert image.image.size == (48, 32)


def test_generation_fires_one_before_loop_and_n_in_loop_callbacks():
    calls = {"before_loop": 0, "in_loop": []}

    class _Recorder:
        def call_before_loop(self, **kwargs):
            calls["before_loop"] += 1

        def call_in_loop(self, t, **kwargs):
            calls["in_loop"].append(t)

    # ToyModel exposes the same plain before_loop/in_loop/interrupt lists engine.py
    # appends its own callback to directly (see _StreamCallback in server/engine.py),
    # so a test double can register on them the exact same way.
    recorder = _Recorder()
    model = ToyModel()
    model.callbacks.before_loop.append(recorder)
    model.callbacks.in_loop.append(recorder)

    model.generate_image(seed=1, prompt="x", num_inference_steps=3, width=16, height=16)

    assert calls["before_loop"] == 1
    assert calls["in_loop"] == [0, 1, 2]
