"""HTTP-level coverage of server.py's endpoints, run against the real FastAPI app with
its module-level `engine` swapped for ToyModel (see tests/conftest.py::client)."""

import base64
import io
import json

from PIL import Image


def test_health_reports_the_configured_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["name"] == "toy-solid-color"
    assert body["model"]["supports_guidance"] is False


def test_generate_non_streaming(client):
    resp = client.post(
        "/mfluxible/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 1, "seed": 5, "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "image"
    assert body["seed"] == 5

    image = Image.open(io.BytesIO(base64.b64decode(body["data"])))
    assert image.size == (32, 32)
    assert len(set(image.get_flattened_data())) == 1


def test_generate_rejects_unsupported_guidance_with_a_400_before_streaming(client):
    resp = client.post("/mfluxible/v1/images/generations", json={"prompt": "a cat", "guidance": 3.5})
    assert resp.status_code == 400
    assert resp.json()["type"] == "error"


def _b64_png(size=(16, 16), color=(0, 0, 255)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_generate_image_to_image_non_streaming(client):
    resp = client.post(
        "/mfluxible/v1/images/generations",
        json={
            "prompt": "a cat",
            "width": 32,
            "height": 32,
            "steps": 1,
            "stream": False,
            "image": _b64_png(),
            "image_strength": 0.6,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "image"
    image = Image.open(io.BytesIO(base64.b64decode(body["data"])))
    assert image.size == (32, 32)


def test_generate_rejects_image_strength_without_image_with_a_400(client):
    resp = client.post(
        "/mfluxible/v1/images/generations", json={"prompt": "a cat", "image_strength": 0.5}
    )
    assert resp.status_code == 400
    assert resp.json()["type"] == "error"


def test_generate_rejects_invalid_base64_image_with_a_400(client):
    resp = client.post(
        "/mfluxible/v1/images/generations", json={"prompt": "a cat", "image": "not!base64!!"}
    )
    assert resp.status_code == 400
    assert resp.json()["type"] == "error"


def _png_bytes(size=(16, 16), color=(0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_openai_edit_non_streaming(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color", "size": "32x32"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "created" in body
    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.size == (32, 32)


def test_openai_edit_accepts_image_strength_extension(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color", "size": "16x16", "image_strength": "0.8"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200


def test_openai_edit_rejects_mask_with_a_400(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color"},
        files={
            "image": ("input.png", _png_bytes(), "image/png"),
            "mask": ("mask.png", _png_bytes(), "image/png"),
        },
    )
    assert resp.status_code == 400
    assert "mask" in resp.json()["error"]["message"]


def test_openai_edit_rejects_wrong_model_name(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "not-the-loaded-model"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert "not-the-loaded-model" in resp.json()["error"]["message"]


def test_openai_edit_rejects_n_greater_than_1(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color", "n": "2"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_openai_edit_rejects_malformed_base64_image_strength_range(client):
    resp = client.post(
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color", "image_strength": "5.0"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_openai_edit_streaming_ends_with_completed_event(client):
    with client.stream(
        "POST",
        "/v1/images/edits",
        data={"prompt": "a cat", "model": "toy-solid-color", "size": "16x16", "stream": "true"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :]) for line in resp.iter_lines() if line.startswith("data: ")
        ]
    assert [e["type"] for e in events] == ["image_generation.completed"]


def test_openai_list_models_shows_only_the_running_model(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1

    model = body["data"][0]
    assert model["id"] == "toy-solid-color"
    assert model["object"] == "model"
    assert model["owned_by"] == "mfluxible"
    assert isinstance(model["created"], int) and model["created"] > 0


def test_openai_retrieve_model_matches_the_running_model(client):
    resp = client.get("/v1/models/toy-solid-color")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "toy-solid-color"
    assert body["object"] == "model"
    assert body["owned_by"] == "mfluxible"


def test_openai_retrieve_model_404s_for_any_other_id(client):
    resp = client.get("/v1/models/not-the-loaded-model")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert "not-the-loaded-model" in error["message"]


def test_openai_generate_non_streaming(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "toy-solid-color", "size": "32x32"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "created" in body
    assert len(body["data"]) == 1

    image = Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.size == (32, 32)


def test_openai_generate_defaults_to_1024_square(client):
    resp = client.post("/v1/images/generations", json={"prompt": "a cat", "model": "toy-solid-color"})
    assert resp.status_code == 200
    image = Image.open(io.BytesIO(base64.b64decode(resp.json()["data"][0]["b64_json"])))
    assert image.size == (1024, 1024)


def test_openai_generate_rejects_wrong_model_name(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "not-the-loaded-model"},
    )
    assert resp.status_code == 400
    assert "not-the-loaded-model" in resp.json()["error"]["message"]


def test_openai_generate_rejects_n_greater_than_1(client):
    resp = client.post("/v1/images/generations", json={"prompt": "a cat", "model": "toy-solid-color", "n": 2})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_openai_generate_rejects_url_response_format(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "toy-solid-color", "response_format": "url"},
    )
    assert resp.status_code == 400


def test_openai_generate_rejects_malformed_size(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "toy-solid-color", "size": "bogus"},
    )
    assert resp.status_code == 400


def test_openai_generate_streaming_ends_with_completed_event(client):
    with client.stream(
        "POST",
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "toy-solid-color", "size": "32x32", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :]) for line in resp.iter_lines() if line.startswith("data: ")
        ]

    assert [e["type"] for e in events] == ["image_generation.completed"]
    assert "b64_json" in events[-1]


def test_openai_generate_streaming_with_partial_images(client):
    # toy-solid-color's default_steps is 2 (see tests/doubles/toy_model.py), so
    # partial_images=2 should land a preview on every step: preview_every =
    # max(1, 2 // 2) == 1.
    with client.stream(
        "POST",
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "model": "toy-solid-color",
            "size": "32x32",
            "stream": True,
            "partial_images": 2,
        },
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :]) for line in resp.iter_lines() if line.startswith("data: ")
        ]

    assert [e["type"] for e in events] == [
        "image_generation.partial_image",
        "image_generation.partial_image",
        "image_generation.completed",
    ]
    assert [e["partial_image_index"] for e in events[:2]] == [0, 1]


def test_harness_html_is_served(client):
    resp = client.get("/harness.html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "mfluxible harness" in resp.text


def test_generate_streaming_sse(client):
    with client.stream(
        "POST",
        "/mfluxible/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2, "seed": 9},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :]) for line in resp.iter_lines() if line.startswith("data: ")
        ]

    assert [e["type"] for e in events] == ["start", "thinking", "thinking", "image"]
    assert events[-1]["seed"] == 9
