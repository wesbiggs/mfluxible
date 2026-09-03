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
        "/v1/images/generations",
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
    resp = client.post("/v1/images/generations", json={"prompt": "a cat", "guidance": 3.5})
    assert resp.status_code == 400
    assert resp.json()["type"] == "error"


def test_harness_html_is_served(client):
    resp = client.get("/harness.html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "mfluxible harness" in resp.text


def test_generate_streaming_sse(client):
    with client.stream(
        "POST",
        "/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2, "seed": 9},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :]) for line in resp.iter_lines() if line.startswith("data: ")
        ]

    assert [e["type"] for e in events] == ["start", "thinking", "thinking", "image"]
    assert events[-1]["seed"] == 9
