"""HTTP-level coverage of server.py's endpoints, run against the real FastAPI app with
its module-level `engine` swapped for ToyModel (see tests/conftest.py::client)."""

import base64
import io
import json

from PIL import Image

from models import MODELS


def test_health_reports_the_configured_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["name"] == "toy-solid-color"
    assert body["model"]["supports_guidance"] is False
    # harness.html and mcp_server.py both gate the fractional-start control on this,
    # and both read it as "unknown" when absent -- so it going missing would silently
    # re-offer a knob the server now rejects rather than failing anywhere visible.
    assert body["model"]["supports_fractional_start"] is True
    assert body["available"] == [m.key for m in MODELS]


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


def test_harness_is_served_at_root(client):
    resp = client.get("/")
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


# Both error envelopes, checked against the same leak. server.py has no `except` clause
# in it at all any more -- an exception carrying mflux's text out to a response body was
# the only way host detail could get into one (see MfluxEngine.request_problem), so
# these pin the shape of the fix rather than just its current wording.
_LEAKY_MESSAGE = "No such file: '/Users/someone/.cache/huggingface/hub/models--x/transformer.safetensors'"


def _make_generation_fail(client):
    import server as server_module

    def boom(*_args, **_kwargs):
        raise FileNotFoundError(_LEAKY_MESSAGE)

    server_module.engine.model.generate_image = boom


def test_generate_failure_is_a_500_that_does_not_quote_the_exception(client):
    _make_generation_fail(client)
    resp = client.post(
        "/mfluxible/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2, "stream": False},
    )
    assert resp.status_code == 500
    assert "/Users/someone" not in resp.text
    assert "server log" in resp.json()["message"]


def test_openai_generate_failure_does_not_quote_the_exception(client):
    _make_generation_fail(client)
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "model": "toy-solid-color", "size": "32x32"},
    )
    assert resp.status_code == 500
    assert "/Users/someone" not in resp.text
    assert resp.json()["error"]["type"] == "api_error"


def test_streaming_failure_does_not_quote_the_exception(client):
    # The stream is the path that can't be fixed at the endpoint: the 200 is already on
    # the wire by the time generation fails, so whatever the engine emits is what ships.
    _make_generation_fail(client)
    with client.stream(
        "POST",
        "/mfluxible/v1/images/generations",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "/Users/someone" not in body
    assert "server log" in body


def test_a_rejected_request_still_says_exactly_why(client):
    # The other half of the invariant: messages server.py *did* write are worth reading,
    # and none of the above is allowed to flatten them into something generic.
    resp = client.post("/mfluxible/v1/images/generations", json={"prompt": "a cat", "image_strength": 0.5})
    assert resp.status_code == 400
    assert resp.json()["message"] == "image_strength requires image to also be set."


# --- The SillyTavern-facing shim (OPTIONS ping + /sdapi/v1/txt2img) ------------------
#
# See server.py's comment on POST /sdapi/v1/txt2img for what SillyTavern's `sdcpp`
# source actually calls and why this endpoint drops fields the native API rejects.


def test_a1111_ping_answers_a_bare_options_request(client):
    # SillyTavern's "Validate" button is a server-side `fetch(url, {method: 'OPTIONS'})`
    # with no Access-Control-Request-Method header, so the CORS middleware doesn't
    # handle it and a missing route would 405. It checks only the status.
    resp = client.options("/v1/images/generations")
    assert resp.status_code == 204
    assert resp.headers["allow"] == "POST, OPTIONS"


def test_a1111_txt2img_returns_a_base64_png_in_images(client):
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 1, "seed": 5, "batch_size": 1},
    )
    assert resp.status_code == 200
    body = resp.json()

    image = Image.open(io.BytesIO(base64.b64decode(body["images"][0])))
    assert image.size == (32, 32)

    # A1111 puts a JSON *string* here, not an object -- a client parses it a second time.
    info = json.loads(body["info"])
    assert info["seed"] == 5
    assert info["steps"] == 1
    assert info["sd_model_name"] == "toy-solid-color"


def test_a1111_txt2img_drops_guidance_the_model_cannot_use(client):
    # toy-solid-color has supports_guidance False, so this is the exact request the
    # native API answers with a 400. Here it must succeed, and `info` is where the
    # caller can see the field had no effect.
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 1, "cfg_scale": 7.0},
    )
    assert resp.status_code == 200
    assert json.loads(resp.json()["info"])["cfg_scale"] is None


def test_a1111_txt2img_drops_a_negative_prompt_the_model_cannot_use(client):
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "a cat",
            "width": 32,
            "height": 32,
            "steps": 1,
            "negative_prompt": "blurry",
        },
    )
    assert resp.status_code == 200
    assert json.loads(resp.json()["info"])["negative_prompt"] == ""


def test_a1111_txt2img_treats_a_negative_seed_as_random(client):
    # A1111 spells "pick one for me" as -1; passing it through would be a literal --
    # and reproducible -- seed. SillyTavern omits the field instead, so this is for
    # every other A1111 client.
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 1, "seed": -1},
    )
    assert resp.status_code == 200
    seed = json.loads(resp.json()["info"])["seed"]
    assert isinstance(seed, int) and seed >= 0


def test_a1111_txt2img_reports_the_models_own_step_count_when_none_was_sent(client):
    resp = client.post("/sdapi/v1/txt2img", json={"prompt": "a cat", "width": 32, "height": 32})
    assert resp.status_code == 200
    # toy-solid-color's default_steps is 2 (see tests/doubles/toy_model.py).
    assert json.loads(resp.json()["info"])["steps"] == 2


def test_a1111_txt2img_rejects_asking_for_more_than_one_image(client):
    # The one thing this endpoint does reject: unlike a dropped cfg_scale, answering
    # "give me 4" with one image is a concretely wrong result to a request that named
    # a number. SillyTavern hardcodes batch_size: 1, so it can never see this.
    for payload in ({"batch_size": 2}, {"n_iter": 2}):
        resp = client.post("/sdapi/v1/txt2img", json={"prompt": "a cat", **payload})
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_request"


def test_a1111_txt2img_ignores_a_stale_model_name(client):
    # SillyTavern only auto-selects from a freshly loaded model list when its stored
    # sd.model is empty, so a name left over from a previously configured source is
    # sent here while the dropdown shows something else. Validating it would turn that
    # invisible stale setting into an unexplained failure -- see the endpoint's comment.
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "a cat",
            "width": 32,
            "height": 32,
            "steps": 1,
            "model": "sd_xl_base_1.0.safetensors",
        },
    )
    assert resp.status_code == 200
    assert json.loads(resp.json()["info"])["sd_model_name"] == "toy-solid-color"


def test_a1111_shim_never_builds_a_request_its_own_model_rejects(monkeypatch):
    """The drops in _a1111_to_generate_request duplicate request_problem's rules, so
    this pins the two together across every model in the real table: whatever
    SillyTavern sends, what comes out the other side must be a request the loaded model
    actually accepts. If request_problem grows a clause this doesn't mirror, this fails
    here rather than as a bare 500 in someone's chat window.

    No weights are touched: MfluxEngine resolves its spec in __init__ and downloads
    nothing until load().
    """
    import server as server_module
    from engine import MfluxEngine
    from schemas import A1111Txt2ImgRequest

    # The maximal payload SillyTavern can send, across the cfg_scale values that
    # matter: its own default, one at the CFG floor (where a negative prompt would be
    # encoded and never consulted -- Krea-2's default guidance is exactly this), and
    # the field omitted entirely so the model's own default applies.
    for spec in MODELS:
        engine = MfluxEngine(model=spec.key, quantize=None, model_cache_dir=None)
        monkeypatch.setattr(server_module, "engine", engine)
        try:
            for cfg_scale in (7.0, 1.0, None):
                req = A1111Txt2ImgRequest(
                    prompt="a cat",
                    negative_prompt="blurry",
                    cfg_scale=cfg_scale,
                    steps=20,
                    seed=-1,
                )
                gen_req = server_module._a1111_to_generate_request(req)
                problem = engine.request_problem(gen_req)
                assert problem is None, f"{spec.key} (cfg_scale={cfg_scale}): {problem}"
        finally:
            engine.shutdown()


def test_a1111_ping_route_does_not_shadow_a_real_cors_preflight(client):
    # The new OPTIONS route sits on a path the CORS middleware also handles. The two
    # don't collide only because Starlette routes a preflight by the presence of
    # Access-Control-Request-Method -- so a browser still gets its
    # Access-Control-Allow-Origin here, rather than the bare 204 above.
    resp = client.options(
        "/v1/images/generations",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_a1111_txt2img_accepts_sillytaverns_actual_default_payload(client):
    """The literal request a stock SillyTavern install emits, as assembled by
    `generateSdcppImage` and then rebuilt by the `sdcpp` backend router (which deletes
    undefined/null/empty-string keys, so an unset negative prompt and a -1 seed never
    arrive at all). Its defaults are 512x512, 20 steps and CFG 7, with sampler and
    scheduler names from a hardcoded stable-diffusion.cpp list that means nothing here.

    Written out in full rather than built from the helpers above: the thing most likely
    to break this integration is a field appearing in that payload that this endpoint
    chokes on, and that only gets caught by sending the real shape.
    """
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={
            "model": "toy-solid-color",
            "prompt": "a cat",
            "width": 32,
            "height": 32,
            "steps": 1,
            "cfg_scale": 7,
            "batch_size": 1,
            "sampler_name": "euler_a",
            "scheduler": "discrete",
        },
    )
    assert resp.status_code == 200
    assert Image.open(io.BytesIO(base64.b64decode(resp.json()["images"][0]))).size == (32, 32)


def test_a1111_txt2img_failure_does_not_quote_the_exception(client):
    # The fourth sink for the invariant the three _LEAKY_MESSAGE tests above cover: a
    # new endpoint that turns a failed generation into a response body is a new place
    # for an upstream message (and the absolute cache path in it) to reach a client.
    _make_generation_fail(client)
    resp = client.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "a cat", "width": 32, "height": 32, "steps": 2},
    )
    assert resp.status_code == 500
    assert "/Users/someone" not in resp.text
    assert "server log" in resp.json()["detail"]
