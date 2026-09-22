"""The object-detection mailbox: the store's own behaviour, and the HTTP surface.

Nothing here runs a detection. The worker shells out to the Claude Code CLI, which is
exactly the part that can't be exercised in CI -- so the seam under test is the
handover: an image goes in one side, a worker claims it, regions come out the other.
"""

import base64
import io
import json
import pathlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mfluxible.vlm import BACKENDS, VlmMailbox
from tests.doubles.toy_model import TOY_MODEL_SPEC


def _png(size=(64, 48), color=(10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture
def mailbox(tmp_path):
    box = VlmMailbox(tmp_path / "regions")
    box.prepare()
    return box


@pytest.fixture
def vlm_client(monkeypatch, mailbox):
    """The real app on the *worker* backend, with the mailbox switched on and both waits
    shortened so a test that exercises a timeout finishes in milliseconds rather than
    three minutes.

    The backend is pinned rather than inherited: it defaults to `local` now, and leaving
    it there would have every test in this file quietly asking an absent vision model to
    answer jobs that were written for a sidecar. `local_client` further down is the
    fixture for the other one.
    """
    import mfluxible.server as server_module
    from mfluxible.engine import MfluxEngine
    from mfluxible.vlm import BACKEND_WORKER

    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    monkeypatch.setattr(server_module, "MAILBOX", mailbox)
    monkeypatch.setattr(server_module, "VLM_BACKEND", BACKEND_WORKER)
    monkeypatch.setattr(server_module, "DETECTOR", None)
    monkeypatch.setattr(server_module, "VLM_TIMEOUT_S", 0.25)
    monkeypatch.setattr(server_module, "VLM_POLL_S", 0.25)
    with TestClient(server_module.app) as test_client:
        yield test_client


# --- the store -------------------------------------------------------------------


async def test_a_submitted_job_is_what_a_worker_claims(mailbox):
    job = mailbox.submit(_png())
    claimed = await mailbox.claim(timeout=0.1)
    assert claimed is not None
    assert claimed.id == job.id
    # The path is the payload: the worker's whole job is to read this file.
    assert claimed.path.is_file()
    assert claimed.path.read_bytes() == _png()


async def test_dimensions_are_measured_with_exif_orientation_applied(mailbox):
    """The frame a box is chosen in has to be the frame the mask is built in.

    A phone photo carries its rotation in a tag rather than in its pixels, and mflux
    rotates before encoding -- so reporting the raw size here would hand the worker one
    frame and the mask canvas another, and every box would be transposed.
    """
    buf = io.BytesIO()
    # Orientation 6: stored landscape, displayed portrait.
    img = Image.new("RGB", (64, 48), (200, 30, 30))
    exif = img.getexif()
    exif[274] = 6
    img.save(buf, format="JPEG", exif=exif)

    job = mailbox.submit(buf.getvalue())
    assert (job.width, job.height) == (48, 64)


async def test_a_second_submit_supersedes_the_first_rather_than_queueing(mailbox):
    """One model, one lock, one pending detection -- a second click replaces the first
    and says so, instead of leaving the old stream to time out."""
    first = mailbox.submit(_png())
    second = mailbox.submit(_png(size=(32, 32)))

    assert first.done.is_set()
    assert "superseded" in first.result["error"]
    assert second.result is None

    claimed = await mailbox.claim(timeout=0.1)
    assert claimed.id == second.id


async def test_completing_delivers_to_the_waiter(mailbox):
    job = mailbox.submit(_png())
    await mailbox.claim(timeout=0.1)
    assert mailbox.complete(job.id, {"regions": [{"label": "sky", "box": [0, 0, 1, 0.5]}]})
    result = await mailbox.wait(job, timeout=0.1)
    assert result["regions"][0]["label"] == "sky"


async def test_a_stale_job_id_is_refused_rather_than_overwriting_the_live_one(mailbox):
    stale = mailbox.submit(_png())
    live = mailbox.submit(_png(size=(32, 32)))
    assert mailbox.complete(stale.id, {"regions": []}) is False
    assert live.result is None


async def test_waiting_with_nothing_attached_says_so(mailbox):
    job = mailbox.submit(_png())
    result = await mailbox.wait(job, timeout=0.05)
    assert "nothing claimed" in result["error"]


async def test_waiting_on_a_claimed_job_blames_the_worker_instead(mailbox):
    job = mailbox.submit(_png())
    await mailbox.claim(timeout=0.1)
    result = await mailbox.wait(job, timeout=0.05)
    assert "worker took too long" in result["error"]


async def test_claim_marks_attendance(mailbox):
    assert mailbox.worker_attached() is False
    await mailbox.claim(timeout=0.01)
    assert mailbox.worker_attached() is True


def test_submit_problem_returns_reasons_rather_than_raising(mailbox):
    """The invariant from CLAUDE.md: nothing an imaging library says can reach a
    response body, so every refusal here is a *returned* string written for a caller."""
    assert mailbox.submit_problem(b"") is not None
    assert mailbox.submit_problem(b"not an image at all") is not None
    assert mailbox.submit_problem(_png()) is None


def test_pruning_keeps_the_live_job_and_drops_stale_files(mailbox):
    import os
    import time

    stale = mailbox.dir / "old.png"
    stale.write_bytes(_png())
    os.utime(stale, (time.time() - 7200, time.time() - 7200))

    job = mailbox.submit(_png())
    assert not stale.exists()
    assert job.path.is_file()


# --- the HTTP surface ------------------------------------------------------------


def test_health_reports_the_feature_off_by_default(client):
    """The plain `client` fixture has no mailbox, which is how a server runs unless
    MFLUXIBLE_VLM_DIR names a directory. harness.html draws its button off this."""
    body = client.get("/health").json()
    assert body["vlm"]["enabled"] is False
    assert body["vlm"]["worker_attached"] is False
    # Nothing is loaded until a detection asks for it, and with the feature off nothing
    # ever will. `backend` still reports which one *would* answer -- it is a property of
    # the configuration, not of the mailbox -- so it is asserted for membership rather
    # than for a value, since this fixture takes it from the ambient environment.
    assert body["vlm"]["model_loaded"] is False
    assert body["vlm"]["backend"] in BACKENDS


def test_the_endpoints_are_absent_when_the_feature_is_off(client):
    assert client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(_png())}).status_code == 404
    assert client.get("/mfluxible/v1/vlm/next").status_code == 404
    assert client.post("/mfluxible/v1/vlm/abc", json={"regions": []}).status_code == 404


def test_health_reports_the_feature_on_when_configured(vlm_client):
    body = vlm_client.get("/health").json()
    assert body["vlm"]["enabled"] is True
    assert body["vlm"]["worker_attached"] is False


def test_a_worker_polling_shows_up_as_attached(vlm_client):
    assert vlm_client.get("/mfluxible/v1/vlm/next").status_code == 204
    assert vlm_client.get("/health").json()["vlm"]["worker_attached"] is True


def test_detect_refuses_a_bad_body_before_the_stream_starts(vlm_client):
    """A 400, not a 200 with an error torn into the body -- the same rule the
    generation endpoint follows, and the reason request_problem runs in the endpoint."""
    resp = vlm_client.post("/mfluxible/v1/vlm/describe", json={"image": "not base64!!"})
    assert resp.status_code == 400
    assert "base64" in resp.json()["message"]

    resp = vlm_client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(b"nope")})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")


def test_a_detection_with_no_worker_streams_pending_then_a_reason(vlm_client):
    with vlm_client.stream(
        "POST", "/mfluxible/v1/vlm/describe", json={"image": _b64(_png())}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[0]["type"] == "pending"
    assert (events[0]["width"], events[0]["height"]) == (64, 48)
    assert events[0]["worker_attached"] is False
    assert events[-1]["type"] == "error"
    assert "timed out" in events[-1]["message"]


def test_next_hands_a_worker_the_path_and_the_measured_size(vlm_client, mailbox):
    job = mailbox.submit(_png())
    body = vlm_client.get("/mfluxible/v1/vlm/next").json()
    assert body["job_id"] == job.id
    assert body["image_path"] == str(job.path)
    assert (body["width"], body["height"]) == (64, 48)


def test_posting_a_superseded_job_is_reported_not_an_error(vlm_client, mailbox):
    """The usual cause is the user clicking the button again mid-detection, so the
    worker gets a plain answer rather than a status code it would have to special-case."""
    stale = mailbox.submit(_png())
    mailbox.submit(_png(size=(32, 32)))
    resp = vlm_client.post(f"/mfluxible/v1/vlm/{stale.id}", json={"regions": []})
    assert resp.status_code == 200
    assert resp.json()["delivered"] is False


def test_regions_round_trip_through_the_schema(vlm_client, mailbox):
    """Boxes survive as fractions in mask_boxes order, which is what lets a chip be
    handed to a generation without conversion."""
    job = mailbox.submit(_png())
    resp = vlm_client.post(
        f"/mfluxible/v1/vlm/{job.id}",
        json={"regions": [{"label": "sky", "box": [0.1, 0.2, 0.8, 0.9]}], "width": 64, "height": 48},
    )
    assert resp.json()["delivered"] is True
    assert job.result["regions"][0]["box"] == (0.1, 0.2, 0.8, 0.9)
    assert (job.result["width"], job.result["height"]) == (64, 48)


# --- the shared reply contract ----------------------------------------------------
#
# Neither backend can run in CI -- one is a subprocess, the other wants a few gigabytes
# of weights -- but everything either side of them can: what a reply has to look like to
# become a chip, and what gets dropped rather than repaired. This lives in
# mfluxible/vlm_reply.py precisely so both backends are held to it by these same tests.


def test_a_reply_is_read_out_of_a_fenced_block():
    from mfluxible.vlm_reply import extract_json

    fenced = 'Here you go:\n```json\n[{"label": "apple", "box": [0.1, 0.2, 0.3, 0.4]}]\n```\nHope that helps.'
    assert extract_json(fenced) == [{"label": "apple", "box": [0.1, 0.2, 0.3, 0.4]}]
    assert extract_json('[{"label": "bare", "box": [0, 0, 1, 1]}]') is not None
    assert extract_json("I could not find any objects.") is None


def test_unusable_boxes_are_dropped_rather_than_coerced():
    """A box outside the frame, or inverted, means the reply wasn't measured against
    the image it was asked about. Clamping it would put a rectangle nobody chose into
    a list the user is about to click."""
    from mfluxible.vlm_reply import clean_regions

    cleaned = clean_regions([
        {"label": "good", "box": [0.1, 0.1, 0.9, 0.9]},
        {"label": "out of frame", "box": [0.1, 0.1, 1.4, 0.9]},
        {"label": "inverted", "box": [0.9, 0.1, 0.2, 0.9]},
        {"label": "zero width", "box": [0.5, 0.1, 0.5, 0.9]},
        {"label": "too short", "box": [0.1, 0.2]},
        {"label": "not numbers", "box": ["a", "b", "c", "d"]},
        "not even an object",
    ])
    assert [c["label"] for c in cleaned] == ["good"]
    assert cleaned[0]["box"] == [0.1, 0.1, 0.9, 0.9]


def test_the_worker_names_a_missing_cli_rather_than_raising():
    """A person reading the harness needs to know it's the binary, not the image."""
    from mfluxible import vlm_worker

    result = vlm_worker.detect("/nonexistent/path/to/an/image.png")
    assert "no longer on disk" in result["error"]


def test_the_number_of_regions_is_capped():
    from mfluxible.vlm_reply import clean_regions

    many = [{"label": f"o{i}", "box": [0.0, 0.0, 0.5, 0.5]} for i in range(20)]
    assert len(clean_regions(many)) == 8


# --- the detection command is a template, not a hardcoded CLI ----------------------


def test_the_default_command_drives_claude_code():
    from mfluxible.vlm_worker import DEFAULT_COMMAND, build_command

    argv = build_command(DEFAULT_COMMAND, image="a.png", prompt="find things")
    assert argv[0] == "claude"
    assert "find things" in argv
    # The model belongs to the default template; a custom command names its own.
    assert "--model" in argv


def test_a_prompts_own_punctuation_cannot_reshape_the_command():
    """Split first, substitute second. Formatting the string and *then* splitting would
    let a quote inside the prompt swallow the rest of the line or split one argument
    into two -- and the prompt is long English prose with quotes and newlines in it."""
    from mfluxible.vlm_worker import build_command

    nasty = 'say "hello" then\nstop --model evil; rm -rf /'
    argv = build_command("tool -p {prompt} --flag", image="a.png", prompt=nasty)
    assert argv == ["tool", "-p", nasty, "--flag"]
    # The prompt is exactly one argument, so nothing inside it is ever a token.
    assert "--model" not in argv
    assert "rm" not in argv


def test_a_detector_template_needs_only_the_image():
    """A tool that isn't prompt-driven simply never substitutes {prompt}."""
    from mfluxible.vlm_worker import build_command

    argv = build_command("detect --format json {image}", image="a.png", prompt="ignored")
    assert argv == ["detect", "--format", "json", "a.png"]
    assert "ignored" not in argv


def test_placeholders_work_inside_a_token():
    from mfluxible.vlm_worker import build_command

    argv = build_command("tool --image={image} --json", image="a.png", prompt="p")
    assert argv == ["tool", "--image=a.png", "--json"]


def test_an_unparseable_template_yields_no_command_rather_than_raising():
    from mfluxible.vlm_worker import build_command

    assert build_command('tool "unclosed', image="a.png", prompt="p") == []
    assert build_command("", image="a.png", prompt="p") == []


def test_a_command_that_cannot_be_run_is_reported_not_raised(tmp_path, monkeypatch):
    """Every failure path returns a body for the server: the harness shows the reason,
    and a misconfigured template is a normal outcome rather than a crashed worker."""
    from mfluxible import vlm_worker

    image = tmp_path / "x.png"
    image.write_bytes(_png())

    monkeypatch.setattr(vlm_worker, "COMMAND", "definitely-not-a-real-binary {image}")
    assert "not on PATH" in vlm_worker.detect(str(image))["error"]

    monkeypatch.setattr(vlm_worker, "COMMAND", 'oops "unclosed')
    assert "unbalanced quotes" in vlm_worker.detect(str(image))["error"]


def test_any_command_printing_the_agreed_json_is_enough(tmp_path, monkeypatch):
    """The whole contract, exercised without Claude: a shell one-liner standing in for
    a local detector. If this passes, so does anything that prints the same array."""
    import sys

    from mfluxible import vlm_worker

    image = tmp_path / "x.png"
    image.write_bytes(_png())

    script = tmp_path / "fake_detector.py"
    script.write_text(
        'import sys\n'
        'print("looking at", sys.argv[1], file=sys.stderr)\n'
        'print(\'[{"label": "a duck", "box": [0.1, 0.2, 0.6, 0.7]}]\')\n'
    )
    monkeypatch.setattr(vlm_worker, "COMMAND", f"{sys.executable} {script} {{image}}")

    result = vlm_worker.detect(str(image))
    # A bare array is the contract's earlier shape and still reads as regions with no
    # prompt, so a wrapper written against it keeps working rather than returning nothing.
    assert result == {"prompt": None, "regions": [{"label": "a duck", "box": [0.1, 0.2, 0.6, 0.7]}]}


# --- the prompt half of the output -----------------------------------------------


def test_an_object_reply_carries_both_a_prompt_and_regions():
    from mfluxible.vlm_reply import clean_payload

    payload = clean_payload({
        "prompt": "  a single red apple on oak, soft window light  ",
        "regions": [{"label": "apple", "box": [0.1, 0.2, 0.6, 0.7]}],
    })
    assert payload["prompt"] == "a single red apple on oak, soft window light"
    assert payload["regions"][0]["label"] == "apple"


def test_a_missing_or_blank_prompt_stays_none_rather_than_empty():
    """The harness tells "no prompt offered" apart from "an empty one": the first hides
    the affordance, the second would put a blank string into the prompt box."""
    from mfluxible.vlm_reply import clean_payload

    assert clean_payload({"regions": []})["prompt"] is None
    assert clean_payload({"prompt": "   ", "regions": []})["prompt"] is None
    assert clean_payload({"prompt": 42, "regions": []})["prompt"] is None


def test_a_prompt_only_reply_is_valid():
    """A tool that captions but doesn't localize is still useful -- the prompt fills the
    box even though there is nothing to click."""
    from mfluxible.vlm_reply import clean_payload

    payload = clean_payload({"prompt": "a misty harbour at dawn"})
    assert payload["prompt"] == "a misty harbour at dawn"
    assert payload["regions"] == []


def test_an_array_of_objects_is_not_mistaken_for_one_object():
    """The outermost bracket decides the shape. Checking for "{...}" first would slice
    from the array's first element to the last "}" inside it -- which parses cleanly and
    silently returns one region where the reply listed several."""
    from mfluxible.vlm_reply import extract_json  # noqa: F401

    reply = '[{"label": "a", "box": [0,0,1,1]}, {"label": "b", "box": [0,0,1,1]}]'
    parsed = extract_json(reply)
    assert isinstance(parsed, list)
    assert len(parsed) == 2

    obj = extract_json('{"prompt": "x", "regions": [{"label": "a", "box": [0,0,1,1]}]}')
    assert isinstance(obj, dict)
    assert obj["prompt"] == "x"


def test_the_prompt_is_capped():
    from mfluxible.vlm_reply import clean_payload

    assert len(clean_payload({"prompt": "x" * 5000})["prompt"]) == 2000


def test_the_result_event_carries_the_prompt(vlm_client, mailbox):
    job = mailbox.submit(_png())
    resp = vlm_client.post(
        f"/mfluxible/v1/vlm/{job.id}",
        json={"prompt": "a teal square", "regions": [], "width": 64, "height": 48},
    )
    assert resp.json()["delivered"] is True
    assert job.result["prompt"] == "a teal square"


def test_a_fresh_mailbox_is_never_attached_however_young_the_clock_is(monkeypatch):
    """`time.monotonic()`'s epoch is arbitrary and counts from boot, so a sentinel of
    0.0 is not "long ago" -- on a machine that booted a minute ago it is a moment ago.
    The original code read that as a worker having just polled.

    Pinned with the clock forced small rather than by waiting for a freshly booted
    machine: on any developer box uptime hides this, which is why it reached main.
    """
    from mfluxible import vlm

    monkeypatch.setattr(vlm.time, "monotonic", lambda: 5.0)
    box = vlm.VlmMailbox(pathlib.Path("/tmp/does-not-need-to-exist"))
    assert box.worker_attached() is False

    # And it still flips to True once something really has polled.
    box._worker_seen = 4.0
    assert box.worker_attached() is True


# --- the in-process backend, over HTTP ----------------------------------------------
#
# The model itself can't run in CI, but the wiring around it is the part that decides
# whether a detection ever reaches it: who claims the job, which timeout applies, and
# what a sidecar polling the wrong server is told. A stand-in detector covers all three
# without a byte of weights, because `engine.run_exclusive` doesn't care what it runs.


class _StandInDetector:
    """Shaped like LocalDetector, answering from a canned reply."""

    def __init__(self, result=None, loaded=True):
        self.result = result if result is not None else {"prompt": "a teal square", "regions": []}
        self.loaded = loaded
        self.seen = []

    def detect_sync(self, path):
        self.seen.append(path)
        return self.result


@pytest.fixture
def local_client(monkeypatch, mailbox):
    """The app with the in-process backend selected and a stand-in for the model."""
    import mfluxible.server as server_module
    from mfluxible.engine import MfluxEngine
    from mfluxible.vlm import BACKEND_LOCAL

    detector = _StandInDetector()
    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    monkeypatch.setattr(server_module, "MAILBOX", mailbox)
    monkeypatch.setattr(server_module, "DETECTOR", detector)
    monkeypatch.setattr(server_module, "VLM_BACKEND", BACKEND_LOCAL)
    monkeypatch.setattr(server_module, "VLM_TIMEOUT_S", 5.0)
    monkeypatch.setattr(server_module, "VLM_LOAD_TIMEOUT_S", 5.0)
    with TestClient(server_module.app) as test_client:
        yield test_client, detector


def _events(resp) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]


def test_the_local_backend_answers_its_own_job(local_client):
    """End to end with no worker anywhere: the same mailbox, filled from inside."""
    client, detector = local_client
    resp = client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(_png())})
    assert resp.status_code == 200

    events = _events(resp)
    assert [e["type"] for e in events] == ["pending", "result"]
    assert events[0]["backend"] == "local"
    assert events[1]["prompt"] == "a teal square"
    # And it was handed the stashed file, not the raw bytes.
    assert len(detector.seen) == 1 and detector.seen[0].exists()


def test_the_local_backend_reports_a_detection_failure_as_an_error_event(local_client):
    """detect_sync returns its reasons rather than raising -- server.py has nowhere to
    catch one, by design -- so an error body has to survive the trip as an event."""
    client, detector = local_client
    detector.result = {"error": "the detection failed -- see the server log for the reason."}

    events = _events(client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(_png())}))
    assert events[-1]["type"] == "error"
    assert "see the server log" in events[-1]["message"]


def test_a_local_job_is_claimed_so_a_timeout_names_the_right_thing(local_client, monkeypatch):
    """"nothing claimed the job" is the worker backend's message and would be a lie
    here: the thing that claims jobs under this backend is the request handler."""
    import mfluxible.server as server_module

    client, detector = local_client
    monkeypatch.setattr(server_module, "VLM_TIMEOUT_S", 0.2)
    monkeypatch.setattr(server_module, "VLM_LOAD_TIMEOUT_S", 0.2)

    def never(path):
        import time as _time

        _time.sleep(5)
        return {"regions": []}

    detector.detect_sync = never
    events = _events(client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(_png())}))
    assert events[-1]["type"] == "error"
    assert "took too long" in events[-1]["message"]


def test_a_cold_model_waits_on_the_load_timeout_rather_than_the_detection_one(
    local_client, monkeypatch
):
    """A first detection downloads a few gigabytes. Failing that after three minutes
    would make a slow connection look like a broken server, which is why the two
    timeouts are separate rather than one generous number."""
    import mfluxible.server as server_module

    client, detector = local_client
    detector.loaded = False
    waited = []

    original = server_module.MAILBOX.wait

    async def record(job, timeout):
        waited.append(timeout)
        return await original(job, timeout)

    monkeypatch.setattr(server_module.MAILBOX, "wait", record)
    monkeypatch.setattr(server_module, "VLM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(server_module, "VLM_LOAD_TIMEOUT_S", 99.0)

    events = _events(client.post("/mfluxible/v1/vlm/describe", json={"image": _b64(_png())}))
    assert waited == [99.0]
    assert events[0]["model_loaded"] is False


def test_health_says_which_backend_answers(local_client):
    client, _ = local_client
    vlm = client.get("/health").json()["vlm"]
    assert vlm == {
        "enabled": True,
        "backend": "local",
        "worker_attached": False,
        "model_loaded": True,
    }


def test_a_worker_polling_a_local_server_is_told_why_it_gets_nothing(local_client):
    """The 404 a worker already handles says "set MFLUXIBLE_VLM_DIR", which here is
    already set and would not help. A sidecar long-polling a server that will never hand
    it a job is exactly the silent nothing this message exists to break."""
    client, _ = local_client

    for resp in (
        client.get("/mfluxible/v1/vlm/next"),
        client.post("/mfluxible/v1/vlm/anything", json={"regions": []}),
    ):
        assert resp.status_code == 404
        assert "in-process" in resp.json()["message"]
        assert "MFLUXIBLE_VLM_BACKEND=worker" in resp.json()["message"]


def test_the_worker_backend_still_hands_out_jobs(vlm_client, mailbox):
    """The other half of the gate above: selecting the worker leaves the two endpoints
    exactly as they were, which is what keeps the sidecar a supported option rather than
    a deprecated one."""
    mailbox.submit(_png())
    resp = vlm_client.get("/mfluxible/v1/vlm/next")
    assert resp.status_code == 200
    assert "image_path" in resp.json()
