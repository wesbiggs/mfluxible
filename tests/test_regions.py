"""The object-detection mailbox: the store's own behaviour, and the HTTP surface.

Nothing here runs a detection. The worker shells out to the Claude Code CLI, which is
exactly the part that can't be exercised in CI -- so the seam under test is the
handover: an image goes in one side, a worker claims it, regions come out the other.
"""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from regions import RegionMailbox
from tests.doubles.toy_model import TOY_MODEL_SPEC


def _png(size=(64, 48), color=(10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture
def mailbox(tmp_path):
    box = RegionMailbox(tmp_path / "regions")
    box.prepare()
    return box


@pytest.fixture
def regions_client(monkeypatch, mailbox):
    """The real app with the mailbox switched on, and both waits shortened so a test
    that exercises a timeout finishes in milliseconds rather than three minutes."""
    import server as server_module
    from engine import MfluxEngine

    monkeypatch.setattr(
        server_module,
        "engine",
        MfluxEngine(model=TOY_MODEL_SPEC, quantize=None, model_cache_dir=None),
    )
    monkeypatch.setattr(server_module, "MAILBOX", mailbox)
    monkeypatch.setattr(server_module, "REGIONS_TIMEOUT_S", 0.25)
    monkeypatch.setattr(server_module, "REGIONS_POLL_S", 0.25)
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
    MFLUXIBLE_REGIONS_DIR names a directory. harness.html draws its button off this."""
    body = client.get("/health").json()
    assert body["regions"] == {"enabled": False, "worker_attached": False}


def test_the_endpoints_are_absent_when_the_feature_is_off(client):
    assert client.post("/mfluxible/v1/regions/detect", json={"image": _b64(_png())}).status_code == 404
    assert client.get("/mfluxible/v1/regions/next").status_code == 404
    assert client.post("/mfluxible/v1/regions/abc", json={"regions": []}).status_code == 404


def test_health_reports_the_feature_on_when_configured(regions_client):
    body = regions_client.get("/health").json()
    assert body["regions"]["enabled"] is True
    assert body["regions"]["worker_attached"] is False


def test_a_worker_polling_shows_up_as_attached(regions_client):
    assert regions_client.get("/mfluxible/v1/regions/next").status_code == 204
    assert regions_client.get("/health").json()["regions"]["worker_attached"] is True


def test_detect_refuses_a_bad_body_before_the_stream_starts(regions_client):
    """A 400, not a 200 with an error torn into the body -- the same rule the
    generation endpoint follows, and the reason request_problem runs in the endpoint."""
    resp = regions_client.post("/mfluxible/v1/regions/detect", json={"image": "not base64!!"})
    assert resp.status_code == 400
    assert "base64" in resp.json()["message"]

    resp = regions_client.post("/mfluxible/v1/regions/detect", json={"image": _b64(b"nope")})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")


def test_a_detection_with_no_worker_streams_pending_then_a_reason(regions_client):
    with regions_client.stream(
        "POST", "/mfluxible/v1/regions/detect", json={"image": _b64(_png())}
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


def test_next_hands_a_worker_the_path_and_the_measured_size(regions_client, mailbox):
    job = mailbox.submit(_png())
    body = regions_client.get("/mfluxible/v1/regions/next").json()
    assert body["job_id"] == job.id
    assert body["image_path"] == str(job.path)
    assert (body["width"], body["height"]) == (64, 48)


def test_posting_a_superseded_job_is_reported_not_an_error(regions_client, mailbox):
    """The usual cause is the user clicking the button again mid-detection, so the
    worker gets a plain answer rather than a status code it would have to special-case."""
    stale = mailbox.submit(_png())
    mailbox.submit(_png(size=(32, 32)))
    resp = regions_client.post(f"/mfluxible/v1/regions/{stale.id}", json={"regions": []})
    assert resp.status_code == 200
    assert resp.json()["delivered"] is False


def test_regions_round_trip_through_the_schema(regions_client, mailbox):
    """Boxes survive as fractions in mask_boxes order, which is what lets a chip be
    handed to a generation without conversion."""
    job = mailbox.submit(_png())
    resp = regions_client.post(
        f"/mfluxible/v1/regions/{job.id}",
        json={"regions": [{"label": "sky", "box": [0.1, 0.2, 0.8, 0.9]}], "width": 64, "height": 48},
    )
    assert resp.json()["delivered"] is True
    assert job.result["regions"][0]["box"] == (0.1, 0.2, 0.8, 0.9)
    assert (job.result["width"], job.result["height"]) == (64, 48)


# --- the worker's own parsing ----------------------------------------------------
#
# The subprocess can't run in CI, but everything either side of it can: what a reply
# has to look like to become a chip, and what gets dropped rather than repaired.


def test_the_worker_reads_an_array_out_of_a_fenced_reply():
    from region_worker import _extract_json_array

    fenced = 'Here you go:\n```json\n[{"label": "apple", "box": [0.1, 0.2, 0.3, 0.4]}]\n```\nHope that helps.'
    assert _extract_json_array(fenced) == [{"label": "apple", "box": [0.1, 0.2, 0.3, 0.4]}]
    assert _extract_json_array('[{"label": "bare", "box": [0, 0, 1, 1]}]') is not None
    assert _extract_json_array("I could not find any objects.") is None


def test_the_worker_drops_unusable_boxes_rather_than_coercing_them():
    """A box outside the frame, or inverted, means the reply wasn't measured against
    the image it was asked about. Clamping it would put a rectangle nobody chose into
    a list the user is about to click."""
    from region_worker import _clean

    cleaned = _clean([
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
    import region_worker

    result = region_worker.detect("/nonexistent/path/to/an/image.png")
    assert "no longer on disk" in result["error"]


def test_the_worker_caps_the_number_of_regions():
    from region_worker import _clean

    many = [{"label": f"o{i}", "box": [0.0, 0.0, 0.5, 0.5]} for i in range(20)]
    assert len(_clean(many)) == 8


# --- the detection command is a template, not a hardcoded CLI ----------------------


def test_the_default_command_drives_claude_code():
    from region_worker import DEFAULT_COMMAND, build_command

    argv = build_command(DEFAULT_COMMAND, image="a.png", prompt="find things")
    assert argv[0] == "claude"
    assert "find things" in argv
    # The model belongs to the default template; a custom command names its own.
    assert "--model" in argv


def test_a_prompts_own_punctuation_cannot_reshape_the_command():
    """Split first, substitute second. Formatting the string and *then* splitting would
    let a quote inside the prompt swallow the rest of the line or split one argument
    into two -- and the prompt is long English prose with quotes and newlines in it."""
    from region_worker import build_command

    nasty = 'say "hello" then\nstop --model evil; rm -rf /'
    argv = build_command("tool -p {prompt} --flag", image="a.png", prompt=nasty)
    assert argv == ["tool", "-p", nasty, "--flag"]
    # The prompt is exactly one argument, so nothing inside it is ever a token.
    assert "--model" not in argv
    assert "rm" not in argv


def test_a_detector_template_needs_only_the_image():
    """A tool that isn't prompt-driven simply never substitutes {prompt}."""
    from region_worker import build_command

    argv = build_command("detect --format json {image}", image="a.png", prompt="ignored")
    assert argv == ["detect", "--format", "json", "a.png"]
    assert "ignored" not in argv


def test_placeholders_work_inside_a_token():
    from region_worker import build_command

    argv = build_command("tool --image={image} --json", image="a.png", prompt="p")
    assert argv == ["tool", "--image=a.png", "--json"]


def test_an_unparseable_template_yields_no_command_rather_than_raising():
    from region_worker import build_command

    assert build_command('tool "unclosed', image="a.png", prompt="p") == []
    assert build_command("", image="a.png", prompt="p") == []


def test_a_command_that_cannot_be_run_is_reported_not_raised(tmp_path, monkeypatch):
    """Every failure path returns a body for the server: the harness shows the reason,
    and a misconfigured template is a normal outcome rather than a crashed worker."""
    import region_worker

    image = tmp_path / "x.png"
    image.write_bytes(_png())

    monkeypatch.setattr(region_worker, "COMMAND", "definitely-not-a-real-binary {image}")
    assert "not on PATH" in region_worker.detect(str(image))["error"]

    monkeypatch.setattr(region_worker, "COMMAND", 'oops "unclosed')
    assert "unbalanced quotes" in region_worker.detect(str(image))["error"]


def test_any_command_printing_the_agreed_json_is_enough(tmp_path, monkeypatch):
    """The whole contract, exercised without Claude: a shell one-liner standing in for
    a local detector. If this passes, so does anything that prints the same array."""
    import sys

    import region_worker

    image = tmp_path / "x.png"
    image.write_bytes(_png())

    script = tmp_path / "fake_detector.py"
    script.write_text(
        'import sys\n'
        'print("looking at", sys.argv[1], file=sys.stderr)\n'
        'print(\'[{"label": "a duck", "box": [0.1, 0.2, 0.6, 0.7]}]\')\n'
    )
    monkeypatch.setattr(region_worker, "COMMAND", f"{sys.executable} {script} {{image}}")

    result = region_worker.detect(str(image))
    assert result == {"regions": [{"label": "a duck", "box": [0.1, 0.2, 0.6, 0.7]}]}
