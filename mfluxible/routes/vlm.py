"""The object-detection mailbox routes. See vlm.py for the design."""

import asyncio
import base64
import json

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mfluxible import runtime as rt
from mfluxible.schemas import VlmRequest, VlmResult
from mfluxible.vlm import BACKEND_WORKER

router = APIRouter()


# ---------------------------------------------------------------------------
# Object detection mailbox
# ---------------------------------------------------------------------------
#
# The harness submits an image and holds a stream; something answers. Which something
# is MFLUXIBLE_VLM_BACKEND (see vlm.py), and the mailbox is deliberately the meeting
# point for both: under `worker` a sidecar claims the job through the two endpoints
# below, and under `local` a task started here fills the same slot from
# mfluxible/vlm_local.py. `_vlm_sse` cannot tell the difference and does not need to.
#
# Note what did *not* change when the local backend arrived: the server still contains
# no detection logic and no `except` clause. It starts a task and waits on a Job, the
# same as it waited on a worker.

_VLM_OFF = {"type": "error", "message": "object detection is not enabled on this server."}

# A different 404, for a worker polling a server that is answering its own detections.
# Worth distinguishing: "switched off" tells its operator to set MFLUXIBLE_VLM_DIR,
# which here is already set and would not help. A sidecar quietly long-polling a server
# that will never hand it a job is the failure this sentence exists to name.
_VLM_NOT_WORKER = {
    "type": "error",
    "message": (
        "this server runs detections in-process (MFLUXIBLE_VLM_BACKEND=local); "
        "it hands out no jobs. Set MFLUXIBLE_VLM_BACKEND=worker to use this sidecar."
    ),
}


def _vlm_disabled(content: dict = _VLM_OFF) -> JSONResponse:
    # 404 rather than 501: with MFLUXIBLE_VLM_DIR unset these routes are not a
    # feature this server has, and /health already says so before anything calls them.
    return JSONResponse(status_code=404, content=content)


_LOCAL_DETECTIONS: set[asyncio.Task] = set()


def _start_local_detection(job) -> None:
    """Answer `job` from the in-process model, off the request's own coroutine.

    `run_exclusive` puts the work on the engine's MLX thread under its lock, so this
    queues behind any running generation -- which is why it cannot be awaited inline:
    the harness's stream has a `pending` event to send first, and it is the thing that
    tells the user a download may be about to happen.
    """

    async def run() -> None:
        rt.MAILBOX.complete(job.id, await rt.engine.run_exclusive(rt.DETECTOR.detect_sync, job.path))

    task = asyncio.create_task(run())
    # The event loop holds only a weak reference to a running task, so without a strong
    # one here a detection can be collected mid-flight and the stream waits out its
    # whole timeout for an answer nothing is still computing.
    _LOCAL_DETECTIONS.add(task)
    task.add_done_callback(_LOCAL_DETECTIONS.discard)


def _decoded_image(b64_data: str) -> bytes | None:
    """The bytes, or None if that string wasn't base64. Returned, not raised: the
    caller turns None into its own curated message, so binascii's never reaches a
    response body (see the invariant in CLAUDE.md)."""
    try:
        return base64.b64decode(b64_data, validate=True)
    except (ValueError, TypeError):
        return None


async def _vlm_sse(job):
    # Sent before anything blocks, so the harness can render "waiting" with the real
    # dimensions and say up front whether a worker is even listening.
    # True when nothing further has to be fetched before work can start: the worker
    # backend is never waiting on weights, and the local one stops being so once its
    # first detection has loaded them.
    warm = rt.DETECTOR is None or rt.DETECTOR.loaded
    yield "data: " + json.dumps({
        "type": "pending",
        "job_id": job.id,
        "width": job.width,
        "height": job.height,
        "backend": rt.VLM_BACKEND,
        "worker_attached": rt.MAILBOX.worker_attached(),
        "model_loaded": warm,
    }) + "\n\n"

    # A cold local backend is about to download a few gigabytes, which is not a thing
    # to fail after three minutes of. See VLM_LOAD_TIMEOUT_S.
    result = await rt.MAILBOX.wait(job, rt.VLM_TIMEOUT_S if warm else rt.VLM_LOAD_TIMEOUT_S)
    if result.get("error"):
        yield "data: " + json.dumps({"type": "error", "message": result["error"]}) + "\n\n"
        return
    # One event carrying everything the tool said. Named `result` rather than `regions`
    # because it no longer only carries those: `prompt` is a text-to-image prompt for
    # the whole frame, and a tool may return either half on its own.
    yield "data: " + json.dumps({
        "type": "result",
        "prompt": result.get("prompt"),
        "regions": result.get("regions") or [],
        # What the worker measured, for the harness to compare against its own oriented
        # size -- the guard against regions arriving for a different image.
        "width": result.get("width"),
        "height": result.get("height"),
    }) + "\n\n"


@router.post("/mfluxible/v1/vlm/describe")
async def vlm_describe(req: VlmRequest):
    if rt.MAILBOX is None:
        return _vlm_disabled()

    raw = _decoded_image(req.image)
    if raw is None:
        return JSONResponse(
            status_code=400, content={"type": "error", "message": "image is not valid base64."}
        )
    problem = rt.MAILBOX.submit_problem(raw)
    if problem is not None:
        return JSONResponse(status_code=400, content={"type": "error", "message": problem})

    # Same shape as a generation: validate fully, *then* start the stream, so a bad
    # request is a 400 rather than a 200 with an error torn into the body.
    # Under the in-process backend this process is what claims the job, so say so up
    # front: a timeout then reports "took too long" rather than "nothing claimed the
    # job", which here would be actively misleading.
    job = rt.MAILBOX.submit(raw, claimed=rt.DETECTOR is not None)
    if rt.DETECTOR is not None:
        _start_local_detection(job)
    return StreamingResponse(_vlm_sse(job), media_type="text/event-stream")


@router.get("/mfluxible/v1/vlm/next")
async def vlm_next():
    """A worker's long-poll. Returns the pending job, or 204 when none arrives.

    This is the one endpoint here that deliberately returns a filesystem path, which
    is the opposite of what the rest of this server does with them. It isn't a leak of
    the kind CLAUDE.md's invariant guards against -- that one is about text from inside
    an imaging library reaching an arbitrary caller through an exception. Here the path
    is the entire payload: the worker's job is to read that file. It stays safe because
    the path only ever names a file this server wrote inside its own regions directory,
    and because this route is gated -- it is not in Caddyfile.example's `@public`
    matcher, which tests/test_proxy_config.py exists to keep true.
    """
    if rt.MAILBOX is None:
        return _vlm_disabled()
    if rt.VLM_BACKEND != BACKEND_WORKER:
        return _vlm_disabled(_VLM_NOT_WORKER)

    job = await rt.MAILBOX.claim(rt.VLM_POLL_S)
    if job is None:
        return Response(status_code=204)
    return {
        "job_id": job.id,
        "image_path": str(job.path),
        "width": job.width,
        "height": job.height,
    }


@router.post("/mfluxible/v1/vlm/{job_id}")
async def vlm_complete(job_id: str, result: VlmResult):
    if rt.MAILBOX is None:
        return _vlm_disabled()
    if rt.VLM_BACKEND != BACKEND_WORKER:
        return _vlm_disabled(_VLM_NOT_WORKER)

    delivered = rt.MAILBOX.complete(job_id, result.model_dump())
    if not delivered:
        # Not an error worth a 4xx on the worker's side: the usual cause is the user
        # clicking the button again, which supersedes the job this worker was running.
        return {"delivered": False, "reason": "that job is no longer the pending one."}
    return {"delivered": True}
