"""The object-detection mailbox: one pending job, handed between two clients.

The harness can see an image and drive the GPU; it has no vision model. A Claude
Code session has vision and can read a file off disk; it has no UI. Neither can
reach the other, so this module is the one thing both of them already talk to --
the harness POSTs an image and waits, a worker claims the job, and the regions it
posts back come out of the harness's open stream.

The server does no detection of its own. It never loads a model, never holds an
API key and never makes an outbound call: it writes a file, holds one slot in
memory and copies a JSON array from one HTTP request to another. Everything that
spends a Claude account lives in `server/vlm_worker.py`, deliberately -- see
the note in CLAUDE.md for why that is a client rather than a server feature.

**One slot, not a queue.** MfluxEngine already serializes generations behind a
single lock and runs one model in one process, so a second detection racing the
first has nowhere useful to go. A new submit supersedes the pending job rather
than queueing behind it, and the superseded stream is closed with a reason
instead of being left to time out -- which is what makes double-clicking the
button harmless.

**The directory is the feature switch.** MFLUXIBLE_VLM_DIR unset means the
endpoints 404 and /health reports the feature off, so the harness never draws a
button that cannot work. It is also the working directory the worker runs
`claude` in, and that is not a coincidence: a stashed image inside it needs no
--add-dir to be readable, and a directory with no CLAUDE.md in it keeps a
one-shot detection from loading this project's (very long) instructions on every
call.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Shared rather than reimplemented, for the same reason engine.py has only one copy:
# mflux rotates an input image before encoding it, so a box measured against the
# unrotated bytes would be applied in a different frame to the one it was chosen in.
# The worker sees what this reports, and the mask is built against the same numbers.
from engine import _oriented

log = logging.getLogger("mfluxible.regions")

# Pillow's own guard against a decompression bomb is a warning, not a refusal, and
# the thing being protected here is a file written to disk before anything looks at
# it. A ceiling on the encoded bytes is the cheap half of that.
MAX_IMAGE_BYTES = 32 * 1024 * 1024

# How long a stashed file may sit before a later submit deletes it. The worker reads
# the file while the job is live, so this only ever collects finished ones.
STASH_TTL_S = 3600.0

# A worker that has long-polled within this window counts as attached. It is only
# ever reported, never enforced -- the harness uses it to say "nothing is listening"
# up front rather than after a detection times out.
WORKER_FRESH_S = 90.0


@dataclass
class Job:
    id: str
    path: Path
    width: int
    height: int
    created: float
    claimed: bool = False
    # Set once, by complete(): {"regions": [...]} or {"error": "<curated reason>"}.
    result: dict | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


def _suffix_for(fmt: str | None) -> str:
    return {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif"}.get(fmt or "", ".png")


class VlmMailbox:
    """One pending detection, plus the directory its images are stashed in."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self._pending: Job | None = None
        # Fires when an unclaimed job lands, so a long-polling worker wakes at once
        # rather than on a poll interval. Re-armed by claim().
        self._arrived = asyncio.Event()
        self._worker_seen = 0.0

    # -- lifecycle -------------------------------------------------------------

    def prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def worker_attached(self) -> bool:
        return (time.monotonic() - self._worker_seen) < WORKER_FRESH_S

    # -- harness side ----------------------------------------------------------

    def submit_problem(self, raw: bytes) -> str | None:
        """Why this image can't be accepted, or None. A *returned* reason, not a
        raised one -- the same rule the rest of this server follows, so nothing an
        imaging library says about a path or a buffer can reach a response body."""
        if not raw:
            return "no image was sent."
        if len(raw) > MAX_IMAGE_BYTES:
            return f"image is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)}MB limit."
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                probe.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            return "that doesn't decode as an image."
        return None

    def submit(self, raw: bytes) -> Job:
        """Stash `raw` and make it the pending job, superseding any other."""
        self._prune()
        with Image.open(io.BytesIO(raw)) as img:
            fmt = img.format
        oriented = _oriented(raw)
        width, height = oriented.size

        job_id = uuid.uuid4().hex
        path = self.dir / f"{job_id}{_suffix_for(fmt)}"
        path.write_bytes(raw)

        previous = self._pending
        job = Job(id=job_id, path=path, width=width, height=height, created=time.monotonic())
        self._pending = job
        self._arrived.set()

        # Close the old stream with a reason rather than leaving it to time out, so a
        # second click reads as "replaced" instead of as a hang.
        if previous is not None and previous.result is None:
            previous.result = {"error": "superseded by a newer detection."}
            previous.done.set()
        return job

    async def wait(self, job: Job, timeout: float) -> dict:
        """The harness's side: block until the worker answers, or give a reason."""
        try:
            await asyncio.wait_for(job.done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            if job.result is None:
                claimed = "the worker took too long" if job.claimed else "nothing claimed the job"
                job.result = {"error": f"timed out -- {claimed}."}
        return job.result or {"error": "no result."}

    # -- worker side -----------------------------------------------------------

    async def claim(self, timeout: float) -> Job | None:
        """Long-poll for work. Returns an unclaimed pending job, or None on timeout.

        Marking attendance here rather than on a separate ping is what makes
        `worker_attached()` free: a worker that is waiting for work has, by
        definition, just called this.
        """
        self._worker_seen = time.monotonic()
        deadline = time.monotonic() + timeout
        while True:
            job = self._pending
            if job is not None and not job.claimed and job.result is None:
                job.claimed = True
                self._arrived.clear()
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self._arrived.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            finally:
                self._worker_seen = time.monotonic()

    def complete(self, job_id: str, result: dict) -> bool:
        """Deliver the worker's answer. False if that job is gone or already done."""
        job = self._pending
        if job is None or job.id != job_id or job.result is not None:
            return False
        job.result = result
        job.done.set()
        return True

    # -- housekeeping ----------------------------------------------------------

    def _prune(self) -> None:
        """Delete stashed images older than STASH_TTL_S. Best-effort: a file that
        cannot be removed is logged and left, never raised -- this runs inside a
        request and losing a detection over a stale temp file would be worse than
        the file surviving."""
        live = self._pending.path if self._pending is not None else None
        cutoff = time.time() - STASH_TTL_S
        for entry in self.dir.glob("*"):
            if entry == live or not entry.is_file():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError as exc:  # one bad file must not stop the sweep
                log.warning("could not prune a stashed detection image: %s", exc.strerror)


def mailbox_from_env() -> VlmMailbox | None:
    """The mailbox MFLUXIBLE_VLM_DIR asks for, or None when it is unset.

    One variable both enables and configures, rather than a boolean beside a path:
    there is no useful "on but nowhere to put anything" state, and the feature
    spends a Claude account, so it stays off until someone names a directory.
    """
    raw = os.environ.get("MFLUXIBLE_VLM_DIR", "").strip()
    if not raw:
        return None
    return VlmMailbox(Path(raw).expanduser())
