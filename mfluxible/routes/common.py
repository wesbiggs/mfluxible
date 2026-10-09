"""Helpers shared by more than one route module."""

from mfluxible import runtime as rt
from mfluxible.schemas import GenerateRequest


async def collect_final_image(req: GenerateRequest) -> dict:
    """Runs a generation to completion and returns its final event -- `image`, or
    `error` if it ended in one instead. Callers check which; both endpoints below turn
    an `error` into their own error shape rather than sharing a response type, since
    the native and OpenAI-compat contracts disagree on what an error body looks like.

    Handed back rather than raised, for the reason in MfluxEngine.request_problem's
    docstring: an exception travelling out to a response body is how text from inside
    mflux would reach a client, so this module has no `except ... : str(exc)` in it at
    all and the messages below are only ever ones engine.py wrote to be read.
    """
    async for event in rt.engine.generate_stream(req):
        if event["type"] in ("image", "error"):
            return event
    return {"type": "error", "message": "generation ended without an image or error event"}
