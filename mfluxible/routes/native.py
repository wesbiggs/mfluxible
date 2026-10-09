"""mfluxible's own generation endpoint."""

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from mfluxible import runtime as rt
from mfluxible.routes.common import collect_final_image
from mfluxible.schemas import GenerateRequest

router = APIRouter()


async def _sse(req: GenerateRequest):
    async for event in rt.engine.generate_stream(req):
        yield f"data: {json.dumps(event)}\n\n"


# This is mfluxible's own API -- streaming `thinking` events with step timings and
# previews, guidance/negative_prompt, etc. -- moved off /v1/images/generations so that
# path can be a genuine OpenAI-compatible endpoint instead (see below). Every bundled
# client (harness.html, stream_client.py/js, mfluxible_mcp/server.py) targets this path.
@router.post("/mfluxible/v1/images/generations")
async def generate(req: GenerateRequest):
    # Checked before the response begins: for a stream, failing once StreamingResponse
    # has started would mean a torn body with a 200 already on the wire.
    problem = rt.engine.request_problem(req)
    if problem is not None:
        return JSONResponse(status_code=400, content={"type": "error", "message": problem})

    if req.stream:
        return StreamingResponse(_sse(req), media_type="text/event-stream")

    final = await collect_final_image(req)
    if final["type"] == "error":
        return JSONResponse(status_code=500, content={"type": "error", "message": final["message"]})
    return final
