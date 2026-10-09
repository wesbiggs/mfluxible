"""FastAPI server exposing a streaming image-generation endpoint over mflux.

One process runs one model, chosen at startup with MFLUXIBLE_MODEL (see models.py
for the table). Weights are only fetched for the model actually selected.

This module assembles the app: lifespan, middleware, and the routers under
`mfluxible/routes/`. The settings and the engine live in `mfluxible/runtime.py`.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mfluxible import __version__
from mfluxible import runtime as rt
from mfluxible.auth import BasicAuthMiddleware, resolve_bind_host, startup_warning
from mfluxible.limits import BodyLimitMiddleware
from mfluxible.routes import a1111, chat, misc, native, openai, vlm

# Nothing configures logging here, so this lands on stderr via logging.lastResort --
# the terminal running uvicorn -- the same way engine.py's does.
log = logging.getLogger("mfluxible.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the weights, so an exposed-and-unauthenticated server says so
    # immediately rather than after a multi-gigabyte download.
    warning = startup_warning(resolve_bind_host(), rt.BASIC_AUTH is not None)
    if warning is not None:
        log.warning(warning)
    # Before the weights too: a bad MFLUXIBLE_VLM_DIR should fail now rather than
    # after a multi-gigabyte download, for the same reason the warning above is here.
    if rt.MAILBOX is not None:
        rt.MAILBOX.prepare()
    await rt.engine.load()
    rt.MODEL_LOADED_AT = int(time.time())
    yield
    rt.engine.shutdown()


app = FastAPI(title="mfluxible", version=__version__, lifespan=lifespan)

# Default: reflect back any http(s)://localhost:<any port> or 127.0.0.1:<any
# port> origin (Starlette's allow_origin_regex does a fullmatch against the
# Origin header and, on match, echoes that exact origin in
# Access-Control-Allow-Origin rather than "*" -- so a local harness page
# served from either hostname on any port just works). Add other specific
# origins via MFLUXIBLE_CORS_ORIGINS (comma-separated) if needed.
CORS_ORIGIN_REGEX = os.environ.get("MFLUXIBLE_CORS_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?")
_raw_cors_origins = os.environ.get("MFLUXIBLE_CORS_ORIGINS", "").strip()
CORS_ORIGINS = [o.strip() for o in _raw_cors_origins.split(",") if o.strip()]

# Order matters, and it is the reverse of how it reads. Starlette's add_middleware
# inserts at the front of the stack, so the middleware added *last* runs *first* --
# meaning CORS below wraps auth here, not the other way round. That is the order
# required: CORSMiddleware answers a browser preflight itself and returns before
# auth ever sees it, which it has to, because a preflight carries no credentials
# by spec and a 401 there would make every cross-origin call fail with no way for
# the page to send them. Pinned by test_cors_preflight_is_answered_without_credentials.
# Read through callables, not captured, so the tests can swap them.
#
# The body limit is the innermost of the three, so an unauthenticated caller is refused
# before it ever reads a byte.
app.add_middleware(BodyLimitMiddleware, limit=lambda: rt.MAX_BODY_BYTES)
app.add_middleware(BasicAuthMiddleware, credentials=lambda: rt.BASIC_AUTH)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX or None,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (misc, native, vlm, openai, chat, a1111):
    app.include_router(module.router)
