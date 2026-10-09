"""The AUTOMATIC1111-shaped `/sdapi/v1/txt2img`, for SillyTavern."""

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mfluxible import runtime as rt
from mfluxible.models import CFG_GUIDANCE_FLOOR
from mfluxible.routes.common import collect_final_image
from mfluxible.schemas import A1111Txt2ImgRequest, GenerateRequest

router = APIRouter()


def _a1111_error(message: str, status_code: int = 400) -> JSONResponse:
    # A1111's own handled-error envelope shape (a short `error` plus a `detail`),
    # minus the fields it fills from a traceback -- which this server would have
    # nothing to put in anyway, that being the point of collect_final_image's
    # docstring. Returned, never raised, for the same reason as everything else here.
    code = "invalid_request" if status_code < 500 else "generation_failed"
    return JSONResponse(status_code=status_code, content={"error": code, "detail": message})


def _a1111_to_generate_request(req: A1111Txt2ImgRequest) -> GenerateRequest:
    """The A1111 payload narrowed to what the loaded model can actually act on.

    Mirrors the two rules request_problem enforces -- guidance needs
    `supports_guidance`; a negative prompt needs both a negative branch and
    classifier-free guidance actually switched on above CFG_GUIDANCE_FLOOR -- as
    *drops* rather than rejections, for the reason in the endpoint's comment below.

    That duplication is deliberate but not unchecked: the endpoint still runs
    request_problem on what this returns, so if those rules ever gain a third clause,
    this drifts into a visible 400 rather than a silently wrong image.
    `test_a1111_shim_never_builds_a_request_its_own_model_rejects` pins that across
    every model in the table.
    """
    spec = rt.engine.spec
    guidance = req.cfg_scale if spec.supports_guidance else None

    negative_prompt = (req.negative_prompt or "").strip()
    if negative_prompt and not spec.supports_negative_prompt:
        negative_prompt = ""
    if negative_prompt:
        # Having a negative branch isn't sufficient on its own -- CFG has to be on for
        # the unconditional prompt to be encoded at all, so at or below the floor this
        # would be accepted and never consulted. Same check as request_problem's, and
        # the reason it tests the *effective* guidance rather than the requested one.
        effective = guidance if guidance is not None else spec.default_guidance
        if effective is not None and effective <= CFG_GUIDANCE_FLOOR:
            negative_prompt = ""

    return GenerateRequest(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        # A1111 spells "pick one for me" as -1, GenerateRequest spells it None; passing
        # -1 through would be a literal seed, and a reproducible one at that.
        seed=req.seed if req.seed is not None and req.seed >= 0 else None,
        guidance=guidance,
        negative_prompt=negative_prompt or None,
        stream=False,
    )


# An AUTOMATIC1111-shaped txt2img, narrow on purpose. It exists for SillyTavern's
# image-generation extension, which as of 1.18.0 has no OpenAI-compatible image source
# at all -- its `openai` source hardcodes api.openai.com inside SillyTavern's own
# backend with no base-URL setting, unlike the "Custom (OpenAI-compatible)" source on
# its chat side. Its `sdcpp` source (for stable-diffusion.cpp's server) is the one
# local-URL source that fits, and it is a hybrid: OpenAI-shaped discovery, A1111-shaped
# generation. It probes `OPTIONS /v1/images/generations` (see that handler above),
# reads the model list from `GET /v1/models` -- already OpenAI-shaped here, and parsed
# there as `data.data.map(m => ({value: m.id, text: m.name || m.id}))` -- and then
# posts *this* shape, reading `images[0]` as a base64 PNG back out. Read out of
# SillyTavern's own source rather than inferred from behavior, the same way the mflux
# and mcp notes in CLAUDE.md were: `public/scripts/extensions/stable-diffusion/
# index.js` (generateSdcppImage) and `src/endpoints/stable-diffusion.js` (sdcpp router).
#
# **Why this endpoint ignores what it cannot do, where the native API rejects it.**
# All three of SillyTavern's sdcpp routes end in `response.sendStatus(500)`, attaching
# the upstream body to an Error's `cause` that is only ever console.error'd on its
# server. So a 400 from here reaches the user as a bare failed generation with no
# reason attached: request_problem's carefully-written messages ("Z-Image-Turbo ignores
# guidance; omit the field") land nowhere a user will look. And this caller sends
# cfg_scale and steps on *every* request, from its own sliders, whose defaults (7 and
# 20) suit neither a guidance-distilled model nor a 9-step one. Rejecting those would
# make the source unusable rather than correcting anyone. Hence: drop what the loaded
# model can't act on, and report what actually took effect in `info`, which is the one
# channel that survives the trip.
#
# The line between ignoring and rejecting is whether the caller could otherwise detect
# the difference. A dropped cfg_scale changes the image, but the caller named no
# specific image and has nothing to compare against; batch_size=4 answered with one
# image is a concretely wrong result to a request that named a number. So the first is
# ignored and the second is a 400 -- one SillyTavern can never trigger, since it
# hardcodes batch_size: 1.
#
# `model` is ignored rather than validated, which is the one place this diverges from
# `POST /v1/images/generations` above. One model runs per process, so there is nothing
# to switch to either way; the difference is that SillyTavern only auto-selects from a
# freshly loaded model list when its stored `sd.model` is *empty* (`index.js`: `if
# (!extension_settings.sd.model && models.length > 0)`), so a value left over from a
# previously configured source survives, is sent here, and matches no option in the
# dropdown the user is looking at. Validating it would turn that invisible stale
# setting into an unexplained failure. What actually ran is reported as
# `sd_model_name` in `info` instead, and `GET /v1/models` remains the honest answer to
# what this server has loaded.
#
# Not implemented, and not needed by this caller: /sdapi/v1/progress and
# /sdapi/v1/interrupt (SillyTavern's `auto` source polls and posts those; its `sdcpp`
# source does neither), along with the rest of A1111's endpoint surface. Adding the
# `auto` source instead would mean roughly ten endpoints including a GET and a POST of
# /sdapi/v1/options -- see docs/clients.md.
@router.post("/sdapi/v1/txt2img")
async def a1111_txt2img(req: A1111Txt2ImgRequest):
    if req.batch_size != 1 or req.n_iter != 1:
        return _a1111_error(
            "mfluxible generates one image per request; batch_size and n_iter must both be 1."
        )

    gen_req = _a1111_to_generate_request(req)

    problem = rt.engine.request_problem(gen_req)
    if problem is not None:
        # Only reachable if the drops above have drifted out of sync with
        # request_problem's rules -- see _a1111_to_generate_request's docstring.
        return _a1111_error(problem)

    final = await collect_final_image(gen_req)
    if final["type"] == "error":
        return _a1111_error(final["message"], status_code=500)

    return {
        "images": [final["data"]],
        "parameters": req.model_dump(),
        # A1111 puts a JSON *string* here, and it is the only part of the response that
        # says what happened rather than what was asked. So it reports the effective
        # values, not an echo of the request: the seed that actually ran, the steps
        # that actually ran, and guidance/negative_prompt as they survived the drops
        # above -- `cfg_scale: null` against a request that sent 7 is how a caller can
        # see that this model ignored it. `sd_model_name` is what `model` would have
        # been validated against if this endpoint validated it.
        "info": json.dumps(
            {
                "prompt": gen_req.prompt,
                "negative_prompt": gen_req.negative_prompt or "",
                "seed": final["seed"],
                "all_seeds": [final["seed"]],
                "width": gen_req.width,
                "height": gen_req.height,
                "steps": gen_req.steps if gen_req.steps is not None else rt.engine.spec.default_steps,
                "cfg_scale": gen_req.guidance,
                "sd_model_name": rt.engine.spec.key,
            }
        ),
    }
