# Schedulers, fractional start and inpainting

Why `schedulers.py` exists, how masked inpainting rides mflux's scheduler slot, and the measurements behind the defaults. Read before touching `schedulers.py`, mask handling in `engine.py`, or `image_strength`/`mask_feather` defaults.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## mfluxible/schedulers.py depends on mflux conditioning the model on sigmas, not on the step index

`fractional_start` works by moving `sigmas[init_time_step]` off the grid, so the input
image is noised to a level that lies *between* two schedule rungs while the loop still
starts on a whole step. That is only coherent because all three variants derive the
transformer's time conditioning from `sigmas[t]` itself -- `z_image.py` computes
`timestep = 1 - sigmas[t]` inline, and `flux_transformer/transformer.py` and
`qwen_transformer.py` both index `config.scheduler.sigmas[...]`. If a future mflux
conditioned on the step index (or on `LinearScheduler._get_timesteps`, which returns a
bare `arange` nothing currently reads), the latents and the model's idea of where they
are would silently desync, and the failure would look like bad images rather than an
error. The module's docstring carries the rest: why `LinearScheduler` is the right base
for the linear-schedule models in `models.py`, and why the interpolation is done on the
request's own shifted schedule rather than by re-deriving mflux's shift math.

**Which models can take a fractional start is a per-model fact, now recorded as
`ModelSpec.default_scheduler`.** Handing mflux `SCHEDULER_PATH` *replaces* the scheduler
the variant would have chosen, so it is only a fractional start on a model that would
have run linear anyway. Elsewhere it is a sampler swap with no error — Z-Image base and
every FLUX.2 Klein default to `flow_match_euler_discrete`, and the failure looks like bad
images — or a `ValueError` thrown inside `generate_image()` on the worker thread with the
SSE headers already on the wire (`Krea2._resolve_scheduler` maps `"linear"` onto
`"er_sde"` and rejects every other name). `request_problem` rejects `fractional_start` up
front wherever `default_scheduler != "linear"`, for the same reason it rejects guidance:
a 400 before `StreamingResponse` starts, not a torn body. Extending it to a flow-match
model means a sibling scheduler subclassing mflux's flow-match class, not relaxing that
check.

The scheduler is selected by handing mflux a dotted import path
(`Config` -> `try_import_external_scheduler`), now `mfluxible.schedulers.*` -- an
ordinary package import, where it used to depend on `server/` being on `sys.path` as
flat modules. It is still a string kept in step with a class by hand, and nothing checks
it until mflux imports it mid-generation, which is why `test_schedulers.py` builds a real
`Config` from it rather than asserting the literal. That path must never be built from
request data -- it is an arbitrary module import in the server process -- which is why
the API takes a bool and `SCHEDULER_PATH` is a constant.

## Inpainting rides mflux's scheduler slot, because callbacks cannot change anything

Regenerating part of an image means writing the untouched region back over the latents
once per denoising step. The callback registry `engine.py` already hooks **cannot** do
that: a variant's loop is `latents = config.scheduler.step(...)` and only *then*
`ctx.in_loop(t, latents)`, and MLX arrays are immutable, so a subscriber can observe the
trajectory and never influence it. `scheduler.step()` is the only call whose return value
becomes the next step's latents, which is why `MaskedBlendLinearScheduler` lives in
`schedulers.py` beside the fractional-start one rather than anywhere near the callbacks.

What gets written back is exact, not an approximation, and that is a property of these
models rather than a design choice: all three variants noise an image by plain
interpolation (`LatentCreator.add_noise_by_interpolation` is
`(1 - sigma) * clean + sigma * noise`), so "the original at the level this step reached"
is that one line evaluated at `sigmas[timestep + 1]`, with the same noise sample the run
started from. The last sigma is 0, so the kept region lands on the encoded original
exactly -- which is what makes "outside the mask is unchanged" true rather than close.

**Two scheduler classes, not one with a flag.** Without a mask, the signal that says "no
fractional start" is engine.py *not passing a scheduler at all*. A mask forces one to be
passed on every request, which destroys that signal -- so a single class would have to
infer the fractional start from `config.image_strength` and would silently switch it on
for anyone whose strength happened to fall between two rungs. `scheduler_path(masked=,
fractional=)` is a fixed four-entry table for the same reason `SCHEDULER_PATH` is a
constant: the value is a dotted path mflux imports, i.e. an arbitrary module import in
the server process, and must never be assembled from request data.

**The job is module-level state, which the fractional-start scheduler deliberately
avoids.** There is no Config route for it: mflux resolves the scheduler from the Config
alone, and a Config carries no mask, no encoded latents and no seed. It is safe only
because of two properties of this server -- `MfluxEngine` serializes generations behind
an `asyncio.Lock`, and all MLX work runs on one dedicated worker thread -- so at most one
job can ever be live. `run()` sets nothing and clears it in its `finally`; the *set* is
in `_StreamCallback.call_before_loop`. If either invariant goes away this has to become
per-Config state rather than a wider lock.

**Why before_loop rather than before generate_image().** The job has to be built at the
dimensions mflux is really generating at, and `Config` floors width and height to a
multiple of 16 -- encoding at the requested size instead yields latents a different shape
to the ones the loop carries. `before_loop` is the earliest hook that knows the floored
values, and by then mflux has already resolved `config.scheduler` to build the starting
latents. So the scheduler necessarily exists before its job does, and `_MaskedBlend`
checks for a missing job in `step` rather than `__init__`. Missing raises: an unmasked
image would look completely fine and quietly ignore the mask, where a raise lands in
`generate_stream`'s existing `except` as a logged traceback and a failed generation.

**The mask reaches the latents through the model's own `pack_latents`, which is what
keeps engine.py free of per-model branching.** Packing is a pure spatial rearrangement,
so replicating one mask value across the channel axis and packing it puts each element's
mask on that element -- whether the variant folds 2x2 patches into channels (FLUX, Qwen)
or only shuffles axes (Z-Image). The latent grid's size comes from the encoder's own
output shape rather than a hardcoded /8, and the downsample is `Image.BOX`: a windowed
filter would ring past [0, 1] at a hard mask edge, and overshoot there means "more than
the original" or "less than nothing" in the blend.

**`image_strength` near 0 is what inpainting wants, and that surprises everyone -- so it
is the default under a mask rather than advice about one.** With a mask the field no longer
governs how much of the *frame* survives -- the mask does -- only how much of the original
content inside the mask does. mflux's convention is already the inverse of other tools', so
the failure mode is that 0.4 (mflux's CLI default, and what `DEFAULT_IMAGE_STRENGTH` still
means for an unmasked request) leaves the old object standing in the region just painted,
which reads as the mask having been ignored.

The reason that is a `DEFAULT_MASKED_IMAGE_STRENGTH = 0.0` in engine.py rather than a
sentence in `docs/api.md` is the *shape* of the failure: a 400 would be fine, but this one
is well-formed. The stream is correct, the image arrives, the generation reports success,
and the only evidence anything went wrong is that the picture looks like the input.
Measured on a 768x768 Z-Image-Turbo run with identical seed, box, feather and prompt: 5.1/255
of change inside the mask at 0.4 against 41.6 at 0.0 -- the cup that was supposed to become
a dinosaur came back a cup. And it was the *default* path, since omitting an optional field
is the ordinary thing to do. An explicit value still wins, because raising it to restyle
what is inside the region rather than replace it is a real request.

`harness.html` also moves its own field to 0 when the box is ticked, which is now belt and
braces -- it always sends an explicit `image_strength`, so the server's default never fires
for it. Keep it: the point there is that the user *sees* the value change.

**Above 0, the useful range is 0.1-0.2 and it is a composition anchor rather than a strength
dial -- which is also where a formula in `docs/api.md` was wrong.** `Config.init_time_step`
gates on `image_strength > 0.0` *before* applying `max(1, int(steps * strength))`, so 0.0
returns 0 and any non-zero strength returns at least 1. The docs stated only the `max()`,
which implies 0.0 -> 1 and makes 0.0 and 0.1 look adjacent; they are a whole rung apart, and
that rung is the entire effect. At 0.0 the masked region starts from pure noise and the
prompt alone decides where the new content lands inside the box. At rung 1 it starts from the
encoded original, so the new content inherits its position, scale and outline.

That is what makes a non-zero strength worth having, and it is not "restyling": it is the
only way to hold a replacement where the original sat when the prompt cannot reproduce the
original composition. A photograph is the case that forces it -- "a man reading a book" will
not be framed the way the photo was, so without an anchor the replacement is correctly
painted and in the wrong place. The cost is bleed-through and it rises with the anchor:
replacing an S chest emblem with an F, 0.0 gave a clean F at a prompt-chosen size, while 0.2
with fractional_start held the shield exactly in place and let the old S contaminate the
letterform. Anchor when the new content should share the old one's geometry, not otherwise;
no single value does both, which is why this stays a documented dial rather than a smarter
default. Note also that 0.1 and 0.2 are the same rung at 9 steps -- `fractional_start` is
what makes them distinguishable. It also makes `fractional_start` worth
more than it is for plain img2img, since the useful range is narrow and still quantized.

**Measured, so it doesn't get re-litigated:** with the blend alone, outside-mask mean
absolute error against the input is ~1.45/255 (p99 4.3) on a 1024x1024 Z-Image-Turbo run
-- at or below the VAE encode/decode floor of ~1.73, i.e. the blend costs nothing and the
residual is the round-trip. `mask_composite` pastes the result back through the mask in
pixel space and takes it to exactly 0. The blend itself is free in time too: 131.3s vs
136.2s for the same 8 steps masked and unmasked.

**`mask_feather` is not a region blend, and a big one is a double exposure.** It looks
like a pixel-space feather and isn't: a grey mask value is a *per-step* blend weight, so
it means "hold this pixel partway to the original at every denoising step" and the model
never commits to anything in that band. Verified against the real model rather than
reasoned about -- radius 8 on a 1024x1024 run gives the clean result, radius 96 with
everything else identical brings the original mug back as a ghost with the new cactus
faded through it. So the field exists to hide the latent grid's 8px staircase along an
edge and wants to stay within a couple of cells of it; harness.html caps its input at 32.
This is the one place where "more of the smoothing parameter" makes the output
qualitatively wrong rather than merely softer.

**The kept region's tone is not a constraint on the new content.** Nothing forces the
model to match the exposure of what it is keeping, so a mask that boxes in a swathe of
flat background gets a rectangle: measured on the same run, the wall inside the mask came
back at luminance ~159 against the input's ~149. Worth knowing precisely because the
instinct is to blame `mask_composite` -- it isn't that; the identical gap is in the raw
decode with compositing off, and the blend's own accuracy outside the mask on that run was
mean 1.43 / p99 4.00, i.e. the VAE floor. Masking tightly around the subject is the
answer, not widening the feather (see above).

**Two further limits are inherent and are documented rather than worked around.** It
fills, it does not erase -- nothing tells the model the region is a hole to continue the background
across, so masking an object and prompting "an empty wooden table" produces a table-shaped
object in the hole (and on a guidance-distilled model there is no negative prompt to say
"nothing here" with). And the mask edge is a hard boundary: content that wants to cross it
is cut off at it, which is a mask-authoring problem, confirmed by re-running the same seed
and prompt with the mask extended and getting the uncut result.

**`POST /v1/images/edits` still rejects `mask`, now for a different reason.** OpenAI marks
the editable region with *transparency*; the native `mask` field marks it *white* and
ignores alpha. Reading one as the other doesn't fail, it inpaints the complement -- every
region the caller meant to keep -- so that endpoint refuses and names the native one.
Honouring OpenAI's convention is real work (alpha handling, plus deciding what a fully
opaque mask means), not an alias.

One harness landmine worth not rediscovering: the progress sweep across the top of the
canvas is `body:has(#submitBtn:disabled) .canvas::before`, which works only because
Generate is disabled for exactly the length of a run. Gating the button on "a mask has
been painted" would latch that bar on forever, so the empty-mask case is refused in the
submit handler with a log line instead. The mask controls' visibility is pure CSS off
`#imagePreview.hidden` and `#inpaint:not(:checked)`, scoped to `body` rather than
`#baseSection` because one of them lives on the result pane.
