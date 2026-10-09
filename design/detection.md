# Object detection: the mailbox and its two backends

Why detection has a mailbox, a lazy in-process backend and a separate sidecar, and which argument applies to which. Read before touching `vlm*.py`, `routes/vlm.py` or the harness's Find objects button.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## Two detection backends, and the sidecar argument only reaches one of them

`mfluxible/vlm.py` holds one pending job; something fills it. Which something is
`MFLUXIBLE_VLM_BACKEND`, and the two options are not competing implementations of one
idea -- they answer different needs, and conflating them is how the default ended up
wrong for two releases.

- **`local`** (the default): `mfluxible/vlm_local.py`, an MLX vision model loaded into
  the server process on the first detection that asks for one.
- **`worker`**: `mfluxible/vlm_worker.py`, the sidecar, which runs an arbitrary command
  line -- `claude -p` by default.

**The sidecar's justification is two arguments that coincide for `claude -p` and come
apart everywhere else.** They read as one, which is why the sidecar looked mandatory:

- **Consent and blast radius.** An arbitrary command with filesystem access that makes
  outbound calls and spends an account must not sit behind an HTTP endpoint. Behind one,
  `POST /.../detect` would be comfortably the most dangerous route in this repo and one
  path away from `tests/test_proxy_config.py`'s worst case -- a path drifting into
  `@public` would stop being an ungated GPU and start being remote command execution.
- **Residency.** A loaded model holds unified memory and an MLX buffer cache for the
  life of the process.

`claude -p` triggers the first and not the second: it loads nothing and exits. A local
MLX model triggers the second and not the first: it runs no command, holds no key and
spends nothing. They are almost perfectly complementary, so "separate process" survives
as the answer for the CLI and stops being one for a model -- where it was buying nothing
but a second thing to start and a feature that silently does nothing when you forget.

**So the local backend is allowed inside the server, and the cost it pays instead is
memory.** That is not a generic worry here, it is the documented one: see the MLX buffer
cache section below, where identical work went from 7s to 105s because the footprint
passed the machine's headroom. Hence three properties, each load-bearing:

- **Lazy.** Nothing is downloaded or loaded until the first detection. A server whose
  users never press the button is byte-for-byte the server it was before, which is what
  makes shipping this on by default defensible at all.
- **`MfluxEngine.run_exclusive`, not a second thread.** Detection runs on the same MLX
  worker thread under the same lock as generation. A detection therefore waits behind a
  running generation and vice versa. That is the trade, not a limitation to route around:
  two MLX consumers interleaving on one device is exactly the unsynchronized overlap the
  single-lock invariant below exists to forbid.
- **Two timeouts.** `MFLUXIBLE_VLM_LOAD_TIMEOUT` (900s) for a cold first detection that
  is downloading gigabytes, `MFLUXIBLE_VLM_TIMEOUT` (180s) once warm. One generous
  number would make a wedged detection take fifteen minutes to fail; one tight number
  makes a slow connection look like a broken server.

**The mailbox is the meeting point, and keeping it for both is what makes `local` a
backend rather than a parallel universe.** The local path has the bytes in memory and
could skip the stash entirely -- it doesn't, because the Job carries a path, superseding
closes the previous stream, the dimension guards compare the same numbers, and
`_vlm_sse` cannot tell which side answered. One of those paths having no file would mean
two lifecycles to keep in step, to save a PNG write against a multi-second inference.
Note also what did *not* change: server.py still contains no detection logic and no new
`except`. It starts a task and waits on a Job, the same as it waited on a worker.

**Qwen's coordinates become fractions in `vlm_local.py`, and that is a driver rather
than a dial.** The rule below -- no coordinate conversion setting, because a
wrongly-scaled box comes back looking plausible -- bans a menu an operator picks from,
not a backend that knows its own model's convention. Qwen3-VL (the default since it
replaced Qwen2.5-VL as `DEFAULT_MODEL`) emits coordinates as integers on a fixed 0-1000
scale, relative to the whole image regardless of how it was resized -- confirmed against
Qwen's own cookbook rather than assumed, quoted in full in `vlm_local.py`'s `BOX_KEY`
comment: "you don't need to calculate the resized_w." That is a reversion to the scheme
Qwen2-VL used before Qwen2.5-VL's absolute pixels replaced it for one release, and it is
why `to_fractions` no longer takes a `width`/`height` at all -- the whole conversion is
dividing by 1000, and it needs no frame from the caller the way the pixel dialect did.

`smart_resize` (Qwen's own sizing algorithm, reimplemented here) and `_budget` (which
reads `min_pixels`/`max_pixels` off the loaded processor instead of hardcoding a
default) still run before every detection, but what they buy changed with the dialect.
Under Qwen2.5-VL's absolute pixels, a wrong budget meant a *silently* mis-scaled box --
the pixel values were divided by the very frame `smart_resize` computed, so a budget
that undershot the processor's own left every box correctly shaped and systematically
too close to the top-left corner. Under Qwen3-VL's frame-independent coordinates that
risk is gone: `to_fractions` never reads the frame, so a wrong budget now only costs
resizing the image once here and again inside the processor, not a wrong answer.
Pre-resizing survives purely for that efficiency -- an oversized photo no longer needs
its frame computed *correctly* to be interpreted correctly, only computed at all, to
avoid the redundant work.

The aspect ratio still drifts 2-3% on an ordinary photograph, because Qwen floors each
axis onto its patch grid independently. That costs nothing in coordinates for the same
reason it always didn't: the resize (ours or the processor's own) stretches width onto
width and height onto height rather than cropping, so a fraction along either axis
survives untouched regardless of which frame the model actually saw -- worth not
re-deriving, because the drift looks like a bug here and is upstream's algorithm.

**`mfluxible/vlm_reply.py` is a module rather than a few functions in `vlm.py`**, because
`vlm.py` imports `engine.py` for `_oriented` and therefore MLX. The worker is the backend
whose whole point is needing none of that, and making it import a GPU framework to parse
a JSON array would be a cost paid by the cheap one.

The rest of this section is about the `worker` backend, and is unchanged.

`mfluxible/vlm.py` holds one pending job and copies a JSON array between two HTTP
requests. It loads no model, holds no key and makes no outbound call -- the harness
POSTs an image and waits on an SSE stream, `mfluxible/vlm_worker.py` claims the job,
shells out to `claude -p`, and posts regions back.

**The worker is a separate process because of what it does, not to keep the package
tidy -- and that is a different claim from where its file sits.** It runs an arbitrary
operator-configured command with filesystem access, which on the default setting also
makes network calls and spends a Claude account. Had that lived behind an endpoint,
`POST /.../detect` would be comfortably the most dangerous route in this repo and one
path away from `tests/test_proxy_config.py`'s worst case -- a path drifting into
`@public` would stop being an ungated GPU and start being remote command execution. As
a sidecar it is unreachable from the network, it is optional (the server reports
`worker_attached: false` and the harness says so), and *starting it* is what consent to
that command running looks like.

It lived in `clients/` first, on the reasoning that `clients/` is "everything that talks
to it over HTTP" and this does. That was the wrong read: everything else there
*consumes* the API to get an image out, while this supplies a capability the server
advertises, has to be co-located, and is co-configured. The safety argument above never
depended on the word -- a sidecar is just as much a separate process -- but the label
did have a consequence, which was keeping its settings out of `docs/server.md` under the
client/server documentation split. They belong there, which is where they now are.

**Which is also why the CLI is a template rather than a hardcoded call.** The worker's
real dependency is narrow -- a program that prints a JSON object of a prompt and
labelled boxes to stdout -- so `MFLUXIBLE_VLM_COMMAND` names the whole command line and
`claude -p` is only its default. That keeps this repo from pinning a fast-moving external at the one
layer where it does not have to, and opens the local detectors (faster, and free) with
no change here.

The *output* contract stays fixed, and that asymmetry is the point. Tools disagree about
coordinates -- pixels, 0-1000, `xywh` -- so a conversion setting would turn a
wrongly-scaled box into a plausible-looking one, silently. A tool that does not already
speak fractions wants a few lines of wrapper instead: one reader, in one place, keeps a
bad box a *visibly* bad box.

`build_command` splits the template with `shlex` **before** substituting the
placeholders, and that order is load-bearing rather than incidental. The prompt is long
English prose containing quotes, braces and newlines; formatting the string first and
splitting after would let a quote inside it swallow the rest of the line or split one
argument in two. Pinned by
`test_a_prompts_own_punctuation_cannot_reshape_the_command`. No shell is spawned, so
`;` and `|` are argument characters rather than syntax.

**One slot, and no identifier.** The obvious design hashes the image so the two sides
can agree which one they mean. That cannot work, and the reason is measured rather than
assumed: an image that reaches a Claude Code session is **re-encoded** on the way into
the session transcript. A 622,650-byte PNG read with the Read tool was stored as a
70,548-byte **JPEG** -- 11% of the size, a different format, so no hash of the original
can ever match. Dimensions survived (768x768), which is the half that matters, because
boxes are fractions and fractions are scale-invariant for the same reason `mask_boxes`
are in the MCP tool.

So the mailbox pairs a waiting stream with an arriving answer and needs no key at all.
That is only coherent because `MfluxEngine` already serializes generations behind one
lock in one process: a second detection has nowhere useful to queue, so it supersedes
the first and closes that stream with a reason instead of leaving it to time out. If
the engine ever runs more than one generation at a time, this needs a real key -- and
the re-encode above is why that key cannot be the image's bytes.

Two guards replace it, for two different failures. Dimensions catch the server and the
browser disagreeing about one image's *oriented* size -- both sides apply EXIF
orientation (`vlm.py` imports engine.py's `_oriented` rather than copying it, since
mflux rotates before encoding and a box chosen in one frame applied in another is
transposed), so a phone photo lines up without either side converting. And
`baseImageEpoch` in harness.html catches the likelier case: the displayed image being
swapped while a detection is still open, which a replacement of identical size would
otherwise slip past.

**Putting an id inside the image -- EXIF or otherwise -- answers a question nobody is
asking, and does not survive to a reader.** The round trip is already identified: the
server issues a `job_id`, stashes the exact bytes, and hands the worker that same file,
so nothing between the two re-encodes anything. The only consumer past that point is a
vision model looking at pixels, which cannot read a tag. And the field it would want is
taken -- mflux embeds its generation metadata in EXIF `UserComment`, which is precisely
what harness.html parses to recover a prompt and seed, so writing there would collide
with the feature one section up. The identity that actually needs tracking never leaves
the page, which is why the counter lives there.

**`MFLUXIBLE_VLM_DIR` is one variable doing two jobs on purpose.** It enables the
feature and it is where images are stashed -- there is no useful "on but nowhere to put
anything" state. It is *also* the working directory the worker runs `claude` in, which
buys two things at once: a stashed image inside the cwd is readable without widening the
CLI's allowed roots, and a directory with no `CLAUDE.md` in it keeps a one-shot
detection from loading this (very long) file on every call.

**The default command uses opus, and the usual latency-for-quality trade does not apply.**
Measured on one 768x768 photograph with an identical prompt, opus was both better and
*faster*: 11.4s against sonnet's 66.6s. Sonnet put the apple at
`[0.28, 0.28, 0.68, 0.62]` -- roughly the right size in the wrong place, clipping the
fruit's bottom third and taking in a band of forearm, which is exactly the failure the
MCP mask-box section already records. Opus gave `[0.305, 0.344, 0.712, 0.736]` against
`[0.310, 0.344, 0.694, 0.729]` measured by hand, to three decimals rather than the round
two-decimal numbers that signal estimating on a grid. Sonnet had produced a good box on
an earlier run of the same image, so it is variance rather than a constant offset --
which is the worse failure, since nothing downstream can tell the two apart. Hence a
chip selection sets the mask and opens the editor rather than being treated as final.

**The reply carries a prompt as well as regions, which is why the names say `VLM`
rather than `REGIONS`.** A tool that can write a prompt able to regenerate the frame is
language-capable by definition, so the earlier name had stopped describing the feature.
`prompt` and `regions` are independently optional: a captioner that localizes nothing
still fills the prompt box, and a detector that only draws boxes still fills the chips.

Two things fall out of accepting two shapes. A **bare array** is still read as
regions-with-no-prompt, because that was this contract's earlier shape and a wrapper
written against it should keep working rather than start returning nothing. And
`_extract_json` picks its container by **which bracket opens first**, not by trying
`{...}` before `[...]`: an array of objects ends with a `}` inside the array, so the
naive order slices from the first element to that brace, parses cleanly, and silently
returns one region where the reply listed several. Caught by a test rather than in the
wild, and pinned by `test_an_array_of_objects_is_not_mistaken_for_one_object`.

**The harness fills its prompt box only when it is untouched, and "untouched" cannot
mean "empty".** The box ships with example text, so an empty-only rule would never fire
and the feature would look broken. `SHIPPED_PROMPT` is read out of the textarea once at
load -- before anything can edit it, and without a second copy of the literal in the
script -- and the box is filled when the field is blank *or* still holds that value.
Anything else is authored text and is left alone; "Use as prompt" is always there for
the case where it should be replaced anyway.

**The chips are independent toggles and the mask is their union, which makes the
selection the mask's definition rather than a one-shot action.** Each toggle repaints
from the whole selection instead of adding to the canvas: painting additively would make
deselect ambiguous, since erasing one region's rectangle also cuts into overlapping
neighbours and any brush work inside it. The cost is that a hand edit does not survive
the next toggle, and `dropChipSelection` is what stops that being a silent loss -- the
first `pointerdown` on the mask surface drops the selection, as do Clear and Invert, so a
pressed chip never describes something the canvas no longer shows. That one hook covers
brush, erase and rect, which all begin there.

**The harness draws the button off `/health`, and reads a missing block as off.** That
inverts the rule the `supports_*` keys follow, deliberately: those default to "unknown,
let the server decide" because offering a field costs nothing if the server accepts it,
while a Find objects button on a server without the mailbox can only ever 404.
