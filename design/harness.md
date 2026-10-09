# The browser harness

Layout notes for `clients/harness.html`. Read before changing its stage, compare handle or progress styling.

Moved verbatim from `CLAUDE.md`; the short form of each rule is still there.

## The harness's stage is a fallback chain, and the compare handle has its own memory

`.canvas` lays its children out in rows, so two visible panes push each other off screen
-- which is why `showMaskEditor` already hid the other two. `#baseWrap` is the fourth and
it is the *fallback* rather than somewhere you switch to: `syncBasePane()` shows it
whenever there is an input image and none of the other three is up, derived from the
`.hidden` classes they already carry rather than from a fifth piece of state. That
derivation is also what keeps `#emptyState` honest, since it is keyed on the same classes.
A run *in flight* is deliberately not one of the states it covers: the submit handler
opens the preview pane blank, which is more specific than the input image.

What it fixes is that dropping a new image left the previous run's result on the stage --
the page claiming a result for an image the form no longer held. `showInputImage()` is
that path, and **"Use as Base Image" deliberately does not go through it**: there the
result on screen *is* the image being adopted, so taking the pane down would cost its
Download link and its compare handle for nothing.

**The blank output pane is an `<img>` with a real intrinsic size, not a CSS box, and
that is the second attempt.** The first sized `#previewFrame` from custom properties
(`aspect-ratio` plus `width: min(100%, calc(var(--pane-max-h) * ...))`) and collapsed to
94px wide: `.pane` is shrink-to-fit inside `.canvas`, so that percentage resolves against
a containing block this very element is sizing, and what actually decided the width was
the tag row underneath. `blankOutput()` returns a transparent SVG data URL at the
output's dimensions instead, so the placeholder lays out through `.pane img` along the
same path a preview does -- measured identical at every transition, blank to preview to
result -- rather than through a second path that could drift from it. Its dimensions go
through `snapToStep` because the server floors both to a multiple of 16; that agreement
is currently free, since the fields carry `step="16"` and the browser's own validation
refuses to submit anything else, so it is what keeps the blank honest if that changes.

The live preview shares `.pane img`'s size cap rather than carrying its own. It used to
be capped at 340px, inherited from the single-column harness where previews were a 200px
thumbnail strip, and that cap threw away resolution the server had already paid to decode
and send -- `docs/api.md` states the contract outright: a preview is always the full
requested resolution and downscaling is a client's business. It also made the result jump
on the one transition where nothing about the framing has changed. The shimmer that
replaces it is keyed on `body:has(#submitBtn:disabled)` rather than on the preview pane
being up, which is the same derivation the progress sweep uses and load-bearing in the
other direction: the `error` event does not take the pane down, so a failed run leaves its
last preview on screen and a shimmer tied to the pane would run over it for ever.

`pendingCompareSrc` is captured in the submit handler rather than read off the form when
the `image` event lands, and the two genuinely diverge: a generation is long enough to
drop a different image or clear one meanwhile, and either would put an unrelated picture
behind the handle. The base is drawn `object-fit: fill` for the same reason it is drawn at
all -- mflux resizes the input to the request's width/height with a plain `resize()`
(`ImageUtil.scale_to_dimensions`), so the stretched copy is the "before" the model saw and
a letterboxed one would be a picture of nothing that happened. The handle's own line is
fixed white-on-dark rather than themed, for the reason `--mask-edge` is its own colour:
it sits over an arbitrary photograph, not over the page.
