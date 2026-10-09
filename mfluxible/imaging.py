"""The Pillow-only half of what engine.py does with images.

Split out of engine.py because none of this touches MLX -- decoding and vetting a client's
image and mask, applying EXIF orientation, encoding the final PNG -- and so none of it
should need MLX to import. That matters beyond tidiness: `vlm.py` and `vlm_local.py` both
need `oriented()` and used to import all of engine.py (and a GPU framework) to get it, and
`vlm_reply.py` exists as a separate module only to keep that import away from the worker.

Everything that *validates* returns a reason instead of raising one, written here for a
client to read, for the reason in MfluxEngine.request_problem's docstring: the real
exception goes to the log, and no path from an exception to a response body is the
invariant (CLAUDE.md).
"""

import base64
import io
import logging
import os
import tempfile

from PIL import Image, ImageFilter, ImageOps

log = logging.getLogger("mfluxible.imaging")


def oriented(raw: bytes) -> Image.Image:
    """An image decoded with its EXIF Orientation applied, the way mflux's own
    ImageUtil.load_image does it.

    Used for the mask as well as for measuring the input, and that is the point: mflux
    rotates the input image before encoding it, so a mask compared or composited
    against the unrotated bytes would be measured in one frame and applied in another.
    A mask straight out of a canvas carries no EXIF and this is a no-op for it.
    """
    with Image.open(io.BytesIO(raw)) as img:
        return ImageOps.exif_transpose(img)


def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def input_image_problem(b64_data: str) -> str | None:
    """Why a client-supplied base64 input image can't be used, or None if it can.

    Called from request_problem so a garbled `image` field gets the same clean 400 a bad
    guidance/negative_prompt request gets rather than surfacing as an opaque error
    mid-stream.

    Returns the reason rather than raising it, and returns a sentence written here
    rather than the underlying exception's own text -- both for the reasons in
    MfluxEngine.request_problem's docstring. The real exception goes to the log instead,
    where it is worth having and harmless: Pillow's carries a repr of the in-memory
    buffer (address included) and base64's describes this decode call, so neither told a
    client anything it could act on anyway.
    """
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        log.warning("rejecting input image: not valid base64", exc_info=True)
        return "image is not valid base64."
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        log.warning("rejecting input image: not a decodable image", exc_info=True)
        return "image could not be decoded as an image."
    return None


def decode_input_image(b64_data: str) -> bytes:
    """The actual bytes to write to disk (see mflux's own image_path parameter -- it
    wants a file path, not bytes or a PIL.Image, so there's no way to hand it a decoded
    image directly).

    No validation of its own: generate_stream, its only caller, runs check_request first
    -- so input_image_problem has already vetted this exact string, and a second round
    of Pillow work per generation would buy nothing.
    """
    return base64.b64decode(b64_data, validate=True)


def decode_mask(b64_data: str, feather: int = 0) -> Image.Image:
    """The client's mask as a single-channel image: white where the model may change
    the image, black where the input must survive.

    Read as luminance, so an RGB mask works and an alpha channel is ignored. That is
    worth stating because OpenAI's own edits endpoint uses the opposite convention --
    transparent means "edit here" -- so a mask drawn for that API would come out
    inverted here rather than failing. See docs/api.md.

    Feathering is applied at the mask's own resolution, before any resampling, which
    is why mask_feather is documented in input-image pixels: request_problem has
    already established that the mask and the input image are the same size.
    """
    mask = oriented(base64.b64decode(b64_data, validate=True)).convert("L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def input_mask_problem(mask_b64: str, image_b64: str) -> str | None:
    """Why a client-supplied mask can't be used, or None if it can. Same contract as
    input_image_problem: a reason written here, never an exception's own text.

    The size check is strict on purpose. mflux scales the *input image* to
    width/height with a plain resize and no aspect-ratio handling, and a mask that
    doesn't match would be stretched the same way -- silently selecting a different
    region than the one the user painted. Every other failure mode here is loud, so
    this one is too.
    """
    try:
        raw = base64.b64decode(mask_b64, validate=True)
    except Exception:
        log.warning("rejecting mask: not valid base64", exc_info=True)
        return "mask is not valid base64."
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except Exception:
        log.warning("rejecting mask: not a decodable image", exc_info=True)
        return "mask could not be decoded as an image."

    mask_img = oriented(raw)
    mask_size = mask_img.size
    # Before feathering, which can only spread white that is already there: an
    # all-black mask asks for nothing to change, which is a request nobody means to
    # make and which would otherwise spend a full generation reproducing the input.
    empty = mask_img.convert("L").getextrema()[1] == 0
    image_size = oriented(base64.b64decode(image_b64, validate=True)).size
    if mask_size != image_size:
        return (
            f"mask is {mask_size[0]}x{mask_size[1]} but image is {image_size[0]}x{image_size[1]}; "
            "they must be the same size, or the mask selects a different region than it looks like."
        )
    if empty:
        return "mask is entirely black, which selects no region to regenerate; omit it or paint some white."
    return None


def encode_final_png_with_metadata(image) -> bytes:
    # `image` (mflux's GeneratedImage, not the bare PIL image) already carries
    # everything -- prompt, seed, steps, model, quantize, LoRA config,
    # generation time -- that its own .save() embeds as EXIF UserComment, XMP,
    # and IPTC (see mflux.utils.image_util.ImageUtil.save_image and
    # mflux.utils.metadata_builder.MetadataBuilder). Reused as-is rather than
    # hand-rolling a second metadata format; .save() is file-path-based (it
    # re-opens and re-saves), so this round-trips through a temp file.
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(tmp_path, overwrite=True)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
