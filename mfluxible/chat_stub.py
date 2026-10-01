"""A fake "model" behind POST /v1/chat/completions that plays exactly one Native-mode
function-calling turn: given a plain user message, always call `generate_image` with
it as the prompt; given a tool result, always reply with a short acknowledgment.

This exists for an OpenAI-compatible chat frontend (Open WebUI is the motivating case)
whose Native tool-calling mode needs *some* chat model behind the connection before a
user can invoke image generation as a tool -- normally that's a real LLM (e.g. Ollama
running llama3.2), which costs real memory just to decide "yes, call the image tool"
on every message, for a use case where that's the only thing this chat model is ever
asked to do. There's no reasoning to replace here: the decision is hardcoded, not
inferred, so there's nothing an LLM gets right that a fixed rule doesn't get right for
free, at zero additional memory (this is pure Python control flow in the same process
as the diffusion model -- no weights, no inference).

Deliberately narrow: this only recognizes a tool named "generate_image" (or `<server>__generate_image`, as
MCP-aware clients namespace it) if
it's actually offered in the request's own `tools` list -- it never guesses at a tool
that wasn't offered this turn. That name isn't part of any OpenAI spec (tool names are
caller-defined) and isn't necessarily unique to Open WebUI either -- it's simply what
Open WebUI's own builtin tool happens to be called (confirmed by reading the source:
https://github.com/open-webui/open-webui/blob/main/backend/open_webui/tools/builtin.py,
`async def generate_image(prompt: str, ...)`), and it has
no fallback behaviour for anything a real chat model would do (general conversation,
other tools, multi-turn reasoning, deciding whether an image was even wanted). If
that's what's needed, point the chat connection at a real model instead -- this is
only for a connection whose sole purpose is triggering mfluxible's own image tool.
"""

import base64
import io
import json
import re
import time
import uuid
from typing import Any, Callable

from PIL import Image

from mfluxible.schemas import ChatCompletionRequest, ChatMessage

GENERATE_IMAGE_TOOL_NAME = "generate_image"
CHECK_IMAGE_TOOL_NAME = "check_image"

# The MCP tool's "still generating" text (clients/mfluxible_mcp/server.py, `_result`)
# is the only channel a handle travels on, so this is a coupling to that sentence's
# wording -- pinned by a test that builds the text from the real format.
_IMAGE_ID = re.compile(r"\bimage_id ([0-9a-f]{32})\b")
EMBED_MAX_SIDE = 768  # a chat transcript carries this every turn; the file on disk is full size


def _embeddable(png_b64: str) -> str:
    """A downscaled JPEG data URL of `png_b64`, small enough to live in a chat history."""
    with Image.open(io.BytesIO(base64.b64decode(png_b64))) as im:
        im = im.convert("RGB")
        im.thumbnail((EMBED_MAX_SIDE, EMBED_MAX_SIDE))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


_PENDING_HANDLE = re.compile(r'Still generating.*?check_image with handle "([^"]+)"', re.S)

_NO_TOOL_OFFERED_TEXT = (
    "mfluxible's chat endpoint only exists to trigger image generation as a tool call, and no "
    "generate_image tool was offered on this request -- in Open WebUI, check that both "
    "Capabilities and Builtin Tools -> Image Generation are enabled for this model."
)
_ACK_TEXT = "Image generated."


def _extract_text(content: str | list[Any] | None) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _extract_image_urls(content: str | list[Any] | None) -> list[str]:
    """`data:` URLs for any images in a tool result's content parts.

    Two shapes, because the result has been through whatever the caller does to an MCP
    `ImageContent`: OpenAI's `image_url` part (a url, or {"url": ...}), and MCP's own
    `image` part (base64 `data` plus `mimeType`/`mime_type`). Only `data:` URLs are
    passed through -- a remote URL in a tool result is not something to echo back."""
    urls: list[str] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if isinstance(url, str) and url.startswith("data:image/"):
                urls.append(url)
        elif part.get("type") == "image" and isinstance(part.get("data"), str):
            mime = part.get("mimeType") or part.get("mime_type") or "image/png"
            urls.append(f"data:{mime};base64,{part['data']}")
    return urls


def _find_offered_tool_name(tools: list[dict[str, Any]] | None, wanted: str) -> str | None:
    """The offered name for `wanted`: an exact match, else the first `<prefix>__<wanted>`.

    The suffix form is how MCP-aware chat clients namespace a server's tools
    (`mfluxible__generate_image`), and the *offered* name is what goes back in the tool
    call, since that is the only one the caller can route."""
    names = [
        tool["function"]["name"]
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    ]
    if wanted in names:
        return wanted
    return next((name for name in names if name.endswith(f"__{wanted}")), None)


def _tool_call_message(name: str, arguments: dict[str, Any]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def build_response_message(
    req: ChatCompletionRequest, image_lookup: Callable[[str], str | None] | None = None
) -> tuple[dict, str]:
    """Returns (message, finish_reason): a tool call inviting the caller to run
    generate_image with the last user message as its prompt, or a plain closing reply
    if that tool call already happened (the last message is a `tool` result) or was
    never on offer to begin with."""
    last: ChatMessage | None = req.messages[-1] if req.messages else None

    if last is not None and last.role == "tool":
        # A generate_image/check_image call that outlived the MCP tool's wait window
        # comes back as a handle, not an image; acknowledging that would claim an image
        # that isn't there yet. Ask for it again instead (check_image blocks, so this
        # is a wait rather than a busy loop).
        match = _PENDING_HANDLE.search(_extract_text(last.content))
        check_name = _find_offered_tool_name(req.tools, CHECK_IMAGE_TOOL_NAME) if match else None
        if match and check_name:
            return _tool_call_message(check_name, {"handle": match.group(1)}), "tool_calls"
        # Inline any image the result carried as markdown, then the MCP tool's caption
        # (where the file went). A client that doesn't render markdown images shows the
        # data URL as text, and one whose tool results are text-only never sends the
        # image at all, in which case the caption alone is what's left.
        result_text = _extract_text(last.content).strip()
        urls = _extract_image_urls(last.content)
        if not urls and image_lookup is not None:
            # skinnyai, for one, swaps the image for "[image content not shown]" before
            # the result reaches here. The caption still carries the `image_id` the
            # server issued, and the server still holds that image.
            found = _IMAGE_ID.search(result_text)
            png = image_lookup(found.group(1)) if found else None
            urls = [_embeddable(png)] if png else []
        images = [f"![Image]({url})" for url in urls]
        reply = "\n\n".join([*images, result_text] if result_text else images)
        return {"role": "assistant", "content": reply or _ACK_TEXT}, "stop"

    tool_name = _find_offered_tool_name(req.tools, GENERATE_IMAGE_TOOL_NAME)
    if tool_name is None:
        return {"role": "assistant", "content": _NO_TOOL_OFFERED_TEXT}, "stop"

    prompt = _extract_text(last.content) if last is not None else ""
    return _tool_call_message(tool_name, {"prompt": prompt}), "tool_calls"


def _completion_id() -> str:
    return f"chatcmpl-mfluxible-{uuid.uuid4().hex[:24]}"


def non_streaming_response(req: ChatCompletionRequest, image_lookup=None) -> dict:
    message, finish_reason = build_response_message(req, image_lookup)
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def stream_response_lines(req: ChatCompletionRequest, image_lookup=None):
    """Yields complete `data: ...\\n\\n` SSE lines for a streaming chat completion. Not
    async -- there's no I/O here, the whole response is computed up front rather than
    produced token by token."""
    message, finish_reason = build_response_message(req, image_lookup)
    cid = _completion_id()
    created = int(time.time())

    def chunk(delta: dict, fr: str | None = None) -> str:
        body = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": fr}],
        }
        return f"data: {json.dumps(body)}\n\n"

    yield chunk({"role": "assistant"})
    if message.get("tool_calls"):
        # One complete tool call in a single delta rather than split across incremental
        # chunks -- valid per OpenAI's own streaming shape (chunks are additive by
        # `index`; nothing requires more than one), and simpler here since the whole
        # call is already known up front rather than generated token by token.
        tool_call = dict(message["tool_calls"][0])
        tool_call["index"] = 0
        yield chunk({"tool_calls": [tool_call]})
    elif message.get("content"):
        yield chunk({"content": message["content"]})
    yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"
