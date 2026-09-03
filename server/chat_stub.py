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

Deliberately narrow: this only recognizes a tool literally named "generate_image" if
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

import json
import time
import uuid
from typing import Any

from schemas import ChatCompletionRequest, ChatMessage

GENERATE_IMAGE_TOOL_NAME = "generate_image"

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


def _find_offered_tool_name(tools: list[dict[str, Any]] | None, wanted: str) -> str | None:
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if function.get("name") == wanted:
            return function["name"]
    return None


def build_response_message(req: ChatCompletionRequest) -> tuple[dict, str]:
    """Returns (message, finish_reason): a tool call inviting the caller to run
    generate_image with the last user message as its prompt, or a plain closing reply
    if that tool call already happened (the last message is a `tool` result) or was
    never on offer to begin with."""
    last: ChatMessage | None = req.messages[-1] if req.messages else None

    if last is not None and last.role == "tool":
        return {"role": "assistant", "content": _ACK_TEXT}, "stop"

    tool_name = _find_offered_tool_name(req.tools, GENERATE_IMAGE_TOOL_NAME)
    if tool_name is None:
        return {"role": "assistant", "content": _NO_TOOL_OFFERED_TEXT}, "stop"

    prompt = _extract_text(last.content) if last is not None else ""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps({"prompt": prompt})},
            }
        ],
    }
    return message, "tool_calls"


def _completion_id() -> str:
    return f"chatcmpl-mfluxible-{uuid.uuid4().hex[:24]}"


def non_streaming_response(req: ChatCompletionRequest) -> dict:
    message, finish_reason = build_response_message(req)
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def stream_response_lines(req: ChatCompletionRequest):
    """Yields complete `data: ...\\n\\n` SSE lines for a streaming chat completion. Not
    async -- there's no I/O here, the whole response is computed up front rather than
    produced token by token."""
    message, finish_reason = build_response_message(req)
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
