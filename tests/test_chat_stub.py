"""HTTP-level coverage of chat_stub.py's Native-mode function-calling state machine,
run against the real FastAPI app (see tests/conftest.py::client)."""

import json

GENERATE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image based on a text prompt.",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
    },
}


def test_first_turn_calls_generate_image_with_the_user_message_as_prompt(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-solid-color",
            "messages": [{"role": "user", "content": "a puffin on a cliff"}],
            "tools": [GENERATE_IMAGE_TOOL],
        },
    )
    assert resp.status_code == 200
    message = resp.json()["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "generate_image"
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args == {"prompt": "a puffin on a cliff"}
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_first_turn_extracts_text_from_multimodal_content_parts(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-solid-color",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "a puffin"}, {"type": "text", "text": " on a cliff"}]}
            ],
            "tools": [GENERATE_IMAGE_TOOL],
        },
    )
    args = json.loads(resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"prompt": "a puffin on a cliff"}


def test_follow_up_turn_after_tool_result_replies_with_plain_text(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-solid-color",
            "messages": [
                {"role": "user", "content": "a puffin on a cliff"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "generate_image", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "generate_image", "content": "ok"},
            ],
            "tools": [GENERATE_IMAGE_TOOL],
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None
    assert choice["message"]["content"]


def test_no_generate_image_tool_offered_falls_back_to_explanatory_text(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "toy-solid-color", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "generate_image" in choice["message"]["content"]


def test_streaming_first_turn_ends_with_tool_calls_and_done(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "toy-solid-color",
            "messages": [{"role": "user", "content": "a puffin"}],
            "tools": [GENERATE_IMAGE_TOOL],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert finish_reasons[-1] == "tool_calls"
    tool_call_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_call_chunks) == 1
    fn = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert fn["name"] == "generate_image"
    assert json.loads(fn["arguments"]) == {"prompt": "a puffin"}
