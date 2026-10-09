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


def test_a_namespaced_tool_name_is_matched_and_echoed_back(client):
    tool = {**GENERATE_IMAGE_TOOL, "function": {**GENERATE_IMAGE_TOOL["function"], "name": "mfluxible__generate_image"}}
    other = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "a puffin"}], "tools": [other, tool]},
    )
    message = resp.json()["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "mfluxible__generate_image"


def test_a_tool_merely_containing_the_name_is_not_matched(client):
    tool = {**GENERATE_IMAGE_TOOL, "function": {**GENERATE_IMAGE_TOOL["function"], "name": "my_generate_image"}}
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [tool]},
    )
    assert "tool_calls" not in resp.json()["choices"][0]["message"]


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_the_handle_pattern_matches_the_mcp_tools_real_wording():
    # Built by the real MCP tool so a reword there fails here instead of silently
    # going back to acknowledging an image that isn't ready.
    import time

    from mfluxible.chat_stub import _PENDING_HANDLE
    from mfluxible_mcp import server as mcp_server

    job = mcp_server._Job(handle="abc123", width=64, height=64, steps=None, model="", started=time.monotonic())
    text = mcp_server._result(job)[0].text
    assert _PENDING_HANDLE.search(text).group(1) == "abc123"


def test_a_pending_handle_triggers_a_check_image_call(client):
    text = 'Still generating (step 3/9). Call check_image with handle "abc123" to collect it -- blah.'
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "c", "content": text}],
            "tools": [_tool("mfluxible__generate_image"), _tool("mfluxible__check_image")],
        },
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    fn = choice["message"]["tool_calls"][0]["function"]
    assert fn["name"] == "mfluxible__check_image"
    assert json.loads(fn["arguments"]) == {"handle": "abc123"}


def test_a_pending_handle_without_check_image_offered_still_acknowledges(client):
    text = 'Still generating (step 3/9). Call check_image with handle "abc123" to collect it.'
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "c", "content": text}],
            "tools": [GENERATE_IMAGE_TOOL],
        },
    )
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


def test_a_finished_image_result_still_acknowledges(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "c", "content": "Generated 768x768 in 40s"}],
            "tools": [_tool("mfluxible__check_image")],
        },
    )
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


def test_the_closing_reply_passes_along_the_tool_results_caption(client):
    caption = "Z-Image-Turbo, 768x768, 9 steps, seed 1, 40.0s. Full-resolution PNG saved to /x/y.png"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": caption}]}],
            "tools": [_tool("mfluxible__check_image")],
        },
    )
    assert resp.json()["choices"][0]["message"]["content"] == caption


def test_the_closing_reply_links_an_image_the_server_serves(client, monkeypatch, tmp_path):
    import mfluxible.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "OUTPUT_DIR", tmp_path)
    (tmp_path / "20261001-1.png").write_bytes(b"png")
    caption = f"seed 1, 40.0s. Full-resolution PNG saved to {tmp_path}/20261001-1.png (shown as JPEG)"
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "tool", "tool_call_id": "c", "content": caption}], "tools": []},
    )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == f"![Image](http://testserver/mfluxible/v1/output/20261001-1.png)\n\n{caption}"
    assert client.get("/mfluxible/v1/output/20261001-1.png").content == b"png"


def test_a_caption_naming_a_file_the_server_does_not_serve_gets_no_link(client, monkeypatch, tmp_path):
    import mfluxible.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "OUTPUT_DIR", tmp_path)
    caption = "seed 1. Full-resolution PNG saved to /Users/me/elsewhere/a.png (x)"
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "tool", "tool_call_id": "c", "content": caption}], "tools": []},
    )
    assert resp.json()["choices"][0]["message"]["content"] == caption


def test_the_link_names_only_the_files_basename():
    from mfluxible.chat_stub import non_streaming_response
    from mfluxible.schemas import ChatCompletionRequest

    seen = []

    def image_url(name):
        seen.append(name)
        return f"http://h/mfluxible/v1/output/{name}"

    caption = "seed 1. Full-resolution PNG saved to /etc/../Users/me/Pictures/mfluxible/20261001-1.png (x)"
    req = ChatCompletionRequest(model="m", messages=[{"role": "tool", "tool_call_id": "c", "content": caption}])
    content = non_streaming_response(req, image_url)["choices"][0]["message"]["content"]
    assert seen == ["20261001-1.png"]
    assert content == f"![Image](http://h/mfluxible/v1/output/20261001-1.png)\n\n{caption}"


def test_the_failure_caption_is_not_mistaken_for_a_saved_file():
    from mfluxible.chat_stub import non_streaming_response
    from mfluxible.schemas import ChatCompletionRequest

    seen = []
    caption = (
        "seed 1. Full-resolution PNG could not be saved to /out "
        "([Errno 13] Permission denied: '/out/20261001-1.png')"
    )
    req = ChatCompletionRequest(model="m", messages=[{"role": "tool", "tool_call_id": "c", "content": caption}])
    content = non_streaming_response(req, lambda name: seen.append(name))["choices"][0]["message"]["content"]
    assert seen == [] and content == caption


def test_the_server_and_the_mcp_tool_default_to_the_same_directory():
    # Two distributions, two settings, one directory: the chat stub links a file the MCP
    # tool saved, so the defaults drifting apart breaks the link with no error anywhere.
    import mfluxible.runtime as runtime_module
    from mfluxible_mcp import server as mcp_server

    assert runtime_module.OUTPUT_DIR == mcp_server.SAVE_DIR
