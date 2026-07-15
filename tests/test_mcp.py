"""MCP drop-in: the harness as an MCP client, and safe-by-default made explicit (Group 5).

Everything here is offline. The stdio end-to-end test spawns a tiny **fake MCP server**
(a Python script speaking newline-delimited JSON-RPC over stdin/stdout) — a real
subprocess, no network — to prove a dropped-in server's tools activate and a tool call
round-trips. The HTTP path is driven through respx at the transport level. The rest is
pure: config parsing, the SSE/result helpers, name sanitization, resolution merging, the
policy filter, and the brief's safety section.
"""

import json
import sys

import httpx
import pytest
import respx

from basecradle_harness import (
    Engine,
    McpServerConfig,
    McpTool,
    Message,
    Policy,
    ResolvedTools,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    compose_brief,
    install,
    load_mcp_configs,
    load_mcp_tools,
    render_safety,
)
from basecradle_harness._basecradle import _apply_safe_policy, _merge_mcp_tools
from basecradle_harness._mcp import (
    HttpMcpClient,
    McpError,
    McpImageStore,
    _render_tool_result,
    _sse_response,
    mcp_tool_name,
)
from basecradle_harness._policy import SHELL

# A 1×1 PNG, base64 — a real, decodable image for the image-block tests.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4"
    "2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# A minimal MCP server over stdio: initialize → tools/list (one `echo` tool) → tools/call.
# Written to a temp file and launched with the test interpreter, so the stdio transport is
# exercised against a real subprocess with no network.
_FAKE_MCP_SERVER = r"""
import json, sys

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid = req.get("id")
    method = req.get("method")
    if rid is None:
        continue  # a notification (e.g. notifications/initialized): nothing to answer
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "fake", "version": "0"},
            "capabilities": {}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": "echo",
            "description": "Echo text back.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                            "required": ["text"]}}]}})
    elif method == "tools/call":
        args = req.get("params", {}).get("arguments", {})
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": "echo: " + str(args.get("text", ""))}],
            "isError": False}})
    else:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no such method"}})
"""


def _write_server_script(tmp_path):
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(_FAKE_MCP_SERVER, encoding="utf-8")
    return script


def _drop_config(home, name, body):
    """Write one ``mcp/<name>.json`` under a config home, scaffolding the dir."""
    mcp_dir = home / "mcp"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    (mcp_dir / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")


# --- config parsing -----------------------------------------------------------


def test_parse_stdio_config(tmp_path):
    _drop_config(tmp_path, "srv", {"command": "uvx", "args": ["some-mcp"], "env": {"K": "v"}})
    configs = load_mcp_configs(tmp_path)
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "srv"
    assert cfg.transport == "stdio"
    assert cfg.command == "uvx"
    assert cfg.args == ("some-mcp",)
    assert cfg.env == {"K": "v"}


def test_parse_http_config(tmp_path):
    _drop_config(tmp_path, "remote", {"url": "https://h/mcp", "headers": {"Authorization": "t"}})
    (cfg,) = load_mcp_configs(tmp_path)
    assert cfg.transport == "http"
    assert cfg.url == "https://h/mcp"
    assert cfg.headers == {"Authorization": "t"}


def test_parse_mcpservers_wrapper_unwraps_single_entry(tmp_path):
    # A Claude-Desktop-style {"mcpServers": {...}} snippet with one entry drops in; the inner
    # key names the server (overriding the filename).
    _drop_config(tmp_path, "file", {"mcpServers": {"inner": {"command": "x"}}})
    (cfg,) = load_mcp_configs(tmp_path)
    assert cfg.name == "inner"
    assert cfg.command == "x"


def test_bad_config_files_are_skipped_not_fatal(tmp_path):
    _drop_config(tmp_path, "ok", {"command": "x"})
    (tmp_path / "mcp" / "broken.json").write_text("{not json", encoding="utf-8")
    _drop_config(tmp_path, "both", {"command": "x", "url": "https://h"})  # exactly-one violated
    _drop_config(tmp_path, "neither", {"args": []})  # no transport
    configs = load_mcp_configs(tmp_path)
    assert [c.name for c in configs] == ["ok"]  # only the valid one survives


def test_no_mcp_dir_yields_no_configs(tmp_path):
    assert load_mcp_configs(tmp_path) == []


# --- stdio end-to-end (the proof case) ----------------------------------------


def test_stdio_server_tools_activate_and_call_round_trips(tmp_path):
    script = _write_server_script(tmp_path)
    _drop_config(tmp_path, "fake", {"command": sys.executable, "args": [str(script)]})

    resolution = load_mcp_tools(tmp_path, timeout=10)
    try:
        # The server's one tool activated, namespaced under the server name.
        assert [t.name for t in resolution.tools] == ["fake__echo"]
        assert not resolution.skipped
        # Opt-out is surfaced: a notice for the active server, and a per-tool manifest note.
        assert len(resolution.notices) == 1
        assert "fake" in resolution.notices[0]
        assert "safe-by-default" in resolution.notices[0]
        assert resolution.manifest[0][0] == "fake__echo"
        assert "MCP server" in resolution.manifest[0][1]
        # A tool call proxies to the server and back.
        tool = resolution.tools[0]
        assert tool.run(text="hi") == "echo: hi"
        # The tool carries the server's declared schema.
        assert tool.parameters["properties"]["text"]["type"] == "string"
    finally:
        for client in resolution.clients:
            client.close()


def test_stdio_image_result_becomes_a_toolresult_and_is_stashed_for_posting(tmp_path):
    # A server whose one tool returns an image block: load_mcp_tools must wire a shared image
    # store onto the resolution, and the tool's run() must return a ToolResult (vision) *and*
    # stash the bytes there for the assets post_image path (issue #318).
    script = tmp_path / "shot.py"
    script.write_text(
        "import json, sys\n"
        "def send(m):\n"
        '    sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()\n'
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line: continue\n"
        "    req = json.loads(line); rid = req.get('id')\n"
        "    if rid is None: continue\n"
        "    if req['method'] == 'initialize':\n"
        "        send({'jsonrpc':'2.0','id':rid,'result':{'protocolVersion':'2025-06-18'}})\n"
        "    elif req['method'] == 'tools/list':\n"
        "        send({'jsonrpc':'2.0','id':rid,'result':{'tools':["
        "{'name':'screenshot','description':'shot','inputSchema':{}}]}})\n"
        "    elif req['method'] == 'tools/call':\n"
        "        send({'jsonrpc':'2.0','id':rid,'result':{'content':["
        f"{{'type':'image','data':'{_PNG_B64}','mimeType':'image/png'}}]}}}})\n",
        encoding="utf-8",
    )
    _drop_config(tmp_path, "browser", {"command": sys.executable, "args": [str(script)]})
    resolution = load_mcp_tools(tmp_path, timeout=10)
    try:
        assert resolution.images is not None  # the store was wired onto the resolution
        (tool,) = resolution.tools
        result = tool.run()
        assert isinstance(result, ToolResult)
        assert len(result.images) == 1
        # The same store the tool stashed into is the one carried on the resolution.
        assert len(resolution.images) == 1
        assert resolution.images.get("latest").mimetype == "image/png"
    finally:
        for client in resolution.clients:
            client.close()


def test_colliding_sanitized_tool_names_dedup_not_crash(tmp_path):
    # Two distinct remote names that sanitize to the same final name ("a.b" and "a b" both
    # → "dup__a_b"). The second must self-exclude with a reason, never produce two tools of
    # the same name (which would crash ToolRegistry.register and take the wake down).
    script = tmp_path / "dup.py"
    script.write_text(
        "import json, sys\n"
        "def send(m):\n"
        '    sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()\n'
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line: continue\n"
        "    req = json.loads(line); rid = req.get('id')\n"
        "    if rid is None: continue\n"
        "    if req['method'] == 'initialize':\n"
        "        send({'jsonrpc':'2.0','id':rid,'result':{'protocolVersion':'2025-06-18'}})\n"
        "    elif req['method'] == 'tools/list':\n"
        "        send({'jsonrpc':'2.0','id':rid,'result':{'tools':["
        "{'name':'a.b','description':'one','inputSchema':{}},"
        "{'name':'a b','description':'two','inputSchema':{}}]}})\n",
        encoding="utf-8",
    )
    _drop_config(tmp_path, "dup", {"command": sys.executable, "args": [str(script)]})
    resolution = load_mcp_tools(tmp_path, timeout=10)
    try:
        assert [t.name for t in resolution.tools] == ["dup__a_b"]  # only the first survives
        names = [t.name for t in resolution.tools]
        assert len(names) == len(set(names))  # no duplicate tool names reach the registry
        assert any("duplicate" in reason for _, reason in resolution.skipped)
        assert resolution.notices[0].startswith(
            "MCP server 'dup' active (1 tool"
        )  # count is loaded
    finally:
        for client in resolution.clients:
            client.close()


def test_failed_server_self_excludes_with_reason(tmp_path):
    # A command that does not exist: the server must self-exclude, not crash the load.
    _drop_config(tmp_path, "missing", {"command": "this-command-does-not-exist-xyz"})
    resolution = load_mcp_tools(tmp_path, timeout=5)
    assert resolution.tools == []
    assert resolution.notices == []
    assert len(resolution.skipped) == 1
    name, reason = resolution.skipped[0]
    assert name == "missing"
    assert "did not load" in reason


def test_handshake_timeout_self_excludes_and_reaps(tmp_path):
    # A server that spawns but never answers initialize: start() must time out, the
    # subprocess must be reaped by _connect's failure teardown, and the server self-excludes
    # — without hanging the load. A short timeout keeps the test fast.
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    _drop_config(tmp_path, "hang", {"command": sys.executable, "args": [str(script)]})
    resolution = load_mcp_tools(tmp_path, timeout=1)
    assert resolution.tools == []
    assert resolution.clients == []  # nothing left live to close
    assert len(resolution.skipped) == 1
    assert resolution.skipped[0][0] == "hang"


def test_server_that_dies_after_start_fails_fast_not_after_timeout(tmp_path):
    # A server that completes the handshake, then exits before answering tools/list. The
    # reader thread sees EOF and queues the sentinel, so tools/list fails *immediately*
    # ("server closed") rather than waiting out the (here, long) timeout. The test would
    # take ~30s if fast-fail regressed; it should finish in well under a second.
    script = tmp_path / "die.py"
    script.write_text(
        "import json, sys\n"
        "req = json.loads(sys.stdin.readline())\n"
        'sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{}}) + "\\n")\n'
        "sys.stdout.flush()\n"
        "sys.exit(0)\n",  # exit right after the initialize response
        encoding="utf-8",
    )
    _drop_config(tmp_path, "dier", {"command": sys.executable, "args": [str(script)]})
    resolution = load_mcp_tools(tmp_path, timeout=30)  # large timeout: fast-fail must beat it
    assert resolution.tools == []
    assert len(resolution.skipped) == 1
    assert "closed" in resolution.skipped[0][1] or "did not load" in resolution.skipped[0][1]


def test_safe_by_default_empty_mcp_dir_is_no_op(tmp_path):
    # A scaffolded-but-empty config home (the shipped safe state): no MCP tools, no notices.
    install(tmp_path)
    resolution = load_mcp_tools(tmp_path)
    assert resolution.tools == []
    assert resolution.notices == []
    assert resolution.skipped == []


# --- HTTP transport (Streamable HTTP, via respx) ------------------------------


@respx.mock
def test_http_transport_lists_and_calls(tmp_path):
    url = "https://mcp.example.com/mcp"

    def handler(request):
        msg = json.loads(request.content)
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:  # notifications/initialized
            return httpx.Response(202)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "sess-1"},
                json={"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18"}},
            )
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "tools": [{"name": "ping", "description": "Ping.", "inputSchema": {}}]
                    },
                },
            )
        if method == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"content": [{"type": "text", "text": "pong"}]},
                },
            )
        return httpx.Response(400)

    route = respx.post(url).mock(side_effect=handler)

    client = HttpMcpClient(McpServerConfig(name="r", url=url), timeout=5)
    client.start()
    try:
        tools = client.list_tools()
        assert [t["name"] for t in tools] == ["ping"]
        # call_tool returns the raw JSON-RPC result dict now; rendering is McpTool.run's job.
        assert _render_tool_result(client.call_tool("ping", {})) == "pong"
        # The session id from initialize is echoed on later requests.
        later = route.calls[-1].request
        assert later.headers.get("mcp-session-id") == "sess-1"
    finally:
        client.close()


@respx.mock
def test_http_transport_parses_sse_response(tmp_path):
    url = "https://mcp.example.com/mcp"

    def handler(request):
        msg = json.loads(request.content)
        rid = msg.get("id")
        if rid is None:
            return httpx.Response(202)
        if msg.get("method") == "initialize":
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {}})}\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        body = (
            "data: " + json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"tools": []}}) + "\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    respx.post(url).mock(side_effect=handler)
    client = HttpMcpClient(McpServerConfig(name="r", url=url), timeout=5)
    client.start()
    try:
        assert client.list_tools() == []
    finally:
        client.close()


# --- the JSON-RPC / rendering helpers -----------------------------------------


def test_render_tool_result_joins_text_blocks():
    result = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _render_tool_result(result) == "a\nb"


def test_render_tool_result_marks_errors_and_non_text():
    assert _render_tool_result(
        {"content": [{"type": "text", "text": "nope"}], "isError": True}
    ) == ("Error: nope")
    # A non-image, non-text block (an embedded resource) keeps the by-type placeholder.
    assert "[resource content]" in _render_tool_result(
        {"content": [{"type": "resource", "resource": {}}]}
    )
    # An image block with undecodable data degrades to a describing placeholder, not a crash.
    assert "could not decode" in _render_tool_result({"content": [{"type": "image", "data": "!!"}]})
    assert _render_tool_result({}) == "(the tool returned no content)"


# --- image content: vision inlining (Req A) + stashing for post_image (Req B) --


def test_render_image_block_inlines_as_vision_and_stashes_with_a_handle():
    store = McpImageStore()
    result = _render_tool_result(
        {
            "content": [
                {"type": "text", "text": "shot"},
                {"type": "image", "data": _PNG_B64, "mimeType": "image/png"},
            ]
        },
        store,
    )
    # It becomes a ToolResult so the engine can route the pixels into vision input.
    assert isinstance(result, ToolResult)
    assert len(result.images) == 1
    assert result.images[0].url.startswith("data:image/png;base64,")
    assert result.images[0].alt == "mcp-image-1"
    # The text carries both the tool's own text and an honest placeholder naming the handle
    # and how to post it — what a *non-vision* model reads in place of the picture.
    assert "shot" in result.text
    assert "mcp-image-1" in result.text
    assert "image/png" in result.text
    assert "post_image" in result.text
    # And the bytes are stashed for the post_image path, keyed by that handle.
    assert len(store) == 1
    assert store.get("mcp-image-1").mimetype == "image/png"


def test_render_image_block_without_a_store_still_inlines_but_offers_no_post_handle():
    # The library/test path (no per-wake store): vision still works; there is just nothing to
    # stash, so the placeholder names no reference and does not advertise post_image.
    result = _render_tool_result(
        {"content": [{"type": "image", "data": _PNG_B64, "mimeType": "image/png"}]}
    )
    assert isinstance(result, ToolResult)
    assert len(result.images) == 1
    assert "post_image" not in result.text
    assert "mcp-image" not in result.text


def test_render_image_block_unsupported_type_is_stashed_but_not_inlined():
    # A non-viewable image type (a model can't take it as input) is described and stashed for the
    # post path, but never inlined as vision — so the result is a plain str with no images.
    store = McpImageStore()
    result = _render_tool_result(
        {"content": [{"type": "image", "data": _PNG_B64, "mimeType": "image/bmp"}]}, store
    )
    assert isinstance(result, str)  # no vision image → no ToolResult
    assert "not viewable" in result
    assert "post_image" in result  # still postable
    assert len(store) == 1


def test_render_image_block_over_the_size_ceiling_is_described_not_stored(monkeypatch):
    # An image over MAX_IMAGE_BYTES is neither inlined nor stashed — just described — so the store
    # (and the transcript) stay bounded.
    import basecradle_harness._mcp as mcp

    monkeypatch.setattr(mcp, "MAX_IMAGE_BYTES", 4)
    store = McpImageStore()
    result = _render_tool_result(
        {"content": [{"type": "image", "data": _PNG_B64, "mimeType": "image/png"}]}, store
    )
    assert isinstance(result, str)
    assert "too large" in result
    assert len(store) == 0


def test_image_store_is_a_bounded_ring_with_latest_and_by_handle_lookup():
    store = McpImageStore(cap=2)
    h1 = store.stash("image/png", b"one")
    h2 = store.stash("image/png", b"two")
    assert (h1, h2) == ("mcp-image-1", "mcp-image-2")
    assert store.get("latest").data == b"two"
    assert store.get(None).data == b"two"  # omitted ref → latest
    assert store.get("mcp-image-1").data == b"one"
    # A third stash evicts the oldest; its handle is never reused.
    h3 = store.stash("image/png", b"three")
    assert h3 == "mcp-image-3"
    assert len(store) == 2
    assert store.get("mcp-image-1") is None  # evicted
    assert store.get("latest").data == b"three"


def test_render_image_result_feeds_the_engines_vision_fork_both_ways():
    """The MCP-rendered `ToolResult` drives the same vision gate `view` does (issues #316/#318).

    Proven end-to-end: a tool returning exactly what `_render_tool_result` produces is shown to a
    vision model and withheld from a text-only one — the fork the harness relies on to keep a
    non-vision agent from ever receiving an image part.
    """
    store = McpImageStore()
    rendered = _render_tool_result(
        {"content": [{"type": "image", "data": _PNG_B64, "mimeType": "image/png"}]}, store
    )

    class _Shot(Tool):
        name = "shot"
        description = "Return a screenshot."

        def run(self, **kwargs):
            return rendered

    class _Scripted:
        def __init__(self, *replies, vision):
            self._replies = list(replies)
            self._vision = vision
            self.seen = []

        def chat(self, messages, tools=None):
            self.seen.append(
                [Message(role=m.role, content=m.content, images=list(m.images)) for m in messages]
            )
            return self._replies.pop(0)

        def supports_vision(self):
            return self._vision

    def _run(vision):
        provider = _Scripted(
            Message.assistant(tool_calls=[ToolCall(id="c1", name="shot", arguments={})]),
            Message.assistant(content="done"),
            vision=vision,
        )
        registry = ToolRegistry()
        registry.register(_Shot())
        Engine(provider, registry).run([Message.user("take a screenshot")])
        return provider

    # Vision model: the pixels reach the model on the second call.
    seen_vision = _run(True).seen[1]
    assert any(m.images for m in seen_vision)
    # Text-only model: no pixels anywhere — an honest note stands in instead.
    seen_blind = _run(False).seen[1]
    assert not any(m.images for m in seen_blind)


def test_sse_response_finds_matching_id_ignoring_notifications():
    body = (
        "data: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/log"}) + "\n\n"
        "data: " + json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}) + "\n\n"
    )
    assert _sse_response(body, 7) == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}


def test_sse_response_raises_when_no_match():
    body = "data: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n\n"
    with pytest.raises(McpError):
        _sse_response(body, 999)


def test_tool_name_namespaced_sanitized_and_bounded():
    assert mcp_tool_name("srv", "echo") == "srv__echo"
    # Illegal characters collapse to underscore; the whole thing is capped at 64.
    assert mcp_tool_name("a b.c", "x/y") == "a_b_c__x_y"
    assert len(mcp_tool_name("s" * 50, "t" * 50)) == 64


# --- resolution merging + the policy filter -----------------------------------


class _ShellTool(Tool):
    name = "danger"
    description = "Needs a shell."
    requires = frozenset({SHELL})

    def run(self, **kwargs):
        return "ran"


class _PlainTool(Tool):
    name = "plain"
    description = "Harmless."

    def run(self, **kwargs):
        return "ok"


def test_apply_safe_policy_drops_and_surfaces_forbidden_tool():
    resolved = ResolvedTools(
        tools=[_PlainTool(), _ShellTool()],
        manifest=[("plain", None), ("danger", None)],
    )
    out = _apply_safe_policy(resolved, Policy.locked())
    assert [t.name for t in out.tools] == ["plain"]  # the shell tool is filtered out
    assert ("danger", "danger") not in [(n, n) for n, _ in out.manifest]
    assert [n for n, _ in out.manifest] == ["plain"]
    assert any(name == "danger" for name, _ in out.skipped)
    assert any("danger" in notice and "safe-by-default" in notice for notice in out.notices)


def test_apply_safe_policy_is_noop_when_all_permitted():
    resolved = ResolvedTools(tools=[_PlainTool()], manifest=[("plain", None)])
    out = _apply_safe_policy(resolved, Policy.locked())
    assert out is resolved  # unchanged identity: nothing refused, nothing to rebuild


def test_merge_mcp_tools_extends_set_and_carries_notices():
    base = ResolvedTools(tools=[_PlainTool()], manifest=[("plain", None)])

    class _FakeMcpResolution:
        tools = [
            McpTool(
                server="s",
                remote_name="echo",
                description="d",
                parameters={},
                client=None,  # not called in this merge-only test
            )
        ]
        manifest = [("s__echo", "via MCP server 's'")]
        skipped = [("dead", "did not load")]
        notices = ["MCP server 's' active"]
        images = McpImageStore()

    fake = _FakeMcpResolution()
    out = _merge_mcp_tools(base, fake)
    assert [t.name for t in out.tools] == ["plain", "s__echo"]
    assert out.skipped == [("dead", "did not load")]
    assert out.notices == ["MCP server 's' active"]
    # The per-wake image store is carried onto the resolved set, so the assets tool can reach it.
    assert out.mcp_images is fake.images


def test_merge_mcp_tools_empty_is_noop():
    base = ResolvedTools(tools=[_PlainTool()], manifest=[("plain", None)])

    class _Empty:
        tools = []
        manifest = []
        skipped = []
        notices = []

    assert _merge_mcp_tools(base, _Empty()) is base


# --- the brief's safety section -----------------------------------------------


def test_render_safety_blocks_for_notices_else_none():
    assert render_safety([]) is None
    assert render_safety(None) is None
    block = render_safety(["MCP server 'x' active", "Tool 'y' refused"])
    assert block is not None
    assert "opt-out" in block
    assert "- MCP server 'x' active" in block
    assert "- Tool 'y' refused" in block


def test_compose_brief_places_safety_after_manifest():
    brief = compose_brief(
        initialize="INIT",
        manifest="TOOLS",
        safety="SAFETY",
        dashboard="DASH",
        system_prompt="CHARTER",
    )
    # Order is load-bearing: guidance → tools → safety opt-out → dashboard → charter.
    assert brief.index("INIT") < brief.index("TOOLS") < brief.index("SAFETY") < brief.index("DASH")
    assert brief.index("SAFETY") < brief.index("CHARTER")


def test_compose_brief_omits_safety_when_absent():
    brief = compose_brief(
        initialize="INIT", manifest="TOOLS", dashboard=None, system_prompt="CHARTER"
    )
    assert "INIT" in brief and "TOOLS" in brief
