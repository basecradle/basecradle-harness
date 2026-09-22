"""MCP drop-in: the harness as an MCP client, and safe-by-default made explicit (Group 5).

Everything here is offline. The stdio end-to-end test spawns a tiny **fake MCP server**
(a Python script speaking newline-delimited JSON-RPC over stdin/stdout) — a real
subprocess, no network — to prove a dropped-in server's tools activate and a tool call
round-trips. The HTTP path is driven through respx at the transport level. The rest is
pure: config parsing, the SSE/result helpers, name sanitization, resolution merging, the
policy filter, and the brief's safety section.
"""

import base64
import json
import os
import shlex
import stat
import sys
import time

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
    _named_files,
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
        notice = resolution.notices[0]
        assert "fake" in notice
        # The audit tail stays loud (issue #322)...
        assert "safe-by-default" in notice
        assert "recorded for audit" in notice
        # ...while the model-facing body sanctions the tools rather than warning against them:
        # no legacy "warning label" wording, and an explicit approval + anti-fabrication line.
        assert "all bets off" not in notice
        assert "external code you opted into" not in notice
        assert "installed and approved for your use" in notice
        assert "never report a tool result you did not get back" in notice
        assert "fake__" in notice  # names the namespace the model calls the tools by
        assert resolution.manifest[0][0] == "fake__echo"
        note = resolution.manifest[0][1]
        assert "MCP server" in note
        assert "approved for your use" in note
        assert "beyond the safe-by-default tool set" not in note  # the 24×-repeated warning is gone
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


# playwright-mcp@0.0.80's screenshot tool, reduced to what it does with a file (issue #552): it
# always saves the capture and prints a link to it, relative to its own working directory, and it
# sends the picture as an image block **only when the model did not name the file** —
# `if (!params.filename) await response.registerImageResult(data, type)`.
_PLAYWRIGHT_LIKE_SERVER = r"""
import base64, json, os, sys

PNG = base64.b64decode(sys.argv[1])

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid = req.get("id")
    if rid is None:
        continue
    if req["method"] == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18"}})
    elif req["method"] == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": "browser_take_screenshot", "description": "shot", "inputSchema": {}}]}})
    elif req["method"] == "tools/call":
        filename = req["params"]["arguments"].get("filename")
        name = filename or os.path.join(".playwright-mcp", "page-1.png")
        os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
        with open(name, "wb") as out:
            out.write(PNG)
        link = name if os.path.dirname(name) else "./" + name
        content = [{"type": "text", "text": (
            "### Result\n- [Screenshot of viewport](" + link + ")\n"
            "### Ran Playwright code\n```js\n// Screenshot viewport and save it as " + link + "\n```"
        )}]
        if not filename:
            content.append({"type": "image", "data": sys.argv[1], "mimeType": "image/png"})
        send({"jsonrpc": "2.0", "id": rid, "result": {"content": content}})
"""


def test_a_named_screenshot_is_postable_like_an_unnamed_one(tmp_path, monkeypatch):
    """A file-link-only result is stashed from the server's working directory (issue #552).

    The live defect, end to end through the real stdio client: the unnamed call stashed, the named
    one did not. Now both do, and the unnamed one — which carries an image block *and* a link to
    its saved copy — still stashes exactly once.
    """
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)  # the harness's cwd is the directory the server is spawned in
    script = tmp_path / "playwright_like.py"
    script.write_text(_PLAYWRIGHT_LIKE_SERVER, encoding="utf-8")
    home = tmp_path / "home"
    _drop_config(home, "pw", {"command": sys.executable, "args": [str(script), _PNG_B64]})
    resolution = load_mcp_tools(home, timeout=10)
    try:
        (tool,) = resolution.tools
        store = resolution.images
        assert store is not None and len(store) == 0
        assert resolution.clients[0].workdir == work.resolve()

        named = tool.run(filename="example.png")
        assert (work / "example.png").is_file()  # the server wrote it in the directory we pinned
        assert isinstance(named, ToolResult)  # vision inlining, exactly as for an image block
        assert len(named.images) == 1
        assert len(store) == 1  # the defect: this stayed 0
        assert store.get("latest").data == base64.b64decode(_PNG_B64)
        assert store.get("latest").mimetype == "image/png"
        assert "image mcp-image-1: image/png" in named.text
        assert "saved as 'example.png'" in named.text
        assert "action='post_image', image='mcp-image-1'" in named.text

        unnamed = tool.run()
        assert isinstance(unnamed, ToolResult)
        assert len(unnamed.images) == 1
        assert len(store) == 2  # the image block, and not the link to its saved copy as well
    finally:
        for client in resolution.clients:
            client.close()


def _png_file(path):
    path.write_bytes(base64.b64decode(_PNG_B64))
    return path


def _linking(*targets, is_error=False):
    """A result that links `targets` the way playwright-mcp prints a saved file, and no image."""
    lines = "\n".join(f"- [Screenshot of viewport]({target})" for target in targets)
    return {"content": [{"type": "text", "text": f"### Result\n{lines}"}], "isError": is_error}


@pytest.mark.parametrize("shape", ["relative", "absolute", "symlink"])
def test_a_named_file_outside_the_servers_working_directory_is_refused(tmp_path, shape):
    """The server's working directory is the root, and a path is judged where it really lands.

    A real PNG outside the root, named and linked three ways — climbing out, naming it absolutely,
    and a symlink *inside* the root pointing at it — is never read: nothing stashed, nothing shown,
    and the model told why.
    """
    work = tmp_path / "work"
    work.mkdir()
    outside = _png_file(tmp_path / "outside.png")
    target = {"relative": "../outside.png", "absolute": str(outside), "symlink": "inside.png"}[
        shape
    ]
    if shape == "symlink":
        (work / "inside.png").symlink_to(outside)
    named = _named_files({"filename": target}, work)
    outside.write_bytes(
        outside.read_bytes()
    )  # "the call wrote it" holds; containment still refuses
    store = McpImageStore()
    result = _render_tool_result(_linking(target), store, named)
    assert isinstance(result, str)  # no ToolResult: no pixels reached the model
    assert len(store) == 0
    assert f"[image file {str(outside)!r} not shared: it is outside the MCP server's working " in (
        result
    )


def test_a_sibling_directory_sharing_the_roots_prefix_is_outside_it(tmp_path):
    """Containment is a path test, never a string prefix: ``work-evil`` is not inside ``work``."""
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "work-evil").mkdir()
    named = _named_files({"filename": "../work-evil/shot.png"}, work)
    _png_file(tmp_path / "work-evil" / "shot.png")
    store = McpImageStore()
    result = _render_tool_result(_linking("../work-evil/shot.png"), store, named)
    assert len(store) == 0
    assert "outside the MCP server's working directory" in result


def test_text_a_page_wrote_can_never_choose_what_is_read(tmp_path):
    """The read is keyed to the call's arguments, never to link-shaped text in its result.

    playwright-mcp emits page-authored strings verbatim at the start of a line — a page error's
    stack, a response body, a storage value — so a page *can* write ``- [x](photo.png)`` into a
    result. An image already in the root, linked that way, is not opened unless the model named it;
    named, it is still not shared unless the call wrote it; and only then is it shown.
    """
    work = tmp_path / "work"
    work.mkdir()
    photo = _png_file(work / "photo.png")
    injected = {
        "content": [
            {
                "type": "text",
                "text": "### Events\n- [ERROR] Error: boom\n- [x](photo.png)\n    at page.js:1",
            }
        ]
    }
    store = McpImageStore()
    unasked = _render_tool_result(injected, store, _named_files({"url": "https://h/"}, work))
    assert unasked == injected["content"][0]["text"]  # untouched: no read, no note
    assert len(store) == 0

    unwritten = _render_tool_result(
        _linking("photo.png"), store, _named_files({"filename": "photo.png"}, work)
    )
    assert "[image file 'photo.png' not shared: this call did not write it]" in unwritten
    assert len(store) == 0

    named = _named_files({"filename": "photo.png"}, work)
    # The call writes it. A different size, so the stamp moves however coarse this filesystem's
    # clock is — a same-size rewrite inside one tick is the one write a stamp cannot see.
    photo.write_bytes(base64.b64decode(_PNG_B64) + b"\0" * 8)
    shared = _render_tool_result(_linking("./photo.png"), store, named)
    assert isinstance(shared, ToolResult)
    assert len(store) == 1


def test_the_bytes_decide_what_is_an_image_never_the_extension(tmp_path):
    """Whatever was named, nothing that is not a picture reaches the store.

    A secret written under a screenshot's name is refused with a note; one written under its own
    name is passed over in silence (a call that saves a non-image — a PDF, a storage state — is
    ordinary); and a real capture saved with no extension at all is shared, because its bytes say
    it is a PNG.
    """
    work = tmp_path / "work"
    work.mkdir()
    secret = "AI_API_KEY=sk-fake-0000000000000000\n"
    store = McpImageStore()

    named = _named_files({"filename": "disguised.png"}, work)
    (work / "disguised.png").write_text(secret, encoding="utf-8")
    disguised = _render_tool_result(_linking("./disguised.png"), store, named)
    assert "not shared: it is not a recognizable image" in disguised

    named = _named_files({"filename": "agent.env"}, work)
    (work / "agent.env").write_text(secret, encoding="utf-8")
    plain = _render_tool_result(_linking("./agent.env"), store, named)
    assert plain == "### Result\n- [Screenshot of viewport](./agent.env)"  # untouched, no note
    assert len(store) == 0
    assert "sk-fake" not in disguised + plain

    named = _named_files({"filename": "capture"}, work)
    _png_file(work / "capture")
    shared = _render_tool_result(_linking("./capture"), store, named)
    assert isinstance(shared, ToolResult)
    assert len(store) == 1


def test_a_named_file_is_read_only_with_a_root_and_only_on_success(tmp_path):
    """No root (an HTTP server, the library path) or a failed call: the link stays text."""
    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "shot.png"}, work)
    _png_file(work / "shot.png")
    store = McpImageStore()
    assert _render_tool_result(_linking("./shot.png"), store) == (
        "### Result\n- [Screenshot of viewport](./shot.png)"
    )
    failed = _render_tool_result(_linking("./shot.png", is_error=True), store, named)
    assert failed.startswith("Error: ")
    assert len(store) == 0


def test_a_named_file_the_store_cannot_hold_is_described_never_raised(tmp_path, monkeypatch):
    """Every way a named file can be unreadable is a reason in the result, never an exception.

    An exception here would replace a call that *finished* — whose side effect happened — with an
    error, and invite the model to do it again.
    """
    import basecradle_harness._mcp as mcp

    work = tmp_path / "work"
    work.mkdir()
    store = McpImageStore()

    def run(filename, make=None, link=None):
        named = _named_files({"filename": filename}, work)
        if make is not None:
            make(work / filename)
        return _render_tool_result(_linking(link or filename), store, named)

    assert "not shared: it is not a regular file" in run("pipe.png", os.mkfifo)
    assert "not shared: it is not a regular file" in run("dir.png", os.mkdir)
    assert "not shared: it was not found" in run("gone.png")
    # A saved directory that is not meant as a picture (playwright's trace resources) is silent.
    assert run("traces", os.mkdir) == "### Result\n- [Screenshot of viewport](traces)"
    # A NUL byte names no file on any Python, and must not raise out of resolve or open.
    assert "\x00" in run("a\x00.png", link="a\x00.png")
    monkeypatch.setattr(mcp, "MAX_IMAGE_BYTES", 8)
    assert "not shared: it is too large to show or share" in run("big.png", _png_file)
    assert len(store) == 0


def test_a_named_file_is_read_only_when_the_result_links_it(tmp_path):
    """The call must name the file *and* the server must say it saved it there.

    A tool that writes a named image as a side effect and never mentions it is not a screenshot
    tool, and its file is left alone.
    """
    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "shot.png"}, work)
    _png_file(work / "shot.png")
    store = McpImageStore()
    quiet = {"content": [{"type": "text", "text": "done"}]}
    assert _render_tool_result(quiet, store, named) == "done"
    assert len(store) == 0


def test_an_image_block_and_a_named_file_stash_the_picture_once(tmp_path):
    """A result that sent the picture inline is not read back from disk as well."""
    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "shot.png"}, work)
    _png_file(work / "shot.png")
    result = _linking("./shot.png")
    result["content"].append({"type": "image", "data": _PNG_B64, "mimeType": "image/png"})
    store = McpImageStore()
    rendered = _render_tool_result(result, store, named)
    assert isinstance(rendered, ToolResult)
    assert len(rendered.images) == 1
    assert len(store) == 1


@pytest.mark.parametrize(
    "text",
    [
        "- [Screenshot of the ](odd) button](./shot.png)",
        "- [Screenshot of line one\nline two](shot.png)  ",
        "- [Screenshot of viewport](./shot.png)\n- [Screenshot of viewport](shots/../shot.png)",
    ],
    ids=["bracket-in-title", "newline-in-title", "linked-twice"],
)
def test_a_link_is_matched_by_where_it_lands_not_how_it_is_spelled(tmp_path, text):
    """Titles, spellings and a symlinked root cannot hide the file the call named.

    An element screenshot is titled with the model's own description of the element, which may
    carry ``](`` or a line break; a file may be linked twice under two spellings; and the root may
    be reached through a symlink (macOS ``/var`` → ``/private/var``). It is shared, once.
    """
    work = tmp_path / "work"
    work.mkdir()
    root = tmp_path / "root-link"
    root.symlink_to(work)
    named = _named_files({"element": "the ](odd) button", "filename": "./shot.png"}, root)
    _png_file(work / "shot.png")
    store = McpImageStore()
    result = _render_tool_result({"content": [{"type": "text", "text": text}]}, store, named)
    assert isinstance(result, ToolResult)
    assert len(result.images) == 1
    assert len(store) == 1
    assert "saved as 'shot.png'" in result.text


def test_a_name_with_spaces_and_parentheses_survives(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "sub/shot two (1).png"}, work)
    (work / "sub").mkdir()
    _png_file(work / "sub" / "shot two (1).png")
    store = McpImageStore()
    result = _render_tool_result(_linking("sub/shot two (1).png"), store, named)
    assert isinstance(result, ToolResult)
    assert "saved as 'sub/shot two (1).png'" in result.text


def test_nothing_but_a_regular_file_is_ever_opened(tmp_path, monkeypatch):
    """A FIFO or a device is judged on its ``lstat`` and never opened — opening one can block or act."""
    import basecradle_harness._mcp as mcp

    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "pipe.png"}, work)
    os.mkfifo(work / "pipe.png")

    def refuse(*args, **kwargs):
        raise AssertionError("a non-regular file was opened")

    monkeypatch.setattr(mcp.os, "open", refuse)
    result = _render_tool_result(_linking("pipe.png"), McpImageStore(), named)
    assert "not shared: it is not a regular file" in result


def test_the_size_bound_holds_on_what_was_read_not_only_on_the_fstat(tmp_path, monkeypatch):
    """A file that grows between the ``fstat`` and the read is still held to the ceiling."""
    import basecradle_harness._mcp as mcp

    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "shot.png"}, work)
    _png_file(work / "shot.png")
    monkeypatch.setattr(mcp, "MAX_IMAGE_BYTES", 32)
    real_fstat = os.fstat

    def shrunk(fd):
        info = real_fstat(fd)
        return os.stat_result((*info[:6], 0, *info[7:10]))  # st_size reads 0

    monkeypatch.setattr(mcp.os, "fstat", shrunk)
    store = McpImageStore()
    result = _render_tool_result(_linking("shot.png"), store, named)
    assert "not shared: it is too large to show or share" in result
    assert len(store) == 0


def test_a_re_shot_of_an_unchanged_page_is_still_written(tmp_path):
    """Identical bytes into the same file move only its write times — and that is enough.

    A second screenshot of a page that has not changed keeps the inode and the size; only
    ``st_mtime_ns``/``st_ctime_ns`` move. The pause is longer than any filesystem clock tick this
    suite runs on; inside one tick the rewrite would read as unwritten, which fails closed.
    """
    work = tmp_path / "work"
    work.mkdir()
    shot = _png_file(work / "shot.png")
    named = _named_files({"filename": "shot.png"}, work)
    time.sleep(0.05)
    shot.write_bytes(shot.read_bytes())
    store = McpImageStore()
    assert isinstance(_render_tool_result(_linking("shot.png"), store, named), ToolResult)
    assert len(store) == 1


def test_a_filename_carrying_the_link_separator_still_matches(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    named = _named_files({"filename": "a](b).png"}, work)
    _png_file(work / "a](b).png")
    store = McpImageStore()
    result = _render_tool_result(_linking("./a](b).png"), store, named)
    assert isinstance(result, ToolResult)
    assert len(store) == 1


def test_what_names_no_file_costs_no_resolve(tmp_path, monkeypatch):
    """Every string argument of every call is a candidate, so the cost has to be bounded.

    A mail body or a base64 attachment past `PATH_MAX`, or a multi-line string, is never resolved;
    and a result's link lines — which a page can fill — are resolved only when their final component
    is one the call's arguments spelled.
    """
    import basecradle_harness._mcp as mcp

    work = tmp_path / "work"
    work.mkdir()
    assert _named_files({"body": "a/" * 3000, "text": "one\ntwo/x.png"}, work).before == {}

    named = _named_files({"filename": "shot.png"}, work)
    resolved = []
    real = mcp._resolve
    monkeypatch.setattr(
        mcp, "_resolve", lambda root, target: resolved.append(target) or real(root, target)
    )
    page = "\n".join(f"- [x](asset-{n}.png)" for n in range(1000)) + "\n- " + "[y](z" * 50000 + ")"
    _render_tool_result({"content": [{"type": "text", "text": page}]}, McpImageStore(), named)
    assert resolved == []


def test_the_open_and_the_read_are_fenced(tmp_path, monkeypatch):
    """The open refuses a swapped-in link and cannot block or take a terminal; the fd always closes;
    a read error is a reason; an oversized file is never read past its magic bytes; and a file that
    turned into something else between the ``lstat`` and the open is refused on the ``fstat``.
    """
    import errno

    import basecradle_harness._mcp as mcp

    work = tmp_path / "work"
    work.mkdir()
    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def spy_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        opened.append((fd, flags))
        return fd

    def spy_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(mcp.os, "open", spy_open)
    monkeypatch.setattr(mcp.os, "close", spy_close)

    def render(name):
        named = _named_files({"filename": name}, work)
        _png_file(work / name)
        return _render_tool_result(_linking(name), McpImageStore(), named)

    assert isinstance(render("ok.png"), ToolResult)
    ((fd, flags),) = opened
    for flag in (os.O_NOFOLLOW, os.O_NONBLOCK, os.O_NOCTTY):
        assert flags & flag
    assert closed == [fd]

    reads = []

    class _Handle:
        def __init__(self, fail):
            self._fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            reads.append(n)
            if self._fail:
                raise OSError(errno.EIO, "Input/output error")
            return base64.b64decode(_PNG_B64)[:n]

    monkeypatch.setattr(mcp, "open", lambda *a, **k: _Handle(fail=True), raising=False)
    assert "not shared: it could not be read (Input/output error)" in render("eio.png")
    assert len(closed) == 2  # closed on the error path too

    monkeypatch.setattr(mcp, "open", lambda *a, **k: _Handle(fail=False), raising=False)
    monkeypatch.setattr(mcp, "MAX_IMAGE_BYTES", 32)
    reads.clear()
    assert "too large to show or share" in render("big.png")
    assert reads == [mcp._SNIFF_BYTES]  # the magic bytes, and not one byte of the body

    fifo_mode = stat.S_IFIFO | 0o644
    real_fstat = os.fstat
    monkeypatch.setattr(
        mcp.os, "fstat", lambda fd: os.stat_result((fifo_mode, *real_fstat(fd)[1:10]))
    )
    assert "not shared: it is not a regular file" in render("swapped.png")


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
            "MCP server 'dup' active with 1 tool(s)"
        )  # count is loaded
    finally:
        for client in resolution.clients:
            client.close()


def _bare_name_launcher(tmp_path, monkeypatch, name="nova-mcp"):
    """A fake MCP server reachable only as `name`, from a bin dir `sys.executable` now sits in.

    The shape of a server installed *into the agent's venv*: its console script lands beside
    the harness's own entry points, in a directory nothing puts on ``PATH`` (issue #409).
    """
    script = _write_server_script(tmp_path)
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / name
    command = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    launcher.write_text(f"#!/bin/sh\n{command}\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    return name


def test_a_stdio_server_beside_the_interpreter_launches_by_bare_name(tmp_path, monkeypatch):
    """A bare `command` resolves against exactly the PATH it is handed — so it carries the venv."""
    _drop_config(tmp_path, "fake", {"command": _bare_name_launcher(tmp_path, monkeypatch)})
    resolution = load_mcp_tools(tmp_path, timeout=10)
    try:
        assert [t.name for t in resolution.tools] == ["fake__echo"]
        assert resolution.skipped == []
    finally:
        for client in resolution.clients:
            client.close()


def test_an_explicit_path_in_the_servers_env_still_wins(tmp_path, monkeypatch):
    """The harness's addition sits *under* the config's env, so an operator's PATH is absolute."""
    name = _bare_name_launcher(tmp_path, monkeypatch)
    _drop_config(tmp_path, "fake", {"command": name, "env": {"PATH": "/usr/bin:/bin"}})
    resolution = load_mcp_tools(tmp_path, timeout=5)
    assert resolution.tools == []
    assert resolution.skipped[0][0] == "fake"


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
        # A remote server's paths name files on its host, never ours (issue #552)...
        assert client.workdir is None
        # ...and the handshake advertises no MCP `roots`, which is what makes playwright-mcp resolve
        # a named file against its own cwd — the directory a stdio server's files are read from.
        initialize = json.loads(route.calls[0].request.content)
        assert initialize["method"] == "initialize"
        assert "roots" not in initialize["params"]["capabilities"]
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
    assert "opt-out" in block  # still an audited opt-out marker
    assert "- MCP server 'x' active" in block
    assert "- Tool 'y' refused" in block


def test_render_safety_header_sanctions_rather_than_warns():
    """The header must read as a provenance record, not a warning label (issue #322).

    A safety-trained model given only a warning-shaped header ("⚠ … beyond the safe set")
    refused its own opted-in MCP tools, denied they existed, and confabulated results around
    them. The header now tells the model an ``active`` server is approved for its use, while
    still naming the audited opt-out — so both line kinds this block mixes read correctly.
    """
    block = render_safety(["MCP server 'x' active with 3 tool(s)"])
    assert block is not None
    # Sanctions the tools to the reader that is supposed to use them.
    assert "not a warning to you" in block
    assert "installed and approved for your use" in block
    assert "first-class" in block
    # The legacy alarm framing that caused the refusals is gone.
    assert "⚠" not in block
    assert "loaded tools beyond the shipped safe set" not in block


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
