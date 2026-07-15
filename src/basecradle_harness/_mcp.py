"""MCP drop-in: the harness as an MCP client, with safe-by-default surfacing.

[MCP](https://modelcontextprotocol.io) (the Model Context Protocol) is the open standard
for exposing a server's tools to a model. This module makes the harness an **MCP client**:
drop a server config into the config home's ``mcp/`` dir and that server's tools become
part of the agent's active tool set on the next wake — no code change, the same
"everything in the folder is active" model as the ``tools/`` overlay (Group 2).

Safe by default, opt-out made explicit
--------------------------------------
The harness ships **safe**: ``mcp/`` is empty, so a fresh install talks to no MCP server
and runs with BaseCradle-only tools. Dropping a server config in is the operator
*knowingly leaving the safe zone* — an MCP server is external code the harness cannot
police (a stdio server is a subprocess; an HTTP server is a remote endpoint). So this
module does not hide the transition: every active server is **logged** and carries an
**opt-out notice** rendered into the persistent Turn-0 brief (`notices`), so "all bets
off" is a stated, auditable choice — never silent. This is orthogonal to the policy gate
(`_policy.py`): the policy still refuses an in-process `Tool` that declares ``SHELL``;
MCP is a *different axis* the operator opts into per-server, and the harness surfaces it.

The config shape (one server per file)
--------------------------------------
Each ``mcp/<name>.json`` declares **one** MCP server; the filename stem is the server
name. The body follows the standard MCP config shape, so a published server's snippet
drops in unmodified:

- **stdio** — ``{"command": "uvx", "args": ["some-mcp"], "env": {"API_KEY": "…"}}``
- **HTTP** (Streamable HTTP) — ``{"url": "https://host/mcp", "headers": {"Authorization": "…"}}``

A single-entry ``{"mcpServers": {"<name>": {…}}}`` wrapper (the shape copied from a
Claude-Desktop-style config) is unwrapped for convenience. Drop-to-add / delete-to-disable,
consistent with the ``tools/`` overlay; ``mcp/`` ships empty so there is nothing for the
conffile upgrader to reconcile and an operator-added file is never touched.

**Secrets.** A server's ``env`` may carry secrets, so the file is ``chmod 600``-friendly
and its values are passed to the subprocess **literally** via ``Popen(env=…)`` — never
shell-sourced or expanded (the basecradle-router#109 lesson: don't interpolate untrusted
values through a shell). ``shell=False`` always.

Lifecycle under the wake model
------------------------------
A wake is one process per platform event, so a stdio server is spawned at tool-resolution
time, kept alive for the wake's tool calls, and reaped when the process exits (an
``atexit`` hook plus the daemon reader thread). The trade is **per-wake startup latency**:
each wake that has MCP configured pays the server's handshake + ``tools/list`` once. With
``mcp/`` empty (the default) a wake pays nothing — only an operator who opts in pays. A
pooled/long-lived server is a possible future optimization, out of scope here.

Failure never crashes the wake
------------------------------
A server that fails to start, handshake, or list its tools **self-excludes**: its tools
are dropped from the active set and recorded in `skipped` with a reason, exactly the
Group-2 activation robustness bar. One flaky server never takes the wake down.
"""

from __future__ import annotations

import atexit
import base64
import itertools
import json
import logging
import os
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._assets import (
    _VIEWABLE_IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    _data_url,
    _media_type,
)
from basecradle_harness._install import config_home
from basecradle_harness._messages import ImageContent, ToolResult
from basecradle_harness._tools import NO_PARAMETERS, Tool
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

# The MCP protocol version this client advertises in the initialize handshake. A server
# may negotiate a different one in its response; we tolerate that (we only use the small,
# stable request/response subset — initialize, tools/list, tools/call).
_PROTOCOL_VERSION = "2025-06-18"

# How long (seconds) to wait for any single MCP request — the handshake, tools/list, or a
# tool call. Bounds a hung server so it degrades to "skipped" / a tool error instead of
# stalling the whole wake. Overridable via HARNESS_MCP_TIMEOUT.
_DEFAULT_TIMEOUT = 20.0
_TIMEOUT_VAR = "HARNESS_MCP_TIMEOUT"

# Function-tool names must match ^[A-Za-z0-9_-]{1,64}$ on the providers, so an MCP tool is
# namespaced ``<server>__<tool>`` (the same convention MCP clients use) with both parts
# sanitized, and the whole truncated to 64. The separator keeps two servers' same-named
# tools — and an MCP tool vs. a built-in — from colliding.
_NAME_MAX = 64
_SEP = "__"

# Queued by the stdio reader thread when the server's stdout closes (it exited/crashed), so
# a request blocked waiting for a response fails fast instead of waiting out the timeout.
_CLOSED = object()

# How many recent MCP-returned images the per-wake `McpImageStore` keeps for the
# ``assets action='post_image'`` path (issue #318). Small on purpose: a wake is one short-lived
# process and the store is in-memory only, but each image can be up to `MAX_IMAGE_BYTES`, so the
# ring is bounded in *count* to keep the worst-case footprint bounded. The common case is one
# capture posted immediately; the ring covers "took several, post one".
_IMAGE_STORE_CAP = 8

# The alias that resolves to the most recent capture, mirroring the assets tool's ``'latest'``.
_LATEST = "latest"


class McpError(Exception):
    """An MCP transport or protocol failure — a failed handshake, a JSON-RPC error, a timeout."""


# --- config -------------------------------------------------------------------


@dataclass(frozen=True)
class McpServerConfig:
    """One MCP server, parsed from a ``mcp/<name>.json`` file.

    Exactly one transport is configured: ``command`` (stdio) or ``url`` (HTTP). `name` is
    the filename stem, used to namespace the server's tools and to label it in logs and the
    opt-out notice.
    """

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def transport(self) -> str:
        """``"stdio"`` when a command is set, else ``"http"`` (a url)."""
        return "stdio" if self.command else "http"


def load_mcp_configs(home: str | os.PathLike[str] | None = None) -> list[McpServerConfig]:
    """Every server declared by ``mcp/*.json`` in the config home, in filename order.

    The ``mcp/`` dir ships empty (safe by default), so a missing dir or an empty one yields
    no servers. A file that fails to parse is logged and skipped — one malformed operator
    file never takes the agent down, the same robustness the ``tools/`` overlay has.
    """
    mcp_dir = config_home(home) / "mcp"
    if not mcp_dir.is_dir():
        return []
    configs: list[McpServerConfig] = []
    for path in sorted(mcp_dir.glob("*.json")):
        try:
            configs.append(_parse_config(path))
        except Exception as exc:  # noqa: BLE001 - a bad operator file is skipped, not fatal
            _log.warning("Skipping MCP server config %s: %s", path.name, exc)
    return configs


def _parse_config(path: Path) -> McpServerConfig:
    """Parse one ``mcp/<name>.json`` into an `McpServerConfig`, validating the shape.

    Accepts a bare server object (``{"command": …}`` or ``{"url": …}``) or a single-entry
    ``{"mcpServers": {"<name>": {…}}}`` wrapper, whose inner key overrides the filename as
    the server name. Exactly one of ``command`` / ``url`` must be present.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    name = path.stem
    servers = data.get("mcpServers")
    if isinstance(servers, dict):
        if len(servers) != 1:
            raise ValueError("a 'mcpServers' wrapper must hold exactly one server")
        name, data = next(iter(servers.items()))
        if not isinstance(data, dict):
            raise ValueError("the server entry must be a JSON object")

    command = data.get("command")
    url = data.get("url")
    if bool(command) == bool(url):
        raise ValueError("set exactly one of 'command' (stdio) or 'url' (http)")
    return McpServerConfig(
        name=str(name),
        command=str(command) if command else None,
        args=tuple(str(a) for a in data.get("args", ())),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        url=str(url) if url else None,
        headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
    )


# --- the per-wake image store (issue #318) ------------------------------------


@dataclass(frozen=True)
class StashedImage:
    """One image an MCP tool returned, held for the ``assets action='post_image'`` path.

    `mimetype` is the bare media type (``image/png``); `data` is the decoded bytes, ready to
    upload to the timeline exactly as the assets tool uploads a generated image.
    """

    mimetype: str
    data: bytes


class McpImageStore:
    """A per-wake, in-memory, bounded ring of images MCP tools returned (issue #318).

    The "show me what you see" half of MCP image support, and it is **independent of the model's
    vision** — the point being that a text-only agent (e.g. @glm-5.2) can still *post* a browser
    screenshot to the timeline even though it cannot itself see it. When an MCP tool result carries
    an image, `_render_tool_result` stashes the decoded bytes here under a short handle
    (``mcp-image-N``) and names that handle in the model-readable placeholder; the assets tool's
    ``post_image`` action then looks the bytes back up by handle and uploads them through the
    existing asset-create path.

    It is deliberately **ephemeral**: never persisted, never replayed, and gone when the wake
    process exits — the same stance the engine takes toward a viewed image's pixels. That is also
    why a ``post_image`` create carries **no** idempotency key and is never re-issued by a recovery
    (`_idempotency`): the bytes live only here, so a killed-and-resumed wake cannot reconstruct
    them — exactly the non-replayable shape a generated image's upload already has. The ring is
    bounded in count (`_IMAGE_STORE_CAP`); the oldest capture is evicted first.
    """

    def __init__(self, cap: int = _IMAGE_STORE_CAP) -> None:
        self._cap = max(1, cap)
        self._by_handle: dict[str, StashedImage] = {}  # insertion-ordered (dict, 3.7+)
        self._count = 0  # monotonic, so a handle is never reused within a wake

    def stash(self, mimetype: str, data: bytes) -> str:
        """Store one image and return its handle (``mcp-image-N``), evicting the oldest if full."""
        self._count += 1
        handle = f"mcp-image-{self._count}"
        self._by_handle[handle] = StashedImage(mimetype=mimetype, data=data)
        while len(self._by_handle) > self._cap:
            del self._by_handle[next(iter(self._by_handle))]  # drop the oldest (FIFO)
        return handle

    def get(self, handle: str | None) -> StashedImage | None:
        """The image for `handle`, the ``'latest'`` alias, or ``None`` if unknown/empty."""
        if handle is None:
            return self.latest()
        if handle.strip().lower() == _LATEST:
            return self.latest()
        return self._by_handle.get(handle.strip())

    def latest(self) -> StashedImage | None:
        """The most recently stashed image, or ``None`` when the store is empty."""
        if not self._by_handle:
            return None
        return self._by_handle[next(reversed(self._by_handle))]

    def __len__(self) -> int:
        return len(self._by_handle)


# --- the JSON-RPC clients -----------------------------------------------------


class McpClient(ABC):
    """A minimal, synchronous MCP client: handshake, list tools, call a tool, close.

    The harness engine is synchronous and one wake is one short-lived process, so this is a
    deliberately small request/response client over JSON-RPC 2.0 — no async event loop, no
    server-initiated streaming. `start` performs the initialize handshake; after it,
    `list_tools` and `call_tool` are plain blocking round-trips bounded by `timeout`.
    """

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        self.config = config
        self.timeout = timeout
        self._ids = itertools.count(1)

    @abstractmethod
    def start(self) -> None:
        """Connect/spawn and run the initialize handshake. Raises `McpError` on failure."""

    @abstractmethod
    def _request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return its ``result``, or raise `McpError`."""

    @abstractmethod
    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""

    @abstractmethod
    def close(self) -> None:
        """Release the transport (terminate the subprocess / close the HTTP client)."""

    def _handshake(self) -> None:
        """The MCP initialize handshake: ``initialize`` then ``notifications/initialized``."""
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "basecradle-harness", "version": __version__},
            },
        )
        self._notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        """The server's tools (``tools/list``): each a dict with ``name``/``description``/``inputSchema``."""
        result = self._request("tools/list")
        tools = result.get("tools")
        return list(tools) if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke ``tools/call`` and return the raw JSON-RPC ``result`` dict.

        The client speaks the protocol; rendering the result into what the *model* reads —
        joining text blocks, and (issue #318) turning an image block into a `ToolResult` plus a
        stashed capture — is `McpTool.run`'s job, because only the tool holds the per-wake
        `McpImageStore`. Raises `McpError` if the call returned a JSON-RPC error.
        """
        return self._request("tools/call", {"name": name, "arguments": arguments})

    @staticmethod
    def _result_of(message: dict) -> dict:
        """The ``result`` of a JSON-RPC response, raising `McpError` if it carried an error."""
        if "error" in message:
            raise McpError(_error_text(message["error"]))
        result = message.get("result")
        return result if isinstance(result, dict) else {}


class StdioMcpClient(McpClient):
    """An MCP client over a spawned stdio subprocess (newline-delimited JSON-RPC).

    The server is launched with ``shell=False`` and an explicit ``env`` (the process
    environment overlaid with the config's literal ``env`` values — never shell-expanded),
    so a secret in the config is passed straight to the child and never interpolated. A
    daemon reader thread drains stdout into a queue, so a blocking ``readline`` can never
    wedge the wake; ``stderr`` is discarded (server logging is not our transport).
    """

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        super().__init__(config, timeout)
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[dict | object] = queue.Queue()
        self._closed = False

    def start(self) -> None:
        assert self.config.command is not None
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - args are an explicit list, shell=False
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**os.environ, **self.config.env},
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise McpError(f"could not start {self.config.command!r}: {exc}") from exc
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()
        self._handshake()

    def _read_loop(self) -> None:
        """Parse each newline-delimited JSON message from stdout onto the queue.

        On EOF — the server's stdout closed because it exited or crashed — a sentinel is
        queued so a request blocked in `_request` fails *immediately* with "server closed"
        rather than waiting out the full timeout for a response that can never come.
        """
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # a non-JSON line (stray server output) is not our concern
        self._queue.put(_CLOSED)  # stdout ended: the server is gone — wake any waiter fast

    def _send(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpError("server is not running")
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise McpError(f"server stdin closed: {exc}") from exc

    def _request(self, method: str, params: dict | None = None) -> dict:
        rid = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(f"timed out after {self.timeout}s waiting for {method!r}")
            try:
                message = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise McpError(f"timed out waiting for {method!r}") from None
            if message is _CLOSED:
                self._queue.put(_CLOSED)  # re-arm so a later request also fails fast
                raise McpError(f"server closed the connection while awaiting {method!r}")
            assert isinstance(message, dict)
            if message.get("id") == rid:  # ignore notifications / unrelated ids
                return self._result_of(message)

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        # Terminate the child *first*. The daemon reader thread is blocked in ``readline``
        # holding stdout's lock; closing that pipe from here would block until the child
        # happened to exit on its own (a sleeping server → a multi-second hang). Killing the
        # child makes its ``readline`` return EOF, so the reader thread ends and releases the
        # pipe. Only stdin (which no other thread touches) is safe to close from here.
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except ProcessLookupError:
            pass  # already gone
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class HttpMcpClient(McpClient):
    """An MCP client over Streamable HTTP: each request is a POST that returns JSON or SSE.

    Speaks the current Streamable-HTTP transport (a single POST endpoint), not the older
    2024 HTTP+SSE two-endpoint transport. A request is POSTed as JSON-RPC; the response is
    either ``application/json`` (one body) or ``text/event-stream`` (SSE events, from which
    we take the one carrying the matching id). The ``Mcp-Session-Id`` the server returns on
    ``initialize`` is echoed on every later request. Uses the SDK's HTTP client stack
    (httpx) so there is no new dependency.
    """

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        super().__init__(config, timeout)
        import httpx  # local import: only an HTTP MCP server needs it

        assert config.url is not None
        self._session_id: str | None = None
        self._client = httpx.Client(
            base_url="",
            headers={**config.headers},
            timeout=timeout,
        )
        self._url = config.url

    def start(self) -> None:
        self._handshake()

    def _post(self, payload: dict, *, expect_response: bool) -> dict | None:
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise McpError(f"HTTP request to {self._url} failed: {exc}") from exc
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        # A non-2xx is a failure on *either* path — including a notification (e.g. the server
        # rejecting `notifications/initialized` with a 4xx): surfacing it fails the handshake
        # cleanly (the server self-excludes) instead of silently proceeding on a half-open
        # session whose first real request would fail anyway.
        if not response.is_success:
            raise McpError(f"HTTP {response.status_code} from {self._url}")
        if not expect_response:
            return None
        rid = payload.get("id")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return _sse_response(response.text, rid)
        return response.json()

    def _request(self, method: str, params: dict | None = None) -> dict:
        rid = next(self._ids)
        message = self._post(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}},
            expect_response=True,
        )
        if message is None:
            raise McpError(f"no response to {method!r}")
        return self._result_of(message)

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._post(
            {"jsonrpc": "2.0", "method": method, "params": params or {}}, expect_response=False
        )

    def close(self) -> None:
        self._client.close()


def _sse_response(body: str, rid: object) -> dict:
    """The JSON-RPC message carrying ``rid`` from an SSE body, or raise `McpError`.

    A Streamable-HTTP response may be a stream of ``data:`` events; we want the one that is
    the response to our request (its ``id`` matches), ignoring any server notifications that
    rode along. Multi-line ``data:`` fields within one event are concatenated per the SSE spec.
    """
    data_lines: list[str] = []

    def flush() -> dict | None:
        if not data_lines:
            return None
        try:
            message = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None
        return message if isinstance(message, dict) and message.get("id") == rid else None

    for raw in body.splitlines():
        if raw.startswith("data:"):
            data_lines.append(raw[len("data:") :].lstrip())
        elif not raw.strip():  # blank line terminates an event
            found = flush()
            if found is not None:
                return found
            data_lines = []
    found = flush()
    if found is not None:
        return found
    raise McpError("no matching JSON-RPC response in the SSE stream")


def _error_text(error: object) -> str:
    """A JSON-RPC error object rendered to a short string for an `McpError`."""
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message", "error")
        return f"{message} (code {code})" if code is not None else str(message)
    return str(error)


def _render_tool_result(result: dict, store: McpImageStore | None = None) -> str | ToolResult:
    """An MCP ``tools/call`` result as what the model reads — text, and any images (issue #318).

    Joins the ``text`` content blocks. An **image** block is no longer collapsed to a bare
    ``[image content]`` placeholder: its bytes are decoded, and

    - handed to the model as **vision input** — the block becomes an `ImageContent` on a
      `ToolResult`, which the engine routes into the model's input exactly as the assets tool's
      ``view`` action does. A model with no vision never receives the pixels: the engine's own
      gate (`_engine._show_images` + `model_sees_images`, issue #316) substitutes an honest
      withheld note, so this always attaches the image and lets the engine decide — the tool has
      no view of the model (the body/brain split); and
    - **stashed** in the per-wake `store` (when one is bound) under a handle the placeholder
      names, so the agent can post the image to the timeline via ``assets action='post_image'``
      *regardless of whether it can see it* (the "show me what you see" path).

    The placeholder text describes the image (type + size + handle) on **every** path, so a
    text-only model still gets an honest, non-empty result rather than nothing or a crash. Any
    other non-text block (an embedded resource, audio) keeps the by-type placeholder — inlining
    those is out of scope. An ``isError`` result is prefixed so the model sees it failed.

    Returns a plain ``str`` when there is nothing to show (the common case, unchanged), and a
    `ToolResult` only when at least one image was inlined as vision input.
    """
    blocks = result.get("content")
    parts: list[str] = []
    images: list[ImageContent] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "image":
                text, image = _render_image_block(block, store)
                parts.append(text)
                if image is not None:
                    images.append(image)
            else:
                parts.append(f"[{block.get('type', 'non-text')} content]")
    text = "\n".join(p for p in parts if p) or "(the tool returned no content)"
    text = f"Error: {text}" if result.get("isError") else text
    return ToolResult(text=text, images=images) if images else text


def _render_image_block(
    block: dict, store: McpImageStore | None
) -> tuple[str, ImageContent | None]:
    """Render one MCP ``image`` content block into (placeholder text, vision image-or-None).

    Decodes the base64 ``data``, stashes the bytes in `store` for the ``post_image`` path (when a
    store is bound), and — for a viewable type within the size ceiling — builds an `ImageContent`
    for vision input. The placeholder always names the type and size (cheap), and, when stashed,
    the handle plus how to post it. A missing/undecodable payload, or one over the ceiling, yields
    a describing placeholder and no image rather than a crash.
    """
    raw_b64 = block.get("data")
    mimetype = _media_type(str(block.get("mimeType") or "")) or "image/png"
    if not isinstance(raw_b64, str) or not raw_b64:
        return "[image content (no data returned)]", None
    try:
        data = base64.b64decode(raw_b64, validate=True)
    except (ValueError, TypeError):
        return f"[image content ({mimetype}, could not decode)]", None
    size = len(data)
    if size <= 0:
        return f"[image content ({mimetype}, empty)]", None
    if size > MAX_IMAGE_BYTES:
        return (
            f"[image: {mimetype}, {_human_bytes(size)} — too large to show or share "
            f"(over the {MAX_IMAGE_BYTES}-byte limit)]",
            None,
        )
    handle = store.stash(mimetype, data) if store is not None else None
    viewable = mimetype in _VIEWABLE_IMAGE_TYPES
    image = ImageContent(url=_data_url(mimetype, data), alt=handle or "image") if viewable else None
    prefix = f"image {handle}" if handle else "image"
    note = "" if viewable else " (type not viewable to a model; described only)"
    share = (
        f" — to share it on this timeline, use the assets tool with "
        f"action='post_image', image='{handle}'"
        if handle is not None
        else ""
    )
    return f"[{prefix}: {mimetype}, {_human_bytes(size)}{note}{share}]", image


def _human_bytes(n: int) -> str:
    """A compact human-readable byte size (``45.2 KB``) for an image placeholder."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# --- the tool wrapper ---------------------------------------------------------


class McpTool(Tool):
    """A discovered MCP tool, exposed to the model as a function tool that proxies to the server.

    Its ``name`` is the namespaced ``<server>__<tool>`` so it cannot collide with a built-in
    tool or another server's; its ``parameters`` are the server-declared ``inputSchema``.
    ``requires`` is empty: an MCP tool is a proxy with no in-process capability to gate, so
    it registers under the locked policy — the safe-by-default *opt-out* is surfaced via the
    brief notice, not enforced by refusing the proxy (the activation-vs-policy split).
    """

    requires: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        server: str,
        remote_name: str,
        description: str,
        parameters: dict,
        client: McpClient,
        images: McpImageStore | None = None,
    ) -> None:
        self.name = mcp_tool_name(server, remote_name)
        self.description = description or f"The {remote_name!r} tool from MCP server {server!r}."
        self.parameters = parameters or NO_PARAMETERS
        self._remote_name = remote_name
        self._client = client
        #: The per-wake image store, shared across every MCP tool of one resolution, so an image
        #: this tool returns can be posted to the timeline via ``assets action='post_image'``
        #: (issue #318). ``None`` on the library/test path — vision inlining still works, but there
        #: is nowhere to stash a capture for the post path.
        self._images = images

    def run(self, **kwargs: object) -> str | ToolResult:
        """Proxy the call to the MCP server and render its result for the model.

        Returns a `ToolResult` (text + vision images) when the server returned an image; a plain
        ``str`` otherwise. Any returned image is also stashed in the per-wake store for the
        ``post_image`` path (issue #318).
        """
        result = self._client.call_tool(self._remote_name, dict(kwargs))
        return _render_tool_result(result, self._images)


def mcp_tool_name(server: str, tool: str) -> str:
    """The model-facing name for an MCP tool: ``<server>__<tool>``, sanitized and ≤64 chars."""
    base = f"{_sanitize(server)}{_SEP}{_sanitize(tool)}"
    return base[:_NAME_MAX]


def _sanitize(part: str) -> str:
    """Coerce a name part to the ``[A-Za-z0-9_-]`` the providers require, collapsing the rest to ``_``."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in part) or "_"


# --- resolution into the active tool set --------------------------------------


@dataclass
class McpResolution:
    """The outcome of loading every configured MCP server, for merging into `ResolvedTools`.

    Args:
        tools: The instantiated `McpTool` proxies for every active server's tools.
        manifest: ``(name, note)`` for each, for the Turn-0 brief's tool block — the note
            marks the tool's MCP provenance and sanctions its use (issue #322).
        skipped: ``(server, reason)`` for every server that failed to load — the visible
            "why isn't this server here?" trail, mirroring Group-2 activation.
        notices: One safe-by-default opt-out line per active server, surfaced in the brief.
        clients: The live clients, closed at process exit (registered with ``atexit``).
        images: The per-wake `McpImageStore` shared by every active server's tools (issue #318),
            carried to the assets tool via the `PlatformContext` so a returned image can be posted
            to the timeline. ``None`` when no server loaded (nothing to stash).
    """

    tools: list[Tool] = field(default_factory=list)
    manifest: list[tuple[str, str | None]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    clients: list[McpClient] = field(default_factory=list)
    images: McpImageStore | None = None


def _timeout_from_env() -> float:
    """The per-request MCP timeout from ``HARNESS_MCP_TIMEOUT``, else the default."""
    raw = os.environ.get(_TIMEOUT_VAR)
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
        return value if value > 0 else _DEFAULT_TIMEOUT
    except ValueError:
        return _DEFAULT_TIMEOUT


def _connect(config: McpServerConfig, timeout: float) -> McpClient:
    """Build and start the client for a config's transport, initialized and ready.

    If `start` fails *after* the transport came up — a stdio subprocess that spawned but
    then timed out on the handshake, an HTTP client whose initialize POST errored — the
    half-open transport is torn down before the error propagates, so a failed server never
    leaks a running subprocess or socket into the wake.
    """
    client: McpClient
    if config.transport == "stdio":
        client = StdioMcpClient(config, timeout)
    else:
        client = HttpMcpClient(config, timeout)
    try:
        client.start()
    except Exception:
        _safe_close(client)
        raise
    return client


def load_mcp_tools(
    home: str | os.PathLike[str] | None = None, *, timeout: float | None = None
) -> McpResolution:
    """Load every configured MCP server's tools, surfacing the safe-by-default opt-out.

    For each ``mcp/<name>.json``: connect, run the handshake, and ``tools/list``; on success
    wrap each tool as an `McpTool` proxy, record a manifest entry and a one-line opt-out
    notice, and register the client for teardown at process exit. On **any** failure the
    server self-excludes — its tools are dropped and the failure is recorded in `skipped`
    with a reason — so a flaky or missing server never crashes the wake.

    The active-server log line and the brief notice are the explicit, auditable surfacing
    of "this agent has left the safe-by-default zone" (Part B): with ``mcp/`` empty the
    whole function is a no-op and nothing is surfaced.
    """
    timeout = _timeout_from_env() if timeout is None else timeout
    resolution = McpResolution()
    # One image store per resolution, shared by every server's tools, so a screenshot from any
    # active MCP tool can be posted via ``assets action='post_image'`` (issue #318). Bound onto
    # the resolution only if a tool actually loads (below) — no MCP tools, nothing to stash.
    store = McpImageStore()
    seen: set[str] = set()  # final tool names already claimed, across all servers
    for config in load_mcp_configs(home):
        client: McpClient | None = None
        try:
            client = _connect(config, timeout)
            discovered = client.list_tools()
        except Exception as exc:  # noqa: BLE001 - a failed server self-excludes, never fatal
            reason = f"MCP server {config.name!r} did not load: {exc}"
            resolution.skipped.append((config.name, reason))
            _log.warning(reason)
            _safe_close(client)
            continue
        loaded = 0
        for spec in discovered:
            tool = McpTool(
                server=config.name,
                remote_name=str(spec.get("name", "")),
                description=str(spec.get("description") or ""),
                parameters=spec.get("inputSchema") or NO_PARAMETERS,
                client=client,
                images=store,
            )
            # Two tools can collide on the *final* name even when their remote names differ —
            # sanitization (``a.b`` and ``a b`` both → ``a_b``) or the 64-char truncation can
            # map them together. A duplicate name would crash `ToolRegistry.register`, so the
            # later one self-excludes with a reason instead of taking the wake down.
            if tool.name in seen:
                resolution.skipped.append(
                    (tool.name, f"duplicate tool name from MCP server {config.name!r}; skipped")
                )
                _log.warning("MCP tool name %r already claimed; skipping the duplicate.", tool.name)
                continue
            seen.add(tool.name)
            resolution.tools.append(tool)
            resolution.manifest.append((tool.name, _tool_note(config.name)))
            loaded += 1
        resolution.notices.append(_opt_out_notice(config.name, loaded))
        resolution.clients.append(client)
        atexit.register(client.close)
        _log.warning(
            "MCP server %r active: %d tool(s) loaded — this agent has extended beyond the "
            "safe-by-default tool set.",
            config.name,
            loaded,
        )
    # Bind the store only if at least one MCP tool loaded — with no MCP tools there is nothing to
    # stash, and the assets ``post_image`` path stays cleanly "no captures available".
    resolution.images = store if resolution.tools else None
    return resolution


def _safe_close(client: McpClient | None) -> None:
    """Tear down a partially-started client, swallowing any teardown error."""
    if client is None:
        return
    try:
        client.close()
    except Exception:  # noqa: BLE001 - teardown of a failed server must not raise
        pass


def _tool_note(server: str) -> str:
    """The per-tool manifest note marking an MCP-sourced tool in the Turn-0 brief.

    Repeated on *every* tool line (a 24-tool server writes it 24 times), so it is a strong
    signal channel — and until issue #322 it repeated a warning ("beyond the safe-by-default
    tool set") 24 times, reinforcing the model's read that these tools were off-limits. It now
    repeats a sanction instead: provenance plus "approved for your use." The audit marker rides
    the per-server notice and the journald log, not this per-line note.
    """
    return f"from MCP server {server!r} — installed and approved for your use"


def _opt_out_notice(server: str, count: int) -> str:
    """The per-server safe-by-default opt-out notice for an active server, surfaced in the brief.

    Sanctions the tools to the model while keeping the audit tail loud (issue #322). The prior
    wording — "external code you opted into; all bets off" — was written for the operator's
    audit trail but *read by the model*, which then refused the very tools this notice announces.
    So it now states plainly that the server was deliberately installed and approved for the
    agent's use, names the ``<server>__…`` namespace the tools are called by (a model handed the
    bare names still would not call them), and closes the observed fabrication hole ("never report
    a tool result you did not get back from a real call"). The "operator opt-out … recorded for
    audit" tail keeps the safe-by-default record loud without poisoning the tools.
    """
    prefix = _sanitize(server)
    return (
        f"MCP server {server!r} active with {count} tool(s), named {prefix}__… — deliberately "
        f"installed and approved for your use. They are first-class, working tools: call them to "
        f"actually perform the work, and never report a tool result you did not get back from a "
        f"real call. (An operator opt-out beyond the safe-by-default tool set, recorded for audit.)"
    )
