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

A named screenshot is still a screenshot (issue #552)
-----------------------------------------------------
An image reaches the model and the ``post_image`` store two ways. The obvious one is an
``image`` content block. The other is a **file**: playwright-mcp's screenshot tool, handed a
``filename``, saves the file, prints ``- [Screenshot of viewport](./shot.png)``, and sends *no*
image block (``if (!params.filename) registerImageResult(…)``). A model that names its screenshot
— which is exactly what a model does when it means to post it — used to lose the ability to post
it. So a file the call **named, wrote, and linked** is read from the server's working directory
and rendered as if it had arrived inline (`_render_file_links`).

**The rule is keyed to the call's own arguments, never to what the result's text says** — and
that is the design, not a detail. A result's text carries third-party strings verbatim and at the
start of a line: a page's error stack, a response body, a storage value, a mail's body. A link in
it therefore proves nothing about who asked, and a rule that followed any link-shaped line would
let a web page choose which of the agent's files the harness opens. The model's arguments are the
one input no page writes. So the harness reads a file only when every one of these holds, and the
failure direction of each is "not shared", never "read anyway":

- **Only a stdio server has a working directory we know.** It is the directory this process
  spawned it in, pinned on the ``Popen`` (`StdioMcpClient.workdir`). An HTTP server's paths name
  a file on *its* host, so they are never read. The harness advertises no MCP ``roots``, and that
  is load-bearing: without them playwright-mcp resolves a file against its own ``process.cwd()``,
  which is therefore the same directory.
- **The model named it** — the path is one of the call's own string arguments (`_named_files`) —
  **and the result links it**: some line ends in a markdown link that resolves to that same path.
  Paths are compared where they land, never as spelled, so ``./a.png``, ``a.png`` and
  ``sub/../a.png`` are one file.
- **The call wrote it.** Its ``lstat`` before the call and after must differ (`_stamp`: inode,
  size, mtime and ctime — a write moves the last two even when it puts back identical bytes, at
  the filesystem's timestamp resolution). This is clock-free, so it holds on a home whose file
  server's clock drifts, and it is what stops a stale file of the same name — a server that wrote
  somewhere else — being passed off as the capture. Its one blind spot fails closed: an identical
  rewrite inside one timestamp tick (a second on some filesystems) reads as "not written".
- **It sits inside that directory** — the working directory, and only that: a file playwright
  saves under an operator's ``--output-dir`` elsewhere is refused, with a note — **and is a regular
  file** — symlinks followed first, then a
  containment test on the result, never a string prefix — opened ``O_NOFOLLOW | O_NONBLOCK |
  O_NOCTTY``, so a final-component link swapped in cannot redirect the read and a FIFO or device
  cannot wedge it. (A same-user actor racing an *intermediate* directory is not defended against:
  anything able to do that could as easily have sent the bytes inline.)
- **The bytes are an image** — read off their magic bytes (`sniff_media_ext`), never the
  extension — within `MAX_IMAGE_BYTES`. Whatever was named, nothing that is not a picture reaches
  the store.

A named-and-linked file that looks like an image (by extension) and fails a check gets a one-line
note saying why, so the model learns the reason rather than meeting it as a failed ``post_image``;
a named file that is not meant as a picture (a saved PDF, a storage-state ``.json``) passes
silently. A named image is shown to a vision model even from a server configured to omit image
responses (playwright's ``--image-responses omit``): that setting governs what the *server* sends,
and the model asked for this file by name. The harness only ever **reads** the file — it is the
agent's, written at the model's request, and never ours to clean up.
"""

from __future__ import annotations

import atexit
import base64
import itertools
import json
import logging
import os
import queue
import stat
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._assets import (
    _VIEWABLE_IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    _data_url,
    _media_type,
)
from basecradle_harness._install import config_home
from basecradle_harness._media import sniff_media_ext
from basecradle_harness._messages import ImageContent, ToolResult
from basecradle_harness._tools import NO_PARAMETERS, Tool
from basecradle_harness._venv import with_interpreter_bin
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

# The extensions that make a file *look like* an image — which decides only whether one that could
# not be shared earns a note (issue #552). Whether a file is shared is decided by its bytes.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})

# The media type for each image format `sniff_media_ext` recognizes. Its video formats are absent on
# purpose: an MP4 is not a picture, and a clip saved to a named file is not a screenshot.
_SNIFFED_IMAGE_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

# Enough leading bytes for every signature `sniff_media_ext` reads (WebP's is at offset 8–12).
_SNIFF_BYTES = 16

# What `_stamp` records of a file: its type and identity, and enough to tell that something wrote to
# it — a write moves ``st_mtime_ns`` and ``st_ctime_ns`` even when it puts back identical bytes, at
# the filesystem's timestamp resolution.
_Stamp = tuple[int, int, int, int, int, int]

# The most of a server's own ``instructions`` the model is shown (issue #553). They ride the Turn-0
# brief, which is re-sent on every step of every wake, and a server is external code: a bound is
# what makes "nothing replayed per wake may be unbounded" true of them. Generous — the Steel
# launcher's whole disclosure is about 1 KB — and an elision says how much was not shown.
_INSTRUCTIONS_CAP = 4096

# The most of a server's ``serverInfo`` name + version shown, on one line.
_LABEL_CAP = 120

# The longest string treated as a path: Linux's PATH_MAX, and more than any path macOS opens. The
# bound is about *cost*: every string argument of every stdio MCP call is a candidate, a mail body
# or a base64 attachment is full of slashes, and resolving one walks every one of them — seconds,
# on the 3.12+ pathlib, for an argument that names no file at all.
_PATH_MAX = 4096


class McpError(Exception):
    """An MCP transport or protocol failure — a failed handshake, a JSON-RPC error, a timeout."""


# --- the withheld tools (issue #553) --------------------------------------------


@dataclass(frozen=True)
class Withholding:
    """One MCP tool the harness withholds by default, and the documented exception that says why.

    Human–AI parity is the default, so a capability withheld from an agent is legal only as an
    **explicit, dated exception naming its decider** — never by oversight. Each entry is that record,
    in the one place the withholding is built, and every word of it reaches the model: the agent is
    told what it does not have, why, who decided, and what to use instead (@origin, 2026-09-22:
    *"agents need to know what they have and why!"*).
    """

    #: What the tool does that makes it withheld — completes "it is withheld because …".
    reason: str
    #: The server's tool that covers the legitimate need, named in every refusal.
    instead: str
    decider: str
    decided: str
    #: The ruling's own word for how long it stands.
    standing: str


#: Every MCP tool the harness can withhold, by the tool's name on its server. **Only these can be
#: named in ``withheld_tools``**: withholding a tool from an agent needs a stated reason, so adding
#: one is an entry here, in code review, with its decider and date — never a free string in config.
#:
#: ``browser_run_code_unsafe`` (Playwright MCP) runs arbitrary JavaScript inside the playwright-mcp
#: **process** — its own description says "RCE-equivalent", and Node's ``vm`` is no boundary, so
#: ``process`` is reachable. On a local browser that is code execution as the agent's OS user, which
#: walks around the ``shell`` tool's opt-in and the NOC's ``verify_unprivileged`` gate; behind the
#: Steel launcher it is the process holding the Steel key, which reaches every profile in the
#: organization. Playwright MCP 0.0.80 has no flag to turn it off. ``browser_evaluate`` is kept: it
#: runs JavaScript in the *page*, which is the browsing capability a human has. Withheld fleet-wide
#: by @origin's ruling of 2026-09-22 (basecradle/basecradle#582, ruling 4: *"approve both (for
#: now)"*), and a default he can waive per agent — *"@briggs is the exception, meaning if he wants
#: access, he gets it"* — which is why the list is per-server configuration and not a constant.
WITHHOLDABLE: Mapping[str, Withholding] = {
    "browser_run_code_unsafe": Withholding(
        reason=(
            "it runs arbitrary code inside the browser server's own process rather than in the "
            "page — code execution on this machine as you, outside the shell tool's opt-in and "
            "its checks, and on a cloud browser the process that holds its credential"
        ),
        instead="browser_evaluate",
        decider="@origin",
        decided="2026-09-22",
        standing="for now",
    ),
}

#: What a server withholds when its config names nothing: every withholdable tool. Keyed by tool
#: name, so on a server that does not offer the tool it is a no-op.
DEFAULT_WITHHELD: tuple[str, ...] = tuple(WITHHOLDABLE)


def withheld_refusal(server: str, tool: str, *, waivable: bool, offered: Iterable[str] = ()) -> str:
    """What the model is told when it calls a withheld tool by name anyway: what, why, instead.

    The tool to use instead is named only when the server `offered` it under a name of its own —
    a refusal pointing at a tool that is not there, or (past the 64-character truncation) at the
    withheld tool's own name, would be a second wrong answer inside the right one.
    """
    rule = WITHHOLDABLE[tool]
    name = mcp_tool_name(server, tool)
    waiver = f" {_waiver(rule)}" if waivable else ""
    instead = mcp_tool_name(server, rule.instead)
    use = (
        f" Use {instead} to run JavaScript in the page itself."
        if rule.instead in set(offered) and instead != name
        else ""
    )
    return (
        f"{name} is withheld from you: {rule.reason}. Withheld by {rule.decider}'s ruling of "
        f"{rule.decided}, {rule.standing}.{waiver}{use}"
    )


def _waiver(rule: Withholding) -> str:
    """The sentence that tells an agent a withheld tool is its for the asking.

    It speaks for the decider, which is why ``withheld_waivable`` is not a convenience flag: setting
    it asserts that the decider has said this agent may have the tool on request (@origin's ruling
    named @briggs). Only whoever records founder decisions for the fleet sets it.
    """
    return f"That is a default, not a lock: {rule.decider} has said it is yours whenever you ask."


# --- config -------------------------------------------------------------------

#: The keys in a server's config that are the operator's, not the transport's (issue #553).
_OPERATOR_KEYS = frozenset({"withheld_tools", "withheld_waivable", "note"})


@dataclass(frozen=True)
class McpServerConfig:
    """One MCP server, parsed from a ``mcp/<name>.json`` file.

    Exactly one transport is configured: ``command`` (stdio) or ``url`` (HTTP). `name` is
    the filename stem, used to namespace the server's tools and to label it in logs and the
    opt-out notice.

    Three keys are the operator's, not the transport's (issue #553). ``withheld_tools`` names the
    server's tools the agent does not get — absent means `DEFAULT_WITHHELD`, ``[]`` means none, and
    only `WITHHOLDABLE` names are accepted. ``withheld_waivable`` tells the agent the withholding is
    a default it may ask to have lifted. ``note`` is the operator's own words to the model about this
    server — what it is and what backs it — shown beside whatever the server says about itself.
    """

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    withheld_tools: tuple[str, ...] = DEFAULT_WITHHELD
    withheld_waivable: bool = False
    note: str | None = None

    @property
    def transport(self) -> str:
        """``"stdio"`` when a command is set, else ``"http"`` (a url)."""
        return "stdio" if self.command else "http"


def load_mcp_configs(home: str | os.PathLike[str] | None = None) -> list[McpServerConfig]:
    """Every server declared by ``mcp/*.json`` in the config home, in filename order.

    The ``mcp/`` dir ships empty (safe by default), so a missing dir or an empty one yields
    no servers. A file that fails to parse is logged and skipped — one malformed operator
    file never takes the agent down, the same robustness the ``tools/`` overlay has. The
    rejected files are reported by `load_mcp_configs_report`.
    """
    return load_mcp_configs_report(home)[0]


def load_mcp_configs_report(
    home: str | os.PathLike[str] | None = None,
) -> tuple[list[McpServerConfig], list[tuple[str, str]]]:
    """The parsed configs, and ``(file stem, reason)`` for every file that was rejected.

    A rejected file is a server its operator declared and the agent does not have, so it must be as
    visible as one that failed to connect (issue #553): `load_mcp_tools` puts it in ``skipped`` and
    ``--resolved-config`` keeps its stem in ``mcp_servers`` — reported from disk, loaded or not. A
    config the harness cannot honor still **fails closed**: loading it on a guess — the default
    withholding, say, in place of a list with a typo in it — could hand the agent the very tool its
    operator meant to withhold.
    """
    mcp_dir = config_home(home) / "mcp"
    if not mcp_dir.is_dir():
        return [], []
    configs: list[McpServerConfig] = []
    rejected: list[tuple[str, str]] = []
    for path in sorted(mcp_dir.glob("*.json")):
        try:
            configs.append(_parse_config(path))
        except Exception as exc:  # noqa: BLE001 - a bad operator file is skipped, not fatal
            _log.warning("Skipping MCP server config %s: %s", path.name, exc)
            rejected.append((path.stem, f"MCP server config {path.name} was rejected: {exc}"))
    return configs, rejected


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
        misplaced = sorted(set(data) & _OPERATOR_KEYS)
        if misplaced:
            # Outside the wrapper they would be silently ignored — and a waiver that does not land,
            # or a withholding that does not, is exactly what an operator must be told about.
            raise ValueError(f"{misplaced} belong inside the server entry, not beside 'mcpServers'")
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
        withheld_tools=_withheld_tools(data.get("withheld_tools", DEFAULT_WITHHELD)),
        withheld_waivable=_flag(data.get("withheld_waivable", False), "withheld_waivable"),
        note=_note(data.get("note")),
    )


def _withheld_tools(value: object) -> tuple[str, ...]:
    """``withheld_tools`` validated: a list of `WITHHOLDABLE` names (issue #553).

    A config the harness cannot honor **fails closed**: the file is skipped like any malformed one,
    so the server does not load — rather than loading with the very tool its operator meant to
    withhold. A name with no documented reason is refused for the same reason and one more: a
    capability withheld from an agent needs its exception written down, and a free string in config
    is not one.
    """
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("'withheld_tools' must be a list of tool names")
    unknown = sorted(set(value) - set(WITHHOLDABLE))
    if unknown:
        raise ValueError(
            f"'withheld_tools' names {unknown}, which the harness has no documented reason to "
            f"withhold; it can withhold {sorted(WITHHOLDABLE)}"
        )
    return tuple(dict.fromkeys(value))


def _flag(value: object, key: str) -> bool:
    """A JSON boolean, or a `ValueError` naming `key` — never a truthy guess at ``"false"``."""
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be true or false")
    return value


def _note(value: object) -> str | None:
    """The operator's ``note``: a string, or absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("'note' must be a string")
    return value.strip() or None


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
    an image — inline, or saved to a file the call named (issue #552) —
    `_render_tool_result` stashes the bytes here under a short handle
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

    #: The directory the server resolves a relative file against, when this process knows it —
    #: which is only for a server it spawned itself (`StdioMcpClient`). ``None`` means the server's
    #: files are not ours to read, so a file its calls save is never read back (issue #552).
    workdir: Path | None = None

    #: What the server said about itself in its ``initialize`` result (issue #553): its
    #: ``serverInfo`` name and version, and its ``instructions`` — the protocol's channel for
    #: telling the model how to use the server. Read in the handshake; the harness used to discard
    #: the whole result, so no server's instructions ever reached a model.
    server_label: str | None = None
    instructions: str | None = None

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        self.config = config
        self.timeout = timeout
        self._ids = itertools.count(1)
        #: The server's tools this client refuses to call (issue #553). The tool list never offers
        #: them, so a model cannot reach one through a registered tool; this is the second fence,
        #: at the one call every path to the server goes through.
        self.withheld = frozenset(config.withheld_tools)
        #: The tool names the server's last ``tools/list`` offered — what a refusal may point to.
        self.listed: frozenset[str] = frozenset()

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
        """The MCP initialize handshake: ``initialize`` then ``notifications/initialized``.

        The result's ``serverInfo`` and ``instructions`` are kept (issue #553) — each bounded,
        because a server is external code and both are shown to the model on every step of every
        wake. The ``capabilities`` sent stay empty, and that is load-bearing: advertising ``roots``
        would move the directory a server like playwright-mcp resolves a named file against, which
        is the directory `workdir` names (issue #552).
        """
        result = self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "basecradle-harness", "version": __version__},
            },
        )
        self.server_label = _server_label(result.get("serverInfo"))
        instructions = result.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            self.instructions = _bounded(instructions.strip(), _INSTRUCTIONS_CAP)
        self._notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        """The server's tools (``tools/list``): each a dict with ``name``/``description``/``inputSchema``."""
        result = self._request("tools/list")
        tools = result.get("tools")
        listed = list(tools) if isinstance(tools, list) else []
        self.listed = frozenset(
            str(spec.get("name", "")) for spec in listed if isinstance(spec, dict)
        )
        return listed

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke ``tools/call`` and return the raw JSON-RPC ``result`` dict.

        The client speaks the protocol; rendering the result into what the *model* reads —
        joining text blocks, and (issue #318) turning an image block into a `ToolResult` plus a
        stashed capture — is `McpTool.run`'s job, because only the tool holds the per-wake
        `McpImageStore`. Raises `McpError` if the call returned a JSON-RPC error, or if `name` is a
        tool this agent's configuration withholds — refused here, before anything is sent.
        """
        if name in self.withheld:
            raise McpError(
                withheld_refusal(
                    self.config.name,
                    name,
                    waivable=self.config.withheld_waivable,
                    offered=self.listed,
                )
            )
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

    That environment carries the harness's own venv ``bin`` on ``PATH`` (`_venv`), because a
    bare ``command`` is resolved against exactly the ``PATH`` it is handed — so a server
    installed *into the agent's venv* is launchable by name rather than only by absolute path.
    It is applied under the config's ``env``, so an operator who sets ``PATH`` explicitly still
    wins outright.

    The child runs in this process's working directory, and it is **pinned** on the ``Popen``
    rather than inherited, so `workdir` is the directory the server was actually handed — the root
    a file its calls save is resolved and contained against (issue #552). If this process's own
    directory is gone, the server still launches (inheriting it, as it always did); it simply has no
    root, and its saved files are not read back.
    """

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        super().__init__(config, timeout)
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[dict | object] = queue.Queue()
        self._closed = False

    def start(self) -> None:
        assert self.config.command is not None
        try:
            self.workdir = Path.cwd().resolve()
        except OSError:  # this process's own directory was removed under it
            self.workdir = None
        try:
            self._proc = subprocess.Popen(  # args are an explicit list, shell=False
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**with_interpreter_bin(os.environ), **self.config.env},
                cwd=self.workdir,
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
        except Exception:  # noqa: BLE001, S110 - best-effort teardown
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


def _render_tool_result(
    result: dict, store: McpImageStore | None = None, named: _NamedFiles | None = None
) -> str | ToolResult:
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

    A result with **no** image block gets each file the call `named` — and wrote, and linked —
    rendered exactly as an inline image would be (issue #552), under the checks the module
    docstring lists. With no `named` (an HTTP server, the library path) nothing is read; a failed
    call's files are never read, because a call that failed wrote nothing we can vouch for. The two
    paths are exclusive by construction: playwright-mcp's *unnamed* screenshot carries an image
    block **and** a link to its saved copy, and reading the copy as well would stash the one picture
    twice.

    Returns a plain ``str`` when there is nothing to show (the common case, unchanged), and a
    `ToolResult` only when at least one image was inlined as vision input.
    """
    blocks = result.get("content")
    parts: list[str] = []
    texts: list[str] = []
    images: list[ImageContent] = []
    saw_image = False
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
                texts.append(parts[-1])
            elif block_type == "image":
                saw_image = True
                text, image = _render_image_block(block, store)
                parts.append(text)
                if image is not None:
                    images.append(image)
            else:
                parts.append(f"[{block.get('type', 'non-text')} content]")
    if named is not None and not saw_image and not result.get("isError"):
        for text, image in _render_file_links(texts, named, store):
            parts.append(text)
            if image is not None:
                images.append(image)
    text = "\n".join(p for p in parts if p) or "(the tool returned no content)"
    text = f"Error: {text}" if result.get("isError") else text
    return ToolResult(text=text, images=images) if images else text


def _render_image_block(
    block: dict, store: McpImageStore | None
) -> tuple[str, ImageContent | None]:
    """Render one MCP ``image`` content block into (placeholder text, vision image-or-None).

    Decodes the base64 ``data`` and hands the bytes to `_render_image_bytes`. A missing or
    undecodable payload yields a describing placeholder and no image rather than a crash.
    """
    raw_b64 = block.get("data")
    mimetype = _media_type(str(block.get("mimeType") or "")) or "image/png"
    if not isinstance(raw_b64, str) or not raw_b64:
        return "[image content (no data returned)]", None
    try:
        data = base64.b64decode(raw_b64, validate=True)
    except (ValueError, TypeError):
        return f"[image content ({mimetype}, could not decode)]", None
    return _render_image_bytes(mimetype, data, store)


def _render_image_bytes(
    mimetype: str, data: bytes, store: McpImageStore | None, source: str | None = None
) -> tuple[str, ImageContent | None]:
    """Render one image's bytes into (placeholder text, vision image-or-None) — both arrival paths.

    Stashes the bytes in `store` for the ``post_image`` path (when a store is bound), and — for a
    viewable type within the size ceiling — builds an `ImageContent` for vision input. The
    placeholder always names the type and size (cheap), `source` (the file a linked image was read
    from, issue #552) when there is one, and, when stashed, the handle plus how to post it. An
    empty image, or one over the ceiling, is described and neither stashed nor shown.
    """
    size = len(data)
    if size <= 0:
        return f"[image content ({mimetype}, empty)]", None
    if size > MAX_IMAGE_BYTES:
        too_large = (
            f"[image: {mimetype}, {_human_bytes(size)} — too large to show or share "
            f"(over the {MAX_IMAGE_BYTES}-byte limit)]"
        )
        return too_large, None
    handle = store.stash(mimetype, data) if store is not None else None
    viewable = mimetype in _VIEWABLE_IMAGE_TYPES
    image = ImageContent(url=_data_url(mimetype, data), alt=handle or "image") if viewable else None
    prefix = f"image {handle}" if handle else "image"
    origin = f", saved as {source!r}" if source else ""
    note = "" if viewable else " (type not viewable to a model; described only)"
    share = (
        f" — to share it on this timeline, use the assets tool with "
        f"action='post_image', image='{handle}'"
        if handle is not None
        else ""
    )
    return f"[{prefix}: {mimetype}, {_human_bytes(size)}{origin}{note}{share}]", image


@dataclass(frozen=True)
class _NamedFiles:
    """The files one call's own arguments name, as they stood before it ran (issue #552).

    `root` is the server's working directory, resolved. `before` maps each named path — resolved,
    symlinks followed — to its `_stamp` from before the call: ``None`` when it did not exist, or
    when it lies outside `root`, where the harness never looks. `names` is every final path
    component those arguments spell, as written and as resolved — the cheap filter that keeps a
    result's link lines, which a page can fill, from each costing a resolve.
    """

    root: Path
    before: Mapping[Path, _Stamp | None]
    names: frozenset[str]


def _named_files(arguments: Mapping[str, object], root: Path) -> _NamedFiles:
    """Stamp every file a call's top-level string `arguments` name under `root`, before it runs.

    Every string argument is a candidate, whatever its key is called — ``filename`` is
    playwright-mcp's spelling, and another server's is its own — because a candidate costs one
    ``lstat`` and is read only if the result then links it *and* the call wrote it. A string that
    is not a path (a URL, a sentence) names no file that changes, and is never read; one longer than
    `_PATH_MAX`, or spanning lines, is not even resolved — no link line could name it.
    """
    root = root.resolve()
    before: dict[Path, _Stamp | None] = {}
    names: set[str] = set()
    for value in arguments.values():
        if not isinstance(value, str) or not value or len(value) > _PATH_MAX or "\n" in value:
            continue
        path = _resolve(root, value)
        if path is not None and path not in before:
            before[path] = _stamp(path) if _inside(path, root) else None
            names.update((_basename(value), path.name))
    return _NamedFiles(root=root, before=before, names=frozenset(names))


def _resolve(root: Path, target: str) -> Path | None:
    """`target` resolved against `root`, symlinks followed; ``None`` for a path the OS rejects.

    Before 3.13 a symlink loop raises `RuntimeError` (after, the path comes back unresolved and the
    open fails ``ELOOP``), and a NUL byte raises `ValueError` on every version — none of them may
    reach the tool call, which would replace a finished call's result with an error.
    """
    try:
        return (root / target).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _inside(path: Path, root: Path) -> bool:
    """Whether resolved `path` is `root` or below it — a test on path *parts*, never a string prefix.

    A prefix would let ``/home/nova-evil`` pass for ``/home/nova``. `Path.is_relative_to` gets that
    right and costs the square of the path's depth on 3.12+ (it walks ``parents``); this is linear.
    """
    return path.parts[: len(root.parts)] == root.parts


def _basename(target: str) -> str:
    """The final component `target` spells, without building a path — a filter, not a resolution."""
    return target.rstrip("/").rpartition("/")[2]


def _stamp(path: Path) -> _Stamp | None:
    """`path`'s type, identity, size and write times (``lstat``), or ``None`` if it is not there."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _linked_paths(texts: Iterable[str], named: _NamedFiles) -> set[Path]:
    """Which of the `named` files some line-ending markdown link in `texts` resolves to (#552).

    Read for *membership* only, so it is lenient about everything the safety does not rest on: every
    ``](`` on a line that ends in ``)`` is tried as the split, so a title carrying ``](`` (an element
    screenshot is titled with the model's own description of the element), a filename carrying one,
    or a title broken across lines cannot hide the link. And it is cheap about everything a page can
    fill: a target is resolved only if it fits `_PATH_MAX` and its final component is one the call's
    arguments spelled, so a result of a thousand link lines the page wrote costs string compares.
    """
    linked: set[Path] = set()
    for text in texts:
        for line in text.splitlines():
            line = line.rstrip()
            if not line.endswith(")"):
                continue
            start = line.find("](", max(0, len(line) - _PATH_MAX - 3))
            while start >= 0:
                target = line[start + 2 : -1]
                if _basename(target) in named.names:
                    path = _resolve(named.root, target)
                    if path in named.before:
                        linked.add(path)
                start = line.find("](", start + 1)
    return linked


def _render_file_links(
    texts: Iterable[str], named: _NamedFiles, store: McpImageStore | None
) -> Iterator[tuple[str, ImageContent | None]]:
    """Render each file the call named and the result links, as if it had arrived inline (#552).

    A named file the result does not link is passed over. One it does link is read by
    `_read_written_image`; an image is rendered by `_render_image_bytes`, exactly as an inline block
    is, naming the file. A linked file that *looks* like an image but could not be shared yields a
    one-line note saying why and a ``WARNING``, so the model is not left to discover it from a failed
    ``post_image``; one that does not (a saved PDF) yields nothing.
    """
    if not named.before:
        return
    linked = _linked_paths(texts, named)
    for path, before in named.before.items():
        if path not in linked:
            continue
        inside = _inside(path, named.root)
        label = str(Path(*path.parts[len(named.root.parts) :])) if inside else str(path)
        outcome = _read_written_image(path, named.root, before)
        if isinstance(outcome, tuple):
            yield _render_image_bytes(*outcome, store, source=label)
        elif path.suffix.lower() in _IMAGE_SUFFIXES:
            _log.warning("MCP tool saved image %r; not shared: %s.", label, outcome)
            yield f"[image file {label!r} not shared: {outcome}]", None


def _read_written_image(path: Path, root: Path, before: _Stamp | None) -> tuple[str, bytes] | str:
    """``(mimetype, bytes)`` for the image the call wrote at `path`, or why it is not shared (#552).

    `path` and `root` are already resolved, so containment (`_inside`) is a test on the real
    location. The ``lstat`` must show a regular file that is not what it was `before` the call,
    so nothing is opened that is not a file this call wrote; the open refuses a final symlink
    swapped in since (``O_NOFOLLOW``) and cannot block or take a terminal (``O_NONBLOCK``,
    ``O_NOCTTY``), and what it opened must still be a regular file. Then the **bytes** decide: the
    leading magic must be a picture format, and the whole must fit `MAX_IMAGE_BYTES` — checked on the
    ``fstat`` so an oversized file is never read, and again on what was read, so one that grew since
    is still bounded. Every OS error is a reason, never a raise.
    """
    if not _inside(path, root):
        return "it is outside the MCP server's working directory"
    after = _stamp(path)
    if after is None:
        return "it was not found"
    if not stat.S_ISREG(after[0]):
        return "it is not a regular file"
    if after == before:
        return "this call did not write it"
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return f"it could not be opened ({exc.strerror or exc})"
    too_large = f"it is too large to show or share (over the {MAX_IMAGE_BYTES}-byte limit)"
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return "it is not a regular file"
        with open(fd, "rb", closefd=False) as handle:
            head = handle.read(_SNIFF_BYTES)
            mimetype = _SNIFFED_IMAGE_TYPES.get(sniff_media_ext(head, ""))
            if mimetype is None:
                return "it is not a recognizable image"
            if info.st_size > MAX_IMAGE_BYTES:
                return too_large
            data = head + handle.read(max(0, MAX_IMAGE_BYTES + 1 - len(head)))
    except OSError as exc:
        return f"it could not be read ({exc.strerror or exc})"
    finally:
        os.close(fd)
    if len(data) > MAX_IMAGE_BYTES:
        return too_large
    return mimetype, data


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
        ``post_image`` path (issue #318) — including one saved to a file the call itself named, read
        from the server's working directory (issue #552). The named files are stamped *before* the
        call, which is what lets the render tell a file this call wrote from one already there.
        """
        root = self._client.workdir
        named = _named_files(kwargs, root) if root is not None else None
        result = self._client.call_tool(self._remote_name, dict(kwargs))
        return _render_tool_result(result, self._images, named)


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
        withheld: ``model-facing name → refusal`` for every tool an active server offered and this
            agent's configuration withholds (issue #553) — what the engine answers a call to a tool
            the model was never offered, so the model is told why rather than "no tool named".
        about: One block per active server that said something about itself or carries an
            operator's ``note`` (issue #553), for the brief's ``mcp`` part.
    """

    tools: list[Tool] = field(default_factory=list)
    manifest: list[tuple[str, str | None]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    clients: list[McpClient] = field(default_factory=list)
    images: McpImageStore | None = None
    withheld: dict[str, str] = field(default_factory=dict)
    about: list[str] = field(default_factory=list)


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
    configs, rejected = load_mcp_configs_report(home)
    resolution.skipped.extend(rejected)
    for config in configs:
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
        offered = [str(spec.get("name", "")) for spec in discovered]
        for spec in discovered:
            if str(spec.get("name", "")) in config.withheld_tools:
                continue
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
        # The withheld tools, settled only now that every name this server registered is known: a
        # sibling whose name sanitizes or truncates onto a withheld tool's is a real, registered tool
        # the engine will dispatch, so a refusal filed under that name would contradict it — in the
        # brief, in the engine, and in `--resolved-config`. (No shipped server does this; the name is
        # the harness's, so the harness keeps it consistent.)
        withheld = [
            remote
            for remote in dict.fromkeys(offered)
            if remote in config.withheld_tools and mcp_tool_name(config.name, remote) not in seen
        ]
        for remote in withheld:
            resolution.withheld[mcp_tool_name(config.name, remote)] = withheld_refusal(
                config.name, remote, waivable=config.withheld_waivable, offered=offered
            )
        resolution.notices.append(
            " ".join(
                [_opt_out_notice(config.name, loaded), *_withholding(config, offered, withheld)]
            )
        )
        about = _about(config, client)
        if about is not None:
            resolution.about.append(about)
        if withheld:
            _log.info(
                "MCP server %r: withheld %s by its configuration (issue #553).",
                config.name,
                ", ".join(withheld),
            )
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
    except Exception:  # noqa: BLE001, S110 - teardown of a failed server must not raise
        pass


def _withholding(
    config: McpServerConfig, offered: Sequence[str], withheld: Sequence[str]
) -> list[str]:
    """What the server's safety line says about each withholdable tool it offers (issue #553).

    Both states are disclosed, because the agent needs to know what it has **and** why. A withheld
    tool gets the refusal the model would get calling it — what, why, who, instead, and the waiver
    when the operator marked it waivable — spelled once, so the brief and the refusal cannot
    disagree. One this agent's configuration hands back is named as present, with why agents do
    not get it by default: a capability deliberately returned is as much a decision as one kept.
    """
    lines = []
    for remote in dict.fromkeys(offered):
        if remote not in WITHHOLDABLE:
            continue
        if remote in withheld:
            lines.append(
                withheld_refusal(
                    config.name, remote, waivable=config.withheld_waivable, offered=offered
                )
            )
            continue
        if remote in config.withheld_tools:
            continue  # its name went to a registered sibling; the tool it names is not this one
        rule = WITHHOLDABLE[remote]
        lines.append(
            f"{mcp_tool_name(config.name, remote)} is available to you, although agents do not get "
            f"it by default: {rule.reason} — withheld by {rule.decider}'s ruling of "
            f"{rule.decided}, {rule.standing}. This agent's configuration hands it back to you."
        )
    return lines


def _about(config: McpServerConfig, client: McpClient) -> str | None:
    """This server's block for the brief's ``mcp`` part, or ``None`` when it has nothing to say.

    Two voices, each labelled with whose it is, because they carry different authority: the config's
    ``note`` describes this box's own setup (a browser's backend, a mailbox's address), and the
    server's ``instructions`` are external text describing the server — worth reading, and no
    instruction from anyone.

    **The server's text is quoted, every line of it.** Attribution by a leading label alone is a
    claim a server can forge: instructions carrying a line that *begins* ``Configuration note`` —
    or a whole ``MCP server 'x' …`` heading — would read as the more-trusted voice, or as another
    server. Inside a ``> `` quote nothing it writes can start a line of the harness's own.
    """
    if config.note is None and client.instructions is None:
        return None
    heading = f"MCP server {config.name!r}"
    if client.server_label:
        heading += f" ({client.server_label})"
    lines = [f"{heading}, whose tools are named {_sanitize(config.name)}__…:"]
    if config.note is not None:
        lines.append(f"Configuration note for this server: {config.note}")
    if client.instructions is not None:
        lines.append("What the server says about itself, quoted:")
        lines.extend(f"> {line}" for line in client.instructions.splitlines())
    return "\n".join(lines)


def _server_label(info: object) -> str | None:
    """``serverInfo``'s name and version as one short line, or ``None`` if it gave neither."""
    if not isinstance(info, dict):
        return None
    words = [str(info[key]).strip() for key in ("name", "version") if info.get(key)]
    label = " ".join(" ".join(word.split()) for word in words if word)
    return _bounded(label, _LABEL_CAP) if label else None


def _bounded(text: str, cap: int) -> str:
    """`text` whole if it fits `cap` characters, else its head and a marker naming the full size."""
    if len(text) <= cap:
        return text
    return f"{text[:cap]} […{len(text) - cap} more characters not shown]"


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
