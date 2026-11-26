"""Shared fixtures. All HTTP is mocked with respx — no test ever touches a model.

The endpoint is a fabricated OpenAI-compatible host; the key is a correctly-shaped fake. respx
mocks the HTTP client at the transport level, so it intercepts each SDK's own client without
any network — the SDK adapter is tested against real, SDK-valid response bodies. Model
responses follow the OpenAI chat-completions / Responses schemas.
"""

import contextlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import sys
import types
from importlib import metadata

import pytest
import respx
import respx.mocks
from respx.mocks import HTTPCoreMocker

from basecradle_harness import OpenAIProvider, _mempalace, _mining, _openai, _venv
from tests._palace_results import got

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """A log line as a human reads it — the ANSI verdict colors stripped back off (issue #414).

    The colors are wrapped around **whole tokens**, so almost every existing assertion (a substring
    search for ``outcome=ok``, for ``wake failed``) matches the colored bytes untouched. The ones
    that need this are the *anchored* ones — ``startswith("wake end")`` — which is exactly the class
    of consumer the token-integrity rule cannot protect. Tests that assert on the color itself read
    the raw record instead; this is for the ones that only ever cared what the line *said*.
    """
    return _ANSI.sub("", text)


#: The repository root, and the directory holding the package: what a child process must import.
REPO = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO / "src"


def child_env(**extra: str) -> dict[str, str]:
    """The environment for a child Python process that imports the harness: the code under test.

    A child inherits the parent's environment but never its ``sys.path``, so a child handed the
    repository root alone imports whatever harness is *installed* — in CI that is this checkout, so
    the gap is invisible, while a suite run from one checkout against another install runs two
    versions at once and fails on the difference (issue #1016). ``src`` comes first so it shadows
    any install; the root follows so a child can import ``tests``; an inherited ``PYTHONPATH`` is
    kept after both, without its empty entries (each would put the child's working directory on its
    path). ``PYTHONPATH`` is this helper's alone, so one passed in `extra` is replaced, never used.
    The suite itself imports ``src`` too (``pythonpath`` in ``pyproject.toml``), so the parent and
    the child run the same code. Every test that spawns a Python child importing the harness uses
    this.
    """
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    path = [str(SRC_ROOT), str(REPO), *filter(None, inherited)]
    return {**os.environ, "NO_COLOR": "1", **extra, "PYTHONPATH": os.pathsep.join(path)}


#: The NOC's log-store canary rule `credential_assignment`, byte for byte as basecradle-noc spells it
#: (`observability/log-canary.json`, read at 4ca50868bd219493c0290b62d89f267915c8d70f, the squash
#: of basecradle-noc#1560, 2026-11-18; `docs/observability.md` §10), and as basecradle-router holds
#: it in its own `tests/conftest.py` (its widened copy moves in basecradle-router#504). Widened by `@origin`'s decision on basecradle-noc#1547
#: (2026-11-17, issue #927): a bare word is a name, and `_BASE` is the one tail after the word. The
#: scrub's definition of a credential's name (`_redaction.CREDENTIAL_NAME`, issue #918) is held
#: to these bytes, and every credential-assignment test uses this pattern as its oracle. Moving it is
#: a change to the NOC's rule: move it here with it.
NOC_CREDENTIAL_ASSIGNMENT = r'[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)(_BASE)?=[^ ",}]{8,}'


@pytest.fixture
def canary() -> re.Pattern[str]:
    """The NOC's `credential_assignment` rule: the oracle no scrubbed text may match."""
    return re.compile(NOC_CREDENTIAL_ASSIGNMENT)


# A fabricated OpenAI-compatible endpoint and a correctly-shaped fake key.
BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
RESPONSES_URL = f"{BASE_URL}/responses"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"


def run_through_engine(tool, **arguments):
    """Run `tool` as a wake does — through the engine's dispatch, which writes the run's ``tool``
    line, with the vendor work its meter recorded (issue #758)."""
    from basecradle_harness._engine import Engine
    from basecradle_harness._tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)
    return Engine(provider=None, tools=registry)._run_tool(tool.name, arguments)


#: The `llm` head, at the start of a message or after its severity token (``WARNING llm …``).
_LLM_HEAD = re.compile(r"(?:^|\s)llm provider=")
#: The NOC's Phase A exclusion (basecradle-noc#1212): an `llm` line with one of these outcomes is
#: not a call, so no series that counts, times or prices calls reads it.
_NOT_A_CALL = re.compile(r"outcome=(retrying|failed)\b")


def is_retry(message: str) -> bool:
    """Is this the `llm` line of an attempt about to be retried (``outcome=retrying``, issue #782)?"""
    text = plain(message)
    return bool(_LLM_HEAD.search(text)) and " outcome=retrying" in text


def is_call(message: str) -> bool:
    """Is this an `llm` line the fleet counts as a **call** — neither retrying nor failed (#782)?

    The NOC's own predicate (basecradle-noc#1212): a line on `` llm provider=`` whose ``outcome`` is
    neither ``retrying`` nor ``failed``. A test that counts calls counts these.
    """
    text = plain(message)
    return bool(_LLM_HEAD.search(text)) and not _NOT_A_CALL.search(text)


def tool_lines(caplog) -> list[str]:
    """Every ``tool`` line a test's capture holds, plain, in order."""
    return [plain(m) for m in (r.getMessage() for r in caplog.records) if m.startswith("tool ")]


class _HTTPCoreBothMocker(HTTPCoreMocker):
    """respx's transport mocker, patching **both** HTTPX families this suite drives.

    ``openai>=3`` moved its client to HTTPX2 — a separate distribution on ``httpcore2``, not a
    new major of ``httpx`` — while everything else the harness speaks HTTP with is still legacy
    ``httpx``: the BaseCradle SDK, the OpenRouter SDK, and the harness's own calls (web_fetch,
    the Grok tools, the ntfy push, asset downloads). A respx router patches **one** family, so a
    default router would silently stop intercepting the model endpoint the moment the SDK moved
    — which is exactly how this arrived: every route still registered, none of them called.

    Both families are patched by one mocker rather than by two routers because the *same* test
    routinely mocks both at once (every wake test mocks the platform **and** the model), and a
    split would make ``assert_all_called`` answer for only half the routes. respx converts each
    intercepted request to an ``httpx.Request`` and hands back an ``httpcore.Response``; HTTPX2
    reads both structurally, so the legacy objects the routes are written against keep working
    on either transport — which is why the suite needed no per-test change.

    Installed as respx's default mocker below, so the ~220 bare ``respx.mock(...)`` call sites
    (and the `router` fixture) get it without naming it — subclass-plus-``DEFAULT_MOCKER`` is the
    extension respx's own maintainer points forked-transport users at (respx#316), and the same
    shape his ``pytest-httpx2`` plugin ships; this one covers both families in a single mocker
    rather than the plugin's httpcore2-only one, which is the part the mixed suite needs.
    """

    name = "httpcore+httpcore2"
    targets = [
        *HTTPCoreMocker.targets,
        "httpcore2._sync.connection.HTTPConnection",
        "httpcore2._sync.connection_pool.ConnectionPool",
        "httpcore2._sync.http_proxy.HTTPProxy",
        "httpcore2._async.connection.AsyncHTTPConnection",
        "httpcore2._async.connection_pool.AsyncConnectionPool",
        "httpcore2._async.http_proxy.AsyncHTTPProxy",
    ]


respx.mocks.DEFAULT_MOCKER = _HTTPCoreBothMocker.name


# The keyword arguments MemPalace's `search_memories` actually accepts. The fake rejects
# anything outside this set, so a kwarg the adapter invents (or one upstream renames) fails
# the suite here rather than raising a TypeError against the real library in production.
_SEARCH_KWARGS = {"n_results", "candidate_strategy", "max_distance"}

# The same for `convo_miner.file_conversation_exchange` (MemPalace 3.9.0 through 3.11.0, every
# argument after the collection keyword-only), so the observe's call is held to its real signature.
_FILE_KWARGS = {
    "wing",
    "room",
    "text",
    "source_file",
    "agent",
    "authored_at",
    "extra_metadata",
}


#: The published ``mcp-mail-server@2.0.2`` ``tools/list`` response, captured verbatim by running
#: the tarball's own server over stdio (issue #496). Its `send_email` schema is the one that killed
#: every wake of @briggs on the native xAI adapter, so it is the fixture three test files reason
#: about — and it lives here so nobody re-types it into a fourth. Provenance is inside the file.
MAIL_SERVER_TOOLS = json.loads(
    (pathlib.Path(__file__).parent / "data" / "mcp_mail_server_2_0_2_tools.json").read_text()
)


def mail_tool(name: str) -> dict:
    """One tool from that capture, by its server-side name."""
    return next(tool for tool in MAIL_SERVER_TOOLS["tools"] if tool["name"] == name)


#: Every ``HARNESS_MEMORY_PROVIDER`` value that selects today's MemPalace adapter (issue #1010).
#: Written out, not read from `_memory_provider`, so the contract is pinned here: a test that
#: selects MemPalace by value takes `mempalace_selector` and runs once under each name, which is
#: the proof the second name behaves identically. `test_memory_provider.py` checks this tuple
#: against the module's own set, so a third name cannot land without its paired runs.
MEMPALACE_SELECTORS = ("mempalace", "mempalace-previous")


@pytest.fixture(params=MEMPALACE_SELECTORS)
def mempalace_selector(request) -> str:
    """One name that selects the MemPalace adapter; the test runs once per name."""
    return request.param


@pytest.fixture
def mempalace_not_installed(monkeypatch):
    """The ``mempalace`` distribution absent, whatever this venv holds (issue #1024).

    A MemPalace fake stands in at ``sys.modules``, but the harness reads the extra's version off
    the installed distribution's metadata (`importlib.metadata.version`), which no fake reaches. A
    test whose premise is "the extra is not installed" takes this fixture, so the premise holds in
    a venv with every extra installed too, rather than only where CI happens to leave it out. Every
    other distribution reads as installed.
    """
    real = metadata.version

    def version(distribution: str) -> str:
        if distribution == "mempalace":
            raise metadata.PackageNotFoundError(distribution)
        return real(distribution)

    monkeypatch.setattr(metadata, "version", version)


@pytest.fixture
def fake_mempalace(monkeypatch):
    """Install fake ``mempalace.convo_miner`` / ``.searcher`` / ``.palace`` / ``.normalize``.

    Returns the first two fakes so a test can assert how the adapter called them.
    ``file_conversation_exchange`` records each drawer it is asked to file in ``filed`` (its
    keyword arguments, plus the palace the collection was opened on, under ``palace``, its id under
    ``id`` and the text MemPalace stores, stripped, under ``stored``) and answers MemPalace's way: a
    drawer id, or ``None`` for text that is blank once stripped. ``mine_convos``
    records its args in ``calls`` (the observe never calls it; the palace check's forecast does).
    ``palace.mine_lock`` records each path it locks in ``convo_miner.locked``, and a writable
    collection answers ``get(where={"source_file": …})`` from ``filed`` and records its deletes in
    ``convo_miner.deleted`` (a test makes either raise by setting ``convo_miner.refuse``). For the
    adapter issue #1011 built it answers the way MemPalace 3.11's typed results do (attribute access
    only, issue #1026): a ``where`` on that adapter's two keys, under ``$and``, from each
    filed drawer's ``extra_metadata``, and a ``where_document`` ``$contains`` on the stored text in a
    wing. ``palace.MineAlreadyRunning`` is the library's lock refusal, and
    ``convo_miner.busy`` is how many filings raise it before one goes through. ``search_memories`` records the
    kwargs it was *passed* (not their defaults — the `max_distance` tests turn on that
    distinction) and returns whatever the test stashes on it.

    The palace is the fleet's (issue #625): MemPalace 3.10.0, a drawers collection that declares
    the cosine metric, so `search` passes ``max_distance=2.0`` by default. A test that needs
    another palace sets ``searcher.metric`` (the collection's ``distance_metric``; an exception
    instance is raised on read) or patches ``_mempalace.mempalace_version``. ``searcher.opened``
    counts how often the collection was opened.
    """
    convo_miner = types.ModuleType("mempalace.convo_miner")
    convo_miner.calls = []
    convo_miner.filed = []
    convo_miner.locked = []
    convo_miner.deleted = []
    convo_miner.refuse = None
    convo_miner.busy = 0

    def mine_convos(convo_dir, palace_path, **kwargs):
        convo_miner.calls.append((convo_dir, palace_path, kwargs))

    def file_conversation_exchange(collection, **kwargs):
        unknown = set(kwargs) - _FILE_KWARGS
        assert not unknown, f"MemPalace's file_conversation_exchange takes no {sorted(unknown)}"
        if convo_miner.busy:
            convo_miner.busy -= 1
            raise MineAlreadyRunning("palace is held by another process")
        if not (kwargs["text"] or "").strip():
            return None
        drawer = f"drawer_{len(convo_miner.filed) + 1}"
        stored = kwargs["text"].strip()  # MemPalace strips the text it files
        convo_miner.filed.append(
            {**kwargs, "palace": collection.palace, "id": drawer, "stored": stored}
        )
        return drawer

    convo_miner.mine_convos = mine_convos
    convo_miner.file_conversation_exchange = file_conversation_exchange
    convo_miner.detect_convo_room = lambda content: "general"

    searcher = types.ModuleType("mempalace.searcher")
    searcher.result = {"results": []}
    searcher.queries = []

    def search_memories(query, palace_path, **kwargs):
        unknown = set(kwargs) - _SEARCH_KWARGS
        assert not unknown, f"MemPalace's search_memories takes no {sorted(unknown)} kwarg"
        searcher.queries.append((query, palace_path, kwargs))
        return searcher.result

    searcher.search_memories = search_memories
    searcher.metric = "cosine"
    searcher.opened = 0

    class Collection:
        @property
        def distance_metric(self):
            if isinstance(searcher.metric, BaseException):
                raise searcher.metric
            return searcher.metric

    class MineAlreadyRunning(RuntimeError):
        """MemPalace's lock refusal (`mempalace.palace.MineAlreadyRunning`)."""

    def Result(drawers):
        """A typed result, read by attribute only (`tests._palace_results`)."""
        return got(
            ids=[d["id"] for d in drawers],
            documents=[d["stored"] for d in drawers],
            metadatas=[
                {
                    "source_file": d["source_file"],
                    "wing": d["wing"],
                    "room": d["room"],
                    **(d.get("extra_metadata") or {}),
                }
                for d in drawers
            ],
        )

    def _matches(drawer, where):
        if "$and" in where:
            return all(_matches(drawer, clause) for clause in where["$and"])
        ((field, wanted),) = where.items()
        meta = {**drawer, **(drawer.get("extra_metadata") or {})}
        return meta.get(field) == wanted

    class WritableCollection:
        def __init__(self, palace_path):
            self.palace = palace_path

        def get(self, *, where, include, where_document=None):
            if convo_miner.refuse is not None:
                raise convo_miner.refuse
            found = [d for d in convo_miner.filed if _matches(d, where)]
            if where_document is not None:
                found = [d for d in found if where_document["$contains"] in d["stored"]]
            return Result(found)

        def delete(self, *, ids):
            if convo_miner.refuse is not None:
                raise convo_miner.refuse
            convo_miner.deleted.extend(ids)

    palace = types.ModuleType("mempalace.palace")

    def get_collection(palace_path, **kwargs):
        if not kwargs:  # the observe's own open: MemPalace's default, which creates the palace
            return WritableCollection(palace_path)
        assert kwargs == {"create": False, "read_only": True}, kwargs
        searcher.opened += 1
        return Collection()

    @contextlib.contextmanager
    def mine_lock(source_file):
        convo_miner.locked.append(source_file)
        yield

    palace.get_collection = get_collection
    palace.mine_lock = mine_lock
    palace.MineAlreadyRunning = MineAlreadyRunning

    normalize = types.ModuleType("mempalace.normalize")
    normalize.normalize_conversations = lambda path: [pathlib.Path(path).read_text("utf-8")]

    parent = types.ModuleType("mempalace")
    parent.convo_miner = convo_miner
    parent.searcher = searcher
    parent.palace = palace
    parent.normalize = normalize

    monkeypatch.setitem(sys.modules, "mempalace", parent)
    monkeypatch.setitem(sys.modules, "mempalace.convo_miner", convo_miner)
    monkeypatch.setitem(sys.modules, "mempalace.searcher", searcher)
    monkeypatch.setitem(sys.modules, "mempalace.palace", palace)
    monkeypatch.setitem(sys.modules, "mempalace.normalize", normalize)
    monkeypatch.setattr(_mempalace, "mempalace_version", lambda: "3.10.0")
    return convo_miner, searcher


@pytest.fixture(autouse=True)
def _isolated_commands_dir(tmp_path_factory, monkeypatch):
    """Make the harness's commands directory (`_venv.commands_dir`) inside the test's own temp tree.

    Every spawn through `_venv.spawn_env` makes one, in the system temp directory by default, and
    the process that made it removes it only at exit: a test must leave nothing outside its own
    ``tmp_path`` (CLAUDE.md, "Whatever creates, cleans up"), and must not reuse the directory an
    earlier test made against a different interpreter directory.
    """
    monkeypatch.setattr(_venv, "_PARENT", str(tmp_path_factory.mktemp("commands")))
    _venv._reset()
    yield
    _venv._reset()


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path_factory, monkeypatch):
    """Point the config home at an empty temp dir so no test reads the real ``$HOME``.

    The charter is now sourced from files under ``$BASECRADLE_CONFIG_HOME`` (default
    ``$HOME/.config/basecradle``). Without this, a dev/CI box that has ever run
    ``basecradle-harness-install`` would leak its real charter into every ``from_env``
    test. Pinning the var to a fresh, empty dir makes the whole suite hermetic; a test
    that exercises the config home overrides it (or passes an explicit ``home=``).
    """
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))
    # The scaffolding catalog reads that home's charter and memoizes the result (`_mining.catalog`),
    # so a catalog built under one test's config home would answer for the next one's.
    _mining._reset()


@pytest.fixture(autouse=True)
def _no_ambient_rerank_config(monkeypatch):
    """Clear every ``HARNESS_MEMPALACE_RERANK_*`` variable the running shell exported.

    A test sets the reranker variables it means, and only those. Since issue #780 two of them are
    mutually exclusive (an order beside a providers list is ``config:conflicting_routing``), so an
    order exported on a fleet box would turn every test that sets a providers list into a dead
    reranker — the same leak `_isolated_config_home` closes for the charter.
    """
    for name in [var for var in os.environ if var.startswith("HARNESS_MEMPALACE_RERANK_")]:
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _no_price_list_from_another_test():
    """Forget every models list an earlier test read (issue #874).

    The last list each host and key answered is kept for the life of the process, so a catalog whose
    own read fails can price from it (`_openai._LAST_LISTS`). The test process is one process, so
    without this a test that reads a list would price the calls of every later test whose read fails.
    """
    _openai._LAST_LISTS.clear()
    yield
    _openai._LAST_LISTS.clear()


class _StructuredLineGuard(logging.Handler):
    """Every structured line a test makes the harness log, rendered through the real formatter.

    `ScrubbingFormatter` quotes every argument of a harness record that is not `Rendered` (issue
    #912), so a structured line passed in as plain text — ``"%s", f"{head(...)} " + kv(...)`` — would
    reach the journal quoted, its head inside a string where no reader finds it, while every test
    that reads the record through ``caplog`` (which formats with its own formatter) stayed green.
    This handler is that missing reader: a record whose message parses as a structured line must
    render exactly as it reads, apart from the scrub the formatter always ran.
    """

    def __init__(self) -> None:
        super().__init__()
        self.broken: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        from basecradle_harness._observability import (
            RECORD_BYTES,
            ScrubbingFormatter,
            _is_harness,
            scrub_line,
        )
        from tests._line_expectations import parse_line

        if not _is_harness(record.name):
            return
        try:
            raw = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad record is logging's own error, not this guard's
            return
        if "\n" in raw or parse_line(plain(raw)) is None:
            return
        copy = logging.makeLogRecord(
            {**record.__dict__, "exc_info": None, "exc_text": None, "stack_info": None}
        )
        rendered = ScrubbingFormatter("%(message)s").format(copy)
        expected = scrub_line(raw)
        if len(expected.encode()) > RECORD_BYTES:
            return  # split under the record budget on purpose (`test_one_line_records.py`)
        if rendered != expected:
            self.broken.append((raw, rendered))


@pytest.fixture(autouse=True)
def _structured_lines_render_as_written():
    guard = _StructuredLineGuard()
    logger = logging.getLogger("basecradle_harness")
    logger.addHandler(guard)
    try:
        yield
    finally:
        logger.removeHandler(guard)
    assert not guard.broken, (
        f"a structured line the formatter would not render as written: {guard.broken}"
    )


# `scripts/isolated_home.py` is repo tooling, not shipped code, so it is loaded by path.
_ISOLATED_HOME_SPEC = importlib.util.spec_from_file_location(
    "isolated_home", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "isolated_home.py"
)
isolated_home = importlib.util.module_from_spec(_ISOLATED_HOME_SPEC)
_ISOLATED_HOME_SPEC.loader.exec_module(isolated_home)


@pytest.fixture(scope="session")
def _mempalace_home():
    """One throwaway ``$HOME`` for every real-MemPalace test in the session, removed at its end."""
    with isolated_home.isolated_home() as home:
        yield home


@pytest.fixture(autouse=True)
def _isolated_mempalace_home(request, monkeypatch):
    """Run every test marked ``mempalace`` with ``HOME`` pointed at a throwaway home (issue #630).

    Every write to a real palace leaves a lock file in ``$HOME/.mempalace/locks`` that MemPalace
    never removes, and each test palace has a fresh path, so before this the real-palace suite left
    one per palace in the developer's home on every run. Keyed on the marker rather than requested
    by name, so a new real-palace test is isolated without anyone remembering to ask. It covers
    writes made inside a test; a wider-scoped fixture that writes a palace must point ``HOME`` at
    ``_mempalace_home`` itself. ``test_isolated_home_real.py`` proves the lock lands here.
    """
    if request.node.get_closest_marker("mempalace") is not None:
        monkeypatch.setenv("HOME", str(request.getfixturevalue("_mempalace_home")))


@pytest.fixture
def router():
    """A respx router; routes are matched by absolute URL."""
    with respx.mock(assert_all_called=True) as r:
        yield r


@pytest.fixture
def provider():
    """The openai-SDK adapter on its **chat** surface, pointed at the fabricated endpoint.

    ``max_retries=0`` keeps error tests single-shot and deterministic (the SDK otherwise
    retries 429/5xx with backoff).
    """
    p = OpenAIProvider(
        model="gpt-4o", api_key=FAKE_KEY, base_url=BASE_URL, surface="chat", max_retries=0
    )
    yield p
    p.close()


@pytest.fixture
def responses_provider():
    """The openai-SDK adapter on its **responses** surface (web_search-capable)."""
    p = OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        surface="responses",
        max_retries=0,
    )
    yield p
    p.close()


def completion(*, content=None, tool_calls=None, finish_reason="stop"):
    """A chat-completions response body, OpenAI-shaped (SDK-valid)."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fake0001",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def wire_tool_call(*, id, name, arguments):
    """A tool call as it appears on the wire — `arguments` is a JSON string."""
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# --- Responses API shapes (the Responses surface) ----------------------------


def responses_body(*output):
    """A Responses-API response body wrapping the given `output` items (SDK-valid).

    Carries the fields the ``openai`` SDK's ``Response`` model needs (``created_at``,
    ``parallel_tool_calls``, ``tool_choice``, ``tools``) so the SDK adapter validates it; the
    extra keys are harmless to the xAI httpx adapter, which parses the raw JSON directly.
    """
    return {
        "id": "resp-fake0001",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-mini",
        "output": list(output),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def out_message(text, *, annotations=None):
    """A Responses `message` output item: assistant text with optional citations.

    ``annotations`` is always present (default ``[]``) — the SDK's ``output_text`` content
    part requires the field.
    """
    content = {"type": "output_text", "text": text, "annotations": annotations or []}
    return {
        "id": "msg-fake0001",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content],
    }


def out_function_call(*, call_id, name, arguments):
    """A Responses `function_call` output item — a custom tool the harness must run."""
    return {
        "id": "fc-fake0001",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def out_web_search_call(*, query="latest news"):
    """A Responses `web_search_call` output item — resolved server-side, never run here."""
    return {
        "id": "ws-fake0001",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": query},
    }


def url_citation(*, url, title, start_index=0, end_index=1):
    """A `url_citation` annotation, as web_search attaches to message text."""
    return {
        "type": "url_citation",
        "start_index": start_index,
        "end_index": end_index,
        "url": url,
        "title": title,
    }


def out_code_interpreter_call(*, code, container_id="cntr_fake0001"):
    """A Responses `code_interpreter_call` output item — code run server-side (issue #172)."""
    return {
        "id": "ci-fake0001",
        "type": "code_interpreter_call",
        "status": "completed",
        "container_id": container_id,
        "code": code,
        "outputs": [],
    }


def container_file_citation(*, file_id, filename, container_id="cntr_fake0001"):
    """A `container_file_citation` annotation — a file the Code Interpreter produced."""
    return {
        "type": "container_file_citation",
        "container_id": container_id,
        "file_id": file_id,
        "filename": filename,
        "start_index": 0,
        "end_index": 1,
    }
