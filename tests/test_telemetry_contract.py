"""The telemetry contract, enforced: every head and field the harness emits is on `docs/telemetry.md`.

The page is the contract (issue #757, @origin's ruling of 2026-10-31: the fleet's telemetry is
contract-first). This module reads it — its per-head field tables, and its **Known Violations**
table as the one allowlist — and checks the harness against it two ways, because neither alone sees
everything:

- **Statically**, every structured emission site in the package (each `kv(...)` that becomes a
  journal record), with its head and its literal field names. This is what makes a new head or field
  *anywhere* fail the build, including on paths no test drives.
- **At runtime**, a representative call through every provider adapter's own log path, rendered by
  the real SDK against respx (the fixtures the adapters' own suites use). This is the half the
  static scan cannot do: the `llm` line's fields come out of a vendor's usage block, not out of
  literals, so only a rendered line says what a vendor's facts became.

The allowlist is held to the same standard as the code. Each violation row names a **witness**, and
a row whose witness no longer holds fails the build: a departure that disappears must be struck from
the table, so the table is always exactly today's departures — never a list of excuses that outlived
their defects.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from basecradle_harness import (
    AnthropicProvider,
    ImageContent,
    Message,
    OpenAIProvider,
    ProviderError,
    XaiSdkProvider,
)
from basecradle_harness._attribution import log_context_attribution
from basecradle_harness._describer import Describer
from basecradle_harness._engine import _reason
from tests.conftest import FAKE_KEY as OPENAI_KEY
from tests.conftest import out_message, out_web_search_call, responses_body
from tests.test_anthropic import FAKE_KEY as ANTHROPIC_KEY
from tests.test_anthropic import MESSAGES as ANTHROPIC_MESSAGES
from tests.test_anthropic import MODEL as ANTHROPIC_MODEL
from tests.test_anthropic import reply as anthropic_reply
from tests.test_anthropic import text as anthropic_text
from tests.test_google import (  # noqa: F401 - private_key_pem is a fixture
    FAKE_TOKEN,
    GENERATE,
    build,
    grounded,
    key_info,
    private_key_pem,
)
from tests.test_google import body as google_body
from tests.test_google import text as google_text
from tests.test_openrouter import CHAT_URL as OPENROUTER_CHAT
from tests.test_openrouter import _provider as openrouter_provider
from tests.test_openrouter import completion as openrouter_completion
from tests.test_rerank import CHAT_URL as RERANK_CHAT
from tests.test_rerank import completion as rerank_completion
from tests.test_rerank import error as rerank_error
from tests.test_rerank import hits, picks, reranker
from tests.test_xai_sdk import _FakeClient

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "telemetry.md"
SRC = ROOT / "src" / "basecradle_harness"

OPENAI_HOST = "https://api.openai.com/v1"

#: The `kv(...)` calls that render text which is **not** a journal record, with why. Keyed by
#: (module, enclosing function), so a second such call in the same function is still checked.
NOT_JOURNAL = {
    # The dry-run report `basecradle-harness-scrub-palace` prints to stdout for a human to read
    # before approving an apply; its journal line is `palace scrub`, which is checked.
    ("_scrub.py", "render"),
}

#: Heads whose field names are partly computed (a usage block's token counts, the attribution
#: partition, the retry diagnostics), so the static scan cannot list them. Each is rendered by the
#: runtime half below; a computed field on any **other** head fails the scan.
RUNTIME_RENDERED = {"llm", "llm retry", "context attribution"}


# === the page ====================================================================================


@dataclass
class Violation:
    id: str
    form: str
    arg: str


@dataclass
class Contract:
    heads: dict[str, set[str]]
    violations: list[Violation]

    def allows(self, head: str, name: str) -> bool:
        fields = self.heads.get(head, set())
        return name in fields or any(
            pattern.endswith("*") and name.startswith(pattern[:-1]) for pattern in fields
        )


_HEAD = re.compile(r"^### `(?P<head>[^`]+)`$")
_FIELD_ROW = re.compile(r"^\| `(?P<field>[^`]+)` \|")
_VIOLATION_ROW = re.compile(
    r"^\| (?P<id>[TF]\d+) \| .+ \| `(?P<witness>[^`]+)` \| Slice \d \|$", re.MULTILINE
)


def parse_contract(text: str) -> Contract:
    body, _, known = text.partition("\n## Known Violations\n")
    assert known, "the contract page has no Known Violations section"
    heads: dict[str, set[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith("#"):
            match = _HEAD.match(line)
            current = match["head"] if match else None
            if current is not None:
                assert current not in heads, f"the head `{current}` is defined twice"
                heads[current] = set()
            continue
        row = _FIELD_ROW.match(line)
        if current is not None and row:
            heads[current].add(row["field"])
    violations = []
    for row in _VIOLATION_ROW.finditer(known):
        form, _, arg = row["witness"].partition(" ")
        violations.append(Violation(row["id"], form, arg))
    return Contract(heads, violations)


@pytest.fixture(scope="module")
def contract() -> Contract:
    return parse_contract(DOC.read_text(encoding="utf-8"))


# === the static half: every structured emission site ==============================================


@dataclass
class Site:
    module: str
    line: int
    head: str
    fields: set[str]
    computed: bool = False
    prose_reason: bool = False


_LEVELS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
_SLUG = re.compile(r"[a-z0-9_:.-]+")


def _constants(tree: ast.Module) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    found[target.id] = node.value.value
    return found


def _named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _head_of(call: ast.Call, parent: ast.AST | None, constants: dict[str, str]) -> str | None:
    """The head a `kv(...)` call's record wears, read off how the record is built."""
    if isinstance(parent, ast.BinOp) and isinstance(parent.left, ast.JoinedStr):
        for value in parent.left.values:  # f"{head(TEXT, COLOR)} " + kv(...)
            if isinstance(value, ast.FormattedValue) and _named(value.value, "head"):
                return _literal(value.value.args[0], constants)
    logged = (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Attribute)
        and parent.func.attr in _LEVELS
    )
    if not logged:
        return None
    args = parent.args[1:] if parent.func.attr == "log" else parent.args
    fmt = _literal(args[0], constants) if args else None
    if fmt == "%s %s":  # _log.x("%s %s", head(TEXT, COLOR) | HEAD, kv(...))
        first = args[1]
        return _literal(first.args[0] if _named(first, "head") else first, constants)
    if fmt and "%s" in fmt:  # _log.x("memory recall %s", kv(...))
        text = fmt.split("%s")[0].strip()
        return text if "%" not in text else None
    return None


def _local_dict_keys(function: ast.AST, name: str) -> set[str] | None:
    """The keys a local dict named `name` can hold, or ``None`` when they are not all literals."""
    keys: set[str] = set()
    for node in ast.walk(function):
        targets: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    targets.extend(zip(target.elts, node.value.elts))
                else:
                    targets.append((target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets.append((node.target, node.value))
        for target, value in targets:
            if isinstance(target, ast.Name) and target.id == name:
                if not isinstance(value, ast.Dict):
                    return None
                for key in value.keys:
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        return None
                    keys.add(key.value)
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
            ):
                if not (
                    isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)
                ):
                    return None
                keys.add(target.slice.value)
    return keys


def _prose(value: ast.AST) -> bool:
    """Whether a ``reason=`` value is shaped like a sentence rather than a slug."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return not _SLUG.fullmatch(value.value)
    return isinstance(value, ast.JoinedStr) or _named(value, "str")


def scan(source: Path = SRC) -> tuple[list[Site], list[str]]:
    """Every structured emission site, and every `kv(...)` call whose head could not be read."""
    sites: list[Site] = []
    unresolved: list[str] = []
    for path in sorted(source.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _constants(tree)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(function):
                if not _named(call, "kv"):
                    continue
                if (path.name, function.name) in NOT_JOURNAL:
                    continue
                if _enclosing_function(call, parents) is not function:
                    continue
                head = _head_of(call, parents.get(call), constants)
                if head is None:
                    unresolved.append(f"{path.name}:{call.lineno}")
                    continue
                site = Site(path.name, call.lineno, head, set())
                for keyword in call.keywords:
                    if keyword.arg is not None:
                        site.fields.add(keyword.arg)
                        if keyword.arg == "reason" and _prose(keyword.value):
                            site.prose_reason = True
                        continue
                    value = keyword.value
                    if isinstance(value, ast.Dict) and all(
                        isinstance(k, ast.Constant) for k in value.keys
                    ):
                        site.fields.update(k.value for k in value.keys)  # type: ignore[union-attr]
                        continue
                    keys = (
                        _local_dict_keys(function, value.id)
                        if isinstance(value, ast.Name)
                        else None
                    )
                    if keys is None:
                        site.computed = True
                    else:
                        site.fields.update(keys)
                sites.append(site)
    return sites, unresolved


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


@pytest.fixture(scope="module")
def sites() -> list[Site]:
    found, unresolved = scan()
    assert not unresolved, (
        "a kv(...) call whose journal head this test cannot read — build the line through head() "
        "or a literal format, or name it in NOT_JOURNAL with why it is not a journal record: "
        f"{unresolved}"
    )
    return found


# === the runtime half: a representative call through every adapter's log path =====================

#: A strict structured line: a head of words, then one or more ``key=value`` fields. Anything else
#: is a prose line, which carries no fields (`docs/telemetry.md` → Prose Lines).
_KV = r'[a-z_][a-z0-9_]*=(?:"[^"]*"|[^\s"]+)'
_LINE = re.compile(
    rf"^(?:PROBE )?(?P<head>[A-Za-z][A-Za-z_-]*(?: [A-Za-z][A-Za-z_-]*)*?)(?P<fields>(?: {_KV})+)$"
)
_PAIR = re.compile(rf" ({_KV})")


@dataclass
class Line:
    head: str
    fields: dict[str, str]
    message: str


def parse_line(message: str) -> Line | None:
    match = _LINE.match(message)
    if match is None:
        return None
    pairs = (pair.split("=", 1) for pair in _PAIR.findall(match["fields"]))
    return Line(match["head"], {key: value.strip('"') for key, value in pairs}, message)


@dataclass
class Rendered:
    lines: dict[str, list[Line]] = field(default_factory=dict)
    prose: dict[str, list[str]] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    def heads(self, scenario: str, head: str) -> list[Line]:
        return [line for line in self.lines[scenario] if line.head == head]

    def llm(self, scenario: str) -> Line:
        (line,) = self.heads(scenario, "llm")
        return line


SCENARIOS: dict[str, Callable[..., None]] = {}


def scenario(fn: Callable[..., None]) -> Callable[..., None]:
    SCENARIOS[fn.__name__] = fn
    return fn


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


@scenario
def openai_chat(router, **_):
    payload = {
        "id": "chatcmpl-ContractChat0001",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6-sol",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }
    router.post(f"{OPENAI_HOST}/chat/completions").mock(return_value=_ok(payload))
    adapter = OpenAIProvider(
        model="gpt-6-sol", api_key=OPENAI_KEY, base_url=OPENAI_HOST, surface="chat", max_retries=0
    )
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def openai_responses(router, **_):
    payload = responses_body(out_web_search_call(), out_message("The news."))
    payload["model"] = "gpt-6-sol"
    payload["usage"] = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
        "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "output_tokens_details": {"reasoning_tokens": 50},
    }
    router.post(f"{OPENAI_HOST}/responses").mock(return_value=_ok(payload))
    adapter = OpenAIProvider(
        model="gpt-6-sol",
        api_key=OPENAI_KEY,
        base_url=OPENAI_HOST,
        surface="responses",
        max_retries=0,
        builtin_tools=["web_search"],
    )
    adapter.chat([Message.user("news?")])
    adapter.close()


@scenario
def openrouter(router, **_):
    payload = openrouter_completion(content="ok")
    payload["usage"] = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "cost": 0.0042,
        "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "completion_tokens_details": {"reasoning_tokens": 50},
        "server_tool_use_details": {"web_search_requests": 2},
    }
    router.post(OPENROUTER_CHAT).mock(return_value=_ok(payload))
    openrouter_provider(retries_disabled=True).chat([Message.user("hi")])


@scenario
def xai(**_):
    # The numbers are xAI's own documented example (docs.x.ai, Chat Completions reference): the
    # reasoning is reported apart from the completion, and the total counts both.
    usage = SimpleNamespace(
        prompt_tokens=32,
        completion_tokens=9,
        reasoning_tokens=94,
        total_tokens=135,
        cached_prompt_text_tokens=6,
        num_sources_used=3,
        server_side_tools_used=[1],
    )
    response = SimpleNamespace(
        content="ok",
        tool_calls=[],
        citations=[],
        usage=usage,
        cost_usd=0.0012,
        id="0199a2b4-7c3e-7f00-8a1b-2c3d4e5f6a7b",
        finish_reason="REASON_STOP",
    )
    XaiSdkProvider(
        "grok-4.3", api_key="xai-test-0123456789abcdef", client=_FakeClient(response)
    ).chat([Message.user("hi")])


@scenario
def google_chat(router, key_file, **_):
    usage = {
        "promptTokenCount": 1200,
        "candidatesTokenCount": 30,
        "thoughtsTokenCount": 70,
        "toolUsePromptTokenCount": 400,
        "totalTokenCount": 1700,
        "cachedContentTokenCount": 1000,
        "trafficType": "ON_DEMAND",
    }
    router.post(GENERATE).mock(return_value=_ok(google_body(google_text("ok"), usage=usage)))
    adapter = build(key_file)
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def google_search(router, key_file, **_):
    router.post(GENERATE).mock(return_value=grounded(google_text("Sunny.")))
    adapter = build(key_file)
    adapter.search("weather in chicago")
    adapter.close()


_ANTHROPIC_USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 200,
    "cache_creation_input_tokens": 3_000,
    "cache_read_input_tokens": 40_000,
    "cache_creation": {"ephemeral_5m_input_tokens": 3_000, "ephemeral_1h_input_tokens": 0},
    "server_tool_use": {
        "web_search_requests": 2,
        "web_fetch_requests": 1,
        "code_execution_requests": 1,
    },
    "output_tokens_details": {"thinking_tokens": 120},
    "inference_geo": "us",
}


@scenario
def anthropic(router, **_):
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=anthropic_reply(anthropic_text("ok"), used=dict(_ANTHROPIC_USAGE))
    )
    adapter = AnthropicProvider(ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096)
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def anthropic_proxy(router, **_):
    router.post("https://proxy.example.com/v1/messages").mock(
        return_value=anthropic_reply(anthropic_text("ok"))
    )
    adapter = AnthropicProvider(
        ANTHROPIC_MODEL,
        api_key=ANTHROPIC_KEY,
        max_tokens=4096,
        base_url="https://proxy.example.com",
    )
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def anthropic_describer(router, **_):
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=anthropic_reply(
            anthropic_text("A tabby cat asleep on a windowsill."), used=dict(_ANTHROPIC_USAGE)
        )
    )
    adapter = AnthropicProvider(ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096)
    image = ImageContent(url="data:image/png;base64,iVBORw0KGgo=", alt="cat.png")
    assert Describer(adapter, ANTHROPIC_MODEL).describe_images([image])
    adapter.close()


@scenario
def rerank(router, **_):
    router.post(RERANK_CHAT).mock(
        side_effect=[
            httpx.Response(429, json=rerank_error(429, "Rate limit exceeded")),
            _ok(rerank_completion(picks(2, 1))),
        ]
    )
    reranker().rerank("what is the plan", hits(3), 2, surface="turn0")


@scenario
def attribution(**_):
    log_context_attribution([Message.user("hi")], source="timeline:contract")


def _gateway_timeout(reasons: dict[str, str], name: str, call: Callable[[], object]) -> None:
    try:
        call()
    except ProviderError as exc:
        reasons[name] = _reason(exc)
    else:  # pragma: no cover - the mock answers 504 and nothing else
        raise AssertionError(f"{name}: a 504 did not raise")


@scenario
def gateway_timeouts(router, key_file, reasons, **_):
    """An HTTP 504 through every adapter that speaks HTTP, read as the retry vocabulary's word."""
    body = {"error": {"message": "upstream timed out", "type": "timeout", "code": 504}}
    router.post(f"{OPENAI_HOST}/chat/completions").mock(return_value=httpx.Response(504, json=body))
    router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(504, json=body))
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=httpx.Response(
            504, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
        )
    )
    router.post(GENERATE).mock(
        return_value=httpx.Response(
            504, json={"error": {"code": 504, "message": "x", "status": "DEADLINE_EXCEEDED"}}
        )
    )
    hi = [Message.user("hi")]
    openai = OpenAIProvider(
        model="gpt-6-sol", api_key=OPENAI_KEY, base_url=OPENAI_HOST, surface="chat", max_retries=0
    )
    _gateway_timeout(reasons, "openai", lambda: openai.chat(hi))
    _gateway_timeout(
        reasons, "openrouter", lambda: openrouter_provider(retries_disabled=True).chat(hi)
    )
    claude = AnthropicProvider(ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096)
    _gateway_timeout(reasons, "anthropic", lambda: claude.chat(hi))
    gemini = build(key_file)
    _gateway_timeout(reasons, "google", lambda: gemini.chat(hi))
    for adapter in (openai, claude, gemini):
        adapter.close()


def _stub_google_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """`tests/test_google.py`'s auth stub: a fake token on refresh, every real road a failure."""
    import datetime

    from google.oauth2 import service_account

    def refresh(self, request):
        self.token = FAKE_TOKEN
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) + datetime.timedelta(hours=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("a test reached a real Google credential or endpoint")

    monkeypatch.setattr(service_account.Credentials, "refresh", refresh)
    monkeypatch.setattr("google.auth.default", forbidden)
    monkeypatch.setattr("google.auth.transport.requests.Request.__call__", forbidden)
    for var in ("AI_CREDENTIALS_FILE", "AI_LOCATION", "AI_PROJECT", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, private_key_pem) -> Iterator[Rendered]:  # noqa: F811
    """Every scenario's records, parsed — rendered once for the module, offline."""
    import json

    import respx

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("NO_COLOR", "1")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "AI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    _stub_google_auth(monkeypatch)
    path = tmp_path_factory.mktemp("google") / "key.json"
    path.write_text(json.dumps(key_info(private_key_pem)))

    result = Rendered()
    handler = _Collector()
    logger = logging.getLogger("basecradle_harness")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        for name, run in SCENARIOS.items():
            handler.records.clear()
            with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
                run(router=router, key_file=path, reasons=result.reasons)
            messages = [record.getMessage() for record in handler.records]
            parsed = [(message, parse_line(message)) for message in messages]
            result.lines[name] = [line for _, line in parsed if line is not None]
            result.prose[name] = [message for message, line in parsed if line is None]
        yield result
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        monkeypatch.undo()


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# === the behavioral witnesses: one per `test` row of the Known Violations table ===================


def _static_tool_fields(sites: list[Site]) -> set[str]:
    return {name for site in sites if site.head == "tool" for name in site.fields}


def _sums(line: Line) -> bool:
    """Whether the line's ``tokens_total`` is exactly ``tokens_in`` + ``tokens_out``."""
    tokens_in, tokens_out, tokens_total = _tokens(line)
    return tokens_total == tokens_in + tokens_out


def _tokens(line: Line) -> tuple[int, int, int]:
    return tuple(int(line.fields[k]) for k in ("tokens_in", "tokens_out", "tokens_total"))  # type: ignore[return-value]


WITNESSES: dict[str, Callable[[Rendered, list[Site]], bool]] = {
    # A server-side tool writes no `tool` line, and the engine's tool line names no executor.
    "T5": lambda r, s: (
        not r.heads("anthropic", "tool") and "executor" not in _static_tool_fields(s)
    ),
    # The vendor reported its server tools; nothing on any line counts them.
    "T6": lambda r, s: (
        not r.heads("xai", "tool")
        and not r.heads("openrouter", "tool")
        and not any("web_search_requests" in r.llm(n).fields for n in ("xai", "openrouter"))
    ),
    "F1": lambda r, s: any(
        "cache_write_tokens" not in r.llm(n).fields
        for n in ("openai_chat", "openai_responses", "openrouter")
    ),
    "F2": lambda r, s: (
        "cache_write_tokens" in r.llm("anthropic").fields
        and "cache_write_tokens" not in r.llm("anthropic_describer").fields
    ),
    "F3": lambda r, s: any(
        "tokens_reasoning" not in r.llm(n).fields
        for n in ("openai_chat", "openai_responses", "openrouter", "xai", "google_chat")
    ),
    "F4": lambda r, s: all(
        "tokens_reasoning" not in r.llm(n).fields for n in ("anthropic", "anthropic_describer")
    ),
    "F5": lambda r, s: not _sums(r.llm("xai")),
    # 1,200 prompt + 400 tool-use prompt tokens were billed as input.
    "F6": lambda r, s: _tokens(r.llm("google_chat"))[0] < 1_600 or not _sums(r.llm("google_chat")),
    "F7": lambda r, s: "endpoint" in r.llm("google_chat").fields,
    "F8": lambda r, s: "residency" not in r.llm("anthropic").fields,
    "F9": lambda r, s: "cost" in r.llm("anthropic_proxy").fields,
    "F10": lambda r, s: len(set(r.reasons.values())) > 1,
}


# === the checks =================================================================================


def _vocabulary_allowed(contract: Contract) -> tuple[set[str], set[tuple[str, str]]]:
    heads = {v.arg for v in contract.violations if v.form == "head"}
    fields = set()
    for v in contract.violations:
        if v.form == "field":
            head, _, name = v.arg.rpartition(".")
            fields.add((head, name))
    return heads, fields


def _observed(sites: list[Site], rendered: Rendered) -> set[tuple[str, str]]:
    seen = {(site.head, name) for site in sites for name in site.fields}
    for lines in rendered.lines.values():
        seen.update((line.head, name) for line in lines for name in line.fields)
    return seen


def test_the_page_parses_into_heads_fields_and_violations(contract):
    assert {"llm", "llm retry", "tool", "wake start", "wake end", "unspoken"} <= set(contract.heads)
    assert {"provider", "purpose", "tokens_in", "cache_write_tokens", "residency", "billing"} <= (
        contract.heads["llm"]
    )
    assert all(fields for fields in contract.heads.values()), "a head with no field table"
    ids = [v.id for v in contract.violations]
    assert len(ids) == len(set(ids)), "a violation id is used twice"
    assert {v.form for v in contract.violations} <= {
        "head",
        "field",
        "prose",
        "reason-prose",
        "test",
    }


def test_every_head_the_package_emits_is_in_the_contract(contract, sites, rendered):
    allowed_heads, _ = _vocabulary_allowed(contract)
    emitted = {site.head for site in sites}
    emitted.update(line.head for lines in rendered.lines.values() for line in lines)
    stray = sorted(emitted - set(contract.heads) - allowed_heads)
    assert not stray, (
        f"heads outside docs/telemetry.md: {stray}. A new head is a contract change: define it on "
        "the page (and raise it needs-capital first), never ship it on its own."
    )


def test_every_field_on_every_head_is_in_the_contract(contract, sites, rendered):
    allowed_heads, allowed_fields = _vocabulary_allowed(contract)
    stray = sorted(
        f"{head}.{name}"
        for head, name in _observed(sites, rendered)
        if head not in allowed_heads
        and (head, name) not in allowed_fields
        and not contract.allows(head, name)
    )
    assert not stray, (
        f"fields outside docs/telemetry.md: {stray}. A new field is a contract change for every "
        "adapter whose vendor has the datum, decided before it ships."
    )


def test_a_computed_field_set_is_rendered_by_this_test(sites):
    computed = sorted({site.head for site in sites if site.computed} - RUNTIME_RENDERED)
    assert not computed, (
        f"heads whose fields the scan cannot read: {computed}. Build their fields from literals, or "
        "render them in a scenario here and add the head to RUNTIME_RENDERED."
    )


def test_every_runtime_rendered_head_is_rendered(rendered):
    heads = {line.head for lines in rendered.lines.values() for line in lines}
    assert heads >= RUNTIME_RENDERED


def test_a_reason_is_a_slug_unless_the_page_lists_the_head(contract, sites):
    listed = {v.arg for v in contract.violations if v.form == "reason-prose"}
    prose = {site.head for site in sites if site.prose_reason}
    assert prose - listed == set(), f"reason= carries a sentence on {sorted(prose - listed)}"


def test_every_violation_row_is_still_true(contract, sites, rendered):
    """A departure that disappeared must be struck from the table: the table is today's, exactly."""
    observed = _observed(sites, rendered)
    observed_heads = {head for head, _ in observed}
    prose_heads = {site.head for site in sites if site.prose_reason}
    stale: list[str] = []
    for v in contract.violations:
        if v.form == "head":
            assert v.arg not in contract.heads, f"{v.id}: `{v.arg}` is both a head and a violation"
            holds = v.arg in observed_heads
        elif v.form == "field":
            head, _, name = v.arg.rpartition(".")
            assert not contract.allows(head, name), f"{v.id}: `{v.arg}` is in the contract"
            holds = (head, name) in observed
        elif v.form == "prose":
            module, _, quoted = v.arg.partition(" ")
            assert quoted.startswith('"') and quoted.endswith('"'), v.id
            holds = quoted[1:-1] in (SRC / module).read_text(encoding="utf-8")
        elif v.form == "reason-prose":
            holds = v.arg in prose_heads
        else:
            assert v.id in WITNESSES, f"{v.id}: a `test` row with no witness in WITNESSES"
            holds = WITNESSES[v.id](rendered, sites)
        if not holds:
            stale.append(v.id)
    assert not stale, (
        f"violations no longer present: {stale}. Strike each from docs/telemetry.md → Known "
        "Violations in the change that removed it."
    )


def test_every_witness_has_its_row(contract):
    rows = {v.id for v in contract.violations if v.form == "test"}
    assert set(WITNESSES) == rows


def test_every_prose_line_a_scenario_wrote_is_prose_by_design(rendered):
    """A line the parser cannot read is prose; it must not *look* like a structured head."""
    for name, messages in rendered.prose.items():
        for message in messages:
            assert not re.match(r"^[a-z][a-z_ -]* [a-z_]+=", message), (name, message)


def test_the_line_parser_reads_the_grammar_and_rejects_prose():
    line = parse_line('tool name=messages duration=0.12s outcome=ok error="a b=c"')
    assert line is not None and line.head == "tool"
    assert line.fields == {
        "name": "messages",
        "duration": "0.12s",
        "outcome": "ok",
        "error": "a b=c",
    }
    assert parse_line("Wake breaker RESET timeline=0199 held=1.00s").head == "Wake breaker RESET"
    assert parse_line("Anthropic refused the turn for x (category=cyber): no.") is None
    assert parse_line("step 1/40: tool calls (1.20s)") is None
