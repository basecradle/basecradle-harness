"""The telemetry contract, enforced: every head and field the harness emits is on `docs/telemetry.md`.

The page is the contract (issue #757, @origin's ruling of 2026-10-31: the fleet's telemetry is
contract-first). This module reads it — its per-head field tables, and its **Known Violations**
table as the one allowlist — and checks the harness against it two ways, because neither alone sees
everything:

- **Statically**, every structured emission site in the package: each `kv(...)` that becomes a
  journal record, each ``%``-formatted line shaped like one, and the ``extra=`` literals handed to
  the `llm` emitters — with its head and its literal field names. This is what makes
  a new head or field *anywhere* fail the build, including on paths no test drives.
- **At runtime**, a representative call through every provider adapter's own log path, rendered by
  the real SDK against respx (the fixtures the adapters' own suites use). This is the half the
  static scan cannot do: the `llm` line's fields come out of a vendor's usage block, not out of
  literals, so only a rendered line says what a vendor's facts became.

The allowlist is held to the same standard as the code. Each violation row names a **witness**, and
a row whose witness no longer holds fails the build: a departure that disappears must be struck from
the table, so the table is always exactly today's departures — never a list of excuses that outlived
their defects. The witness forms are documented on the page, above the table.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

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
from basecradle_harness._engine import _REASONS, Engine, _reason
from basecradle_harness._exceptions import ProviderResponseError, ProviderServerError
from basecradle_harness._observability import (
    _VERDICT_VALUES,
    BLUE,
    DELIVERY_ID_ENV,
    GREEN,
    NO_COLOR_ENV,
    RED,
    RESET,
    YELLOW,
    ComputedCost,
    at_step,
    in_wake,
    kv,
    metered,
)
from basecradle_harness._retry import fault_slug
from basecradle_harness._tools import Tool, ToolRegistry
from tests.conftest import FAKE_KEY as OPENAI_KEY
from tests.conftest import (
    out_code_interpreter_call,
    out_message,
    out_web_search_call,
    responses_body,
    wire_tool_call,
)
from tests.test_anthropic import FAKE_KEY as ANTHROPIC_KEY
from tests.test_anthropic import MESSAGES as ANTHROPIC_MESSAGES
from tests.test_anthropic import MODEL as ANTHROPIC_MODEL
from tests.test_anthropic import _no_stray_anthropic_env  # noqa: F401 - autouse fixture
from tests.test_anthropic import reply as anthropic_reply
from tests.test_anthropic import text as anthropic_text
from tests.test_generation_id import OPENROUTER_CHAT_URL as GENERATION_CHAT_URL
from tests.test_generation_id import _openrouter as _generation_openrouter
from tests.test_generation_id import _openrouter_body
from tests.test_google import (  # noqa: F401 - fixtures: the Google suite's own key and auth stub
    GENERATE,
    _no_real_google_auth,
    build,
    grounded,
    key_file,
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
XAI_HOST = "https://api.x.ai/v1"

#: Functions whose `kv(...)` calls render text that is **not** a journal record, with why. Every
#: `kv` call inside a listed function is skipped, so a function belongs here only if none of its
#: calls is a journal line.
NOT_JOURNAL = {
    # The dry-run report `basecradle-harness-scrub-palace` prints to stdout for a human to read
    # before approving an apply; its journal line is `palace scrub`, written by `main`.
    ("_scrub.py", "render"),
}

#: Heads whose field names are partly computed (a usage block's token counts, the attribution
#: partition, a caller's own extras), so the static scan cannot list all of them. Each is rendered
#: by the runtime half below; a computed field on any **other** head fails the scan.
RUNTIME_RENDERED = {"llm", "context attribution"}

#: The emitters that take an ``extra=`` mapping of trailing fields, and the head each writes.
EXTRA_EMITTERS = {
    "log_llm_call": "llm",
    "log_llm_retry": "llm",
    "Retry": "llm",
    "log_brain_failure": "llm",
}

#: Field names the NOC extracts off **every** line, whatever its head. On a prose line they still
#: feed a column, so a prose format string carrying one must be named by a `prose` row.
GLOBALLY_EXTRACTED = ("provider", "source")


# === the page ====================================================================================


@dataclass
class Violation:
    id: str
    form: str
    arg: str
    scope: str | None = None
    fields: frozenset[str] = frozenset()
    module: str = ""
    text: str = ""
    until: str | None = None


@dataclass
class Contract:
    heads: dict[str, set[str]]
    violations: list[Violation]

    def allows(self, head: str, name: str) -> bool:
        fields = self.heads.get(head, set())
        return name in fields or any(
            pattern.endswith("*") and name.startswith(pattern[:-1]) for pattern in fields
        )

    def of(self, form: str) -> list[Violation]:
        return [v for v in self.violations if v.form == form]


_HEAD = re.compile(r"^### `(?P<head>[^`]+)`$")
_FIELD_ROW = re.compile(r"^\| `(?P<field>[^`]+)` \|")
_VIOLATION_ROW = re.compile(
    r"^\| (?P<id>[TFS]\d+) \| .+ \| `(?P<witness>[^`]+)` \| (?:Slice \d|\[#\d+\]\(\S+\)) \|$",
    re.MULTILINE,
)
_PROSE_WITNESS = re.compile(r'^(?P<module>\S+) "(?P<text>.*)"(?: until (?P<until>.+))?$')
_FIELD_WITNESS = re.compile(r"^(?P<head>.+)\.(?P<field>[a-z_]+)(?: @(?P<scope>[a-z_]+))?$")


def _violation(row_id: str, witness: str) -> Violation:
    form, _, arg = witness.partition(" ")
    v = Violation(row_id, form, arg)
    if form == "head":
        head, _, names = arg.partition(": ")
        v.arg, v.fields = head, frozenset(names.split())
        assert v.fields, f"{row_id}: a `head` witness names the fields it allows"
    elif form == "field":
        match = _FIELD_WITNESS.match(arg)
        assert match, f"{row_id}: unreadable field witness {arg!r}"
        v.arg, v.scope = f"{match['head']}.{match['field']}", match["scope"]
    elif form == "prose":
        match = _PROSE_WITNESS.match(arg)
        assert match, f"{row_id}: unreadable prose witness {arg!r}"
        v.module, v.text, v.until = match["module"], match["text"], match["until"]
    return v


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
    violations = [_violation(row["id"], row["witness"]) for row in _VIOLATION_ROW.finditer(known)]
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
    unescaped: bool = False


_LEVELS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
_SLUG = re.compile(r"[a-z0-9_:.-]+")
#: A ``%``-formatted line shaped like a structured one: a head of words, then ``key=`` fields.
_RAW = re.compile(r"^(?P<head>[a-z][a-z_-]*(?: [a-z][a-z_-]*)*?)(?P<fields>(?: [a-z_]+=\S+)+)$")


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level string constants — the only names a head is ever spelled through."""
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Constant):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    found[target.id] = node.value.value
    return found


def _called(node: ast.AST) -> str | None:
    """The name a call is made through: ``f(...)`` → ``f``, ``obj.f(...)`` → ``f``."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _log_args(call: ast.AST) -> list[ast.expr] | None:
    """A logging call's arguments after the level, or ``None`` when `call` is not one."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
        return None
    if call.func.attr not in _LEVELS:
        return None
    return call.args[1:] if call.func.attr == "log" else call.args


def _head_of(call: ast.Call, parent: ast.AST | None, constants: dict[str, str]) -> str | None:
    """The head a `kv(...)` call's record wears, read off how the record is built."""
    if isinstance(parent, ast.BinOp) and isinstance(parent.left, ast.JoinedStr):
        for value in parent.left.values:  # f"{head(TEXT, COLOR)} " + kv(...)
            if isinstance(value, ast.FormattedValue) and _named(value.value, "head"):
                return _literal(value.value.args[0], constants)
    args = _log_args(parent) if parent is not None else None
    if not args:
        return None
    fmt = _literal(args[0], constants)
    if fmt == "%s %s":  # _log.x("%s %s", head(TEXT, COLOR) | HEAD, kv(...))
        first = args[1]
        return _literal(first.args[0] if _named(first, "head") else first, constants)
    if fmt and "%s" in fmt:  # _log.x("memory recall %s", kv(...))
        text = fmt.split("%s")[0].strip()
        return text if "%" not in text else None
    return None


class _Module:
    """One source module, parsed once, with the lookups the scan needs."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.constants = _module_constants(self.tree)
        self.parents = {
            child: node for node in ast.walk(self.tree) for child in ast.iter_child_nodes(node)
        }
        self.functions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def enclosing(self, node: ast.AST) -> ast.AST | None:
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    def calls_to(self, name: str) -> list[ast.Call]:
        return [n for n in ast.walk(self.tree) if isinstance(n, ast.Call) and _called(n) == name]


def _assigned(function: ast.AST, name: str) -> list[ast.expr]:
    """Every value a local `name` is assigned in `function` (tuple unpacking included)."""
    values: list[ast.expr] = []
    for node in ast.walk(function):
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    pairs.extend(zip(target.elts, node.value.elts))
                else:
                    pairs.append((target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs.append((node.target, node.value))
        values.extend(v for t, v in pairs if isinstance(t, ast.Name) and t.id == name)  # type: ignore[misc]
    return values


def _dict_keys(value: ast.AST, function: ast.AST | None) -> set[str] | None:
    """The literal keys a dict expression can hold, or ``None`` when they cannot all be read."""
    if isinstance(value, ast.Dict):
        keys: set[str] = set()
        for key, item in zip(value.keys, value.values):
            if key is None:  # a ``**splat`` inside the literal
                inner = _dict_keys(item, function)
                if inner is None:
                    return None
                keys |= inner
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            else:
                return None
        return keys
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):  # ``(x or {})``
        keys = set()
        for item in value.values:
            inner = _dict_keys(item, function)
            if inner is None:
                return None
            keys |= inner
        return keys
    if isinstance(value, ast.Name) and function is not None:
        keys = set()
        assignments = _assigned(function, value.id)
        subscripts = [
            n.slice.value
            for n in ast.walk(function)
            if isinstance(n, ast.Subscript)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == value.id
            and isinstance(n.slice, ast.Constant)
        ]
        if not assignments and not subscripts:
            return None
        for item in assignments:
            inner = _dict_keys(item, function)
            if inner is None:
                return None
            keys |= inner
        return keys | set(subscripts)
    return None


def _parameters(function: ast.AST) -> list[str]:
    args = function.args  # type: ignore[attr-defined]
    return [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]


def _prose(value: ast.AST, module: _Module, depth: int = 3) -> bool:
    """Whether a ``reason=`` value is shaped like a sentence rather than a slug.

    A literal is read; an f-string or an exception's ``str()`` is prose; a local is read through its
    assignments; a parameter is read through every call to its function in the same module, a few
    levels up. Anything else (an attribute, a subscript, a loop variable) is not judged.
    """
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return not _SLUG.fullmatch(value.value)
    if isinstance(value, ast.JoinedStr) or _named(value, "str"):
        return True
    if not isinstance(value, ast.Name) or depth == 0:
        return False
    function = module.enclosing(value)
    if function is None:
        return False
    if any(_prose(v, module, depth - 1) for v in _assigned(function, value.id)):
        return True
    parameters = _parameters(function)
    if value.id not in parameters:
        return False
    index = parameters.index(value.id)
    for call in module.calls_to(function.name):  # type: ignore[attr-defined]
        offset = 1 if parameters[:1] == ["self"] and isinstance(call.func, ast.Attribute) else 0
        passed = next((k.value for k in call.keywords if k.arg == value.id), None)
        if passed is None and 0 <= index - offset < len(call.args):
            passed = call.args[index - offset]
        if passed is not None and _prose(passed, module, depth - 1):
            return True
    return False


def _kv_sites(module: _Module) -> tuple[list[Site], list[str]]:
    sites: list[Site] = []
    unresolved: list[str] = []
    for call in ast.walk(module.tree):
        if not _named(call, "kv"):
            continue
        function = module.enclosing(call)
        if function is not None and (module.name, function.name) in NOT_JOURNAL:  # type: ignore[attr-defined]
            continue
        head = _head_of(call, module.parents.get(call), module.constants)
        if head is None:
            unresolved.append(f"{module.name}:{call.lineno}")
            continue
        site = Site(module.name, call.lineno, head, set())
        for keyword in call.keywords:
            if keyword.arg is not None:
                site.fields.add(keyword.arg)
                site.prose_reason |= keyword.arg == "reason" and _prose(keyword.value, module)
                continue
            keys = _dict_keys(keyword.value, function)
            if keys is None:
                site.computed = True
            else:
                site.fields |= keys
        sites.append(site)
    return sites, unresolved


def _raw_sites(module: _Module) -> list[Site]:
    """Lines spelled as ``%``-format strings that read as structured — never through `kv`."""
    sites = []
    for call in ast.walk(module.tree):
        args = _log_args(call)
        fmt = _literal(args[0], module.constants) if args else None
        if not fmt or not args or any(_named(a, "kv") for a in args[1:]):
            continue
        match = _RAW.match(fmt)
        if match:
            names = {pair.split("=", 1)[0] for pair in match["fields"].split()}
            sites.append(Site(module.name, call.lineno, match["head"], names, unescaped=True))
    return sites


def _extra_sites(module: _Module) -> list[Site]:
    """The ``extra=`` literals handed to the `llm` emitters, read as fields."""
    sites = []
    for call in ast.walk(module.tree):
        head = EXTRA_EMITTERS.get(_called(call) or "")
        if head is None:
            continue
        for keyword in call.keywords:  # type: ignore[attr-defined]
            if keyword.arg == "extra":
                keys = _dict_keys(keyword.value, module.enclosing(call))
                if keys:
                    sites.append(Site(module.name, call.lineno, head, keys))
    if module.name == "_retry.py":  # `diagnostics` rides every failed attempt's `llm` line
        (function,) = [f for f in module.functions if f.name == "diagnostics"]  # type: ignore[attr-defined]
        (returned,) = [n.value for n in ast.walk(function) if isinstance(n, ast.Return)]
        keys = _dict_keys(returned, function) or set()
        sites.append(Site(module.name, function.lineno, "llm", set(keys)))  # type: ignore[attr-defined]
    return sites


def _prose_formats(module: _Module) -> list[tuple[int, str]]:
    """Prose format strings that carry a field the NOC extracts off every line."""
    found = []
    for call in ast.walk(module.tree):
        args = _log_args(call)
        fmt = _literal(args[0], module.constants) if args else None
        if fmt and not _RAW.match(fmt) and any(f"{key}=" in fmt for key in GLOBALLY_EXTRACTED):
            found.append((call.lineno, fmt))
    return found


@dataclass
class Scan:
    sites: list[Site]
    unresolved: list[str]
    prose_formats: dict[str, list[tuple[int, str]]]


def scan(source: Path = SRC) -> Scan:
    result = Scan([], [], {})
    for path in sorted(source.glob("*.py")):
        module = _Module(path)
        kv_sites, unresolved = _kv_sites(module)
        result.sites += kv_sites + _raw_sites(module) + _extra_sites(module)
        result.unresolved += unresolved
        result.prose_formats[module.name] = _prose_formats(module)
    return result


@pytest.fixture(scope="module")
def scanned() -> Scan:
    found = scan()
    assert not found.unresolved, (
        "a kv(...) call whose journal head this test cannot read — build the line through head() "
        "or a literal format, or name its function in NOT_JOURNAL with why it is not a journal "
        f"record: {found.unresolved}"
    )
    return found


@pytest.fixture(scope="module")
def sites(scanned) -> list[Site]:
    return scanned.sites


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


def _chat_completion(model: str, usage: dict, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


@scenario
def openai_chat(router, **_):
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "completion_tokens_details": {"reasoning_tokens": 50},
    }
    router.post(f"{OPENAI_HOST}/chat/completions").mock(
        return_value=_ok(_chat_completion("gpt-6-sol", usage, call_id="chatcmpl-Contract0001"))
    )
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


#: xAI's own documented example (docs.x.ai, Chat Completions reference): the reasoning is reported
#: apart from the completion, and the total counts both — 32 + 9 + 94 = 135.
_XAI_USAGE = {"prompt_tokens": 32, "completion_tokens": 9, "reasoning_tokens": 94, "total": 135}


@scenario
def xai(**_):
    usage = SimpleNamespace(
        prompt_tokens=_XAI_USAGE["prompt_tokens"],
        completion_tokens=_XAI_USAGE["completion_tokens"],
        reasoning_tokens=_XAI_USAGE["reasoning_tokens"],
        total_tokens=_XAI_USAGE["total"],
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
def xai_openai(router, **_):
    """The other xAI cell: the ``openai`` SDK on xAI's host, reading the same documented usage."""
    usage = {
        "prompt_tokens": _XAI_USAGE["prompt_tokens"],
        "completion_tokens": _XAI_USAGE["completion_tokens"],
        "total_tokens": _XAI_USAGE["total"],
        "prompt_tokens_details": {"cached_tokens": 6},
        "completion_tokens_details": {"reasoning_tokens": _XAI_USAGE["reasoning_tokens"]},
    }
    router.post(f"{XAI_HOST}/chat/completions").mock(
        return_value=_ok(
            _chat_completion("grok-4.3", usage, call_id="0199a2b4-7c3e-7f00-8a1b-2c3d4e5f6a7c")
        )
    )
    adapter = OpenAIProvider(
        model="grok-4.3",
        api_key="xai-test-0123456789abcdef",
        base_url=XAI_HOST,
        provider="xai",
        surface="chat",
        max_retries=0,
    )
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def google_chat(router, key, **_):
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
    adapter = build(key)
    adapter.chat([Message.user("hi")])
    adapter.close()


@scenario
def google_search(router, key, **_):
    router.post(GENERATE).mock(return_value=grounded(google_text("Sunny.")))
    adapter = build(key)
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
def anthropic_refusal(router, **_):
    """A content refusal: the vendor's final word, written as a counted fact on its own head."""
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=anthropic_reply(
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
            message_id="msg_01Refused",
        )
    )
    adapter = AnthropicProvider(ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096)
    adapter.chat([Message.user("…")])
    adapter.close()


@scenario
def anthropic_describer(router, **_):
    used = {k: v for k, v in _ANTHROPIC_USAGE.items() if k != "server_tool_use"}
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=anthropic_reply(
            anthropic_text("A tabby cat asleep on a windowsill."), used=used
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


#: The wake a scenario's model calls are made in: a real UUIDv7 and a router-shaped delivery id.
WAKE_TIMELINE = "019a4e2c-7b1d-7f3e-9c2a-5d8e1f6a3b40"
WAKE_DELIVERY = "dlv-019a4e2c7b1d"


@contextmanager
def _a_wake():
    """Inside a wake, as `Wake.wake` runs one: every `llm` line names its timeline and delivery."""
    previous = os.environ.get(DELIVERY_ID_ENV)
    os.environ[DELIVERY_ID_ENV] = WAKE_DELIVERY
    try:
        with in_wake(WAKE_TIMELINE):
            yield
    finally:
        if previous is None:
            os.environ.pop(DELIVERY_ID_ENV, None)
        else:
            os.environ[DELIVERY_ID_ENV] = previous


@scenario
def brain_failure(router, **_):
    """A brain call the vendor keeps failing (5xx): one retry, then the engine's give-up (#782)."""
    body = {"error": {"message": "upstream unavailable", "code": 503}}
    router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(503, json=body))
    engine = Engine(
        openrouter_provider(retries_disabled=True),
        ToolRegistry(),
        response_retries=1,
        sleep=lambda _s: None,
    )
    with _a_wake(), pytest.raises(ProviderServerError):
        engine.run([Message.user("hi")])


def _cut_tool_call_answer(router, finish_reason: str) -> None:
    """A tool call whose arguments stop mid-string, with the stop reason the vendor gave."""
    cut_off = wire_tool_call(id="call_0", name="get_weather", arguments={"city": "Dallas"})
    cut_off["function"]["arguments"] = '{"city": "Dal'
    router.post(GENERATION_CHAT_URL).mock(
        return_value=_ok(_openrouter_body(tool_calls=[cut_off], finish_reason=finish_reason))
    )
    engine = Engine(_generation_openrouter(), ToolRegistry(), response_retries=0)
    with _a_wake(), pytest.raises(ProviderResponseError):
        engine.run([Message.user("weather?")])


@scenario
def unreadable_answer(router, **_):
    """The vendor answers (and bills) a tool call whose arguments do not parse, having stopped of
    its own accord: the one billed line says `fallback reason=invalid_response`, and the give-up is
    its own line."""
    _cut_tool_call_answer(router, "stop")


@scenario
def truncated_answer(router, **_):
    """The same unreadable tool call, but the vendor says it stopped at the output cap (issue
    #788): a runaway cut off, so both lines say `reason=truncated`, never `invalid_response`."""
    _cut_tool_call_answer(router, "length")


@scenario
def rerank_unusable(router, **_):
    """A rerank the vendor answered with nothing the reranker can use: the caller's ``fallback``."""
    router.post(RERANK_CHAT).mock(return_value=_ok(rerank_completion("no picks in here")))
    with _a_wake(), at_step(3, 24):
        reranker().rerank("what is the plan", hits(3), 2, surface="tool")


@scenario
def attribution(**_):
    log_context_attribution([Message.user("hi")], session="timeline:contract")


_CODE_USE = {
    "type": "server_tool_use",
    "id": "srvtoolu_01ContractCode",
    "name": "bash_code_execution",
    "input": {"command": "echo 2"},
}
_CODE_RAN = {
    "type": "bash_code_execution_tool_result",
    "tool_use_id": "srvtoolu_01ContractCode",
    "content": {
        "type": "bash_code_execution_result",
        "stdout": "2\n",
        "stderr": "",
        "return_code": 0,
        "content": [],
    },
}


@scenario
def anthropic_sandbox(router, **_):
    """Code run with neither web tool declared — container time Anthropic bills and never states —
    beside a fetch that failed inside the 200."""
    fetch = {
        "type": "server_tool_use",
        "id": "srvtoolu_01ContractFetch",
        "name": "web_fetch",
        "input": {"url": "https://example.com/gone"},
    }
    failed = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_01ContractFetch",
        "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible"},
    }
    router.post(ANTHROPIC_MESSAGES).mock(
        return_value=anthropic_reply(
            _CODE_USE, _CODE_RAN, fetch, failed, anthropic_text("2"), message_id="msg_01Sandbox"
        )
    )
    adapter = AnthropicProvider(
        ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096, builtin_tools=["code_execution"]
    )
    adapter.chat([Message.user("run it, and read the page")])
    adapter.close()


@scenario
def openai_container(router, **_):
    payload = responses_body(out_code_interpreter_call(code="print(1)"), out_message("1"))
    payload["model"] = "gpt-6-sol"
    router.post(f"{OPENAI_HOST}/responses").mock(return_value=_ok(payload))
    adapter = OpenAIProvider(
        model="gpt-6-sol",
        api_key=OPENAI_KEY,
        base_url=OPENAI_HOST,
        surface="responses",
        max_retries=0,
        builtin_tools=["code_interpreter"],
    )
    adapter.chat([Message.user("run it")])
    adapter.close()


@scenario
def google_builtins(router, key, **_):
    """Gemini's own built-ins on a main turn: a code run, and a page it could not read."""
    response = google_body(
        {"executableCode": {"language": "PYTHON", "code": "print(2**10)"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1024\n"}},
        google_text("It is 1024."),
    )
    response["candidates"][0]["urlContextMetadata"] = {
        "urlMetadata": [
            {
                "retrievedUrl": "https://example.com/gone",
                "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_ERROR",
            }
        ]
    }
    router.post(GENERATE).mock(return_value=_ok(response))
    adapter = build(key, builtin_tools=["code_execution", "url_context"])
    adapter.chat([Message.user("2^10, and read example.com/gone")])
    adapter.close()


class _ImageTool(Tool):
    """A harness tool that buys one image, as every media tool does: through the meter."""

    name = "generate_image"
    description = "Generate an image."

    def run(self, **_: object) -> str:
        with metered(provider="openai", kind="image.generate", model="gpt-image-2.5-flare") as call:
            call.cost = ComputedCost(0.12505)
        return "Generated and posted."


class _McpTool(Tool):
    """A tool an MCP server serves through the harness's proxy (`McpTool` declares the same)."""

    name = "workmail__send_email"
    description = "Send an email."
    executor = "mcp"

    def run(self, **_: object) -> str:
        return "sent"


@scenario
def harness_and_mcp_tools(**_):
    """The engine's own dispatch — the path every harness and MCP tool takes in a wake."""
    registry = ToolRegistry()
    for tool in (_ImageTool(), _McpTool()):
        registry.register(tool)
    engine = Engine(provider=None, tools=registry)  # type: ignore[arg-type]
    engine._run_tool("generate_image", {"prompt": "a red cube"})
    engine._run_tool("workmail__send_email", {"to": "john@example.com"})
    engine._run_tool("no_such_tool", {})


#: What a fault-slug probe hands the adapter: an HTTP status, or no answer at all.
TRANSPORT = "transport"
#: One probe per row of the page's Fault Slugs table, and a second where a row names two statuses
#: or a range, so a row the adapters read differently from the page is caught on either half.
FAULT_PROBES: tuple[int | str, ...] = (
    TRANSPORT, 401, 403, 402, 404, 413, 429, 504, 500, 503, 400, 422
)  # fmt: skip
#: The HTTP adapters, and the reranker, whose line is read rather than its classifier.
FAULT_ADAPTERS = ("openai", "openrouter", "anthropic", "google")
#: Every word the retry line can say: the engine's table, and the two a connection failure splits.
RETRY_REASONS = frozenset({reason for _, reason in _REASONS} | {"timeout", "transport"})


def _fault(reasons: dict[str, str], key: str, call: Callable[[], object]) -> None:
    try:
        call()
    except ProviderError as exc:
        slug = fault_slug(exc)
        # A transient fault's retry line and its final line name it in the same word, and a fault
        # the engine never retries has no retry word.
        retry = _reason(exc)
        expected = slug if slug in RETRY_REASONS else "provider_error"
        assert retry == expected, f"{key}: retrying says {retry}, failed says {slug}"
        reasons[key] = slug
    else:  # pragma: no cover - every probe fails the call
        raise AssertionError(f"{key}: the probe did not raise")


def _probe(router, probe: int | str) -> None:
    """Every adapter's endpoint answers the probe: the status with its vendor's error shape."""
    routes = {
        f"{OPENAI_HOST}/chat/completions": {"error": {"message": "x", "type": "x", "code": probe}},
        OPENROUTER_CHAT: {"error": {"message": "x", "code": probe}},
        ANTHROPIC_MESSAGES: {"type": "error", "error": {"type": "api_error", "message": "x"}},
        GENERATE: {"error": {"code": probe, "message": "x", "status": "UNKNOWN"}},
    }
    for url, body in routes.items():
        if probe == TRANSPORT:
            router.post(url).mock(side_effect=httpx.ConnectError("connection refused"))
        else:
            router.post(url).mock(return_value=httpx.Response(probe, json=body))


#: A scenario that renders no line by design — it is read through `Rendered.reasons`.
SILENT: set[str] = set()


@scenario
def fault_slugs(router, key, reasons, **_):
    """Every probe through every adapter that speaks HTTP, read as the `reason` it names.

    An adapter's fault is read through `_retry.fault_slug` (its `failed` line's word) and
    `_engine._reason` (its retry line's); the reranker, which has its own mapper, writes its
    `failed` line, and that line is read. The native xAI adapter speaks gRPC, whose codes no HTTP
    table names, so it is not probed.
    """
    hi = [Message.user("hi")]
    openai = OpenAIProvider(
        model="gpt-6-sol", api_key=OPENAI_KEY, base_url=OPENAI_HOST, surface="chat", max_retries=0
    )
    claude = AnthropicProvider(ANTHROPIC_MODEL, api_key=ANTHROPIC_KEY, max_tokens=4096)
    gemini = build(key)
    adapters: dict[str, Callable[[], object]] = {
        "openai": lambda: openai.chat(hi),
        "openrouter": lambda: openrouter_provider(retries_disabled=True).chat(hi),
        "anthropic": lambda: claude.chat(hi),
        "google": lambda: gemini.chat(hi),
    }
    for probe in FAULT_PROBES:
        _probe(router, probe)
        for name, call in adapters.items():
            _fault(reasons, f"{name} {probe}", call)
        # A fresh reranker per probe: a `config:` failure after the first in its life is DEBUG.
        reranker().rerank("what is the plan", hits(3), 2, surface="turn0")
    for adapter in (openai, claude, gemini):
        adapter.close()


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def rendered(monkeypatch, key_file) -> Rendered:  # noqa: F811 - the Google suite's fixture
    """Every scenario's lines, parsed — offline, under the suite's own isolation and stubs."""
    monkeypatch.setenv("NO_COLOR", "1")
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
                run(router=router, key=key_file, reasons=result.reasons)
            parsed = [parse_line(record.getMessage()) for record in handler.records]
            result.lines[name] = [line for line in parsed if line is not None]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return result


# === the behavioral witnesses: one per `test` row of the Known Violations table ===================


def _tokens(line: Line) -> tuple[int, int, int]:
    return (
        int(line.fields["tokens_in"]),
        int(line.fields["tokens_out"]),
        int(line.fields["tokens_total"]),
    )


#: One `test` row's witness: true exactly while its departure stands. Slice 3 (issue #759) struck
#: F1–F10, and issue #796 struck S1 (a bare 402 is `config:billing` on every adapter); each became
#: a regression test below.
WITNESSES: dict[str, Callable[[Rendered, list[Site]], bool]] = {}


# === the checks =================================================================================


def _in_scope(v: Violation, *, module: str | None = None, scenario: str | None = None) -> bool:
    if v.scope is None:
        return True
    return module == f"_{v.scope}.py" or bool(scenario and scenario.startswith(v.scope))


def _allowed(contract: Contract, head: str, name: str, **where: str) -> bool:
    if contract.allows(head, name):
        return True
    if any(v.arg == head and name in v.fields for v in contract.of("head")):
        return True
    return any(v.arg == f"{head}.{name}" and _in_scope(v, **where) for v in contract.of("field"))


def _emitted(sites: list[Site], rendered: Rendered) -> list[tuple[str, str, dict[str, str]]]:
    """Every (head, field, where) the package emits, statically and as rendered."""
    seen = [(s.head, name, {"module": s.module}) for s in sites for name in s.fields]
    for scenario, lines in rendered.lines.items():
        seen += [
            (line.head, name, {"scenario": scenario}) for line in lines for name in line.fields
        ]
    return seen


def test_the_page_parses_into_heads_fields_and_violations(contract):
    assert {"llm", "tool", "wake start", "wake end", "unspoken"} <= set(contract.heads)
    # Issue #782: a retry and a refusal are outcomes of the call, never heads of their own.
    assert not {"llm retry", "llm refused"} & set(contract.heads)
    assert {"provider", "purpose", "tokens_in", "cache_write_tokens", "residency", "billing"} <= (
        contract.heads["llm"]
    )
    assert all(fields for fields in contract.heads.values()), "a head with no field table"
    ids = [v.id for v in contract.violations]
    assert len(ids) == len(set(ids)), "a violation id is used twice"
    forms = {"head", "field", "prose", "reason-prose", "unescaped", "test"}
    assert {v.form for v in contract.violations} <= forms


def test_every_head_the_package_emits_is_in_the_contract(contract, sites, rendered):
    allowed = set(contract.heads) | {v.arg for v in contract.of("head")}
    emitted = {site.head for site in sites}
    emitted |= {line.head for lines in rendered.lines.values() for line in lines}
    stray = sorted(emitted - allowed)
    assert not stray, (
        f"heads outside docs/telemetry.md: {stray}. A new head is a contract change: define it on "
        "the page (and raise it needs-capital first), never ship it on its own."
    )


def test_every_field_on_every_head_is_in_the_contract(contract, sites, rendered):
    stray = sorted(
        {
            f"{head}.{name} ({next(iter(where.values()))})"
            for head, name, where in _emitted(sites, rendered)
            if not _allowed(contract, head, name, **where)
        }
    )
    assert not stray, (
        f"fields outside docs/telemetry.md: {stray}. A new field is a contract change for every "
        "adapter whose vendor has the datum, decided before it ships."
    )


def test_a_computed_field_set_is_rendered_by_this_test(sites, rendered):
    computed = sorted({site.head for site in sites if site.computed} - RUNTIME_RENDERED)
    assert not computed, (
        f"heads whose fields the scan cannot read: {computed}. Build their fields from literals, or "
        "render them in a scenario here and add the head to RUNTIME_RENDERED."
    )
    heads = {line.head for lines in rendered.lines.values() for line in lines}
    assert heads >= RUNTIME_RENDERED


def test_every_scenario_renders_a_line(rendered):
    """A scenario that silently rendered nothing would make every check over it vacuous."""
    silent = sorted(
        name for name, lines in rendered.lines.items() if not lines and name not in SILENT
    )
    assert not silent, silent
    assert len(rendered.reasons) == len(FAULT_ADAPTERS) * len(FAULT_PROBES)


def test_a_reason_is_a_slug_unless_the_page_lists_the_head(contract, sites):
    listed = {v.arg for v in contract.of("reason-prose")}
    prose = {site.head for site in sites if site.prose_reason}
    assert not prose - listed, f"reason= carries a sentence on {sorted(prose - listed)}"


def test_every_line_is_built_through_kv_unless_the_page_lists_the_head(contract, sites):
    listed = {v.arg for v in contract.of("unescaped")}
    raw = {site.head for site in sites if site.unescaped}
    assert not raw - listed, (
        f"%-formatted structured lines on {sorted(raw - listed)}: build them through kv(), which "
        "flattens, scrubs, bounds and quotes every value."
    )


def test_a_prose_line_carrying_a_globally_extracted_field_is_listed(contract, scanned):
    listed = [(v.module, v.text) for v in contract.of("prose")]
    unlisted = [
        f"{module}:{line} {fmt!r}"
        for module, found in scanned.prose_formats.items()
        for line, fmt in found
        if not any(module == m and text in fmt for m, text in listed)
    ]
    assert not unlisted, (
        f"prose lines carrying {GLOBALLY_EXTRACTED} (the NOC reads them off every line): {unlisted}"
    )


def test_every_violation_row_is_still_true(contract, sites, rendered):
    """A departure that disappeared must be struck from the table: the table is today's, exactly."""
    emitted = _emitted(sites, rendered)
    heads = {head for head, _, _ in emitted}
    stale: list[str] = []
    for v in contract.violations:
        if v.form == "head":
            assert v.arg not in contract.heads, f"{v.id}: `{v.arg}` is both a head and a violation"
            holds = v.arg in heads
        elif v.form == "field":
            head, _, name = v.arg.rpartition(".")
            assert not contract.allows(head, name), f"{v.id}: `{v.arg}` is in the contract"
            holds = any(h == head and n == name and _in_scope(v, **w) for h, n, w in emitted)
        elif v.form == "prose":
            until_head, _, until_field = (v.until or "").partition(".")
            until_field, _, until_value = until_field.partition("=")
            if v.until is not None:
                assert (
                    contract.allows(until_head, until_field)
                    if until_field
                    else (until_head in contract.heads)
                ), f"{v.id}: `{v.until}` is not in the contract"
            present = v.text in (SRC / v.module).read_text(encoding="utf-8")
            if until_value:  # a value is a rendered fact: only a rendered line can carry it
                replaced = any(
                    line.head == until_head and line.fields.get(until_field) == until_value
                    for lines in rendered.lines.values()
                    for line in lines
                )
            else:
                replaced = v.until is not None and any(
                    h == until_head and (not until_field or n == until_field) for h, n, _ in emitted
                )
            holds = present and not replaced
        elif v.form == "reason-prose":
            holds = any(site.head == v.arg and site.prose_reason for site in sites)
        elif v.form == "unescaped":
            holds = any(site.head == v.arg and site.unescaped for site in sites)
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
    assert set(WITNESSES) == {v.id for v in contract.of("test")}


#: The tool lines each scenario must render, as ``(name, executor, outcome)``: every adapter's
#: server-side tools and the engine's harness and MCP tools, on the one head (issue #758).
EXPECTED_TOOLS = {
    "openai_responses": {("web_search", "server", "ok")},
    "openai_container": {("code_execution", "server", "ok")},
    "openrouter": {("web_search", "server", "ok")},
    "xai": {("web_search", "server", "ok")},
    "google_builtins": {("code_execution", "server", "ok"), ("url_context", "server", "error")},
    "anthropic": {
        ("web_search", "server", "ok"),
        ("web_fetch", "server", "ok"),
        ("code_execution", "server", "ok"),
    },
    "anthropic_sandbox": {("code_execution", "server", "ok"), ("web_fetch", "server", "error")},
    "harness_and_mcp_tools": {
        ("generate_image", "harness", "ok"),
        ("workmail__send_email", "mcp", "ok"),
        ("no_such_tool", "harness", "error"),
    },
}


def test_every_executor_writes_the_one_tool_line(rendered):
    for name, expected in EXPECTED_TOOLS.items():
        lines = [line for line in rendered.heads(name, "tool") if "outcome" in line.fields]
        found = {
            (line.fields["name"], line.fields["executor"], line.fields["outcome"]) for line in lines
        }
        assert found == expected, name
        for line in lines:
            if line.fields["executor"] == "server":
                # Timed by its call's `llm` line, and joined to it by that call's id.
                assert "duration" not in line.fields and int(line.fields["count"]) >= 1, line
                assert line.fields["generation_id"] == rendered.llm(name).fields["generation_id"]
                assert line.fields["provider"] == rendered.llm(name).fields["provider"]
            else:
                assert "duration" in line.fields and "generation_id" not in line.fields, line
            assert ("reason" in line.fields) == (line.fields["outcome"] == "error"), line
    # No other scenario writes a tool line it was not expected to.
    assert {n for n in rendered.lines if rendered.heads(n, "tool")} == set(EXPECTED_TOOLS)


def test_a_tool_fee_rides_its_tool_line_and_never_the_llm_line(rendered):
    (search,) = [
        line for line in rendered.heads("anthropic", "tool") if line.fields["name"] == "web_search"
    ]
    assert (search.fields["kind"], search.fields["cost"]) == ("search.web", "0.02")
    assert search.fields["cost_basis"] == "computed"
    (image,) = [
        l for l in rendered.heads("harness_and_mcp_tools", "tool") if "outcome" in l.fields
    ][:1]
    assert image.fields == {
        "name": "generate_image",
        "executor": "harness",
        "provider": "openai",
        "kind": "image.generate",
        "model": "gpt-image-2.5-flare",
        "duration": image.fields["duration"],
        "outcome": "ok",
        "cost": "0.12505",
        "cost_basis": "computed",
    }
    retired = {"web_search_requests", "web_fetch_requests", "code_execution_requests"}
    assert not any(
        retired & set(line.fields) for lines in rendered.lines.values() for line in lines
    )


def test_an_unpriced_container_is_one_counted_line_on_both_vendors(rendered):
    for name, provider in (("anthropic_sandbox", "anthropic"), ("openai_container", "openai")):
        (gap,) = rendered.heads(name, "cost unpriced")
        assert gap.fields == {
            "provider": provider,
            "charge": "container_time",
            "reason": "not_reported",
        }


#: The `charge` head's field set: the grammar posted for the NOC on issue #766, which the Steel
#: launcher (basecradle-noc#1153) builds against. A row added to or struck from the page's table, or
#: a field added to or dropped from its grammar block, fails unless this set changes with it, so the
#: page and the posted grammar cannot drift apart.
CHARGE_FIELDS = frozenset(
    {
        "provider",
        "kind",
        "executor",
        "unit",
        "count",
        "duration",
        "started",
        "resource",
        "generation_id",
        "outcome",
        "reason",
        "cost",
        "cost_basis",
        "vendor_usage",
        "vendor_unit",
    }
)
_CHARGE_ROW = re.compile(r"^\| `(?P<field>[^`]+)` \|.*\| (?P<absent>[^|]+) \|$")
_ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_SECONDS = re.compile(r"\d+\.\d+s")
_DOLLARS = re.compile(r"\d+(?:\.\d+)?")


def _section(head: str) -> str:
    section = DOC.read_text(encoding="utf-8").split(f"\n### `{head}`\n", 1)[1]
    return section.split("\n### ", 1)[0].split("\n## ", 1)[0]


def _charge_section() -> str:
    return _section("charge")


def _row(head: str, field: str) -> str:
    """The table row for one field of one head, whole."""
    (row,) = [line for line in _section(head).splitlines() if line.startswith(f"| `{field}` |")]
    return row


def _charge_blocks() -> list[str]:
    return re.findall(r"```text\n(.*?)```", _charge_section(), re.DOTALL)


def _charge_required() -> set[str]:
    """The fields whose row's Absent When is "Never": every `charge` line carries them."""
    rows = [_CHARGE_ROW.match(line) for line in _charge_section().splitlines()]
    return {row["field"] for row in rows if row and row["absent"].startswith("Never")}


def _charge_examples() -> list[Line]:
    """The journal lines the page shows under `charge`, parsed the way the NOC reads the journal."""
    records = [
        record.removeprefix("INFO ")
        for block in _charge_blocks()
        for record in block.splitlines()
        if record.startswith("INFO ")
    ]
    lines = [parse_line(record) for record in records]
    assert all(lines), records
    return lines  # type: ignore[return-value]


def _stray(contract: Contract, line: Line) -> set[str]:
    return {name for name in line.fields if not _allowed(contract, line.head, name)}


def test_charge_carries_exactly_the_posted_field_set(contract):
    assert contract.heads["charge"] == CHARGE_FIELDS


def test_the_charge_grammar_block_is_the_table(contract):
    """The one-line grammar the NOC builds from names every row, and brackets exactly the optional."""
    (grammar,) = [block.strip() for block in _charge_blocks() if block.startswith("charge ")]
    named, required, depth = set(), set(), 0
    for token in re.finditer(r"\[|\]|(?P<key>[a-z_]+)=", grammar):
        if token[0] == "[":
            depth += 1
        elif token[0] == "]":
            depth -= 1
        else:
            named.add(token["key"])
            if depth == 0:
                required.add(token["key"])
    assert depth == 0, grammar
    assert named == CHARGE_FIELDS
    assert required == _charge_required()
    assert {"provider", "kind", "executor", "unit", "count", "outcome", "cost"} == required
    assert "stated" not in grammar  # `cost_basis` is `computed`, or absent for a stated cost


def test_every_charge_example_on_the_page_is_a_conforming_line(contract):
    examples = _charge_examples()
    # Steel's clean session, Steel's crashed one, then container time.
    assert [line.head for line in examples] == ["charge", "charge", "charge"]
    for line in examples:
        fields = line.fields
        assert not _stray(contract, line), line.message
        assert _charge_required() <= set(fields), line.message
        assert ("vendor_usage" in fields) == ("vendor_unit" in fields), line.message
        assert ("reason" in fields) == (fields["outcome"] == "error"), line.message
        # `cost_basis` means one thing fleet-wide: `computed`, or absent for a vendor-stated cost.
        assert fields.get("cost_basis", "computed") == "computed", line.message
        # The grammar's fixed units.
        assert _DOLLARS.fullmatch(fields["cost"]), line.message
        assert "duration" not in fields or _SECONDS.fullmatch(fields["duration"]), line.message
        assert "started" not in fields or _ISO_UTC.fullmatch(fields["started"]), line.message


#: The `charge` line's `reason` vocabulary (issue #768): generic words every vendor's failures map
#: onto, posted for the NOC's Steel launcher (basecradle-noc#1153). A slug added to or struck from
#: the page's row fails unless this set changes with it.
CHARGE_REASONS = frozenset({"crashed", "startup_failed", "timeout"})
#: What a `cost unpriced` line may name as carrying no price (`browser_time`: issue #768).
UNPRICED_CHARGES = frozenset({"tokens", "container_time", "browser_time", "grounding", "search"})
UNPRICED_REASONS = frozenset({"unknown_model", "unpriced_tier", "unpriced_host", "not_reported"})


def _vocabulary(head: str, field: str, after: str, before: str) -> set[str]:
    """The slugs a row lists between two phrases of its Meaning cell."""
    listed = _row(head, field).split(after, 1)[1].split(before, 1)[0]
    return set(re.findall(r"`([a-z_]+)`", listed))


def test_the_charge_reason_vocabulary_is_the_page(contract):
    # A slug of the vocabulary is defined, not merely named: `<slug>` (its meaning).
    assert set(re.findall(r"`([a-z_]+)` \(", _row("charge", "reason"))) == CHARGE_REASONS
    for line in _charge_examples():
        assert line.fields.get("reason", "crashed") in CHARGE_REASONS, line.message


def test_the_cost_unpriced_vocabulary_is_the_page_and_every_gap_speaks_it(contract, rendered):
    assert _vocabulary("cost unpriced", "charge", "no price:", " |") == UNPRICED_CHARGES
    assert _vocabulary("cost unpriced", "reason", "| `reason` |", " |") == UNPRICED_REASONS
    gaps = [
        line for lines in rendered.lines.values() for line in lines if line.head == "cost unpriced"
    ]
    assert gaps  # the scenarios render gaps, so this check reads real lines
    for line in gaps:
        assert line.fields["charge"] in UNPRICED_CHARGES, line.message
        assert line.fields["reason"] in UNPRICED_REASONS, line.message
    # The browser gap the page shows under `charge` is a conforming `cost unpriced` line.
    (record,) = [
        record.removeprefix("WARNING ")
        for block in _charge_blocks()
        for record in block.splitlines()
        if record.startswith("WARNING ")
    ]
    gap = parse_line(record)
    assert gap is not None and gap.head == "cost unpriced", record
    assert not _stray(contract, gap), record
    assert gap.fields == {
        "provider": "steel",
        "charge": "browser_time",
        "reason": "unpriced_tier",
    }


def test_a_charge_field_outside_the_contract_is_refused(contract):
    steel = _charge_examples()[0]
    # The launcher's pre-contract spend fields, and `session`, which already means the harness
    # session's channel key on `context attribution` and so cannot name a vendor's resource here.
    for name in ("billed_minutes", "credits_used", "proxy_bytes", "release_reason", "session"):
        line = parse_line(f"{steel.message} {name}=x")
        assert line is not None and _stray(contract, line) == {name}


def test_the_line_parser_reads_the_grammar_and_rejects_prose():
    line = parse_line('tool name=messages duration=0.12s outcome=ok error="a b=c"')
    assert line is not None and line.head == "tool"
    assert line.fields == {
        "name": "messages",
        "duration": "0.12s",
        "outcome": "ok",
        "error": "a b=c",
    }
    reset = parse_line("Wake breaker RESET timeline=0199 held=1.00s")
    assert reset is not None and reset.head == "Wake breaker RESET"
    assert parse_line("Anthropic refused the turn for x (category=cyber): no.") is None
    assert parse_line("step 1/40: tool calls (1.20s)") is None


def test_the_scan_reads_a_reason_through_its_callers(tmp_path):
    """The reason check follows a parameter to the calls that fill it, and a local to its value."""
    (tmp_path / "_sample.py").write_text(
        "def drop(self, why):\n"
        '    _log.error("%s %s", head("dropped", RED), kv(reason=why))\n'
        "def caller(self):\n"
        '    self.drop("the wake died")\n'
        "def defer(reason):\n"
        '    _log.info("%s %s", head("wake deferred", YELLOW), kv(reason=reason))\n'
        'defer(reason="truncated")\n'
        "def stall():\n"
        '    reason = f"stalled after {n}"\n'
        '    _log.info("%s %s", head("wake stalled", YELLOW), kv(reason=reason))\n'
        "def note(size):\n"
        '    logger.info("direct_message sent bytes=%d", size)\n'
    )
    found = {site.head: site for site in scan(tmp_path).sites}
    assert found["dropped"].prose_reason and found["wake stalled"].prose_reason
    assert not found["wake deferred"].prose_reason
    assert found["direct_message sent"].unescaped and found["direct_message sent"].fields == {
        "bytes"
    }


# === the fields with one meaning on every vendor (issue #759, rows F1–F10 struck) ================

#: Each scenario's token fields, exactly — what every vendor's usage reads as in the one vocabulary.
#: ``tokens_total`` is ``tokens_in`` + ``tokens_out`` on every line, never the vendor's own total.
EXPECTED_TOKENS = {
    # OpenAI and OpenRouter state a cache write and reasoning on both wires (F1, F3).
    "openai_chat": {
        "tokens_in": 1000,
        "tokens_out": 200,
        "tokens_total": 1200,
        "cached_tokens": 600,
        "cache_write_tokens": 300,
        "tokens_reasoning": 50,
    },
    "openai_responses": {
        "tokens_in": 1000,
        "tokens_out": 200,
        "tokens_total": 1200,
        "cached_tokens": 600,
        "cache_write_tokens": 300,
        "tokens_reasoning": 50,
    },
    "openrouter": {
        "tokens_in": 1000,
        "tokens_out": 200,
        "tokens_total": 1200,
        "cached_tokens": 600,
        "cache_write_tokens": 300,
        "tokens_reasoning": 50,
    },
    # xAI's reasoning is billed as output and reported beside it: 32 + (9 + 94) = 135 (F5).
    "xai": {
        "tokens_in": 32,
        "tokens_out": 103,
        "tokens_total": 135,
        "cached_tokens": 6,
        "tokens_reasoning": 94,
    },
    "xai_openai": {
        "tokens_in": 32,
        "tokens_out": 103,
        "tokens_total": 135,
        "cached_tokens": 6,
        "tokens_reasoning": 94,
    },
    # 1,200 prompt + 400 tool-use prompt tokens are input; 30 reply + 70 thinking are output (F6).
    "google_chat": {
        "tokens_in": 1600,
        "tokens_out": 100,
        "tokens_total": 1700,
        "cached_tokens": 1000,
        "tokens_reasoning": 70,
    },
    # Anthropic's prompt in its three parts, its write, and its thinking — on every purpose (F2, F4).
    "anthropic": {
        "tokens_in": 44000,
        "tokens_out": 200,
        "tokens_total": 44200,
        "cached_tokens": 40000,
        "cache_write_tokens": 3000,
        "tokens_reasoning": 120,
    },
    "anthropic_describer": {
        "tokens_in": 44000,
        "tokens_out": 200,
        "tokens_total": 44200,
        "cached_tokens": 40000,
        "cache_write_tokens": 3000,
        "tokens_reasoning": 120,
    },
}

_TOKEN_FIELD_NAMES = (
    "tokens_in",
    "tokens_out",
    "tokens_total",
    "cached_tokens",
    "cache_write_tokens",
    "tokens_reasoning",
)


def test_every_vendor_fills_the_token_fields_with_one_meaning(rendered):
    for name, expected in EXPECTED_TOKENS.items():
        fields = rendered.llm(name).fields
        found = {key: int(fields[key]) for key in _TOKEN_FIELD_NAMES if key in fields}
        assert found == expected, name


def test_a_helper_line_carries_every_token_field_its_adapters_main_line_does(rendered):
    main, helper = rendered.llm("anthropic").fields, rendered.llm("anthropic_describer").fields
    assert {k: main.get(k) for k in _TOKEN_FIELD_NAMES} == {
        k: helper.get(k) for k in _TOKEN_FIELD_NAMES
    }


def test_tokens_total_is_exactly_in_plus_out_on_every_line(rendered):
    for name, lines in rendered.lines.items():
        for line in lines:
            if line.head == "llm" and "tokens_total" in line.fields:
                tokens_in, tokens_out, tokens_total = _tokens(line)
                assert tokens_total == tokens_in + tokens_out, (name, line.message)


def test_endpoint_is_the_serving_upstream_and_residency_where_the_vendor_ran(rendered):
    for name in ("google_chat", "google_search", "anthropic", "anthropic_describer"):
        fields = rendered.llm(name).fields
        assert "endpoint" not in fields, name
        assert fields["residency"] == "us", name


def test_a_computed_cost_is_stated_only_for_the_host_its_table_transcribes(rendered):
    assert "cost" not in rendered.llm("anthropic_proxy").fields
    (gap,) = rendered.heads("anthropic_proxy", "cost unpriced")
    assert gap.fields == {
        "provider": "anthropic",
        "model": ANTHROPIC_MODEL,
        "charge": "tokens",
        "reason": "unpriced_host",
    }


def _fault_table() -> list[tuple[str, str]]:
    """The page's Fault Slugs rows, as (what the producer saw, the `reason` it writes)."""
    section = DOC.read_text(encoding="utf-8").split("\n#### Fault Slugs\n", 1)[1]
    section = section.split("\n#", 1)[0]
    rows = re.findall(r"^\| ([^|`]*(?:`[^`]*`[^|`]*)*) \| `([a-z_:]+)` \|$", section, re.MULTILINE)
    return [(saw, slug) for saw, slug in rows if saw != "What the Producer Saw"]


def _page_slug(probe: int | str) -> str:
    """The slug the page's table gives a probe: its own row, else the range or catch-all it falls in."""
    rows = _fault_table()
    if probe == TRANSPORT:
        (slug,) = [slug for saw, slug in rows if saw.startswith("No HTTP answer")]
        return slug
    named = [slug for saw, slug in rows if f"`{probe}`" in saw]
    if named:
        return named[0]
    family = "`5xx`" if 500 <= int(probe) < 600 else "status"
    (slug,) = [slug for saw, slug in rows if saw.startswith("Any other") and saw.endswith(family)]
    return slug


def test_the_fault_slug_table_is_one_vocabulary():
    """The page states the ruled table (issue #786), and every slug in it is in the `reason` row."""
    assert dict(_fault_table()) == {
        "No HTTP answer: the connection failed, or broke before an answer was read "
        "(DNS, TCP, TLS, a reset)": "transport",
        "HTTP `401` or `403`": "config:auth",
        "HTTP `402`": "config:billing",
        "HTTP `404`": "config:model_not_found",
        "HTTP `413`": "payload_too_large",
        "HTTP `429`": "rate_limited",
        "HTTP `504`": "timeout",
        "Any other HTTP `5xx`": "server_error",
        "Any other HTTP status": "api_error",
    }
    listed = _row("llm", "reason")
    for _, slug in _fault_table():
        cause = "config:<cause>" if slug.startswith("config:") else slug
        assert f"`{cause}`" in listed, slug


def test_every_http_adapter_names_a_fault_by_the_page(rendered):
    """A 504 is `timeout` on every vendor, a 401 `config:auth`, … — the table, through each SDK."""
    expected = {
        f"{name} {probe}": _page_slug(probe) for probe in FAULT_PROBES for name in FAULT_ADAPTERS
    }
    assert rendered.reasons == expected


def test_the_reranker_names_a_fault_by_the_page(rendered):
    """The reranker's own mapper (`_rerank._fault_of`) writes the table's words on its line."""
    failed = [
        line.fields["reason"]
        for line in rendered.heads("fault_slugs", "llm")
        if line.fields["purpose"] == "memory" and line.fields["outcome"] == "failed"
    ]
    assert failed == [_page_slug(probe) for probe in FAULT_PROBES]


def test_a_refusal_is_the_calls_own_line_and_outcome(rendered):
    """One refused attempt, one line (issue #782): it was billed, so it is a call, and its verdict."""
    (line,) = rendered.heads("anthropic_refusal", "llm")
    assert line.fields["outcome"] == "refused"
    assert line.fields["category"] == "cyber"
    assert line.fields["generation_id"] == "msg_01Refused"
    assert "reason" not in line.fields


#: The fields every `llm` line carries, whatever its outcome.
LLM_ALWAYS = frozenset({"provider", "purpose", "model", "outcome"})


def _outcomes() -> dict[str, tuple[set[str], set[str]]]:
    """The page's Outcomes table: each outcome's required and never-carried fields."""
    table = DOC.read_text(encoding="utf-8").split("\n#### Outcomes\n", 1)[1].split("\n### ", 1)[0]
    found: dict[str, tuple[set[str], set[str]]] = {}
    for row in table.splitlines():
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            names = [set(re.findall(r"`([a-z_]+)`", cell)) for cell in cells]
            (outcome,) = names[0]
            found[outcome] = (names[2], names[3])
    return found


def test_the_outcomes_table_is_the_vocabulary_and_names_only_llm_fields(contract):
    outcomes = _outcomes()
    assert set(outcomes) == {"ok", "refused", "fallback", "retrying", "failed"}
    for required, never in outcomes.values():
        assert required <= contract.heads["llm"] and never <= contract.heads["llm"]
        assert not required & never
    assert LLM_ALWAYS <= contract.heads["llm"]


def test_every_llm_line_carries_exactly_what_its_outcome_requires(rendered):
    """Strict fields per outcome, checked on rendered lines (issue #782): the contract stays
    verifiable because an outcome's line can neither omit what it owes nor carry what it must not."""
    outcomes = _outcomes()
    seen: set[str] = set()
    for scenario, lines in rendered.lines.items():
        for line in (line for line in lines if line.head == "llm"):
            fields = set(line.fields)
            outcome = line.fields.get("outcome")
            assert outcome in outcomes, (scenario, line.message)
            required, never = outcomes[outcome]
            assert LLM_ALWAYS | required <= fields, (scenario, line.message)
            assert not never & fields, (scenario, line.message)
            seen.add(outcome)
    assert seen == set(outcomes), f"no scenario renders {sorted(set(outcomes) - seen)}"


def test_an_unreadable_answer_is_billed_once_and_the_give_up_bills_nothing(rendered):
    """One billed attempt, one billed line (issue #782 review): the answer could not be read, so
    its line is `fallback` with its tokens and its id, never `ok`; the engine's give-up follows as a
    line of its own with nothing billed, naming the same generation."""
    billed, given_up = rendered.heads("unreadable_answer", "llm")
    assert (billed.fields["outcome"], billed.fields["reason"]) == ("fallback", "invalid_response")
    assert "tokens_in" in billed.fields and billed.fields["purpose"] == "main"
    assert (given_up.fields["outcome"], given_up.fields["reason"]) == ("failed", "invalid_response")
    assert "tokens_in" not in given_up.fields and "cost" not in given_up.fields
    assert billed.fields["generation_id"] == given_up.fields["generation_id"]


def test_an_answer_cut_off_at_the_cap_is_written_as_truncated_on_both_lines(rendered):
    """A runaway is told apart from an unreadable answer (issue #788): the cap fired, so the billed
    line is `fallback reason=truncated` with its bill, and the decision that follows says the same,
    naming the same generation."""
    billed, given_up = rendered.heads("truncated_answer", "llm")
    assert (billed.fields["outcome"], billed.fields["reason"]) == ("fallback", "truncated")
    assert "tokens_in" in billed.fields
    assert (given_up.fields["outcome"], given_up.fields["reason"]) == ("failed", "truncated")
    assert "tokens_in" not in given_up.fields and "cost" not in given_up.fields
    assert billed.fields["generation_id"] == given_up.fields["generation_id"]


def test_a_call_made_in_a_wake_names_its_wake_and_its_step(rendered):
    """The wake context rides every model call made inside it, ambiently (issue #782)."""
    retrying, failed = rendered.heads("brain_failure", "llm")
    for line in (retrying, failed):
        assert line.fields["timeline"] == WAKE_TIMELINE, line.message
        assert line.fields["delivery"] == WAKE_DELIVERY, line.message
        assert line.fields["step"] == "1/24", line.message
        assert line.fields["purpose"] == "main", line.message
    assert retrying.fields["outcome"] == "retrying" and retrying.fields["attempt"] == "1/2"
    assert failed.fields["outcome"] == "failed" and failed.fields["attempt"] == "2/2"
    assert failed.fields["reason"] == "server_error"
    (rerank,) = rendered.heads("rerank_unusable", "llm")
    assert rerank.fields["outcome"] == "fallback" and rerank.fields["reason"] == "parse"
    assert (rerank.fields["timeline"], rerank.fields["step"]) == (WAKE_TIMELINE, "3/24")
    # Outside a wake there is no wake to name: an adapter's own call carries none of the three.
    assert not {"timeline", "delivery", "step"} & set(rendered.llm("anthropic").fields)


# === the palette =================================================================================
#
# `docs/telemetry.md` → The Palette defines each color tier once for the whole fleet (issue #799).
# The harness is held to it from both sides: every tier's SGR is the constant the harness writes,
# and the `outcome=` vocabulary the page assigns is exactly the one `kv` colors — a value colored
# that the page does not name, or named that the harness leaves plain, fails here.

_PALETTE_TIER = re.compile(r"^\| (?P<tier>[A-Z][A-Za-z' ]+) \| [A-Za-z]+ \| `(?P<sgr>[^`]+)` \| ")
_PALETTE_MEMBERS = re.compile(r"^\| (?P<tier>[A-Z][A-Za-z' ]+) \| .+ \| (?P<outcomes>[^|]+) \|$")


def _palette() -> tuple[dict[str, str], dict[str, str]]:
    """The page's tiers (name → SGR) and its `outcome=` values (value → tier name)."""
    text = DOC.read_text(encoding="utf-8")
    section = text.partition("\n## The Palette\n")[2].partition("\n## ")[0]
    assert section, "the contract page has no Palette section"
    tiers: dict[str, str] = {}
    outcomes: dict[str, str] = {}
    for line in section.splitlines():
        if tier := _PALETTE_TIER.match(line):
            sgr = tier["sgr"].encode().decode("unicode_escape")
            tiers[tier["tier"]] = sgr
        elif members := _PALETTE_MEMBERS.match(line):
            for value in re.findall(r"`([a-z_]+)`", members["outcomes"]):
                assert value not in outcomes, f"`outcome={value}` is in two tiers"
                outcomes[value] = members["tier"]
    return tiers, outcomes


def test_every_palette_tier_is_the_sgr_the_harness_writes():
    tiers, _ = _palette()

    assert tiers == {
        "Success": GREEN,
        "Bookend": BLUE,
        "Warning": YELLOW,
        "Operator's Choice": YELLOW,
        "Failure": RED,
    }


def test_the_outcome_vocabulary_is_the_palettes_in_both_directions(monkeypatch):
    monkeypatch.delenv(NO_COLOR_ENV, raising=False)
    tiers, outcomes = _palette()

    assert set(outcomes) == set(_VERDICT_VALUES["outcome"])
    for value, tier in outcomes.items():
        assert kv(outcome=value) == f"{tiers[tier]}outcome={value}{RESET}", value


def test_a_stopped_wake_is_the_operators_choice_never_a_failure():
    # The tier the ruling added (issue #799): a deliberate stop shares Yellow's SGR, never Red's.
    _, outcomes = _palette()

    assert outcomes["stopped"] == "Operator's Choice"
