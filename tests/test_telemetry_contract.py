"""The telemetry contract, enforced: every head and field the harness emits is on `docs/telemetry.md`.

The page is the contract (issue #757, @origin's ruling of 2026-10-31: the fleet's telemetry is
contract-first). This module reads it — its per-head field tables, and its **Known Violations**
table as the one allowlist — and checks the harness against it two ways, because neither alone sees
everything:

- **Statically**, every structured emission site in the package: each `kv(...)` that becomes a
  journal record, each ``%``-formatted line shaped like one, and the ``extra=`` literals handed to
  the `llm` and `llm retry` emitters — with its head and its literal field names. This is what makes
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
import re
from collections.abc import Callable
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
from basecradle_harness._engine import _reason
from tests.conftest import FAKE_KEY as OPENAI_KEY
from tests.conftest import out_message, out_web_search_call, responses_body
from tests.test_anthropic import FAKE_KEY as ANTHROPIC_KEY
from tests.test_anthropic import MESSAGES as ANTHROPIC_MESSAGES
from tests.test_anthropic import MODEL as ANTHROPIC_MODEL
from tests.test_anthropic import _no_stray_anthropic_env  # noqa: F401 - autouse fixture
from tests.test_anthropic import reply as anthropic_reply
from tests.test_anthropic import text as anthropic_text
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
RUNTIME_RENDERED = {"llm", "llm retry", "context attribution"}

#: The emitters that take an ``extra=`` mapping of trailing fields, and the head each writes.
EXTRA_EMITTERS = {"log_llm_call": "llm", "log_llm_retry": "llm retry", "Retry": "llm retry"}

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
    r"^\| (?P<id>[TF]\d+) \| .+ \| `(?P<witness>[^`]+)` \| Slice \d \|$", re.MULTILINE
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
    """The ``extra=`` literals handed to the `llm` / `llm retry` emitters, read as fields."""
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
    if module.name == "_retry.py":  # `diagnostics` rides both heads (`_retry.diagnostics`)
        (function,) = [f for f in module.functions if f.name == "diagnostics"]  # type: ignore[attr-defined]
        (returned,) = [n.value for n in ast.walk(function) if isinstance(n, ast.Return)]
        keys = _dict_keys(returned, function) or set()
        sites += [Site(module.name, function.lineno, h, set(keys)) for h in ("llm", "llm retry")]  # type: ignore[attr-defined]
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


#: A scenario that renders no line by design — it is read through `Rendered.reasons`.
SILENT = {"gateway_timeouts"}


@scenario
def gateway_timeouts(router, key, reasons, **_):
    """An HTTP 504 through every adapter that speaks HTTP, read as the retry vocabulary's word."""
    body = {"error": {"message": "upstream timed out", "type": "timeout", "code": 504}}
    claude_body = {"type": "error", "error": {"type": "api_error", "message": "x"}}
    vertex_body = {"error": {"code": 504, "message": "x", "status": "DEADLINE_EXCEEDED"}}
    router.post(f"{OPENAI_HOST}/chat/completions").mock(return_value=httpx.Response(504, json=body))
    router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(504, json=body))
    router.post(ANTHROPIC_MESSAGES).mock(return_value=httpx.Response(504, json=claude_body))
    router.post(GENERATE).mock(return_value=httpx.Response(504, json=vertex_body))
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
    gemini = build(key)
    _gateway_timeout(reasons, "google", lambda: gemini.chat(hi))
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


def _sums(line: Line) -> bool:
    """Whether the line's ``tokens_total`` is exactly ``tokens_in`` + ``tokens_out``."""
    tokens_in, tokens_out, tokens_total = _tokens(line)
    return tokens_total == tokens_in + tokens_out


def _static_tool_fields(sites: list[Site]) -> set[str]:
    return {name for site in sites if site.head == "tool" for name in site.fields}


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
        for n in (
            "openai_chat",
            "openai_responses",
            "openrouter",
            "xai",
            "xai_openai",
            "google_chat",
        )
    ),
    "F4": lambda r, s: all(
        "tokens_reasoning" not in r.llm(n).fields for n in ("anthropic", "anthropic_describer")
    ),
    "F5": lambda r, s: any(not _sums(r.llm(n)) for n in ("xai", "xai_openai")),
    # 1,200 prompt + 400 tool-use prompt tokens were billed as input.
    "F6": lambda r, s: _tokens(r.llm("google_chat"))[0] < 1_600 or not _sums(r.llm("google_chat")),
    "F7": lambda r, s: "endpoint" in r.llm("google_chat").fields,
    "F8": lambda r, s: any(
        "residency" not in r.llm(n).fields for n in ("anthropic", "google_chat")
    ),
    "F9": lambda r, s: "cost" in r.llm("anthropic_proxy").fields,
    "F10": lambda r, s: len(set(r.reasons.values())) > 1,
}


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
    assert {"llm", "llm retry", "tool", "wake start", "wake end", "unspoken"} <= set(contract.heads)
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
    assert len(rendered.reasons) == 4


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
            if v.until is not None:
                assert (
                    contract.allows(until_head, until_field)
                    if until_field
                    else (until_head in contract.heads)
                ), f"{v.id}: `{v.until}` is not in the contract"
            present = v.text in (SRC / v.module).read_text(encoding="utf-8")
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
