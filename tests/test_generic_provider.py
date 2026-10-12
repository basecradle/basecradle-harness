"""``AI_PROVIDER`` names the vendor that bills us, and a vendor the harness does not know is legal.

Issue #670. The token was a closed enum of four vendors and also the key into every vendor
convention, so the first agent on a fifth vendor (`@pong`, the ``openai`` SDK aimed at Wiro) could
only be configured as ``openai`` — and was charted, cache-keyed, cost-warned and briefed as OpenAI.
The founder's ruling: the token is the biller, the set opens, the dialect is derived from the
vendor. The four known vendors keep every convention; any other slug gets the **generic**
OpenAI-compatible profile, which these tests pin from the outside in.

The wire claims are read off a **real loopback HTTP server** (issue #433's rule): the adapter's own
view of what it decided is not evidence of what left the process.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import threading

import pytest

from basecradle_harness import Message
from basecradle_harness._basecradle import (
    GENERIC_DEFAULT_SURFACE,
    KNOWN_PROVIDERS,
    _config_from_env,
    _provider_from_config,
    normalize_provider,
)
from basecradle_harness._brief import render_brain
from basecradle_harness._caching import bind_conversation
from basecradle_harness._install import (
    INSTALLED,
    install,
    packaged_tool_sources,
    plugin_opts_in,
    plugin_source_providers,
)
from basecradle_harness._openai import _AFFINITY, CACHE_KEY_FIELD, CONVERSATION_HEADER
from basecradle_harness._openrouter import ROUTING_METADATA_HEADER
from basecradle_harness._plugins import (
    ActivationContext,
    OpenAIKey,
    OpenAISurface,
    Sdk,
    Vendor,
)
from basecradle_harness._resolve import resolve_stems
from basecradle_harness._verify import _active_provider

from .conftest import completion, responses_body

VENDOR = "wiro"
MODEL = "qwen/qwen3-8-27b-obliterated"
FAKE_KEY = "sk-wiro-0123456789abcdef0123456789abcdef"
SESSION = "timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2"
#: What Wiro states on the wire: tokens, and its own price for them.
STATED_COST = 0.000412

#: Every byte a known vendor's conventions could put on a request — a body field or a header name.
#: A generic vendor's request must carry none of them.
VENDOR_BODY_FIELDS = (CACHE_KEY_FIELD, "search_parameters", "provider")
VENDOR_HEADERS = (
    CONVERSATION_HEADER.lower(),
    *(name.lower() for name in ROUTING_METADATA_HEADER),
    "http-referer",
    "x-title",
)


# --- the wire server ------------------------------------------------------------------------------


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _handler(seen: list[dict], stated_cost: float | None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            seen.append(
                {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )
            usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
            if stated_cost is not None:
                usage["cost"] = stated_cost
            if self.path.endswith("/responses"):
                reply = responses_body()
                reply["usage"] = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
            else:
                reply = completion(content="ok")
                reply["usage"] = usage
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # keep the test output clean
            pass

    return Handler


@contextlib.contextmanager
def _serve(stated_cost: float | None):
    seen: list[dict] = []
    server = _Server(("127.0.0.1", 0), _handler(seen, stated_cost))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def wire():
    """A fake OpenAI-compatible vendor on loopback that states its price, as Wiro does: its API
    root, and every request it received."""
    with _serve(STATED_COST) as served:
        yield served


@pytest.fixture
def unpriced_wire():
    """The same vendor, stating tokens and no price — the case that reaches the rate-table path."""
    with _serve(None) as served:
        yield served


@pytest.fixture
def generic_env(monkeypatch):
    """An agent configured exactly as `@pong` will be: the biller's slug, the openai SDK, a host."""
    monkeypatch.setenv("AI_PROVIDER", VENDOR)
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _build():
    """The brain the wake would build — through the config triple and the one factory."""
    provider, sdk, surface = _config_from_env()
    return _provider_from_config(provider, sdk, surface)


# --- the slug -------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["wiro", "WIRO", " wiro ", "together.ai", "my_vendor-2"])
def test_a_well_formed_slug_is_a_provider(raw):
    assert normalize_provider(raw) == raw.strip().lower()


@pytest.mark.parametrize("raw", ["Wiro.AI/", "wi ro", "wiro:prod", "   "])
def test_a_malformed_token_is_refused_by_its_shape(raw):
    """The NOC reads ``provider=([A-Za-z0-9._-]+)``: a token outside it would split the chart."""
    with pytest.raises(ValueError, match="not a provider slug"):
        normalize_provider(raw)


def test_unset_is_still_the_default_vendor():
    assert normalize_provider(None) == normalize_provider("") == "openai"


def test_the_known_vendors_are_exactly_the_four_with_conventions():
    assert KNOWN_PROVIDERS == ("openai", "xai", "openrouter", "google")


# --- the generic profile's preconditions ----------------------------------------------------------


def test_a_generic_vendor_without_a_host_refuses_at_startup(generic_env):
    """No default host exists for a vendor the harness never heard of, and none is guessed."""
    with pytest.raises(ValueError, match=r"AI_PROVIDER='wiro'.*set AI_BASE_URL"):
        _config_from_env()


def test_the_build_refuses_a_generic_vendor_with_no_host_too():
    """The factory checks it as well: a describer reaches it with an endpoint of its own."""
    with pytest.raises(ValueError, match="set AI_BASE_URL"):
        _provider_from_config(VENDOR, "openai", "chat", model=MODEL, api_key=FAKE_KEY)


@pytest.mark.parametrize("sdk", ["openrouter", "xai-sdk", "google-genai"])
def test_a_native_sdk_cannot_reach_a_generic_vendor(generic_env, sdk):
    generic_env.setenv("AI_SDK", sdk)
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    with pytest.raises(ValueError, match=rf"needs AI_SDK=openai \(got '{sdk}'"):
        _config_from_env()
    with pytest.raises(ValueError, match="needs AI_SDK=openai"):
        _provider_from_config(VENDOR, sdk, "", model=MODEL, api_key=FAKE_KEY)


def test_an_active_server_side_builtin_is_a_config_error_not_a_silent_drop(generic_env):
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    with pytest.raises(ValueError, match=r"runs no server-side built-in.*web_search"):
        _provider_from_config(VENDOR, "openai", "chat", builtins=["web_search"])


# --- the surface ----------------------------------------------------------------------------------


def test_a_generic_vendor_defaults_to_the_chat_surface(generic_env):
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    assert _config_from_env() == (VENDOR, "openai", GENERIC_DEFAULT_SURFACE)
    assert GENERIC_DEFAULT_SURFACE == "chat"


def test_responses_is_an_explicit_opt_in_for_a_generic_vendor(generic_env):
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("AI_SDK_SURFACE", "responses")
    assert _config_from_env()[2] == "responses"


@pytest.mark.parametrize("vendor", ["openai", "xai"])
def test_the_known_vendors_keep_the_responses_default(monkeypatch, vendor):
    monkeypatch.setenv("AI_PROVIDER", vendor)
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert _config_from_env() == (vendor, "openai", "responses")


# --- the wire -------------------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_a_generic_vendor_puts_nothing_vendor_specific_on_the_wire(generic_env, wire, surface):
    """The defect's sharpest consequence, closed: under a bound conversation the ``openai`` label
    sent OpenAI's ``prompt_cache_key`` to Wiro on every call. A generic vendor has no affinity cell,
    so a bound session puts no routing key anywhere — read off the request the server received."""
    base, seen = wire
    generic_env.setenv("AI_BASE_URL", base)
    generic_env.setenv("AI_SDK_SURFACE", surface)
    provider = _build()
    bind_conversation(provider, SESSION)

    provider.chat([Message.user("Hi")])

    (request,) = seen
    assert request["path"] == (
        "/v1/responses" if surface == "responses" else "/v1/chat/completions"
    )
    assert request["body"]["model"] == MODEL
    for field in VENDOR_BODY_FIELDS:
        assert field not in request["body"], field
    for header in VENDOR_HEADERS:
        assert header not in request["headers"], header
    assert SESSION not in json.dumps(request)  # the session id reached the wire nowhere at all
    assert "tools" not in request["body"]  # no server-side built-in was offered
    provider.close()


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_a_known_vendor_on_the_same_server_still_sends_its_key(monkeypatch, wire, surface):
    """The control: the server can see the field, so its absence above is a fact about the vendor."""
    base, seen = wire
    monkeypatch.setenv("AI_BASE_URL", base)
    provider = _provider_from_config("openai", "openai", surface, model=MODEL, api_key=FAKE_KEY)
    bind_conversation(provider, SESSION)

    provider.chat([Message.user("Hi")])

    assert seen[0]["body"][CACHE_KEY_FIELD] == SESSION
    provider.close()


def test_a_generic_output_cap_rides_max_tokens(generic_env, wire):
    base, seen = wire
    generic_env.setenv("AI_BASE_URL", base)
    provider = _provider_from_config(VENDOR, "openai", "chat", max_output_tokens=512)

    provider.chat([Message.user("Hi")])

    assert seen[0]["body"]["max_tokens"] == 512
    assert "max_completion_tokens" not in seen[0]["body"]
    provider.close()


# --- the bill -------------------------------------------------------------------------------------


def test_the_llm_line_names_the_biller_and_its_stated_cost_with_no_cost_gap_warning(
    generic_env, wire, caplog
):
    """What *Spend by Provider* reads: ``provider=wiro`` and the vendor's own ``usage.cost`` — and
    no WARNING about OpenAI's rate table not applying, which was never this vendor's table."""
    base, _seen = wire
    generic_env.setenv("AI_BASE_URL", base)
    provider = _build()

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        provider.chat([Message.user("Hi")])

    lines = [r.getMessage() for r in caplog.records]
    llm = [line for line in lines if line.startswith("llm provider=")]
    assert len(llm) == 1
    assert llm[0].startswith(f"llm provider={VENDOR} ")
    assert f"cost={STATED_COST}" in llm[0]
    assert "cost_basis=" not in llm[0]  # stated by the vendor, never computed
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not [line for line in lines if "api.openai.com" in line]
    provider.close()


def test_a_generic_vendor_that_states_no_price_gets_no_cost_and_no_warning(
    generic_env, unpriced_wire, caplog
):
    """The path the label used to reach: no stated cost, so the adapter would consult OpenAI's
    rates and warn that this host is not OpenAI's. A generic vendor never had a table to miss."""
    base, _seen = unpriced_wire
    generic_env.setenv("AI_BASE_URL", base)
    provider = _build()

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        provider.chat([Message.user("Hi")])

    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm provider=")]
    assert line.startswith(f"llm provider={VENDOR} ")
    assert "cost=" not in line
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    provider.close()


# --- the gates and the installer ------------------------------------------------------------------


def _ctx(provider: str) -> ActivationContext:
    return ActivationContext(
        provider=provider,
        sdk="openai",
        surface="responses",
        model=MODEL,
        env={"AI_API_KEY": FAKE_KEY},
    )


def test_no_vendor_gate_is_met_by_a_generic_vendor():
    ctx = _ctx(VENDOR)
    for vendor in KNOWN_PROVIDERS:
        assert not Vendor(vendor).met(ctx), vendor
    assert not OpenAIKey().met(ctx)  # the key is present; the vendor is not OpenAI
    assert OpenAIKey().met(_ctx("openai"))  # the control: same key, OpenAI's label
    # Not vendor gates, and not what keeps a vendor tool off a generic agent:
    assert OpenAISurface("responses").met(ctx)
    assert Sdk("openai").met(ctx)


def test_the_resolver_activates_no_vendor_coupled_tool_for_a_generic_vendor():
    """Every powerful default granted at once: a generic vendor still resolves no built-in and no
    vendor-coupled tool, while the vendor-neutral set is all there."""
    generic = resolve_stems(provider=VENDOR, opt_in=_powerful_stems())
    openai = resolve_stems(provider="openai", opt_in=_powerful_stems())

    assert generic["ai_provider"] == VENDOR
    assert generic["ai_sdk_surface"] == "chat"
    assert generic["builtins"] == []
    for vendor_tool in ("generate_image", "edit_image", "grok_generate_image", "xai_search"):
        assert vendor_tool not in generic["tools"], vendor_tool
    assert "generate_image" in openai["tools"]  # the control: the same grant under OpenAI
    assert {"messages", "assets", "timelines"} <= set(generic["tools"])


def _powerful_stems() -> str:
    return ",".join(
        sorted(stem for stem, text in packaged_tool_sources().items() if plugin_opts_in(text))
    )


def test_the_installer_keeps_only_the_vendor_neutral_defaults_for_a_generic_vendor(tmp_path):
    generic = install(tmp_path / "generic", provider=VENDOR, opt_in=_powerful_stems().split(","))
    known = install(tmp_path / "openai", provider="openai", opt_in=_powerful_stems().split(","))

    def tools(report) -> set[str]:
        return {
            path.removeprefix("tools/").removesuffix(".py")
            for path in report.of(INSTALLED)
            if path.startswith("tools/")
        }

    generic_tools, openai_tools = tools(generic), tools(known)
    sources = packaged_tool_sources()
    neutral = {stem for stem, text in sources.items() if plugin_source_providers(text) is None}
    assert generic_tools == neutral  # every vendor-neutral default kept, nothing else laid down
    assert "generate_image" in openai_tools  # the control: a vendor-affine stem, under its vendor
    assert generic_tools < openai_tools


def test_verify_passes_a_generic_token_through(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "Wiro")
    assert _active_provider(tmp_path, None) == ("wiro", "environment")
    assert _active_provider(tmp_path, "wiro") == ("wiro", "argument")


# --- the report -----------------------------------------------------------------------------------


def test_resolved_config_reports_the_biller(generic_env, tmp_path, capsys):
    from basecradle_harness._wake import main

    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("HARNESS_HOME", str(tmp_path))
    generic_env.setenv("BASECRADLE_TOKEN", "bc_live_0123456789abcdef0123456789abcdef")

    assert main(["--resolved-config"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["ai_provider"] == VENDOR
    assert report["ai_sdk"] == "openai"
    assert report["ai_sdk_surface"] == "chat"
    assert report["ai_base_url"] == "https://llm.example.com/v1"
    assert report["builtins"] == []


def test_a_describer_riding_a_generic_brain_shares_its_host(generic_env):
    from basecradle_harness._describer import describer_from_env

    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("HARNESS_DESCRIBER_MODEL", "qwen/qwen3-vl")
    generic_env.setenv("HARNESS_DESCRIBER_API_KEY", "sk-wiro-describer0123456789abcdef")
    generic_env.setenv("HARNESS_DESCRIBER_PROVIDERS", "wiro")

    describer = describer_from_env()

    assert describer.fault is None
    assert describer.provider.provider == VENDOR
    assert describer.provider.base_url == "https://llm.example.com/v1"


def test_no_generic_vendor_is_a_cell_of_the_affinity_table():
    """The table's vendors are known vendors — a generic one resolves to no entry by construction."""
    assert {vendor for vendor, _surface in _AFFINITY} <= set(KNOWN_PROVIDERS)


def test_the_generic_build_says_so_once(generic_env, caplog):
    """A misspelled known vendor (``openroutr``) is a legal slug and runs generic; this line is the
    only trace that it lost that vendor's conventions."""
    generic_env.setenv("AI_PROVIDER", "openroutr")
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _build().close()

    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("provider ")]
    assert line == "provider profile=generic provider=openroutr surface=chat host=llm.example.com"


def test_a_mistyped_sdk_is_named_as_no_adapter_not_as_another_vendors(generic_env):
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("AI_SDK", "opnai")
    with pytest.raises(ValueError, match="ships no adapter by that name"):
        _config_from_env()


def test_the_resolver_refuses_a_generic_vendor_on_a_native_sdk():
    with pytest.raises(ValueError, match="needs AI_SDK=openai"):
        resolve_stems(provider=VENDOR, sdk="xai-sdk")


def test_the_operators_model_params_still_reach_a_generic_vendor(generic_env, wire, tmp_path):
    """The generic profile adds nothing of its own; the operator's words for this vendor pass."""
    base, seen = wire
    home = tmp_path / "config"
    home.mkdir()
    (home / "model_params.json").write_text(
        json.dumps(
            {"temperature": 0.2, "extra_body": {"top_k": 20}, "extra_headers": {"X-Tier": "a"}}
        )
    )
    generic_env.setenv("BASECRADLE_CONFIG_HOME", str(home))
    generic_env.setenv("AI_BASE_URL", base)
    provider = _build()

    provider.chat([Message.user("Hi")])

    (request,) = seen
    assert request["body"]["temperature"] == 0.2
    assert request["body"]["top_k"] == 20
    assert request["headers"]["x-tier"] == "a"
    provider.close()


def test_the_brief_names_the_biller(generic_env):
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    provider = _build()
    brain = render_brain(provider)
    assert f"- Provider: `{VENDOR}`" in brain
    assert "- Surface: `chat`" in brain
    provider.close()


def test_resolved_config_states_no_cost_basis_for_a_generic_vendor(generic_env, tmp_path, capsys):
    """Whether a generic vendor states a price is a fact about its responses, not its config."""
    from basecradle_harness._wake import main

    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("HARNESS_HOME", str(tmp_path))
    assert main(["--resolved-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["cost_basis"], report["cost_rates_source"]) == (None, None)


def test_resolved_config_refuses_a_builtin_a_generic_wake_would_refuse(
    generic_env, tmp_path, capsys
):
    """An operator plugin offering a built-in with no vendor gate: every wake would refuse it, so
    the probe must not read green over it."""
    from basecradle_harness._wake import main

    home = tmp_path / "config"
    install(home, provider=VENDOR)
    (home / "tools" / "my_search.py").write_text(
        "from basecradle_harness import ToolPlugin\n\n"
        'PLUGIN = ToolPlugin(builtin="web_search", requires=())\n'
    )
    generic_env.setenv("BASECRADLE_CONFIG_HOME", str(home))
    generic_env.setenv("AI_BASE_URL", "https://llm.example.com/v1")
    generic_env.setenv("HARNESS_HOME", str(tmp_path))

    assert main(["--resolved-config"]) == 1
    assert "runs no server-side built-in" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("env", "detail"),
    [
        (
            {"HARNESS_DESCRIBER_PROVIDER": "acme", "HARNESS_DESCRIBER_SDK": "openai"},
            "inherits no AI_BASE_URL",
        ),
        (
            {"HARNESS_DESCRIBER_PROVIDER": "acme", "HARNESS_DESCRIBER_SDK": "openrouter"},
            "HARNESS_DESCRIBER_SDK=openai",
        ),
    ],
    ids=["other-generic-vendor", "generic-vendor-on-a-native-sdk"],
)
def test_a_describer_on_its_own_generic_vendor_names_its_own_variables(monkeypatch, env, detail):
    """Its fault must name the describer's variables: the build's message names the brain's, and
    following it (set AI_BASE_URL) would point the brain at the describer's vendor."""
    from basecradle_harness._describer import describer_from_env

    monkeypatch.setenv("AI_PROVIDER", "openai")
    for var in ("AI_SDK", "AI_SDK_SURFACE", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", "qwen/qwen3-vl")
    monkeypatch.setenv("HARNESS_DESCRIBER_API_KEY", "sk-acme-describer0123456789abcdef")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDERS", "acme")
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    describer = describer_from_env()

    assert describer.fault == "config:no_provider"
    assert detail in describer.detail
    assert "HARNESS_DESCRIBER_PROVIDER='acme'" in describer.detail
