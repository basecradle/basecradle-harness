"""The vendor's own id for a model call, on the `llm` line and the `llm retry` line (issue #634).

On 2026-10-03 three calls served by one upstream ran for 10 to 14 minutes each and came back at the
131,072-token output cap (basecradle-noc#963). OpenRouter's feedback and refund path is *per
generation*, and the journal had the time, model, endpoint, tokens and cost of every one of those
calls and no id, so the complaint went by email with timestamps. A call that fails, or that is
rejected as ``invalid_response``, is exactly the one whose id is needed.

What is pinned here, and the "obviously fine" broken form of each:

- **Every adapter writes the id its vendor gave**, read off the response the line already reads —
  OpenRouter's ``gen-`` id, OpenAI's ``chatcmpl-`` / ``resp_`` id, xAI's response id. The ids in the
  fixtures below are the shapes measured live on 2026-10-03.
- **A refused or unparseable attempt names its generation too.** OpenRouter sets ``X-Generation-Id``
  on every response, a 4xx included (measured live), so the error a mapper raises carries it, and the
  retry line and the final fallback line print it. A field filled only on success would be absent on
  the very lines the issue was filed for.
- **The #963 shape end to end:** a body that arrives, is logged, and then fails to parse (tool-call
  arguments cut off at the cap) puts the *same* id on the `llm` line and on the retry line after it.
- **Absent, never a placeholder, and never anything that is not an id.** The value is vendor-written
  text on a line whose columns are partial regex matches; a value that is not id-shaped is omitted.
- **It moves no NOC column.** It renders last, and the fleet's own extraction expressions read the
  same values off a line with it as without it.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import httpx
import pytest
from openrouter import OpenRouter

from basecradle_harness import (
    Message,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ToolSpec,
    XaiSdkProvider,
)
from basecradle_harness._describer import Describer
from basecradle_harness._exceptions import ProviderError
from basecradle_harness._observability import (
    MAIN,
    RETRY_HEAD,
    generation_id,
    generation_id_header,
    log_llm_call,
)
from basecradle_harness._rerank import MemPalaceReranker
from basecradle_harness._retry import Retry, diagnostics
from tests.conftest import CHAT_URL as OPENAI_CHAT_URL
from tests.conftest import FAKE_KEY as OPENAI_KEY
from tests.conftest import (
    RESPONSES_URL,
    completion,
    out_message,
    responses_body,
    wire_tool_call,
)
from tests.test_openrouter import BASE_URL as OPENROUTER_URL
from tests.test_openrouter import CHAT_URL as OPENROUTER_CHAT_URL
from tests.test_openrouter import FAKE_KEY as OPENROUTER_KEY
from tests.test_openrouter import completion as openrouter_completion

#: Real shapes, measured live 2026-10-03 — fabricated values, never a call that happened.
OPENROUTER_ID = "gen-1791030923-8abbrsSEsIk06gI224Fi"
OPENAI_CHAT_ID = "chatcmpl-EUtFhTOVDnWKIKbLRDA1MgH3Y77Sx"
OPENAI_RESPONSE_ID = "resp_00801893e3e64149006ac0f69a6f5487d19c92bd67f172520b"
XAI_ID = "0199a8f2-5c3e-7b1d-9f4a-2e6c8d0b1a37"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _llm_line(caplog) -> str:
    """The one `llm` line — never an ``llm retry`` one, which shares the first word."""
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm provider=")]
    assert len(lines) == 1, lines
    return lines[0]


def _retry_line(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(f"{RETRY_HEAD} ")]
    assert len(lines) == 1, lines
    return lines[0]


def _field(line: str, key: str) -> str | None:
    return next((f.split("=", 1)[1] for f in line.split() if f.startswith(f"{key}=")), None)


def _last_field(line: str) -> str:
    return line.split()[-1]


# === the readers ==============================================================


@pytest.mark.parametrize("value", [OPENROUTER_ID, OPENAI_CHAT_ID, OPENAI_RESPONSE_ID, XAI_ID])
def test_every_vendors_id_is_read_off_a_mapping_and_an_object(value):
    """One reader for every SDK's answer: a ``model_dump()`` dict, a typed model, a proto wrapper."""
    assert generation_id({"id": value}) == value
    assert generation_id(SimpleNamespace(id=value)) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        42,
        "gen-1 kind=evil",  # a quoted value would still feed `kind=` to a column
        "gen-1\nINFO llm provider=forged",  # one record, one line
        '"gen-1"',
        "x" * 129,  # longer than any id a vendor issues
        "-leading-dash",
    ],
)
def test_anything_that_is_not_an_id_is_omitted_never_rendered(value):
    assert generation_id({"id": value}) is None


def test_no_response_and_no_id_field_are_both_absent():
    assert generation_id(None) is None
    assert generation_id({"object": "chat.completion"}) is None
    assert generation_id(SimpleNamespace()) is None


def test_the_header_is_read_whatever_its_case_and_container():
    """An SDK error may hand its headers back as ``httpx.Headers`` or as a plain dict."""
    assert generation_id_header(httpx.Headers({"X-Generation-Id": OPENROUTER_ID})) == OPENROUTER_ID
    assert generation_id_header({"X-Generation-Id": OPENROUTER_ID}) == OPENROUTER_ID
    assert generation_id_header({"x-generation-id": OPENROUTER_ID}) == OPENROUTER_ID
    assert generation_id_header({"x-request-id": "req_602a97e8"}) is None  # not a generation id
    assert generation_id_header(None) is None
    assert generation_id_header("not headers") is None


# === the lines ================================================================


def test_the_id_renders_last_on_the_llm_line(caplog):
    """Last, so that no column written before it moves."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="z-ai/glm-5.3",
            seconds=1.0,
            endpoint="Sail Research",
            outcome="ok",
            extra={"surface": "turn0"},
            generation_id=OPENROUTER_ID,
        )
    assert _last_field(_llm_line(caplog)) == f"generation_id={OPENROUTER_ID}"


def test_an_id_carried_in_extra_renders_once_last_and_the_argument_wins(caplog):
    """A fallback line's `diagnostics` carries one too; a second value under one keyword would be a
    ``TypeError`` raised from inside an ``except`` block in a wake."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="m",
            seconds=1.0,
            extra={"generation_id": "gen-from-diagnostics", "subject": "cat.png"},
            generation_id=OPENROUTER_ID,
        )
        log_llm_call(
            provider="openrouter",
            model="m",
            seconds=1.0,
            extra={"generation_id": "gen-from-diagnostics", "subject": "cat.png"},
        )
    first, second = (r.getMessage() for r in caplog.records)
    assert first.count("generation_id=") == 1
    assert _last_field(first) == f"generation_id={OPENROUTER_ID}"
    assert _last_field(second) == "generation_id=gen-from-diagnostics"


def test_a_call_with_no_id_is_the_line_it_always_was(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openai", model="gpt-5.4-mini", seconds=1.0)
    assert "generation_id" not in _llm_line(caplog)


def test_the_retry_line_carries_the_refused_attempts_id_last(caplog):
    error = ProviderRateLimitError("limited", status_code=429)
    error.generation_id = OPENROUTER_ID
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        Retry(
            provider="openrouter",
            model="z-ai/glm-5.3",
            purpose=MAIN,
            sleep=lambda _s: None,
            extra={"surface": "turn0"},
        ).again(error, reason="rate_limited")
    line = _retry_line(caplog)
    assert _last_field(line) == f"generation_id={OPENROUTER_ID}"
    assert diagnostics(error)["generation_id"] == OPENROUTER_ID


def test_a_failure_with_no_id_leaves_the_retry_line_as_it_was(caplog):
    assert ProviderError.generation_id is None
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        Retry(provider="openai", model="m", purpose=MAIN, sleep=lambda _s: None).again(
            ProviderServerError("oops", status_code=503), reason="server_error"
        )
    assert "generation_id" not in _retry_line(caplog)


# === OpenRouter, through the real SDK ========================================


def _openrouter(**kw):
    client = OpenRouter(api_key=OPENROUTER_KEY, server_url=OPENROUTER_URL, retry_config=None)
    return OpenRouterProvider("z-ai/glm-5.3", client=client, base_url=OPENROUTER_URL, **kw)


def _openrouter_body(**kw):
    body = openrouter_completion(**kw)
    body["id"] = OPENROUTER_ID
    return body


def test_openrouter_writes_its_generation_id(router, caplog):
    router.post(OPENROUTER_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_openrouter_body(content="Hi."),
            headers={"X-Generation-Id": OPENROUTER_ID},
        )
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _openrouter().chat([Message.user("hello")])
    assert _field(_llm_line(caplog), "generation_id") == OPENROUTER_ID


def test_the_963_shape_names_one_generation_on_the_llm_line_and_the_retry_line(router, caplog):
    """A body arrives at the output cap with its tool-call arguments cut off: the `llm` line is
    written, the parse then fails, and the ``invalid_response`` retry names the same generation."""
    cut_off = wire_tool_call(id="call_0", name="get_weather", arguments={"city": "Dallas"})
    cut_off["function"]["arguments"] = '{"city": "Dal'
    router.post(OPENROUTER_CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=_openrouter_body(tool_calls=[cut_off], finish_reason="length")
        )
    )
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        with pytest.raises(ProviderResponseError) as raised:
            _openrouter().chat([Message.user("weather?")], tools=[WEATHER_TOOL])
        Retry(
            provider="openrouter", model="z-ai/glm-5.3", purpose=MAIN, sleep=lambda _s: None
        ).again(raised.value, reason="invalid_response")

    assert raised.value.generation_id == OPENROUTER_ID
    assert _field(_llm_line(caplog), "generation_id") == OPENROUTER_ID
    assert _field(_retry_line(caplog), "generation_id") == OPENROUTER_ID


def test_a_refused_attempt_carries_the_header_openrouter_set_on_it(router):
    router.post(OPENROUTER_CHAT_URL).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "rate limited", "code": 429}},
            headers={"X-Generation-Id": OPENROUTER_ID},
        )
    )
    with pytest.raises(ProviderRateLimitError) as raised:
        _openrouter().chat([Message.user("hello")])
    assert raised.value.generation_id == OPENROUTER_ID


def test_a_body_the_sdk_cannot_parse_still_names_its_generation(router):
    """The SDK's own validation failure: the typed model refuses the body before the adapter reads
    it, so the header is the only id there is."""
    body = _openrouter_body(content="Hi.")
    del body["system_fingerprint"]  # required by the SDK's typed ChatResult
    router.post(OPENROUTER_CHAT_URL).mock(
        return_value=httpx.Response(200, json=body, headers={"X-Generation-Id": OPENROUTER_ID})
    )
    with pytest.raises(ProviderResponseError) as raised:
        _openrouter().chat([Message.user("hello")])
    assert raised.value.generation_id == OPENROUTER_ID


def test_a_refusal_with_no_header_carries_no_id(router):
    router.post(OPENROUTER_CHAT_URL).mock(
        return_value=httpx.Response(503, json={"error": {"message": "down", "code": 503}})
    )
    with pytest.raises(ProviderServerError) as raised:
        _openrouter().chat([Message.user("hello")])
    assert raised.value.generation_id is None


# === the openai SDK, both surfaces ===========================================


def _openai(surface):
    return OpenAIProvider(
        model="gpt-5.4-mini",
        api_key=OPENAI_KEY,
        base_url=OPENAI_CHAT_URL.rsplit("/chat/completions", 1)[0],
        surface=surface,
        max_retries=0,
    )


def test_openai_chat_writes_its_completion_id(router, caplog):
    body = completion(content="Hi.")
    body["id"] = OPENAI_CHAT_ID
    router.post(OPENAI_CHAT_URL).mock(return_value=httpx.Response(200, json=body))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _openai("chat").chat([Message.user("hello")])
    assert _field(_llm_line(caplog), "generation_id") == OPENAI_CHAT_ID


def test_openai_responses_writes_its_response_id(router, caplog):
    body = responses_body(out_message("Hi."))
    body["id"] = OPENAI_RESPONSE_ID
    router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=body))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _openai("responses").chat([Message.user("hello")])
    assert _field(_llm_line(caplog), "generation_id") == OPENAI_RESPONSE_ID


def test_the_openai_sdk_aimed_at_openrouter_carries_the_refusals_header(router):
    """The same capability read, not a vendor branch: the header is honored wherever it is set."""
    router.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "rate limited", "code": 429}},
            headers={"X-Generation-Id": OPENROUTER_ID},
        )
    )
    with pytest.raises(ProviderRateLimitError) as raised:
        _openai("chat").chat([Message.user("hello")])
    assert raised.value.generation_id == OPENROUTER_ID


def test_an_openai_refusal_names_no_generation(router):
    """OpenAI sends ``x-request-id``, a different id with a different use; it is not passed off as
    a generation id."""
    router.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "rate limited", "type": "rate_limit"}},
            headers={"x-request-id": "req_602a97e879424ec7806935ac6119c8be"},
        )
    )
    with pytest.raises(ProviderRateLimitError) as raised:
        _openai("chat").chat([Message.user("hello")])
    assert raised.value.generation_id is None


# === the native xAI SDK, through its real Response ===========================


def test_xai_writes_the_id_off_the_sdks_own_response(caplog):
    """Read through the real ``xai_sdk.chat.Response`` over a real proto, because the attribute
    name is exactly what the shared reader has to get right."""
    from xai_sdk.chat import Response
    from xai_sdk.proto import chat_pb2

    proto = chat_pb2.GetChatCompletionResponse(
        id=XAI_ID,
        outputs=[chat_pb2.CompletionOutput(message=chat_pb2.CompletionMessage(content="Hi."))],
    )
    response = Response(proto, 0)

    class _Client:
        chat = SimpleNamespace(create=lambda **_kw: SimpleNamespace(sample=lambda: response))

        def close(self):
            pass

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        XaiSdkProvider("grok-4.3", api_key="xai-" + "0" * 32, client=_Client()).chat(
            [Message.user("hello")]
        )
    assert _field(_llm_line(caplog), "generation_id") == XAI_ID


# === the reranker and the describer, the two callers that write their own line ==


def _reranker():
    client = OpenRouter(api_key=OPENROUTER_KEY, server_url=OPENROUTER_URL, retry_config=None)
    return MemPalaceReranker(
        client=client,
        model="z-ai/glm-5.3-flash",
        api_key=OPENROUTER_KEY,
        providers=("deepinfra", "baseten"),
        sleep=lambda _s: None,
    )


def test_a_rerank_names_the_generation_that_answered_it(router, caplog):
    body = _openrouter_body(content='{"picks": [2, 1]}')
    router.post(OPENROUTER_CHAT_URL).mock(return_value=httpx.Response(200, json=body))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _reranker().rerank("cats", [{"text": "a"}, {"text": "b"}], 2, surface="turn0")
    assert _last_field(_llm_line(caplog)) == f"generation_id={OPENROUTER_ID}"


def test_a_rerank_that_falls_back_names_every_refused_generation(router, caplog):
    """Each refusal's id rides its own retry line, and the last one rides the fallback line."""
    refused = [f"gen-1791030923-refused{n}" for n in range(3)]
    router.post(OPENROUTER_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                429,
                json={"error": {"message": "rate limited", "code": 429}},
                headers={"X-Generation-Id": gen},
            )
            for gen in refused
        ]
    )
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        _reranker().rerank("cats", [{"text": "a"}, {"text": "b"}], 2, surface="turn0")

    retries = [r.getMessage() for r in caplog.records if r.getMessage().startswith(RETRY_HEAD)]
    assert [_last_field(line) for line in retries] == [f"generation_id={g}" for g in refused[:2]]
    fallback = _llm_line(caplog)
    assert "outcome=fallback" in fallback
    assert fallback.count("generation_id=") == 1
    assert _last_field(fallback) == f"generation_id={refused[2]}"


class _NamedDescriberProvider:
    """A describer adapter that logs its call the way a real one does, naming its generation."""

    provider = "openrouter"
    model = "google/gemini-3.8-flash"

    def supports_vision(self):
        return True

    def supports_video(self):
        return False

    def chat(self, messages, tools=None):
        log_llm_call(
            provider=self.provider, model=self.model, seconds=1.5, generation_id=OPENROUTER_ID
        )
        return Message.assistant(content="A tabby cat asleep on a windowsill.")


def test_a_describe_carries_the_id_its_adapter_recorded(caplog):
    """The adapter's line is held back by the capture seam; the id must survive onto the caller's."""
    from basecradle_harness import ImageContent

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        Describer(_NamedDescriberProvider(), "d/model").describe_images(
            [ImageContent(url="x", alt="cat.png")]
        )
    line = _llm_line(caplog)
    assert "purpose=helper" in line
    assert _last_field(line) == f"generation_id={OPENROUTER_ID}"


# === the NOC's columns ========================================================

#: Every expression the fleet dashboard reads off an `llm` or `llm retry` line, byte-for-byte as
#: `basecradle-noc` ``observability/ai-box.json`` spells them (read 2026-10-03). ClickHouse's
#: ``match()``/``extract()`` are partial matches, which is what `re.search` is.
NOC_EXPRESSIONS = (
    r" llm provider=",
    r"purpose=([a-z]+)",
    r"kind=([a-z.]+)",
    r"provider=([A-Za-z0-9._-]+)",
    r"model=([A-Za-z0-9._/:-]+)",
    r"endpoint=(\"[^\"]*\"|[A-Za-z0-9._-]+)",
    r"duration=([0-9.]+)s",
    r"tokens_in=([0-9]+)",
    r"tokens_out=([0-9]+)",
    r"cached_tokens=([0-9]+)",
    r"cost=([0-9.]+)",
    r"outcome=([a-z]+)",
    r"source=([A-Za-z0-9_-]+)",
    r"stage=([a-z_]+)",
    r"agent=([A-Za-z0-9._-]+)",
    r" rerank=on",
)


@pytest.mark.parametrize("generation", [OPENROUTER_ID, OPENAI_CHAT_ID, OPENAI_RESPONSE_ID, XAI_ID])
def test_the_field_moves_no_noc_column(caplog, generation):
    def line(**extra):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            log_llm_call(
                provider="openrouter",
                purpose="memory",
                kind="rerank",
                model="z-ai/glm-5.3-flash",
                seconds=1.25,
                endpoint="Sail Research",
                usage={"prompt_tokens": 4812, "completion_tokens": 611, "cost": 0.000846},
                outcome="ok",
                extra={"surface": "turn0", "pool": 20, "picked": 10},
                **extra,
            )
        return _llm_line(caplog)

    without, with_id = line(), line(generation_id=generation)
    assert with_id == f"{without} generation_id={generation}"
    for expression in NOC_EXPRESSIONS:
        before, after = re.search(expression, without), re.search(expression, with_id)
        assert (before and before.group(0)) == (after and after.group(0)), expression
