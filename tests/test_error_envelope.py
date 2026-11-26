"""A vendor's error envelope is a fault, never a billed call (issue #1029).

On 2026-11-26 Refuseless answered every call to ``kimi-k3.1-pro-abliterated`` with the Responses
API's own ``error`` field and nothing else: no ``id``, no ``output``, no ``usage``. The adapter
could not read a turn out of it, and `read_answer` turned that into ``outcome=fallback
reason=invalid_response`` on the **billed** line, a call with no tokens. The fleet's *Log
Extraction Broken* alert counts exactly that, and paged three times.

The rule these tests hold: **a billed line needs evidence of an answer**. An answer that was read
is its own evidence. An answer that could not be read leaves only the vendor's bill (the usage it
stated, a charge it named); with neither, the body was no answer, and the attempt writes only the
harness's decision line (`retrying` or `failed`). Both shapes are held here, side by side, on every
surface that reads a body: an envelope writes no billed line, and an answer the vendor billed and
the parser tripped on still writes its ``fallback`` with its tokens.

The envelopes are the fleet's (basecradle-noc#1790). ``model_unavailable`` is as the failed line
quoted it; of ``gateway_error`` the fleet's lines kept only the code, so its message is a stand-in.
"""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from basecradle_harness import Message, OpenAIProvider
from basecradle_harness._exceptions import ProviderResponseError, ProviderTruncatedError
from basecradle_harness._observability import billed, log_answered_call

from .conftest import plain
from .test_openrouter import CHAT_URL as OPENROUTER_CHAT
from .test_openrouter import _provider as openrouter_provider
from .test_refuseless import FAKE_KEY, HOST, LIST, VENDOR

MODEL = "kimi-k3.1-pro-abliterated"

#: Refuseless's envelope of 2026-11-26, as the `failed` line's ``detail`` quoted it.
MODEL_UNAVAILABLE = {
    "error": {
        "code": "model_unavailable",
        "message": "The model is unavailable.",
        "type": "api_error",
    }
}

#: Refuseless's envelope of 2026-11-11 (`ernie`, `bert`). The code is the fleet's; the message is a
#: stand-in, because the lines that named the code kept nothing else of the body.
GATEWAY_ERROR = {
    "error": {"code": "gateway_error", "message": "Upstream gateway error.", "type": "api_error"}
}

ENVELOPES = pytest.mark.parametrize(
    ("envelope", "code"),
    [(MODEL_UNAVAILABLE, "model_unavailable"), (GATEWAY_ERROR, "gateway_error")],
    ids=["model_unavailable", "gateway_error"],
)

SURFACES = pytest.mark.parametrize(
    ("surface", "path"), [("responses", "/responses"), ("chat", "/chat/completions")]
)


def _refuseless(surface: str) -> OpenAIProvider:
    return OpenAIProvider(MODEL, api_key=FAKE_KEY, base_url=HOST, provider=VENDOR, surface=surface)


def _llm_lines(caplog) -> list[str]:
    return [plain(r.getMessage()) for r in caplog.records if r.getMessage().startswith("llm ")]


@ENVELOPES
@SURFACES
def test_an_error_envelope_writes_no_billed_line(envelope, code, surface, path, caplog):
    """The envelope is the vendor declining to serve: no billed line, and the error the engine
    retries (`ProviderResponseError`, so `invalid_response`, the class it always was) names the
    vendor's own code and message rather than dumping a hundred characters of ``None``."""
    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(return_value=httpx.Response(200, json=LIST))
        sent = router.post(f"{HOST}{path}").mock(return_value=httpx.Response(200, json=envelope))
        provider = _refuseless(surface)
        with pytest.raises(ProviderResponseError) as raised:
            provider.chat([Message.user("hi")])
        provider.close()
    assert sent.call_count == 1
    assert type(raised.value) is ProviderResponseError  # retried as it always was
    assert str(raised.value).startswith("The vendor sent an error in place of an answer: ")
    assert f"{code}: {envelope['error']['message']}" in str(raised.value)
    assert _llm_lines(caplog) == []


@ENVELOPES
def test_an_error_envelope_through_the_router_writes_no_billed_line(envelope, code, caplog):
    """The native `openrouter` adapter reads a chat body through the same reader and the same rule.
    Its SDK's typed model accepts a body only with the fields it requires, and it drops a field it
    does not name, the ``error`` included, so the error cannot quote the vendor here: the rule does
    not need it, because what decides is the missing bill."""
    body = {
        "id": "gen-0123456789abcdef",
        "object": "chat.completion",
        "created": 1791030923,
        "model": "z-ai/glm-5.3",
        "system_fingerprint": "fp_test",
        "choices": [],
        **envelope,
    }
    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True) as router:
        router.post(OPENROUTER_CHAT).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(ProviderResponseError) as raised:
            openrouter_provider(retries_disabled=True).chat([Message.user("hi")])
    assert type(raised.value) is ProviderResponseError
    assert raised.value.generation_id == "gen-0123456789abcdef"  # still names the attempt
    assert _llm_lines(caplog) == []


@SURFACES
def test_a_billed_answer_that_cannot_be_read_still_writes_its_fallback(surface, path, caplog):
    """The other shape, beside the first: the vendor answered and billed, and the parser could not
    read it (the `ping` lines of 2026-11-03). Its line is the billed ``fallback`` with its tokens,
    as before; only a body with no bill lost its line."""
    if surface == "responses":
        body = {"id": "resp_0123456789abcdef", "object": "response", "status": "completed"}
        body["usage"] = {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100}
    else:
        body = {"id": "chatcmpl-0123456789abcdef", "object": "chat.completion", "choices": []}
        body["usage"] = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}
    body["model"] = MODEL
    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(return_value=httpx.Response(200, json=LIST))
        router.post(f"{HOST}{path}").mock(return_value=httpx.Response(200, json=body))
        provider = _refuseless(surface)
        with pytest.raises(ProviderResponseError) as raised:
            provider.chat([Message.user("hi")])
        provider.close()
    assert not str(raised.value).startswith("The vendor sent an error")
    (line,) = _llm_lines(caplog)
    assert " tokens_in=90 tokens_out=10 " in line
    assert " outcome=fallback reason=invalid_response " in line


def test_an_envelope_beside_a_bill_is_still_the_billed_fallback(caplog):
    """An ``error`` field is not what decides: the bill is. A body that states usage was billed,
    whatever else it carries, so it keeps its line and the parser's own message."""
    body = {
        "id": "chatcmpl-0123456789abcdef",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [],
        "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
        **MODEL_UNAVAILABLE,
    }
    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(return_value=httpx.Response(200, json=LIST))
        router.post(f"{HOST}/chat/completions").mock(return_value=httpx.Response(200, json=body))
        provider = _refuseless("chat")
        with pytest.raises(ProviderResponseError) as raised:
            provider.chat([Message.user("hi")])
        provider.close()
    assert str(raised.value).startswith("Malformed chat completion response")
    (line,) = _llm_lines(caplog)
    assert " tokens_in=90 " in line and " outcome=fallback reason=invalid_response " in line


# === the rule itself, for every adapter that writes through it ==================


def test_the_evidence_of_an_answer():
    """`billed`, case by case: a read answer needs nothing more; an unread one needs a bill."""
    unread = ProviderResponseError("no answer in the body")
    assert billed(None) and billed(None, usage=None)  # a read answer is its own evidence (#491)
    assert billed(unread, usage={"prompt_tokens": 90, "completion_tokens": 10})
    assert billed(unread, cost=0.0021)  # xAI states its charge on the response
    assert not billed(unread)
    assert not billed(unread, usage={"prompt_tokens": 0, "completion_tokens": 0}, cost=0)
    assert not billed(ProviderTruncatedError("cut off"), usage=None)


def test_log_answered_call_reads_the_lines_own_usage_and_cost(caplog):
    """The evidence is what the line would print, so an adapter cannot pass one and print another.
    A stated charge with no usage keeps the line (xAI's case); nothing at all writes none."""
    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    unread = ProviderResponseError("no answer in the body")
    log_answered_call(unread, provider="xai", model="grok-4.3", seconds=0.4, cost=0.0021)
    log_answered_call(unread, provider="xai", model="grok-4.3", seconds=0.4)
    log_answered_call(None, provider="xai", model="grok-4.3", seconds=0.4)
    charged, read = _llm_lines(caplog)
    assert " cost=0.0021 " in charged and charged.endswith(
        " outcome=fallback reason=invalid_response"
    )
    assert " outcome=ok" in read


def test_the_describer_writes_only_its_decisions_for_an_envelope(caplog):
    """The describer writes a captured call's line itself, and writes a billed one only for an
    attempt its adapter recorded (`_bill_unreadable`). The adapter records nothing for an envelope,
    so the describe's lines are its decisions alone, with no tokens and no cost on any of them."""
    from basecradle_harness._describer import Describer
    from basecradle_harness._messages import ImageContent

    caplog.set_level(logging.DEBUG, logger="basecradle_harness")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(return_value=httpx.Response(200, json=LIST))
        router.post(f"{HOST}/chat/completions").mock(
            return_value=httpx.Response(200, json=MODEL_UNAVAILABLE)
        )
        provider = _refuseless("chat")
        describer = Describer(provider, MODEL, sleep=lambda _s: None)
        image = ImageContent(url="data:image/png;base64,iVBORw0KGgo=", alt="a.png")
        assert describer.describe_images([image]) is None
        provider.close()
    lines = _llm_lines(caplog)
    assert lines and all("outcome=retrying" in m or "outcome=failed" in m for m in lines)
    assert not any("tokens_in=" in m or "cost=" in m for m in lines)
    assert "model_unavailable" in lines[-1]
