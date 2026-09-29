"""One timeout policy, on every adapter — issue #589.

On 2026-09-29 @glm-5.2's ninth step needed more than a minute, and every adapter's answer to "how
long may a call take?" was a constant that knew nothing about the call: a flat 60 s on the `openai`
and `openrouter` adapters, none at all on the native xAI one. The wall cut the answer off, the retry
re-sent the identical request into the identical wall, and the wake died — four wakes in a row.

Three things are pinned here, in the order the issue's definition of done names them:

- **the policy** — a short fixed connect, and a generation budget *fitted* to the call;
- **every adapter applies it, the same way** — the tests iterate the adapters, and for each one the
  budget is read off **what the transport actually carried** (the request's timeout on the HTTP
  wire, the deadline a real gRPC server sees), never off what the adapter decided;
- **the failure itself, reproduced and then survived** — a real loopback server that answers after
  the budget, first as the wall that killed the wake, then as the one larger-budget retry that
  answers it.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time
from concurrent import futures

import grpc
import httpx
import httpx2
import pytest
import respx
from xai_sdk.proto import chat_pb2, chat_pb2_grpc

from basecradle_harness import (
    Engine,
    Message,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderConnectionError,
    ProviderServerError,
    ProviderTimeoutError,
    ToolRegistry,
    ToolSpec,
    XaiSdkProvider,
    _timeouts,
)
from basecradle_harness._context import request_chars
from basecradle_harness._observability import RETRY_HEAD
from basecradle_harness._retry import connection_reason
from basecradle_harness._timeouts import (
    CONNECT_TIMEOUT,
    GENERATION_CEILING,
    GENERATION_FLOOR,
    GENERATION_STEP,
    METADATA_TIMEOUT,
    TIMEOUT_RETRY_SCALE,
    CallTimeout,
    bind_scale,
    call_timeout,
    generation_timeout,
    last_timeout,
    output_cap,
)
from tests.conftest import completion, out_message, responses_body
from tests.test_openrouter import completion as openrouter_completion

FAKE_KEY = "sk-test-0123456789abcdef0123456789abcdef"
OPENAI_BASE = "https://openai.test/v1"
OPENROUTER_BASE = "https://openrouter.test/api/v1"

TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


# --- the policy ---------------------------------------------------------------------------------


def test_no_call_gets_less_time_than_the_old_wall():
    """The floor is the 60 s every call used to get, so the fit can only ever add time — and since
    anything above the floor rounds up a whole step, the least any real call gets is two minutes."""
    assert generation_timeout(0, output_tokens=1) >= GENERATION_FLOOR
    assert generation_timeout(0, output_tokens=1) == GENERATION_FLOOR + GENERATION_STEP


def test_the_incidents_call_gets_the_time_it_needed():
    """~40 K tokens in, uncapped — the call that ran out of a flat minute, fitted.

    Measured the way `request_chars` measures (characters as the model reads them) and converted at
    `_context.WORST_CASE_CHARS_PER_TOKEN`, so 120 K characters is 40 K tokens. The fit is the floor,
    40 s to read it, and ~410 s to write an uncapped answer at the slow-but-alive decode floor —
    510 s, rounded up to whole minutes.
    """
    assert generation_timeout(120_000, output_tokens=None) == 540.0


def test_the_fit_grows_with_the_request_and_with_the_answer():
    small = generation_timeout(3_000, output_tokens=None)
    assert generation_timeout(600_000, output_tokens=None) > small  # more to read
    assert generation_timeout(3_000, output_tokens=2_048) < small  # a cap is less to write


def test_the_fit_is_capped_and_rounded_up_to_whole_steps():
    for chars in (0, 7_777, 120_000, 5_000_000):
        budget = generation_timeout(chars, output_tokens=None)
        assert budget % GENERATION_STEP == 0
        assert budget <= GENERATION_CEILING


def test_the_retry_budget_is_materially_larger_even_at_the_ceiling():
    """The scale applies *after* the cap, or a ceiling-sized call's retry would be no larger."""
    first = generation_timeout(50_000_000, output_tokens=None)
    assert first == GENERATION_CEILING
    assert generation_timeout(50_000_000, output_tokens=None, scale=TIMEOUT_RETRY_SCALE) == (
        TIMEOUT_RETRY_SCALE * GENERATION_CEILING
    )


def test_connect_is_fixed_whatever_the_call():
    """Reaching an endpoint does not get slower as a conversation grows."""
    for chars in (0, 120_000, 5_000_000):
        assert call_timeout(chars, output_tokens=None).connect == CONNECT_TIMEOUT


def test_the_retry_doubles_every_phase_so_no_timeout_is_retried_identically():
    """A connect timeout is a timeout too: its one retry gets more time to connect, not the same."""
    first = call_timeout(120_000, output_tokens=None)
    retry = call_timeout(120_000, output_tokens=None, scale=TIMEOUT_RETRY_SCALE)
    assert retry.connect == TIMEOUT_RETRY_SCALE * first.connect
    assert retry.generation == TIMEOUT_RETRY_SCALE * first.generation


def test_a_fixed_timeout_overrides_the_fit_and_still_scales():
    assert call_timeout(120_000, output_tokens=None, fixed=30.0).generation == 30.0
    assert call_timeout(0, output_tokens=None, fixed=30.0, scale=2.0).generation == 60.0


def test_the_read_phase_is_the_generation():
    """A non-streaming response sends nothing until it is done, so the read wait *is* the answer."""
    assert CallTimeout(connect=10.0, generation=540.0).phases() == {
        "connect": 10.0,
        "read": 540.0,
        "write": 540.0,
        "pool": 10.0,
    }


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_the_output_cap_is_read_in_every_spelling(key):
    assert output_cap({key: 2_048, "temperature": 0.2}) == 2_048


@pytest.mark.parametrize("value", [True, 0, -5, "2048", 2048.0, None])
def test_a_value_that_is_not_a_cap_never_shortens_the_budget(value):
    assert output_cap({"max_tokens": value}) is None


def test_bind_scale_is_a_capability_and_never_breaks_a_wake(caplog):
    class Fits:
        scale = None

        def bind_timeout_scale(self, scale):
            self.scale = scale

    class Raises:
        def bind_timeout_scale(self, scale):
            raise RuntimeError("no")

    fits = Fits()
    bind_scale(fits, 2.0)
    assert fits.scale == 2.0
    bind_scale(object(), 2.0)  # an adapter that does not fit its timeouts: nothing happens
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        bind_scale(Raises(), 2.0)
    assert "timeout scale" in caplog.text


def test_last_timeout_reads_only_a_number():
    class Reports:
        last_timeout = 540.0

    class Lies:
        last_timeout = True

    assert last_timeout(Reports()) == 540.0
    assert last_timeout(Lies()) is None
    assert last_timeout(object()) is None


# --- every adapter applies it, the same way -----------------------------------------------------
#
# The HTTP adapters are read at the transport (respx sees the request's own timeout extension, which
# is what the HTTP client enforces); the native xAI adapter is read at a real gRPC server on
# loopback, which sees the deadline the SDK put on the call. Neither asks the adapter what it did.

HTTP_CELLS = ("openai-responses", "openai-chat", "openrouter")


def _http_provider(cell: str, **kw):
    if cell == "openrouter":
        return OpenRouterProvider("z-ai/glm-5.2", api_key=FAKE_KEY, base_url=OPENROUTER_BASE, **kw)
    surface = "responses" if cell == "openai-responses" else "chat"
    return OpenAIProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=OPENAI_BASE, surface=surface, **kw
    )


def _http_url(cell: str) -> str:
    if cell == "openrouter":
        return f"{OPENROUTER_BASE}/chat/completions"
    return (
        f"{OPENAI_BASE}/responses"
        if cell == "openai-responses"
        else f"{OPENAI_BASE}/chat/completions"
    )


def _http_ok(cell: str) -> dict:
    if cell == "openrouter":
        return openrouter_completion(content="ok")
    return (
        responses_body(out_message("ok"))
        if cell == "openai-responses"
        else completion(content="ok")
    )


def _read_timeout(cell: str) -> Exception:
    """A read timeout from the HTTP family each SDK really runs on (the `openai` SDK: HTTPX2)."""
    if cell == "openrouter":
        return httpx.ReadTimeout("The read operation timed out")
    return httpx2.ReadTimeout("The read operation timed out")


MESSAGES = [Message.system("You are helpful."), Message.user("What is the weather in Dallas?")]


@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_every_http_adapter_puts_the_fitted_budget_on_the_wire(cell):
    expected = call_timeout(request_chars(MESSAGES, [TOOL]), output_tokens=None).phases()
    provider = _http_provider(cell)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(_http_url(cell)).mock(
            return_value=httpx.Response(200, json=_http_ok(cell))
        )
        provider.chat(MESSAGES, tools=[TOOL])

    assert route.calls.last.request.extensions["timeout"] == expected
    assert provider.last_timeout == expected["read"]


@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_every_http_adapter_fits_its_output_cap(cell):
    """A capped call is given the time to write its cap, and no more."""
    expected = call_timeout(request_chars(MESSAGES, None), output_tokens=256).generation
    cap = {"openai-responses": "max_output_tokens"}.get(cell, "max_tokens")
    provider = _http_provider(cell, **{cap: 256})
    with respx.mock(assert_all_called=True) as router:
        route = router.post(_http_url(cell)).mock(
            return_value=httpx.Response(200, json=_http_ok(cell))
        )
        provider.chat(MESSAGES)

    assert route.calls.last.request.extensions["timeout"]["read"] == expected


@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_every_http_adapter_gives_the_one_timeout_retry_twice_the_budget(cell):
    base = call_timeout(request_chars(MESSAGES, None), output_tokens=None).generation
    provider = _http_provider(cell)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(_http_url(cell)).mock(
            return_value=httpx.Response(200, json=_http_ok(cell))
        )
        provider.bind_timeout_scale(TIMEOUT_RETRY_SCALE)
        provider.chat(MESSAGES)
        provider.bind_timeout_scale(1.0)
        provider.chat(MESSAGES)

    budgets = [call.request.extensions["timeout"] for call in route.calls]
    assert [b["read"] for b in budgets] == [TIMEOUT_RETRY_SCALE * base, base]  # and back again
    assert [b["connect"] for b in budgets] == [
        TIMEOUT_RETRY_SCALE * CONNECT_TIMEOUT,
        CONNECT_TIMEOUT,
    ]


@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_every_http_adapter_names_a_timeout_a_timeout(cell):
    provider = _http_provider(cell)
    with respx.mock() as router:
        router.post(_http_url(cell)).mock(side_effect=_read_timeout(cell))
        with pytest.raises(ProviderTimeoutError) as raised:
            provider.chat(MESSAGES)

    assert isinstance(raised.value, ProviderConnectionError)  # every transport `except` still holds
    assert connection_reason(raised.value) == "timeout"


@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_no_http_adapter_retries_inside_its_sdk(cell):
    """The engine's retry is the one policy; an SDK retrying underneath it composed attempts and —
    on the `openai` SDK — re-sent a timed-out request with the identical budget."""
    provider = _http_provider(cell)
    with respx.mock() as router:
        route = router.post(_http_url(cell)).mock(
            return_value=httpx.Response(500, json={"error": {"code": 500, "message": "boom"}})
        )
        with pytest.raises(ProviderServerError):
            provider.chat(MESSAGES)

    assert route.call_count == 1


class _DeadlineChat(chat_pb2_grpc.ChatServicer):
    """A real gRPC chat service that records the deadline each call arrived with, and can stall."""

    def __init__(self) -> None:
        self.remaining: list[float] = []
        self.delay = 0.0
        self.calls = 0

    def GetCompletion(self, request, context):  # the gRPC method name, PascalCase by protocol
        self.calls += 1
        self.remaining.append(context.time_remaining())
        if self.delay:
            time.sleep(self.delay)
        return chat_pb2.GetChatCompletionResponse(
            id="resp-1",
            model="grok-4.3",
            outputs=[
                chat_pb2.CompletionOutput(
                    message=chat_pb2.CompletionMessage(
                        role=chat_pb2.MessageRole.ROLE_ASSISTANT, content="ok"
                    )
                )
            ],
        )


@pytest.fixture
def grpc_chat():
    servicer = _DeadlineChat()
    pool = futures.ThreadPoolExecutor(max_workers=4)
    server = grpc.server(pool)
    chat_pb2_grpc.add_ChatServicer_to_server(servicer, server)
    port = server.add_secure_port(
        "localhost:0", grpc.local_server_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    )
    server.start()
    try:
        yield f"localhost:{port}", servicer
    finally:
        server.stop(None).wait()
        pool.shutdown(wait=False)


def test_the_native_xai_adapter_puts_the_fitted_deadline_on_the_wire(grpc_chat):
    """Issue #589's fourth defect: this adapter passed no timeout at all, so the SDK's 27 minutes
    applied — one harness, two rules. The server reads the deadline the call actually carried."""
    host, servicer = grpc_chat
    expected = call_timeout(request_chars(MESSAGES, None), output_tokens=None).generation
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, api_host=host)
    try:
        provider.chat(MESSAGES)
        provider.bind_timeout_scale(TIMEOUT_RETRY_SCALE)
        provider.chat(MESSAGES)
    finally:
        provider.close()

    # gRPC encodes a deadline in whole units on the wire, so what the server reads is within a
    # second or so of the budget either way — nowhere near the SDK's own 1,620 s.
    first, retried = servicer.remaining
    assert first == pytest.approx(expected, abs=5)
    assert retried == pytest.approx(TIMEOUT_RETRY_SCALE * expected, abs=5)


def test_the_native_xai_adapter_names_a_deadline_a_timeout(grpc_chat):
    """A real deadline, really exceeded — not a mapped status code handed to a fake."""
    host, servicer = grpc_chat
    servicer.delay = 1.0
    provider = XaiSdkProvider("grok-4.3", api_key=FAKE_KEY, api_host=host, timeout=0.2)
    try:
        with pytest.raises(ProviderTimeoutError) as raised:
            provider.chat(MESSAGES)
    finally:
        provider.close()

    assert connection_reason(raised.value) == "timeout"
    assert servicer.calls == 1  # the SDK's own gRPC retry re-sends only UNAVAILABLE, never this


# --- the failure, reproduced and survived -------------------------------------------------------
#
# A real HTTP server on loopback that answers only after `ANSWER_AFTER` seconds — the shape of the
# incident's slow generation, scaled from minutes to a second. The policy's constants are scaled
# with it (`_small_policy`), so the fit it produces is measured against a clock, not asserted.

ANSWER_AFTER = 1.2  # the slow answer
SMALL_FLOOR = 0.8  # the scaled floor: a small request's fit lands below the answer
SMALL = [Message.user("hi")]


class _SlowServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _slow_handler(counter: list[int]):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            counter.append(1)
            self.rfile.read(int(self.headers.get("content-length") or 0))
            time.sleep(ANSWER_AFTER)
            if self.path.endswith("/responses"):
                body = responses_body(out_message("the real answer"))
            elif self.path.startswith("/api/"):
                body = openrouter_completion(content="the real answer")
            else:
                body = completion(content="the real answer")
            payload = json.dumps(body).encode()
            try:
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client gave up at its budget, which is the point of the first attempt

        def log_message(self, *args):  # keep the test output clean
            pass

    return Handler


@pytest.fixture
def slow_server():
    counter: list[int] = []
    server = _SlowServer(("127.0.0.1", 0), _slow_handler(counter))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", counter
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def _small_policy(monkeypatch):
    """The fit, scaled from minutes to fractions of a second: a floor of `SMALL_FLOOR`, a read rate
    that makes a large request's size count, and no output term."""
    monkeypatch.setattr(_timeouts, "GENERATION_FLOOR", SMALL_FLOOR)
    monkeypatch.setattr(_timeouts, "GENERATION_STEP", 0.1)
    monkeypatch.setattr(_timeouts, "UNCAPPED_OUTPUT_TOKENS", 0)
    monkeypatch.setattr(_timeouts, "PREFILL_TOKENS_PER_SECOND", 1_000)


def _slow_provider(cell: str, base: str):
    if cell == "openrouter":
        return OpenRouterProvider("z-ai/glm-5.2", api_key=FAKE_KEY, base_url=f"{base}/api/v1")
    surface = "responses" if cell == "openai-responses" else "chat"
    return OpenAIProvider(
        model="gpt-5.4-mini", api_key=FAKE_KEY, base_url=f"{base}/v1", surface=surface
    )


@pytest.mark.usefixtures("_small_policy")
@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_a_budget_shorter_than_the_answer_is_the_wall_that_killed_the_wake(cell, slow_server):
    """The failure, reproduced: a budget below the answer's time raises a timeout on every adapter.

    This is the 60 s wall, scaled down — and before issue #589 it was also every retry's budget, so
    the engine re-sent the request into it twice more and gave up.
    """
    base, counter = slow_server
    provider = _slow_provider(cell, base)
    fit = generation_timeout(request_chars(SMALL), output_tokens=None)
    assert fit < ANSWER_AFTER  # the premise: this budget is the wall

    with pytest.raises(ProviderTimeoutError):
        provider.chat(SMALL)

    assert provider.last_timeout == pytest.approx(fit)
    assert counter == [1]


@pytest.mark.usefixtures("_small_policy")
@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_the_one_larger_budget_retry_answers_the_slow_call(cell, slow_server, caplog):
    """The failure, survived: the engine retries the timeout once at twice the budget, and the slow
    answer lands. The retry line says which kind of retry it was, in numbers."""
    base, counter = slow_server
    engine = Engine(_slow_provider(cell, base), ToolRegistry(), sleep=lambda _seconds: None)
    fit = generation_timeout(request_chars(SMALL), output_tokens=None)
    assert fit < ANSWER_AFTER < TIMEOUT_RETRY_SCALE * fit  # the premise: only the retry fits it

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        reply = engine.run(list(SMALL))

    assert reply.content == "the real answer"
    assert len(counter) == 2  # the attempt that ran out of time, and its one retry
    retry = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(RETRY_HEAD))
    assert "reason=timeout" in retry
    assert (
        f"timeout={fit:.2f}s timeout_scale=2 next_timeout={TIMEOUT_RETRY_SCALE * fit:.2f}s"
    ) in retry


@pytest.mark.usefixtures("_small_policy")
@pytest.mark.parametrize("cell", HTTP_CELLS)
def test_a_large_request_is_given_the_time_its_size_needs(cell, slow_server):
    """The fit, doing its job: the same slow answer to a large request lands on the first attempt,
    because reading a large request is part of what the budget is fitted to."""
    base, counter = slow_server
    provider = _slow_provider(cell, base)
    large = Message.user("weather " * 400)  # ~3.2 K characters → ~1.1 s of reading at this rate

    assert provider.chat([large]).content == "the real answer"
    assert counter == [1]
    assert provider.last_timeout > ANSWER_AFTER


# --- a metadata read is not a model call --------------------------------------------------------


def test_a_metadata_read_keeps_its_flat_minute_before_and_after_a_turn():
    """A model's context window is a catalog lookup, not a generation, and is not fitted.

    With no client-wide timeout left on the HTTP adapters it would otherwise take whatever the
    client held: `httpx`'s own five seconds before the first turn, and the last turn's fitted
    budget after it — neither of them a decision anybody made.
    """
    from tests.test_openrouter import ENDPOINTS_URL, _endpoint, _endpoints_body

    metadata = dict.fromkeys(("connect", "read", "write", "pool"), METADATA_TIMEOUT)
    provider = OpenRouterProvider(
        "z-ai/glm-5.2", api_key=FAKE_KEY, base_url="https://openrouter.test/api/v1"
    )
    with respx.mock(assert_all_called=True) as router:
        endpoints = router.get(ENDPOINTS_URL).mock(
            return_value=httpx.Response(
                200, json=_endpoints_body(_endpoint(name="Novita", context_length=1_048_576))
            )
        )
        router.post(f"{OPENROUTER_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=_http_ok("openrouter"))
        )
        assert provider.context_limit() == 1_048_576
        provider.chat(MESSAGES)
        assert provider.context_limit() == 1_048_576

    assert [call.request.extensions["timeout"] for call in endpoints.calls] == [metadata, metadata]


def test_an_openai_metadata_read_keeps_its_flat_minute():
    provider = _http_provider("openai-chat")
    with respx.mock(assert_all_called=True) as router:
        models = router.get(f"{OPENAI_BASE}/models/gpt-5.4-mini").mock(
            return_value=httpx.Response(
                200,
                json={"id": "gpt-5.4-mini", "object": "model", "created": 0, "owned_by": "x"},
            )
        )
        provider.context_limit()

    assert models.calls.last.request.extensions["timeout"]["read"] == METADATA_TIMEOUT
