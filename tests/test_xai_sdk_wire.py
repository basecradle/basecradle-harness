"""The affinity key on the **actual gRPC wire** — issue #433 (NOC finding basecradle#512).

Every other test of the native adapter injects a fake client, which is exactly the blind spot this
file exists to cover. Version 0.110.0 shipped the cache-affinity key as
``chat.create(conversation_id=...)`` with a full suite of passing tests behind it — and ``xai-sdk``
1.19.0 *accepts* that keyword and then spends it on an OpenTelemetry span attribute. It reaches no
request proto, so it reached no xAI server, so the ~75% cached-prefix discount stayed unearned while
everything looked green. **A key a Python signature accepts is not a key on the wire.**

So these tests do not assert against a fake. They stand up a **real gRPC server** on loopback and
drive a **real** ``xai_sdk.Client`` — built by the real `XaiSdkProvider` — at it, then read back the
metadata the SDK actually put on the connection (``context.invocation_metadata()``). The one thing
that is not production is the channel *credential*: ``xai_sdk`` swaps TLS for
``grpc.local_channel_credentials()`` when the host is ``localhost:``, which is its own doing and
leaves the metadata path (``create_channel_credentials`` → ``_APIAuthPlugin``) byte-identical to the
one that talks to ``api.x.ai``. No network leaves the box, so this runs in the default offline suite.

What only the live endpoint can answer — whether xAI *routes* on the key it now receives — is the
``cached_tokens=`` probe in `test_xai_sdk_live.py` and the organic hit-rate check on a real agent.
"""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest
from xai_sdk.proto import chat_pb2, chat_pb2_grpc

from basecradle_harness import Message
from basecradle_harness._xai_sdk import CONVERSATION_METADATA_KEY, XaiSdkProvider

FAKE_KEY = "xai-test-0123456789abcdef"
SESSION = "timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2"
OTHER_SESSION = "timeline:019f6e88-0d31-77a2-8b41-2c5a90c1e7d4"


class _RecordingChat(chat_pb2_grpc.ChatServicer):
    """A real gRPC chat service that answers ``ok`` and remembers the headers it was called with."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, str]] = []

    def GetCompletion(self, request, context):  # the gRPC method name, PascalCase by protocol
        self.metadata.append(dict(context.invocation_metadata()))
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
def xai_server():
    """A loopback gRPC server speaking xAI's real chat service; yields ``(host, servicer)``."""
    servicer = _RecordingChat()
    pool = futures.ThreadPoolExecutor(max_workers=2)
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
        pool.shutdown(wait=False)  # `server.stop` does not own the executor it was handed


def _provider(host: str) -> XaiSdkProvider:
    """A real adapter — real ``xai_sdk.Client``, real channel — pointed at the loopback server."""
    return XaiSdkProvider(model="grok-4.3", api_key=FAKE_KEY, api_host=host, timeout=10)


def _keys(servicer: _RecordingChat) -> list[str | None]:
    return [md.get(CONVERSATION_METADATA_KEY) for md in servicer.metadata]


def test_the_bound_conversation_reaches_the_server_as_grpc_metadata(xai_server):
    """The whole fix, proven where 0.110.0 could not be: the key is on the connection xAI reads.

    Also round-trips a real response back through the adapter, so this is the genuine end-to-end
    path — `XaiSdkProvider.chat` → `xai_sdk` → HTTP/2 → a server → a harness `Message` — and not a
    request that merely got built.
    """
    host, servicer = xai_server
    provider = _provider(host)
    provider.bind_conversation(SESSION)
    try:
        reply = provider.chat([Message.user("Hi")])
    finally:
        provider.close()

    assert reply.content == "ok"
    assert _keys(servicer) == [SESSION]


def test_nothing_rides_the_wire_until_a_conversation_is_bound(xai_server):
    """Unbound the header is **absent**, not blank: a fabricated id is a fresh conversation to xAI
    on every call, which trades a lucky hit for a guaranteed miss."""
    host, servicer = xai_server
    provider = _provider(host)
    try:
        provider.chat([Message.user("Hi")])
    finally:
        provider.close()

    assert CONVERSATION_METADATA_KEY not in servicer.metadata[0]


def test_consecutive_calls_in_one_session_carry_the_same_key(xai_server):
    """Affinity is only worth anything if it *holds* — one key across the session's whole life is
    what routes call two back to the server call one warmed."""
    host, servicer = xai_server
    provider = _provider(host)
    provider.bind_conversation(SESSION)
    try:
        provider.chat([Message.user("Hi")])
        provider.bind_conversation(SESSION)  # what `Session._drive` does before every turn
        provider.chat([Message.user("Again")])
    finally:
        provider.close()

    assert _keys(servicer) == [SESSION, SESSION]


def test_rebinding_switches_the_key_on_the_wire(xai_server):
    """One adapter serves every session, so a switch has to actually take — a stale key would herd
    a second timeline's prefix onto the first one's server, which is the anti-pattern, not the fix.

    This is also the test that pins the *mechanism*: the SDK bakes metadata into a client at
    construction, so the only way the second call differs from the first is a rebuilt client.
    """
    host, servicer = xai_server
    provider = _provider(host)
    try:
        provider.bind_conversation(SESSION)
        provider.chat([Message.user("Hi")])
        provider.bind_conversation(OTHER_SESSION)
        provider.chat([Message.user("Hi")])
    finally:
        provider.close()

    assert _keys(servicer) == [SESSION, OTHER_SESSION]


def test_clearing_the_binding_takes_the_key_back_off_the_wire(xai_server):
    """`None` clears, all the way to the connection — the `Provider` capability's stated contract,
    not merely a field left out of a payload."""
    host, servicer = xai_server
    provider = _provider(host)
    try:
        provider.bind_conversation(SESSION)
        provider.chat([Message.user("Hi")])
        provider.bind_conversation(None)
        provider.chat([Message.user("Hi")])
    finally:
        provider.close()

    assert _keys(servicer) == [SESSION, None]


def test_a_key_grpc_would_reject_costs_the_hit_and_not_the_wake(xai_server):
    """The guard, proven against the thing it guards: grpc refuses a non-printable-ASCII metadata
    value **at call time** (verified — the same client with `timeline:\\u30a6` or an embedded newline
    dies with an `RpcError`, which `_mapped_errors` turns into a failed model call and a failed
    wake). Affinity is an optimization, so an unusable key is dropped on the way in and the call
    goes out unbound: no hit, but an answer.
    """
    host, servicer = xai_server
    provider = _provider(host)
    provider.bind_conversation("timeline:ウィ")  # not ASCII; grpc could not carry it
    try:
        reply = provider.chat([Message.user("Hi")])
    finally:
        provider.close()

    assert reply.content == "ok"
    assert CONVERSATION_METADATA_KEY not in servicer.metadata[0]
