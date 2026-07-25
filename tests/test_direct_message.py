"""The send_direct_message_to_origin tool: the push, the byte cap, the failure modes (issue #341).

All HTTP is mocked with respx — no test reaches ntfy.sh. The token here is a correctly-shaped
fake; the real one lives only in an agent's `agent.env`.

Four things these pin, in rough order of what would hurt most if it broke:

- The **wire shape**, header for header. @origin's phone renders the title, and the title carries
  the *sending agent's own* handle read off the live platform identity — never a hardcoded name.
- The **byte cap is refused, never truncated**, and its error names the real byte count so the
  model can shorten and retry. Past 4,096 bytes ntfy silently converts the message into a `.txt`
  attachment, so a "successful" oversize send is a broken DM.
- **Nothing fails silently, and nothing leaks the token** — every failure comes back as readable
  text, and the credential appears in no result string and no log record.
- It is **opt-in**, and it records **nothing** in the speech ledger: a push notification is not a
  timeline action, and counting it as one would make a silent wake look like it spoke.
"""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from basecradle_harness import (
    DirectMessageTool,
    PlatformContext,
    PlatformError,
    Policy,
    ToolRegistry,
)
from basecradle_harness._direct_message import (
    DEFAULT_BASE_URL,
    DEFAULT_TOPIC,
    MAX_BODY_BYTES,
    TOKEN_ENV,
)
from basecradle_harness._unspoken import SpeechLedger

TOKEN = "tk_fake_ntfy_publish_token_000"
NTFY_URL = f"{DEFAULT_BASE_URL}/{DEFAULT_TOPIC}"

# The fabricated cast: Nova Digital is the AI doing the sending (CLAUDE.md → Conventions).
NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"


class _Identity:
    def __init__(self, handle):
        self.uuid = NOVA_UUID
        self.handle = handle


class _Dashboard:
    def __init__(self, handle):
        self.identity = _Identity(handle)


class _FakeClient:
    """The bits of the BaseCradle SDK client this tool touches: `me.identity.handle`, no more."""

    def __init__(self, handle="nova", fail=False):
        self._handle = handle
        self._fail = fail
        self.me_reads = 0

    @property
    def me(self):
        self.me_reads += 1
        if self._fail:
            raise RuntimeError("the platform is unreachable")
        return _Dashboard(self._handle)


def _context(handle="nova", client=None, speech=None):
    return PlatformContext(
        client=client or _FakeClient(handle),
        timeline=TIMELINE_UUID,
        handle=handle,
        speech=speech,
    )


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """No ambient credential leaks into a test — each drives the token explicitly."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)


@pytest.fixture
def tool():
    """A bound tool with the token injected and the retry pause stubbed out."""
    tool = DirectMessageTool(token=TOKEN, sleep=lambda _seconds: None)
    tool.bind(_context())
    return tool


def _accepted(request=None):
    """ntfy's success response: the stored message, echoed back."""
    return httpx.Response(200, json={"id": "fake-msg-id", "topic": DEFAULT_TOPIC})


# --- the wire shape ----------------------------------------------------------


@respx.mock
def test_the_push_carries_the_topic_headers_and_a_plain_text_body(tool):
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    result = tool.run(body="The deploy is green.")

    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.headers["Title"] == "BaseCradle DM from @nova"
    assert request.headers["Priority"] == "5"
    assert request.headers["Tags"] == "speech_balloon"
    assert request.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert "Markdown" not in request.headers  # plain text, never a markdown flag
    assert request.content == b"The deploy is green."
    assert "Sent" in result


@respx.mock
def test_the_title_names_this_agents_own_handle_never_a_hardcoded_one():
    # The whole point of resolving identity: from @briggs it must say @briggs.
    tool = DirectMessageTool(token=TOKEN)
    tool.bind(_context(handle="briggs"))
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="hello")

    assert route.calls.last.request.headers["Title"] == "BaseCradle DM from @briggs"


@respx.mock
def test_a_context_with_no_handle_falls_back_to_one_cached_me_read():
    # A hand-wired context (a library embedding) carries no handle; the tool asks the platform
    # once and caches it, rather than paying a round-trip per notification.
    client = _FakeClient(handle="nova")
    tool = DirectMessageTool(token=TOKEN)
    tool.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="one")
    tool.run(body="two")

    assert client.me_reads == 1
    assert route.calls.last.request.headers["Title"] == "BaseCradle DM from @nova"


@respx.mock
def test_an_unresolvable_handle_still_delivers_and_says_the_title_is_unlabelled():
    # Degrade, never collapse: a message that arrives less well-labelled beats no message.
    tool = DirectMessageTool(token=TOKEN)
    tool.bind(PlatformContext(client=_FakeClient(fail=True), timeline=TIMELINE_UUID))
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    result = tool.run(body="hello")

    assert route.calls.last.request.headers["Title"] == "BaseCradle DM"
    assert "Sent" in result
    assert "handle could not be resolved" in result


@respx.mock
def test_a_hostile_handle_cannot_inject_a_header(tool):
    # Identity comes from the platform, not the model — but a header is still a header.
    tool.bind(_context(handle="nova\r\nX-Injected: 1"))
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="hello")

    request = route.calls.last.request
    assert request.headers["Title"] == "BaseCradle DM from @novaX-Injected1"
    assert "X-Injected" not in request.headers


# --- the byte cap ------------------------------------------------------------


@respx.mock
def test_an_oversize_body_is_refused_with_its_real_byte_count_and_never_sent(tool):
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)
    body = "x" * (MAX_BODY_BYTES + 117)

    result = tool.run(body=body)

    assert not route.called  # refused before the request, never truncated into one
    assert result.startswith("Error:")
    assert f"{MAX_BODY_BYTES + 117:,} bytes" in result  # the model's actual size
    assert f"{MAX_BODY_BYTES:,}-byte limit" in result  # and the cap to get under
    assert "117 bytes" in result  # ...and how much to cut


@respx.mock
def test_the_cap_is_bytes_not_characters(tool):
    # ntfy's limit is on the wire, so a 3-byte-per-character script hits it three times sooner.
    # A character count would have waved this through and let ntfy turn it into a .txt file.
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)
    body = "あ" * 1400  # 1,400 characters, 4,200 bytes

    result = tool.run(body=body)

    assert not route.called
    assert "4,200 bytes" in result


@respx.mock
def test_a_body_exactly_at_the_cap_is_sent(tool):
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    result = tool.run(body="x" * MAX_BODY_BYTES)

    assert route.called
    assert "Sent" in result


@respx.mock
def test_non_ascii_text_survives_the_round_trip_as_utf_8(tool):
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="配備は完了しました — déployé ✅")

    assert route.calls.last.request.content == "配備は完了しました — déployé ✅".encode()


# --- refusals the model can act on -------------------------------------------


@respx.mock
@pytest.mark.parametrize("body", [None, "", "   \n  "])
def test_an_empty_body_is_refused_before_any_request(tool, body):
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    result = tool.run(body=body)

    assert not route.called
    assert result.startswith("Error:")
    assert "non-empty 'body'" in result


@respx.mock
def test_a_missing_token_is_a_readable_error_never_a_silent_drop():
    tool = DirectMessageTool()  # nothing injected, and the env var is cleared
    tool.bind(_context())
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    result = tool.run(body="hello")

    assert not route.called
    assert result.startswith("Error:")
    assert TOKEN_ENV in result
    assert "Nothing was sent" in result


@respx.mock
def test_the_token_is_read_from_the_environment_when_not_injected(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    tool = DirectMessageTool()
    tool.bind(_context())
    route = respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="hello")

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


# --- delivery failures: one retry, and only for a transient fault -------------


@respx.mock
def test_a_5xx_is_retried_once_and_can_succeed(tool):
    route = respx.post(NTFY_URL).mock(
        side_effect=[httpx.Response(503, text="upstream unavailable"), _accepted()]
    )

    result = tool.run(body="hello")

    assert route.call_count == 2
    assert "Sent" in result


@respx.mock
def test_a_transport_error_is_retried_once_and_can_succeed(tool):
    route = respx.post(NTFY_URL).mock(side_effect=[httpx.ConnectTimeout("timed out"), _accepted()])

    result = tool.run(body="hello")

    assert route.call_count == 2
    assert "Sent" in result


@respx.mock
def test_a_transient_failure_is_retried_exactly_once_then_reported(tool):
    # At most one retry — a phone notification, not a delivery queue.
    route = respx.post(NTFY_URL).mock(return_value=httpx.Response(502, text="bad gateway"))

    result = tool.run(body="hello")

    assert route.call_count == 2
    assert result.startswith("Couldn't deliver")
    assert "502" in result
    assert "Nothing was sent" in result


@respx.mock
def test_a_4xx_is_not_retried_because_resending_the_same_bytes_cannot_help(tool):
    route = respx.post(NTFY_URL).mock(return_value=httpx.Response(401, text="unauthorized"))

    result = tool.run(body="hello")

    assert route.call_count == 1
    assert result.startswith("Couldn't deliver")
    assert "401" in result


@respx.mock
def test_an_unreachable_server_is_reported_never_swallowed(tool):
    respx.post(NTFY_URL).mock(side_effect=httpx.ConnectError("no route to host"))

    result = tool.run(body="hello")

    assert result.startswith("Couldn't deliver")
    assert "ntfy.sh" in result


# --- the credential never leaks ----------------------------------------------


@respx.mock
def test_a_response_echoing_the_token_is_redacted_out_of_the_error(tool):
    # ntfy has no reason to echo the credential; the guard is mechanical so no later edit
    # has to remember to be careful.
    respx.post(NTFY_URL).mock(return_value=httpx.Response(400, text=f"bad token {TOKEN} sent"))

    result = tool.run(body="hello")

    assert TOKEN not in result
    assert "[redacted]" in result


@respx.mock
@pytest.mark.parametrize("response", [httpx.Response(200), httpx.Response(401, text="nope")])
def test_no_log_record_ever_carries_the_token(tool, caplog, response):
    respx.post(NTFY_URL).mock(return_value=response)

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        tool.run(body="hello")

    assert caplog.records  # the send is logged either way...
    assert not any(TOKEN in record.getMessage() for record in caplog.records)  # ...but never this


# --- what it is, and what it is not ------------------------------------------


@respx.mock
def test_a_push_is_not_timeline_speech_and_records_nothing_in_the_ledger():
    # The ledger answers "did this wake put something on the *timeline*?" — what `posted=` counts
    # and what the no-reply informer reads. A phone notification is not on a timeline.
    speech = SpeechLedger()
    tool = DirectMessageTool(token=TOKEN)
    tool.bind(_context(speech=speech))
    respx.post(NTFY_URL).mock(side_effect=_accepted)

    tool.run(body="hello")

    assert speech.actions == 0
    assert speech.posts == [] and speech.acts == []


def test_it_loads_under_the_shipped_locked_profile():
    # It needs the platform only to know its own name — no shell, no exec — so the safe default
    # profile registers it once an operator has opted it in.
    registry = ToolRegistry(Policy.locked())
    registry.register(DirectMessageTool())
    assert "send_direct_message_to_origin" in registry


def test_an_unbound_tool_says_so_rather_than_raising_an_attribute_error():
    # The PlatformTool contract: a tool that was never wired explains itself.
    with pytest.raises(PlatformError):
        DirectMessageTool(token=TOKEN).run(body="hello")
