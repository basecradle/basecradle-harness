"""Ring @origin's phone: a direct message that leaves the platform entirely (issue #341).

Every other way an agent speaks lands on a **timeline** — a place @origin has to go and
look. This is the one channel that goes *to him*: a real push notification on his iPhone,
delivered through [ntfy.sh](https://ntfy.sh) to a topic reserved under his own account. It
is the persona-to-founder counterpart of the fleet's GitHub `needs-human` alert, which
shipped on the same transport.

Three things follow from "this rings a human's phone", and they are the whole design:

- **It is powerful, so it is opt-in** (`opt_in=True`, issue #168). Off by default on every
  provider; it reaches an agent only when its plugin is dropped into that persona's
  `tools/` overlay. An interruption channel that arrives switched on for everyone is a
  spam channel.
- **It is an interruption, not a channel.** The description says so to the model in the
  plainest words there are — use it only when @origin has asked for a DM — because the
  only real guard on volume is the model understanding what it is holding. Nothing here
  forces or rate-limits it (the Unspoken Channel's stance: inform, never force); a persona
  that abuses it gets the tool taken away, which is a decision for a human, not a counter.
- **It never speaks for the agent on a timeline**, so it records nothing in the
  `SpeechLedger`. The ledger answers "did this wake put something on the *timeline*?" — the
  question the wake's `posted=` bookend and the no-reply informer both read. A push
  notification is not on a timeline, and counting it there would make a silent wake look
  like it spoke.

**The sender identifies itself, and never by a hardcoded name.** The notification title is
``BaseCradle DM from @<handle>``, where the handle is *this* agent's own platform handle,
read from the live identity the hosting agent already resolved (`PlatformContext.handle`).
That is why this is a `PlatformTool` despite never calling a BaseCradle endpoint: it needs
to know who it is. When the harness was wired without an identity (a bare context in a
library embedding), it falls back to one `bc.me` read, and if even that is unavailable it
**still delivers** under a plain ``BaseCradle DM`` title and says so in its result — a
message that arrives slightly less well-labelled beats a message that never arrives.

**The 4,096-byte cap is ntfy's, and it is enforced here rather than discovered there.**
Over that size ntfy silently converts the message into a `.txt` *attachment* — the
notification still "succeeds", and what lands on the phone is a file instead of a message.
So an oversize body is refused **before** the request, as an error naming the body's actual
byte count and the cap, and it is **never truncated**: the words are the model's to choose,
and picking which of them to drop is not this tool's call. Bytes are the right unit here
precisely because it is a *wire* limit — unlike a context cap, which is measured in what
the model reads (see CLAUDE.md → Context Discipline).

**The credential never leaves.** `NTFY_DM_TOKEN` is read at call time, sent in one
`Authorization` header, and interpolated into no log line and no error string; ntfy's own
response text is scrubbed of it before it is shown to the model, so the invariant is
mechanical rather than a promise to be careful.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable

import httpx

from basecradle_harness._platform import PlatformTool

logger = logging.getLogger("basecradle_harness")

#: The ntfy server. Overridable only for a test double or a self-hosted relay.
DEFAULT_BASE_URL = "https://ntfy.sh"

#: The reserved topic. @origin's ntfy account restricts publish *and* subscribe on it, so a
#: valid `NTFY_DM_TOKEN` is required to reach it and nobody else can listen in. A fixed
#: contract with the account, not a setting — the constructor argument exists for tests.
DEFAULT_TOPIC = "BaseCradle-Direct-Message"

#: The env var carrying the publish token. A **fixed name**, provisioned per-agent in that
#: agent's `agent.env`; it is a dedicated second token, independent of the org secret behind
#: the fleet's GitHub `needs-human` alert.
TOKEN_ENV = "NTFY_DM_TOKEN"

#: ntfy's hard message cap. Past it the message is silently turned into a `.txt` attachment,
#: which is a broken DM wearing a success code — so it is enforced before the request.
MAX_BODY_BYTES = 4096

#: Per-request timeout (seconds). A phone notification is worth a short wait and no more; a
#: turn must never hang on one.
DEFAULT_TIMEOUT = 10.0

#: Pause before the single retry (seconds). One retry, one short pause — this is a
#: notification, not a delivery queue.
RETRY_DELAY = 1.0

#: Everything a notification header may carry. A handle is drawn from platform identity, not
#: from the model, but a header is still a header: anything outside this set is dropped, so no
#: identity value can inject a header line or smuggle a non-ASCII byte into `Title`.
_HANDLE_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class _DeliveryFailed(Exception):
    """ntfy could not be reached or refused the message, with a model-readable reason.

    Its message is always safe to show: it names the status or the transport failure, and any
    response text it carries has been through `_redact`.
    """


class DirectMessageTool(PlatformTool):
    """`send_direct_message_to_origin` — push a message to @origin's iPhone.

    A `PlatformTool` that calls no BaseCradle endpoint: it takes the platform handle from the
    bound context so the notification can say who sent it, and does its one piece of work over
    plain HTTPS to ntfy.sh. Opt-in everywhere (its plugin declares `opt_in=True`), because it
    interrupts a human.

    Every failure comes back as readable text — a missing token, an oversize body, a refusal
    from ntfy, an unreachable server — and none of them carry the token.

    Args:
        token: The ntfy publish token. ``None`` (the default) reads `NTFY_DM_TOKEN` from the
            environment at call time.
        base_url: The ntfy server root (for a test double or a self-hosted relay).
        topic: The reserved topic to publish to.
        timeout: Per-request timeout in seconds.
        retry_delay: Seconds to wait before the one retry.
        sleep: The sleep used for that pause (injectable for tests).
    """

    name = "send_direct_message_to_origin"
    description = (
        "Send a message as a real push notification straight to @origin's iPhone. It reaches "
        "him wherever he is, off the platform entirely — no timeline involved. Use it only "
        "when @origin has explicitly asked you to send him a direct message: it interrupts a "
        "human, so do not overuse it or spam it. Takes one 'body' parameter — plain text (no "
        "markdown), at most 4,096 bytes of UTF-8. Nothing is ever truncated for you: an "
        "oversize body comes back as an error naming your actual byte count so you can shorten "
        "it and retry, and every other failure comes back to you as readable text too."
    )
    parameters = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": (
                    "The message to push to @origin's phone — plain text, at most 4,096 bytes "
                    "of UTF-8. This is the whole notification, so make it self-contained."
                ),
            },
        },
        "required": ["body"],
    }

    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        topic: str = DEFAULT_TOPIC,
        timeout: float = DEFAULT_TIMEOUT,
        retry_delay: float = RETRY_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._topic = topic
        self._timeout = timeout
        self._retry_delay = retry_delay
        self._sleep = sleep
        self._discovered_handle: str | None = None  # cached `bc.me` fallback (see `_sender`)

    def run(self, body: str | None = None) -> str:
        """Push `body` to @origin's phone, or return the reason it did not go."""
        if not body or not body.strip():
            return (
                f"Error: {self.name!r} needs a non-empty 'body' — the message to send. "
                "Nothing was sent."
            )

        payload = body.encode("utf-8")
        size = len(payload)
        if size > MAX_BODY_BYTES:
            return (
                f"Error: that body is {size:,} bytes of UTF-8, over the "
                f"{MAX_BODY_BYTES:,}-byte limit ntfy enforces — shorten it by at least "
                f"{size - MAX_BODY_BYTES:,} bytes and send it again. Nothing was sent, and "
                "nothing was truncated for you: which words to drop is your call, not this "
                "tool's. (Note bytes, not characters — non-ASCII text costs more than one byte "
                "per character.)"
            )

        token = self._token or os.environ.get(TOKEN_ENV)
        if not token:
            return (
                f"Error: {self.name!r} has no publish credential — {TOKEN_ENV} is unset or "
                "empty in this agent's environment, so it cannot reach @origin's phone. "
                "Nothing was sent, and this is not something you can fix from here."
            )

        sender = self._sender()
        title = f"BaseCradle DM from @{sender}" if sender else "BaseCradle DM"

        try:
            self._publish(token, title, payload)
        except _DeliveryFailed as exc:
            logger.warning("direct_message failed bytes=%d reason=%s", size, exc)
            return f"Couldn't deliver the direct message to @origin: {exc} Nothing was sent."

        logger.info("direct_message sent bytes=%d title=%r", size, title)
        sent = (
            f"Sent — that message is a push notification on @origin's phone now ({size:,} bytes)."
        )
        if sender:
            return sent
        return (
            f"{sent} One caveat: your own platform handle could not be resolved, so the "
            "notification is titled 'BaseCradle DM' with no sender. Say who you are in the "
            "body next time."
        )

    # --- who is sending -------------------------------------------------------

    def _sender(self) -> str | None:
        """This agent's own platform handle, header-safe — or ``None`` if unresolvable.

        The hosting agent (`WakeAgent`/`TimelineAgent`) already reads `bc.me` at startup and
        puts the handle on the context, so the common path costs nothing. The `bc.me` fallback
        is for a context wired by hand (a library embedding, a test); it is cached, and a
        failure degrades to an unlabelled title rather than costing @origin the message.
        """
        handle = self.context.handle or self._discover()
        safe = _HANDLE_SAFE.sub("", handle or "")
        return safe or None

    def _discover(self) -> str | None:
        """One cached `bc.me` read, for a context that carries no handle. Never raises."""
        if self._discovered_handle is None:
            try:
                self._discovered_handle = (
                    getattr(self.context.client.me.identity, "handle", "") or ""
                )
            except Exception:  # noqa: BLE001 - identity is a nicety here; the message is the point
                logger.debug("direct_message could not resolve its own handle", exc_info=True)
                self._discovered_handle = ""
        return self._discovered_handle or None

    # --- delivery -------------------------------------------------------------

    def _publish(self, token: str, title: str, payload: bytes) -> None:
        """POST the notification, retrying once on a transient failure. Raises `_DeliveryFailed`.

        Transient means the request never got an answer, or ntfy answered for itself (5xx). A
        4xx is ntfy's verdict on *this* request — a bad token, a refused topic — and re-sending
        the identical bytes cannot change it, so it fails on the first answer.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Title": title,
            "Priority": "5",
            "Tags": "speech_balloon",
            "Content-Type": "text/plain; charset=utf-8",
        }
        url = f"{self._base_url}/{self._topic}"

        reason = "no attempt was made."
        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(2):  # one try, at most one retry
                try:
                    response = client.post(url, content=payload, headers=headers)
                except httpx.RequestError as exc:
                    reason = f"couldn't reach ntfy.sh ({exc})."
                else:
                    if response.status_code < 300:
                        return
                    reason = _refusal(response, token)
                    if response.status_code < 500:
                        break  # ntfy's verdict on this request; a retry sends the same thing
                if attempt == 0:
                    self._sleep(self._retry_delay)
        raise _DeliveryFailed(reason)


def _refusal(response: httpx.Response, token: str) -> str:
    """Why ntfy refused, as one model-readable sentence with the token scrubbed out."""
    detail = _redact(response.text, token).strip()[:200]
    if detail:
        return f"ntfy.sh returned HTTP {response.status_code}: {detail}."
    return f"ntfy.sh returned HTTP {response.status_code}."


def _redact(text: str, token: str) -> str:
    """`text` with the publish token replaced, so no path can leak it into a log or a result.

    ntfy has no reason to echo the credential back, which is exactly why this is here: the
    invariant that the token never appears in log or error text should hold *mechanically*,
    not because every future edit to this file remembers it.
    """
    return text.replace(token, "[redacted]") if token else text
