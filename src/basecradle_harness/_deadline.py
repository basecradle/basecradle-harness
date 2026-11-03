"""A model call's budget bounds the whole attempt, not a silence (issue #788).

On 2026-11-03 `@ping`'s brain call ran for **941 s** and wrote **131,072** output tokens before the
harness saw an answer it could not use, although the call's fitted budget (`_timeouts`) was about
540 s. Nothing errored, because nothing was measuring the attempt. Each HTTP adapter hands its SDK
the fitted budget as an ``httpx`` ``Timeout``, and an ``httpx`` read timeout is an **idle** timeout:
it bounds the wait between two received chunks, never the whole response. A non-streaming OpenRouter
request is answered with its headers at once, then a keepalive chunk every three seconds until the
body is ready (measured live, `z-ai/glm-5.2`, 2026-11-03), so the idle timer is reset every three
seconds and an attempt runs for as long as the model keeps writing. The budget the design called
"the generation" was only ever enforced against a vendor that sends nothing while it works.

So the budget is also a **deadline**, enforced at the transport, on every HTTP client a fitted model
call drives: the request's own timeout says how long the attempt may take (``connect`` + ``read``,
the two budgets `_timeouts.CallTimeout` sets), and the attempt fails with the client's own
``ReadTimeout`` the moment that time has passed. That is the error each SDK already raises and each
adapter already maps to `ProviderTimeoutError`, so the engine's one timeout retry (twice the budget,
issue #589) is what follows, with no new class anywhere. The native xAI adapter needs none of this:
a gRPC deadline already bounds the whole call.

Why event hooks, and why the request's own timeout
--------------------------------------------------
- **The deadline is read off the request**, never handed in a second time: the budget an adapter put
  on a call is the budget the call is held to, and a retry's larger budget (`bind_scale`) needs no
  second path. A request carrying no read timeout carries no deadline.
- **Event hooks are the clients' public extension point**, and every SDK the harness drives keeps its
  own client configuration (connection limits, redirects) when they are added. A response hook runs
  after the headers arrive and **before** the body is read, so it can check the deadline once and
  then hand the body to a stream that checks it again at every chunk. Both HTTPX families (``httpx``
  for `openrouter` and `google-genai`, HTTPX2 for `openai` and `anthropic`) read a body by iterating
  ``response.stream``, so the one mechanism serves both; each raises its own family's ``ReadTimeout``.
- **The check runs when a byte arrives.** Against a keepalive the deadline is therefore exact to
  within one keepalive interval. Against a vendor that sends nothing at all, the idle timer still
  fires as it always did, so the worst case is unchanged there: the silence must outlast the read
  budget, measured from the last byte received.
- **The clock starts before the request is sent**, so uploading it spends the same budget. The
  ``write`` phase's own allowance still bounds a stalled upload; a merely slow one now counts
  against the attempt, which for a request of a few megabytes is seconds of a budget of minutes.
"""

from __future__ import annotations

import sys
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from types import ModuleType
from typing import Any

#: The clock the deadline is measured on. A module attribute, so a test can stand in for it.
_clock: Callable[[], float] = time.monotonic


def bound_attempts(client: Any) -> bool:
    """Hold every request `client` sends to the deadline its own timeout states.

    Added through the client's public ``event_hooks`` property, on both sides of the hooks already
    on it, and each side's order is load-bearing:

    - the **request** hook runs **last**, because a hook before it may set the timeout this one
      reads (the Vertex adapter's `_phased` does);
    - the **response** hook runs **first**, because a hook after it may read the body (the
      OpenRouter adapter's citation capture does), and a body read before the deadline is installed
      is read with no deadline at all — the keepalive's 941 s again, one hook early.

    ``False`` when `client` is not an ``httpx``-family client (a test double): nothing was
    installed.
    """
    family = _family(client)
    if family is None:
        return False
    deadlines: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()

    def start(request: Any) -> None:
        budget = _budget(request.extensions.get("timeout"))
        if budget is not None:
            deadlines[request] = _clock() + budget

    def hold(response: Any) -> None:
        request = response.request
        deadline = deadlines.pop(request, None)
        if deadline is None:
            return
        _check(family, request, deadline)
        response.stream = _bounded_stream(family, response.stream, request, deadline)

    hooks = client.event_hooks
    client.event_hooks = {
        "request": [*hooks.get("request", ()), start],
        "response": [hold, *hooks.get("response", ())],
    }
    return True


def _family(client: Any) -> ModuleType | None:
    """The ``httpx``-family distribution `client` is a ``Client`` of, or ``None``.

    Read off the client's own class rather than by importing every family and asking
    ``isinstance`` of each: the `openai` and `anthropic` SDKs run on HTTPX2, whose classes are no
    subclasses of ``httpx``'s (issue #545's trap), and a list of packages here would put an optional
    extra's HTTP client in the core and miss the next family silently. The hooks must raise that
    family's own ``ReadTimeout`` (the timeout its SDK recognizes) and hand back that family's own
    ``SyncByteStream`` (the only kind its ``Response`` will iterate).
    """
    for cls in type(client).__mro__:
        if cls.__name__ != "Client":
            continue
        module = sys.modules.get(cls.__module__.partition(".")[0])
        if (
            module is not None
            and getattr(module, "Client", None) is cls
            and isinstance(getattr(module, "ReadTimeout", None), type)
            and isinstance(getattr(module, "SyncByteStream", None), type)
        ):
            return module
    return None


def _budget(timeout: Any) -> float | None:
    """The whole attempt's budget, in seconds, from a request's timeout extension, or ``None``.

    ``connect`` + ``read``: the time to reach the endpoint and the time the answer may take, which
    is how `_timeouts.CallTimeout` divides one call. No read budget means no deadline, because there
    is then nothing to say how long the answer may take.
    """
    if not isinstance(timeout, Mapping):
        return None
    read = timeout.get("read")
    if not _seconds(read):
        return None
    connect = timeout.get("connect")
    return float(read) + (float(connect) if _seconds(connect) else 0.0)


def _seconds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _check(family: ModuleType, request: Any, deadline: float) -> None:
    if _clock() >= deadline:
        raise family.ReadTimeout(
            "The model call ran past its whole-attempt deadline: the endpoint kept the connection "
            "open without finishing its answer.",
            request=request,
        )


def _bounded_stream(family: ModuleType, inner: Any, request: Any, deadline: float) -> Any:
    """`inner`, read under `deadline`: every chunk that arrives after it ends the read."""

    class Bounded(family.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for chunk in inner:
                _check(family, request, deadline)
                yield chunk

        def close(self) -> None:
            inner.close()

    return Bounded()
