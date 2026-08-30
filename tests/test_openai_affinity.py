"""Cache affinity on the `openai` SDK adapter, read off the **actual request** (issue #435).

`bind_conversation` shipped on one adapter in #433, leaving `OpenAIProvider` — the one aimed at
three vendors — sending no routing key anywhere. Each cell was decided from its own vendor's
guidance (`_openai._AFFINITY`, reasoned in `_caching`), and these tests hold that decision to the
standing rule the native adapter's near-miss produced:

> **An adapter has not implemented `bind_conversation` until something outside the adapter proves
> the bytes left.**

Version 0.110.0 bound a key the `xai_sdk` accepted and spent on a telemetry span; every test passed
and no server ever saw it. So nothing here asserts on a mock's arguments. respx intercepts at the
transport, so every assertion below reads the **recorded HTTP request** — its real JSON body, its
real headers — exactly as the endpoint would have.

The cell that sends *nothing* is tested just as hard as the ones that send something: OpenRouter's
pin was measured and rejected (#372), and a silent regression there costs 2.75× at production scale.
"""

import json

import httpx
import pytest

from basecradle_harness import Harness, Message, OpenAIProvider, Session
from basecradle_harness._basecradle import OPENAI_SDK_PROVIDERS
from basecradle_harness._caching import bind_conversation
from basecradle_harness._openai import (
    _AFFINITY,
    CACHE_KEY_FIELD,
    CONVERSATION_HEADER,
    SURFACES,
    _Affinity,
)

from .conftest import BASE_URL, CHAT_URL, FAKE_KEY, RESPONSES_URL, completion, responses_body

SESSION = "timeline:019f6e71-2a12-7b69-a204-0fec1497b9c2"
OTHER_SESSION = "timeline:019f6e88-0d31-77a2-8b41-2c5a90c1e7d4"


def _provider(vendor, surface, **kwargs):
    return OpenAIProvider(
        model="grok-4.3" if vendor == "xai" else "gpt-5.4-mini",
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        provider=vendor,
        surface=surface,
        max_retries=0,
        **kwargs,
    )


def _chat_route(router):
    return router.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=completion(content="ok"))
    )


def _responses_route(router):
    return router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body()))


def _body(route):
    return json.loads(route.calls.last.request.content)


# === the cells that send a body field ==========================================================


@pytest.mark.parametrize("vendor", ["openai", "xai"])
def test_the_responses_body_carries_prompt_cache_key(router, vendor):
    """Both vendors that take the Responses form spell it the same, and it is on the real body."""
    route = _responses_route(router)
    provider = _provider(vendor, "responses")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert _body(route)[CACHE_KEY_FIELD] == SESSION
    provider.close()


def test_the_openai_chat_body_carries_prompt_cache_key(router):
    """OpenAI takes the same field on Chat Completions; xAI does not (see the header test)."""
    route = _chat_route(router)
    provider = _provider("openai", "chat")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert _body(route)[CACHE_KEY_FIELD] == SESSION
    provider.close()


# === the cell that sends a header ==============================================================


def test_the_xai_chat_request_carries_the_grok_conversation_header(router):
    """xAI's Chat Completions surface routes on a header, not a body field — so it must be one.

    Both halves are asserted: the header is really on the wire, **and** the body field is not.
    Sending an unknown top-level field to an endpoint that never asked for it is a 400 on every
    wake, which is the failure this table exists to prevent.
    """
    route = _chat_route(router)
    provider = _provider("xai", "chat")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert route.calls.last.request.headers[CONVERSATION_HEADER] == SESSION
    assert CACHE_KEY_FIELD not in _body(route)
    provider.close()


def test_the_affinity_header_composes_with_the_clients_own_headers(router):
    """A per-call header must not replace the ones the config layer already set on the client.

    OpenRouter's routing metadata and an operator's own ``extra_headers`` ride as client defaults;
    the SDK merges a per-call ``extra_headers`` over them, and this pins that it stays a merge.
    """
    route = _chat_route(router)
    provider = _provider("xai", "chat", extra_headers={"X-Operator-Note": "keep me"})
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    headers = route.calls.last.request.headers
    assert headers[CONVERSATION_HEADER] == SESSION
    assert headers["X-Operator-Note"] == "keep me"
    provider.close()


# === the cell that deliberately sends nothing ==================================================


def test_openrouter_sends_no_routing_key_however_it_is_bound(router):
    """#372 measured a pin here and rejected it: nothing goes on this wire, bound or not.

    The absence is the decision, so it is tested as a decision — a later "consistency" edit that
    adds a key to every cell fails here rather than costing 2.75× in production.
    """
    route = _chat_route(router)
    provider = _provider("openrouter", "chat")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert CACHE_KEY_FIELD not in _body(route)
    assert CONVERSATION_HEADER not in route.calls.last.request.headers
    provider.close()


def test_a_vendor_label_nobody_wired_sends_nothing(router):
    """An unrecognized `provider` label falls out of the table, and the fallback is silence.

    Fail-safe in the same direction `cache_mode` fails: the only thing this feature can do is put
    a vendor field on the wire, and doing that at an endpoint that never asked for it is a 400.
    """
    route = _chat_route(router)
    provider = _provider("some-new-gateway", "chat")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert CACHE_KEY_FIELD not in _body(route)
    assert CONVERSATION_HEADER not in route.calls.last.request.headers
    provider.close()


# === binding semantics =========================================================================


def test_unbound_sends_nothing_rather_than_a_fabricated_id(router):
    """`None` means omit the field. A made-up id is a fresh conversation on every call."""
    route = _chat_route(router)
    provider = _provider("openai", "chat")

    provider.chat([Message.user("Hi")])

    assert CACHE_KEY_FIELD not in _body(route)
    provider.close()


@pytest.mark.parametrize("cleared", [None, ""])
def test_clearing_the_binding_stops_sending_the_key(router, cleared):
    route = _chat_route(router)
    provider = _provider("openai", "chat")
    provider.bind_conversation(SESSION)
    provider.chat([Message.user("one")])
    assert _body(route)[CACHE_KEY_FIELD] == SESSION

    provider.bind_conversation(cleared)
    provider.chat([Message.user("two")])

    assert CACHE_KEY_FIELD not in _body(route)
    provider.close()


def test_rebinding_switches_the_key_on_the_next_call(router):
    """The binding is sticky until the next one — one session's calls, then another's."""
    route = _chat_route(router)
    provider = _provider("openai", "chat")

    provider.bind_conversation(SESSION)
    provider.chat([Message.user("one")])
    assert _body(route)[CACHE_KEY_FIELD] == SESSION

    provider.bind_conversation(OTHER_SESSION)
    provider.chat([Message.user("two")])
    assert _body(route)[CACHE_KEY_FIELD] == OTHER_SESSION
    provider.close()


def test_the_bound_key_wins_over_a_construction_time_default(router):
    """Harness wiring beats call tuning — the same precedence the native adapter states.

    An operator never reaches this (`_basecradle._OWNED_OPENAI` strips `prompt_cache_key` from
    `model_params.json` with a warning); it is what stops a library caller's static value from
    silently pinning every session on the box to one server.
    """
    route = _chat_route(router)
    provider = _provider("openai", "chat", prompt_cache_key="a-static-value")
    provider.bind_conversation(SESSION)

    provider.chat([Message.user("Hi")])

    assert _body(route)[CACHE_KEY_FIELD] == SESSION
    provider.close()


# === every buildable cell has a decided answer ==================================================


def test_every_buildable_cell_has_a_decided_affinity_answer():
    """The mechanical form of the DoD: a documented "no" is fine; silence is not.

    Enumerated from the config layer's own wired-provider list and the adapter's own `SURFACES`,
    never a second copy of either — so wiring a fourth vendor, or adding a third surface, fails
    here until somebody reads that endpoint's guidance and writes the answer down. Absence and a
    deliberate `None` are indistinguishable to a reader of a plain dict, which is exactly the
    green-while-absent shape this repo keeps paying for.
    """
    buildable = {
        (vendor, surface)
        for vendor in OPENAI_SDK_PROVIDERS
        for surface in SURFACES
        # `_provider_from_config` refuses OpenRouter on anything but chat (its Responses API is
        # beta upstream), so that pair is not a cell and needs no answer.
        if not (vendor == "openrouter" and surface != "chat")
    }

    assert buildable <= set(_AFFINITY), f"undecided cells: {sorted(buildable - set(_AFFINITY))}"


def test_a_decided_no_is_a_present_key_not_a_missing_one():
    """OpenRouter's "send nothing" is recorded as a decision, and stays one."""
    assert ("openrouter", "chat") in _AFFINITY
    assert _AFFINITY[("openrouter", "chat")] is None


def test_each_carrier_names_exactly_one_place_to_put_the_key():
    """A body field or a header, never both — the two are read by different code paths."""
    for cell, affinity in _AFFINITY.items():
        if affinity is None:
            continue
        assert bool(affinity.field) != bool(affinity.header), cell


@pytest.mark.parametrize("bad", [{}, {"field": "a", "header": "b"}])
def test_a_malformed_carrier_is_refused_at_construction(bad):
    """The table is a module-level literal, so a bad entry must fail the import, not a wake.

    Neither-nor would reach the SDK as ``create(**{None: ...})``; both-and would silently send the
    key twice, in two places, to an endpoint that asked for one.
    """
    with pytest.raises(ValueError):
        _Affinity(**bad)


def test_a_session_driven_turn_puts_its_own_id_on_the_wire(router):
    """End to end, which is the only claim that matters: `Session` → capability → HTTP request.

    Every other test here binds the adapter by hand. This one drives a real `Session`, which is
    what actually calls `_caching.bind_conversation` in production (`Session._drive`) — so it fails
    if the chain is broken anywhere between the session's `source` and the bytes on the wire, not
    only inside the adapter.
    """
    route = _chat_route(router)
    provider = _provider("openai", "chat")
    session = Session(SESSION, Harness(provider).engine)

    session.send("Hi")

    assert _body(route)[CACHE_KEY_FIELD] == SESSION
    provider.close()


def test_the_capability_is_reached_through_the_shared_helper(router):
    """`_caching.bind_conversation` is how the engine asks — by capability, never a vendor branch."""
    route = _chat_route(router)
    provider = _provider("openai", "chat")

    bind_conversation(provider, SESSION)
    provider.chat([Message.user("Hi")])

    assert _body(route)[CACHE_KEY_FIELD] == SESSION
    provider.close()
