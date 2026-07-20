"""The body's senses and voice: connect the engine to a BaseCradle timeline.

A `TimelineAgent` watches one timeline, hands each new message from someone else
to a `Harness`, and posts the reply back — all through the `basecradle` SDK, never
raw HTTP. This is the v0 way an agent lives on the platform: a poll loop for a
single local agent. No webhooks, no router, no multi-tenancy — those are later,
and their own repos.

Configuration is environment-first (see `TimelineAgent.from_env`):

- ``BASECRADLE_TOKEN``        — the platform credential (read by the SDK). Preferred;
  reused as-is when set, minted only when missing or dead (see `_client_from_env`).
- ``BASECRADLE_EMAIL`` + ``BASECRADLE_PASSWORD`` — the credential fallback: when no
  token is set, mint one on startup; also what a live token re-mints from if it dies
  mid-run (see `_token`). A credential-only AI comes up with no pre-minted token and
  no human in the loop.
- ``BASECRADLE_ENV_FILE``     — optional; the file the agent sources its env from (its
  ``agent.env``). A minted/re-minted token is written back to its ``BASECRADLE_TOKEN=``
  line so the next wake reuses it. Unset → the token is not persisted (a warning is
  logged) and a credential-only agent mints once per wake.
- ``BASECRADLE_SESSION_NAME`` — optional; labels the credential minted from a
  password so it can be told apart later (the SDK's ``login(name=…)``).
- ``BASECRADLE_TIMELINE``     — the uuid of the timeline to watch.

The model config is **three independent axes** (issue #158) — one name per concept,
identical in the env, the code, and the docs:

- ``AI_PROVIDER``             — the vendor whose endpoint + key the agent uses:
  ``openai`` (default) | ``xai`` | ``openrouter``. Both ``openai`` and ``xai`` are wired through
  the one ``openai`` SDK adapter (xAI's endpoint speaks the same wire — ``AI_PROVIDER=xai`` points
  the SDK at ``api.x.ai``, issue #163); ``openrouter`` is a later milestone.
- ``AI_SDK``                  — the **library/package name** of the SDK the harness imports to
  reach the model: ``openai`` (default), and ``xai-sdk`` (the committed next phase, #165). The value is the importable
  package, which also disambiguates it from the provider token (``AI_PROVIDER=xai`` selects
  xAI's *endpoint*; ``AI_SDK=xai-sdk`` selects xAI's *native SDK*, #165). The harness reaches an
  LLM **only** through a vendor SDK; with the named SDK not installed it comes up with no way to
  reach a model and says so. Two adapters ship: ``openai`` (the OpenAI-wire SDK — also xAI over
  ``api.x.ai``) and ``xai-sdk`` (xAI's native gRPC SDK, #165).
- ``AI_MODEL``                — the model id (e.g. ``gpt-5.4-mini``).
- ``AI_API_KEY``             — the provider's API key.
- ``AI_BASE_URL``            — optional; override the provider's endpoint.
- ``AI_SDK_SURFACE``          — optional; **SDK-scoped**, not a top-level config axis. The
  active SDK adapter declares its own ``SURFACES`` + ``DEFAULT_SURFACE``; this var selects among
  them (omitted → the adapter's default; provided-but-unlisted → a hard fail). The ``openai``
  adapter has two — ``responses`` (default, @jt's surface — the one that runs ``web_search`` and
  sees images) and ``chat``; a single-surface SDK never sets it. See `_resolve_surface`,
  `_provider_from_config`, and `basecradle_harness._plugins`.
- ``HARNESS_SYSTEM_PROMPT``   — **legacy fallback** for the agent's standing charter. The
  charter is now sourced from real files under the config home —
  ``prompts/system-prompt.md`` + ``prompts/initialize.md`` (see `basecradle_harness._install`)
  — and this env var is consulted only when the config home was never installed.
- ``HARNESS_CONTEXT_MESSAGES`` — optional; how many backlog messages to seed as
  context (an int, or ``all`` for the whole timeline). Unset → the default.
- ``HARNESS_ONBOARD``         — optional; wake seeded with Dashboard orientation
  (default on). Set falsy (``0``/``false``/``no``/``off``) to wake with only the
  operator's charter.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from basecradle import BaseCradle

from basecradle_harness._code import CODE_EXECUTION_BUILTIN, CodeExecutionBridge
from basecradle_harness._context import Compactor, ContextBudget
from basecradle_harness._engine import (
    DEFAULT_MAX_STEPS,
    DEFAULT_RESPONSE_RETRIES,
    compose_hooks,
)
from basecradle_harness._harness import Harness
from basecradle_harness._install import charter_from_env, reconcile_on_upgrade
from basecradle_harness._mcp import McpImageStore, McpResolution, load_mcp_tools
from basecradle_harness._memory_provider import MemoryProvider, memory_provider_from_env
from basecradle_harness._messages import Message
from basecradle_harness._model_params import load_model_params
from basecradle_harness._observability import log_unspoken
from basecradle_harness._openai import (
    DEFAULT_SURFACE as OPENAI_DEFAULT_SURFACE,
)
from basecradle_harness._openai import (
    SURFACES as OPENAI_SURFACES,
)
from basecradle_harness._openai import (
    OpenAIProvider,
)
from basecradle_harness._openrouter import (
    DEFAULT_SURFACE as OPENROUTER_DEFAULT_SURFACE,
)
from basecradle_harness._openrouter import (
    ROUTING_METADATA_HEADER as OPENROUTER_ROUTING_METADATA_HEADER,
)
from basecradle_harness._openrouter import (
    SURFACES as OPENROUTER_SURFACES,
)
from basecradle_harness._openrouter import (
    WEB_SEARCH_BUILTIN,
    OpenRouterProvider,
)
from basecradle_harness._platform import PlatformContext, bind_platform_tools
from basecradle_harness._plugins import (
    ActivationContext,
    ResolvedTools,
    load_plugins_report,
    resolve_plugins,
)
from basecradle_harness._policy import Policy
from basecradle_harness._provider import Provider
from basecradle_harness._search_params import load_search_params
from basecradle_harness._token import SelfHealingBaseCradle, mint_token
from basecradle_harness._tools import Tool
from basecradle_harness._unspoken import NoReplyInformer, SpeechLedger, is_one_on_one
from basecradle_harness._xai_sdk import (
    DEFAULT_SURFACE as XAI_SDK_DEFAULT_SURFACE,
)
from basecradle_harness._xai_sdk import (
    SURFACES as XAI_SDK_SURFACES,
)
from basecradle_harness._xai_sdk import (
    XaiSdkProvider,
)

_log = logging.getLogger("basecradle_harness")

DEFAULT_POLL_INTERVAL = 2.0

# How many of the timeline's prior messages to seed as context by default. 50 is
# the API's page size, so the default seed is exactly one page — bounded token
# cost and a single startup fetch, while still giving the agent recent history.
# Operators who want the full backlog pass ``context_messages=None`` (env: ``all``).
DEFAULT_CONTEXT_MESSAGES = 50


class TimelineAgent:
    """Runs a `Harness` against one BaseCradle timeline by polling it.

    On construction it resolves the timeline and its own identity, reads the
    timeline as it stands, and does two things with it: marks the newest message
    as the high-water mark — so it *replies* only to messages that arrive after
    it joins, never to history — and seeds the agent's context with (a bounded
    slice of) the backlog, so it *knows* what was said before it joined, the way
    a human who joins a channel scrolls up before answering.

    It also *onboards* itself on its Dashboard: the same `bc.me` read that answers
    "who am I?" also answers "what is this place?", and (when `onboard` is on) that
    orientation is prepended to the agent's charter — so a freshly-woken peer comes
    up already knowing what BaseCradle is and where the docs/API live.

    Args:
        harness: The agent brain + tools.
        timeline: The uuid of the timeline to watch.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
        context_messages: How many of the most recent backlog messages to seed
            as context (oldest-first in `history`). The default bounds token cost
            and startup fetching on long timelines; `None` seeds the whole
            backlog (the pre-cap behavior). The high-water mark is always the
            true newest message, regardless of this cap — seeding less never
            makes the agent reply to history.
        onboard: When `True` (the default), prepend a bounded orientation drawn
            from the agent's Dashboard (what BaseCradle is, what the agent is
            here, where the docs/API live) to `harness.system_prompt`, composing
            with the operator's prompt rather than replacing it. Set `False` to
            wake with only the operator's charter. A Dashboard that carries no
            orientation (e.g. an older API) leaves the charter untouched either
            way. This mutates the harness's charter, so it takes effect for
            sessions created after construction (the timeline's own session
            included); it does not retroactively reseed a session created before.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
        onboard: bool = True,
        code_bridge: CodeExecutionBridge | None = None,
        mcp_images: McpImageStore | None = None,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.code_bridge = code_bridge
        # No `self.timeline` handle: it existed solely to auto-post the model's final text
        # (`timeline.messages.create`), and nothing posts on the agent's behalf any more (issue
        # #293). Keeping it would be a `GET /timelines/{uuid}` on every startup for a value nobody
        # reads. The agent reaches the timeline through its tools, which bind their own client.

        # What this agent has put on the timeline — the ledger its tools record into, and the only
        # truthful answer to "did it speak?" now that nothing posts on its behalf (issue #293).
        self.speech = SpeechLedger()

        # Wire the live platform handle into every platform-aware tool now that the
        # client and current timeline are resolved. This is the seam every Phase-2
        # tool reuses; a plain tool (memory) is skipped. One timeline per agent, so
        # binding once is correct — cross-timeline use is an explicit op argument.
        context = PlatformContext(
            client=self.client,
            timeline=self.timeline_uuid,
            home=self.harness.home,
            code_bridge=code_bridge,
            speech=self.speech,
            # The per-wake MCP image store (issue #318), so the assets ``post_image`` action can
            # post a screenshot an MCP tool returned. ``None`` unless an MCP server's tools loaded.
            mcp_images=mcp_images,
        )
        bind_platform_tools(self.harness.tools, context)
        if code_bridge is not None:
            code_bridge.bind(context)

        # One Dashboard read answers "who am I?" and, when onboarding, "what is this
        # place?" The Dashboard is the literal page a fresh peer wakes on; reading
        # `bc.me` once serves both — `me` is uncached, so we never fetch it twice.
        dashboard = self.client.me
        self.me_uuid = dashboard.identity.uuid

        # The no-reply informer (issues #293, #332), composed onto whatever turn hook is already
        # wired — the poll loop gets the identical behavior the wake path does, because there is one
        # framework here, not two. See `WakeAgent.__init__` for why it composes rather than replaces.
        #
        # The one-on-one arm needs the timeline's live viewer set, and the poll agent deliberately
        # keeps no `self.timeline` (nothing posts on its behalf any more — issue #293). So it is
        # fetched once here, at startup: a real reader now exists for it, which is exactly the
        # condition under which that `GET /timelines/{uuid}` is worth paying (once per process, never
        # per poll). A fetch failure must not sink the agent — degrade to mention-only (`False`).
        try:
            one_on_one = is_one_on_one(self.client.timelines.get(self.timeline_uuid), self.me_uuid)
        except Exception:  # noqa: BLE001 - the informer is a backstop; never let it break startup
            one_on_one = False
        self.informer = NoReplyInformer(
            handle=getattr(dashboard.identity, "handle", None),
            speech=self.speech,
            one_on_one=one_on_one,
        )
        # Composed onto the engine's **base** hook, never its live one — chaining accretes, and a
        # second agent over the same `Harness` would stack a second informer holding a dead ledger.
        engine = self.harness.engine
        engine.turn_hook = compose_hooks(engine.base_turn_hook, self.informer.on_turn)
        if onboard:
            # Prepend the Dashboard orientation to the operator's charter (orientation
            # first — the standing instructions speak to an agent that already knows
            # where it is). Mutating before the seed below is deliberate: that seed is
            # the first session access, so the composed charter reaches every session.
            self.harness.system_prompt = _compose_prompt(
                _orientation(dashboard), self.harness.system_prompt
            )

        # One newest-first read serves both jobs. The high-water mark needs only
        # the newest message; the seed wants the most recent `context_messages`.
        # `_recent` is lazy and auto-paginating, so a capped seed fetches just the
        # pages it needs — never the whole timeline — and always includes the true
        # newest message so the mark is right even when the seed is empty (cap 0).
        recent = _recent(self.client.messages.filter(timeline=self.timeline_uuid), context_messages)
        self._last_seen: str | None = recent[0].content.uuid if recent else None

        to_seed = recent if context_messages is None else recent[:context_messages]
        for message in reversed(to_seed):  # oldest-first into history
            self.harness.history.append(_as_turn(message, self.me_uuid))

    @classmethod
    def from_env(cls) -> TimelineAgent:
        """Build a fully wired agent (provider + tool plugins + timeline) from env vars.

        The memory provider's tools are already folded into ``resolved.tools``
        (`_resolve_tools_and_provider`), so the poll loop keeps the memory tool it always
        had. Its `observe`/`context` middleware hooks are a wake-mode property (the boundary
        of this group), so the poll loop does not fire them — the provider object is built
        but not held here.
        """
        provider, resolved, _memory, bridge = _resolve_tools_and_provider()
        # The deploy-selected profile (issue #256): the same `_profile_from_env` read that gated
        # tool resolution builds the registry, so a policy-forbidden opted-in tool never reaches
        # `Harness` construction under `unlocked` only to raise on the locked default.
        _, policy = _profile_from_env()
        harness = Harness(
            provider,
            system_prompt=charter_from_env(),
            tools=resolved.tools,
            policy=policy,
            max_steps=_max_steps_from_env(),
            response_retries=_response_retries_from_env(),
            server_builtins=resolved.builtins,
            turn_hook=bridge.on_reply if bridge is not None else None,
            # The context budget (issue #276): the poll loop's session is as long-lived as a
            # wake-mode one — it is the same transcript — so it is bounded the same way.
            compactor=_compactor_from_env(provider),
        )
        return cls(
            harness,
            timeline=os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
            code_bridge=bridge,
            # The per-wake MCP image store (issue #318): ``None`` unless an MCP server loaded.
            mcp_images=resolved.mcp_images,
        )

    def poll_once(self) -> list[object]:
        """Handle every new message once: think, and act if the agent decides to. Returns its posts.

        **The agent speaks by calling the `messages` tool — nothing here posts for it** (the
        Unspoken Channel, issue #293). This loop used to auto-post the model's final text as the
        reply, which is the defect that inversion removes: the harness owned an implicit reply
        channel the model could not see, so a model that spoke through the tool *and* ended with
        its usual narration said everything twice. The turn's final text is now **unspoken** —
        journaled for the operator, never posted.

        The consequence is worth stating plainly, because it changes what a hand-built agent needs:
        an agent with no `messages` tool **cannot speak**. `from_env` wires it (it is a shipped
        default), so the fleet path is unaffected; a `Harness` assembled by hand must register
        `MessagesTool` to give its agent a voice. That is the kit's contract now — a capability is
        a tool you hand it, and speech is no longer the one exception.

        Returns the messages the agent posted *through its tools* this poll (empty when it chose
        silence, which is a legitimate outcome, not a failure).
        """
        posted: list[object] = []
        for message in self._new_messages():
            if message.user.uuid == self.me_uuid:
                continue  # never engage with ourselves
            text = _incoming_text(message)
            # Per *message*, not per poll: each is its own turn, with its own answer to "was this
            # addressed to me, and did I act?" The ledger is what the informer reads, so it cycles
            # with it. Every message reaching here is a peer's (own posts are skipped just above),
            # so `counterpart_message=True` — which arms the one-on-one nudge on a two-viewer
            # timeline (issue #332), the poll-path twin of the wake path's message batch.
            self.speech.reset()
            self.informer.arm(text, counterpart_message=True)
            narration = self.harness.send(text)
            kind = "reserve" if self.harness.engine.reserve_used else "narration"
            log_unspoken(narration, timeline=self.timeline_uuid, kind=kind)
            posted.extend(self.speech.posts)
        return posted

    def run(self, *, interval: float = DEFAULT_POLL_INTERVAL, max_polls: int | None = None) -> None:
        """Poll forever (or `max_polls` times), sleeping `interval` seconds between polls."""
        count = 0
        while max_polls is None or count < max_polls:
            self.poll_once()
            count += 1
            if max_polls is not None and count >= max_polls:
                return
            time.sleep(interval)

    # --- reading new messages, newest-first, up to the high-water mark --------

    def _new_messages(self) -> list[object]:
        """Messages newer than the high-water mark, in chronological order."""
        fresh = _messages_since(
            self.client.messages.filter(timeline=self.timeline_uuid), self._last_seen
        )
        if fresh:
            self._last_seen = fresh[-1].content.uuid
        return fresh


# --- shared message helpers (used by both the poll loop and wake mode) -------


def _recent(messages: object, cap: int | None) -> list[object]:
    """The most recent `cap` messages from a newest-first iterable; `None` → all.

    `messages` is the SDK's lazy, auto-paginating filter, so a finite cap fetches
    only the pages it needs rather than the whole timeline. `max(cap, 1)` keeps the
    true newest message in the result even at a cap of 0, so a high-water mark
    derived from it is still correct when no context is seeded.
    """
    if cap is None:
        return list(messages)
    return list(itertools.islice(messages, max(cap, 1)))


def _messages_since(messages: object, mark: str | None) -> list[object]:
    """Messages from a newest-first iterable that are newer than `mark`, chronological.

    Walks newest-first and stops at the high-water mark (`mark`), so it reads only
    the unseen head of the timeline, then reverses to chronological order. A `mark`
    of `None` (or one no longer present) yields everything it iterates.
    """
    fresh = []
    for message in messages:
        if message.content.uuid == mark:
            break
        fresh.append(message)
    fresh.reverse()
    return fresh


def _parse_created_at(value: str) -> datetime:
    """Parse a timeline item's raw ISO-8601 ``created_at`` string into an aware UTC datetime.

    The SDK is a wire-exact passthrough — it hands ``created_at`` back as the raw string the
    platform sent (e.g. ``2026-06-04T00:00:00.000Z``), never a parsed ``datetime`` — so any
    age arithmetic has to parse it. Two robustness fixes over a bare
    ``datetime.fromisoformat``, both load-bearing so the read-pace math (`ReadPacer`) never
    crashes a wake on a real-world stamp:

    - **Trailing ``Z``.** ``fromisoformat`` did not accept the ``Z`` military-UTC suffix until
      Python 3.11, but we target 3.10+, so ``Z`` is normalized to ``+00:00`` first.
    - **A naive stamp is assumed UTC.** Every agent runs UTC on the box and the platform emits
      UTC; a stamp that somehow arrives without an offset is made tz-aware as UTC, so
      subtracting it from an aware ``now`` never raises the naive/aware ``TypeError``.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _incoming_text(message: object) -> str:
    """Another peer's message as the agent hears it: when it arrived, then who spoke.

    The leading ``[created_at]`` stamp is the item's own timeline timestamp, which the model
    reads against the brief's `Current Time:` anchor to reason about how old the message is.
    """
    return f"[{message.created_at}] {message.user.handle}: {message.content.body}"


def _as_turn(message: object, me_uuid: str) -> Message:
    """A timeline message as a conversation turn for the engine.

    The agent's own posts become assistant turns; everyone else's become user turns
    tagged with the speaker, so the model can tell a multi-party conversation apart.
    """
    if message.user.uuid == me_uuid:
        return Message.assistant(content=message.content.body)
    return Message.user(content=_incoming_text(message))


def _client_from_env() -> BaseCradle:
    """Build the BaseCradle client the environment asks for — one token lifecycle.

    The founder directive: *use the existing token for everything, and mint a new one
    only when there is no token or the token is dead.* This factory is the front half
    (reuse vs. mint); the back half (re-mint on a 401) lives in `SelfHealingBaseCradle`,
    which this returns so both paths self-heal. See `_token` for the whole loop.

    1. **Reuse path (preferred, the default).** If ``BASECRADLE_TOKEN`` is set, use it
       as-is — no mint, no write, unchanged precedence and least privilege. The client
       still carries any ``BASECRADLE_EMAIL`` / ``BASECRADLE_PASSWORD`` so that if this
       token later dies mid-run, it can re-mint rather than strand the agent.
    2. **Mint path (the self-bootstrap fallback).** If no token is set but
       ``BASECRADLE_EMAIL`` and ``BASECRADLE_PASSWORD`` are, mint a fresh token via the
       SDK's ``login`` *and persist it* to ``BASECRADLE_ENV_FILE`` (the file the agent
       sources its env from), so the next wake reuses it instead of minting again — a
       credential-only AI mints exactly once, not once per wake. ``BASECRADLE_SESSION_NAME``
       optionally labels the minted credential.

    The password is read straight into the login call — never logged, never placed on the
    agent's reasoning surface. The agent ends up holding a *token*, not the cleartext
    secret; persistence writes only that minted token back.
    """
    token = os.environ.get("BASECRADLE_TOKEN")
    email = os.environ.get("BASECRADLE_EMAIL")
    password = os.environ.get("BASECRADLE_PASSWORD")
    session_name = os.environ.get("BASECRADLE_SESSION_NAME")
    env_file = os.environ.get("BASECRADLE_ENV_FILE")

    if token:
        # Reuse — no mint, no write. Carry the credentials + env-file path so a *later*
        # 401 (the token dies mid-run) can re-mint and re-persist instead of stranding us.
        return SelfHealingBaseCradle(
            token, email=email, password=password, session_name=session_name, env_file=env_file
        )
    if email and password:
        # No token: mint one (persisted + mirrored to the env by `mint_token`) and reuse it.
        token = mint_token(
            email=email, password=password, session_name=session_name, env_file=env_file
        )
        return SelfHealingBaseCradle(
            token, email=email, password=password, session_name=session_name, env_file=env_file
        )
    raise ValueError(
        "No BaseCradle credentials in the environment. Set BASECRADLE_TOKEN to use an "
        "existing token (preferred), or set BASECRADLE_EMAIL + BASECRADLE_PASSWORD to "
        "mint one on startup."
    )


# The Dashboard documentation links worth putting in front of a fresh peer, as
# (label, wire-field) pairs in the order they read. Each is included only if the
# Dashboard actually returned it, so an older API contributes only what it has.
_DOC_LINKS = (
    ("User guide", "user_guide"),
    ("API", "api"),
    ("API reference", "reference"),
    ("OpenAPI", "openapi"),
    ("Changelog", "changelog"),
)


def _orientation(dashboard: object) -> str | None:
    """A bounded startup briefing built from the agent's Dashboard.

    The Dashboard answers "what is this place, and what am I here?" — its
    ``environment`` (name, summary, what you are) plus the ``documentation`` links.
    We render only the fields the Dashboard actually returned (the SDK raises on a
    field the API omitted, so each is read defensively), and only short, fixed
    pieces — never unbounded content. Returns ``None`` when the Dashboard carries
    no orientation at all (e.g. an older API form), so the caller leaves the
    charter untouched rather than seeding an empty heading.
    """
    lines: list[str] = []

    env = getattr(dashboard, "environment", None)
    if env is not None:
        name = getattr(env, "name", None)
        summary = getattr(env, "summary", None)
        you_are = getattr(env, "you_are", None)
        if name and summary:
            lines.append(f"You are on {name} — {summary}")
        elif summary:
            lines.append(summary)
        if you_are:
            lines.append(f"Here, you are {you_are}.")

    docs = getattr(dashboard, "documentation", None)
    if docs is not None:
        doc_lines = [
            f"- {label}: {url}"
            for label, field in _DOC_LINKS
            if (url := getattr(docs, field, None))
        ]
        if doc_lines:
            lines.append("Documentation:")
            lines.extend(doc_lines)

    if not lines:
        return None
    return "Your BaseCradle orientation:\n" + "\n".join(lines)


def _compose_prompt(orientation: str | None, system_prompt: str | None) -> str | None:
    """Join the Dashboard orientation and the operator's charter, orientation first.

    Either may be absent: with neither, the charter stays `None`; with one, it is
    used alone — so onboarding never fabricates a prompt where there was none.
    """
    parts = [part for part in (orientation, system_prompt) if part]
    return "\n\n".join(parts) if parts else None


#: The config defaults: the @jt stack (OpenAI vendor, openai SDK, Responses surface).
DEFAULT_PROVIDER = "openai"
DEFAULT_SDK = "openai"
_PROVIDERS = ("openai", "xai", "openrouter")

#: ``surface`` is an **SDK-scoped** concept: each SDK adapter declares its own allowed
#: ``SURFACES`` and ``DEFAULT_SURFACE`` (so the *next* multi-surface SDK follows the same
#: contract without re-litigation), keyed by the ``AI_SDK`` package name. ``AI_SDK_SURFACE``
#: selects among the active adapter's surfaces; a single-surface SDK simply isn't listed here
#: (no surfaces to pick from) and never sets the var. v0 ships only the ``openai`` adapter.
_SDK_SURFACES: dict[str, tuple[tuple[str, ...], str]] = {
    "openai": (OPENAI_SURFACES, OPENAI_DEFAULT_SURFACE),
    # The native xai-sdk speaks a single (gRPC) surface; `AI_SDK_SURFACE` is left unset for it,
    # and any other value fails clearly against this one-element set (issue #165).
    "xai-sdk": (XAI_SDK_SURFACES, XAI_SDK_DEFAULT_SURFACE),
    # The native openrouter SDK ships the OpenAI-compatible chat wire only (its Responses API is
    # beta upstream), so it too declares a single surface and leaves `AI_SDK_SURFACE` unset.
    "openrouter": (OPENROUTER_SURFACES, OPENROUTER_DEFAULT_SURFACE),
}


def _resolve_surface(sdk: str) -> str:
    """Resolve ``AI_SDK_SURFACE`` against the active SDK adapter's declared surfaces.

    The uniform, SDK-scoped contract (issue #163): the active adapter owns its surface set, so
    the openai-shaped default no longer lives in this generic reader. The single rule —

    - ``AI_SDK_SURFACE`` **omitted/empty** → the adapter's ``DEFAULT_SURFACE``.
    - **provided** → validated against the adapter's ``SURFACES``; anything not in the set is a
      **hard fail** with a clear message.

    catches both a typo (``responsess``) and a surface set on a single-surface SDK (one whose
    adapter declares no surface set here — e.g. a future native ``xai-sdk`` or ``anthropic``).
    For an SDK with no surface declaration, an *unset* var resolves to ``""`` and the precise
    "no adapter yet" error is left to the provider build; a *set* var is the clear error above.
    """
    raw = (os.environ.get("AI_SDK_SURFACE") or "").strip().lower()
    spec = _SDK_SURFACES.get(sdk)
    if spec is None:
        if raw:
            raise ValueError(
                f"AI_SDK_SURFACE={raw!r} is set, but AI_SDK={sdk!r} declares no surfaces "
                "(it is single-surface, or ships no adapter yet). Unset AI_SDK_SURFACE."
            )
        return ""
    surfaces, default = spec
    if not raw:
        return default
    if raw not in surfaces:
        raise ValueError(
            f"Unknown AI_SDK_SURFACE {raw!r} for AI_SDK={sdk!r}; expected one of {surfaces}."
        )
    return raw


def _config_from_env() -> tuple[str, str, str]:
    """The ``(provider, sdk, surface)`` config triple from the environment, validated.

    Read in one place so the provider build and the plugin activation context agree on every
    axis. ``AI_PROVIDER`` is validated here against the known providers; ``AI_SDK_SURFACE`` is
    resolved **SDK-scoped** (`_resolve_surface`) — an unrecognized value is a clear error, never
    a silent fall-through. The SDK name is not constrained here — whether an adapter exists for
    it is decided when the provider is built (`_provider_from_config`), so the error names the
    missing adapter.
    """
    provider = (os.environ.get("AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown AI_PROVIDER {provider!r}; expected one of {_PROVIDERS}.")
    sdk = (os.environ.get("AI_SDK") or DEFAULT_SDK).strip().lower()
    surface = _resolve_surface(sdk)
    return provider, sdk, surface


#: A provider's canonical endpoint, supplied as the **default** ``base_url`` so a persona's
#: ``.env`` needn't hardcode it (``AI_BASE_URL`` always overrides for a proxy/gateway/self-host).
#: ``openai`` is absent → the SDK targets OpenAI's own default. xAI's compat endpoint speaks the
#: Responses **and** Chat wire, so the ``openai`` SDK reaches grok here over either surface.
_PROVIDER_BASE_URLS = {
    "xai": "https://api.x.ai/v1",
    # OpenRouter's OpenAI-compatible chat endpoint — the default base_url whether reached through
    # the native openrouter SDK or the openai SDK pointed here (both speak its chat wire).
    "openrouter": "https://openrouter.ai/api/v1",
}

#: xAI Live-Search sources, keyed by the built-in tool name the plugins activate.
_XAI_SEARCH_SOURCES = {"web_search": "web", "x_search": "x"}


def _xai_search_parameters(builtins: Sequence[str]) -> dict[str, Any] | None:
    """The active search built-ins → xAI's ``search_parameters`` Live-Search body field.

    The web_search wiring **diverges by endpoint vendor** (issue #163, verified against
    docs.x.ai): OpenAI's Responses runs web search from a ``tools:[{"type":"web_search"}]``
    entry, but xAI's endpoint runs Live Search from a top-level ``search_parameters`` object on
    **both** its Responses and Chat surfaces — it does *not* accept the OpenAI tools entry. So a
    Grok-via-``openai``-SDK persona's ``web_search``/``x_search`` built-ins are translated here
    into ``search_parameters`` and forwarded through the SDK's ``extra_body`` (see
    `_provider_from_config`), rather than offered as tools. Returns ``None`` when no search
    built-in is active, so nothing is sent.

    NOTE: the harness asserts *what it sends*; the exact ``search_parameters`` sub-shape is
    xAI's and is ground-truthed by the capital's live verification on the Grok persona.
    """
    sources = [src for name in builtins if (src := _XAI_SEARCH_SOURCES.get(name)) is not None]
    # De-dup while preserving order (web before x), in case a built-in is listed twice.
    sources = list(dict.fromkeys(sources))
    if not sources:
        return None
    return {"mode": "on", "sources": sources, "return_citations": True}


# The keys the harness sets itself on each build path — every one of these present in the
# operator's ``model_params.json`` is stripped with a WARNING (D3): a tuning file must never
# override the harness's own wiring. Each set unions the branch's constructor args (``model``,
# ``base_url``/``api_host``, ``timeout``, …) with the per-call args the adapter fills
# (``messages``/``input``/``tools``). ``model`` is in every set — model identity is ``AI_MODEL``,
# and it gets dedicated warning wording. ``stream`` is in every set too: all three adapters are
# **non-streaming by contract** (they call ``.model_dump()`` on a single response), so a
# ``stream: true`` would not just override wiring but crash the turn — strip it everywhere.
# ``extra_body`` is *not* listed here: it is lifted out separately (D4) — merged on the openai
# SDK, warned-and-dropped on the SDKs where it is not a legal concept.
_OWNED_OPENAI = frozenset(
    {
        "model",
        "api_key",
        "base_url",
        # The endpoint-vendor label the adapter logs (AI_PROVIDER decides it, never a tuning file);
        # it is also a real constructor arg, so leaving it unowned would make a `"provider"` key
        # in model_params.json a hard TypeError rather than a warned-and-dropped collision.
        "provider",
        "surface",
        "timeout",
        "max_retries",
        "builtin_tools",
        "code_container",
        "messages",
        "input",
        "tools",
        "stream",
    }
)
_OWNED_XAI_SDK = frozenset(
    {
        "model",
        "api_key",
        "api_host",
        "timeout",
        "builtin_tools",
        "client",
        "messages",
        "tools",
        "stream",
    }
)
_OWNED_OPENROUTER = frozenset(
    {
        "model",
        "api_key",
        "base_url",
        "timeout",
        "client",
        "builtin_tools",
        "web_search_params",
        "messages",
        "tools",
        "stream",
        # The harness passes this to ``chat.send`` itself (the routing-metadata header, issue #280).
        # Left un-owned, an operator key of the same name would arrive as a *second* value for the
        # same keyword — ``TypeError: got multiple values for keyword argument`` — which is not the
        # "unexpected keyword" shape `_ErrorMapper` reframes, so it would crash a wake raw.
        "http_headers",
    }
)


def _split_model_params(
    raw: Mapping[str, Any], *, owned: frozenset[str], sdk_label: str
) -> tuple[dict[str, Any], Any]:
    """Strip harness-owned keys from the operator's model params, returning ``(tuning, extra_body)``.

    Implements the collision policy (D3): every key in ``owned`` that the operator set in
    ``model_params.json`` is popped with a WARNING and never reaches the SDK call — the harness's
    own value stands, because this file is call *tuning*, not a way to override wiring. ``model``
    gets dedicated wording (model identity is ``AI_MODEL``). Splatting a params ``model`` into a
    constructor that already receives ``model`` positionally would be a hard ``TypeError: got
    multiple values`` — so the strip is what keeps the "warn and win" promise from becoming a crash.

    ``extra_body`` is *lifted* out (D4) rather than plain-stripped: it is returned separately so the
    caller can merge it (the openai SDK, whose ``extra_body`` is a real passthrough) or warn-and-drop
    it (an SDK where it is not a legal concept). ``sdk_label`` names the active SDK in the warnings.
    """
    params = dict(raw)
    extra_body = params.pop("extra_body", None)
    for key in sorted(set(params) & owned):
        if key == "model":
            _log.warning(
                "model_params.json sets 'model', but the model identity is AI_MODEL, not a "
                "model_params.json key — ignoring it. model_params.json is call tuning only."
            )
        else:
            _log.warning(
                "model_params.json sets %r, which the harness controls for %s — ignoring it "
                "(harness-owned keys always win).",
                key,
                sdk_label,
            )
        params.pop(key)
    return params, extra_body


def _merge_extra_body(params_extra_body: Any, harness_extra_body: Any) -> Any:
    """Combine the operator's ``extra_body`` with one the harness composes (D4) — harness wins.

    When both are present (an ``model_params.json`` ``extra_body`` *and* a harness-built one — e.g.
    xAI's ``search_parameters`` when a Grok-via-openai persona has search opted in), they merge
    key-by-key with the **harness value winning** any overlapping key, and each overlap logs a
    WARNING. When only one exists, it is used as-is; when neither, the result is ``None`` so nothing
    is sent.
    """
    if params_extra_body and harness_extra_body:
        for key in sorted(set(params_extra_body) & set(harness_extra_body)):
            _log.warning(
                "model_params.json extra_body sets %r, which the harness sets itself for this "
                "provider (e.g. xAI search_parameters) — the harness value wins.",
                key,
            )
        return {**params_extra_body, **harness_extra_body}
    return harness_extra_body or params_extra_body or None


def _merge_extra_headers(params_headers: Any, harness_headers: Any) -> Any:
    """Combine the operator's ``extra_headers`` with one the harness composes — harness wins.

    The header-side twin of `_merge_extra_body`, and it exists for the same two reasons. **Merging
    rather than owning**: an operator's ``extra_headers`` works today on this adapter's OpenAI and
    xAI endpoints (the ``openai`` SDK takes it per call), so confiscating the key would break a
    working setup to fix a collision that only exists on the OpenRouter one. **Harness wins on
    overlap**: the routing-metadata header is what makes ``endpoint=`` truthful (issue #280), and an
    operator who unsets it would silently blind the fleet's routing review rather than break
    anything visible — the failure mode this whole issue is about.

    ``None`` when neither side has headers, so nothing is sent and the SDK client is built exactly
    as it was before.
    """
    if params_headers and harness_headers:
        for key in sorted(set(params_headers) & set(harness_headers)):
            _log.warning(
                "model_params.json extra_headers sets %r, which the harness sets itself for this "
                "provider (the routing-metadata header that makes endpoint= truthful) — the "
                "harness value wins.",
                key,
            )
        return {**params_headers, **harness_headers}
    return harness_headers or params_headers or None


def resolved_model_params(sdk: str) -> tuple[dict[str, Any], list[str]]:
    """The operator's ``model_params.json`` as loaded, plus the keys the active SDK's build drops.

    The read-only introspection twin of the collision policy `_provider_from_config` enforces at
    build (via `_split_model_params`) — but **pure**: it logs nothing and builds no provider, so
    ``--resolved-config`` can report the loaded tuning without a WARNING storm or a model call.
    Params are non-secret by contract (secrets live in ``agent.env``), so emitting them is safe.

    Returns ``(model_params, stripped)``:

    - ``model_params`` — the ``model_params.json`` object **verbatim** (``{}`` when the file is
      absent), so a verifier sees exactly what the operator wrote.
    - ``stripped`` — the sorted keys that would **not** reach the SDK call: the harness-owned
      collisions for this SDK's build path (`_OWNED_OPENAI`/`_OWNED_XAI_SDK`/`_OWNED_OPENROUTER`,
      keyed on the SDK, since the openai adapter serves openai/xai/openrouter alike), **plus**
      ``extra_body`` on the SDKs whose build warns-and-drops it (``xai-sdk``, ``openrouter``; the
      openai SDK passes ``extra_body`` through, so it is not counted there). The effective tuning
      the SDK receives is thus ``model_params`` minus ``stripped``.

    A malformed ``model_params.json`` raises `ValueError` here (from `load_model_params`) — the
    same failure a wake would hit, surfaced at verify time instead: `resolved_config`'s caller
    turns it into a clean non-zero exit, so the NOC catches the misconfiguration before it goes live.
    """
    loaded = load_model_params()
    owned = {"xai-sdk": _OWNED_XAI_SDK, "openrouter": _OWNED_OPENROUTER}.get(sdk, _OWNED_OPENAI)
    stripped = set(loaded) & owned
    if sdk in ("xai-sdk", "openrouter") and "extra_body" in loaded:
        stripped.add("extra_body")
    return loaded, sorted(stripped)


def _provider_from_config(
    provider: str,
    sdk: str,
    surface: str,
    *,
    builtins: Sequence[str] = (),
    code_bridge: CodeExecutionBridge | None = None,
) -> Provider:
    """Build the model provider the config selects — the @jt OpenAI-SDK stack by default.

    The harness reaches an LLM **only** through a vendor SDK, so the **SDK picks the adapter**
    and the **provider picks the endpoint** (its default ``base_url`` + key). Two adapters ship:

    - ``AI_SDK=openai`` → `OpenAIProvider`, the official ``openai`` SDK. It serves **both**
      wired providers, differing only by endpoint and how the search built-in is wired:
      - ``AI_PROVIDER=openai`` (default) → OpenAI's own endpoint; ``builtins`` (e.g.
        ``web_search``) pass through as ``builtin_tools`` and apply on the Responses surface.
      - ``AI_PROVIDER=xai`` → the same ``openai`` client pointed at ``api.x.ai`` (issue #163;
        ``grok-4.3`` over the ``responses`` *or* ``chat`` surface). xAI's Live-Search built-ins
        (``web_search`` / ``x_search``) are translated to a ``search_parameters`` body field
        (`_xai_search_parameters`) sent via ``extra_body`` — xAI's wiring, not OpenAI's.
    - ``AI_SDK=xai-sdk`` → `XaiSdkProvider`, the **native** xAI SDK (gRPC), the Grok personas'
      end-state brain (issue #165). It talks **only** to ``AI_PROVIDER=xai``; the opted-in search
      built-ins become xAI **Agent Tool** entries on the chat ``tools`` list inside the adapter
      (issue #171 — the native ``SearchParameters`` object is deprecated). Its single native
      surface means ``AI_SDK_SURFACE`` is unset.
    - ``AI_SDK=openrouter`` → `OpenRouterProvider`, OpenRouter's **native** first-party SDK, the
      @glm-5.2 peer's brain. It talks **only** to ``AI_PROVIDER=openrouter``; it speaks a single
      ``chat`` surface (OpenRouter's Responses API is beta upstream), so ``AI_SDK_SURFACE`` is unset.
      The opted-in ``web_search`` built-in becomes OpenRouter's ``openrouter:web_search`` **server
      tool** on the chat ``tools`` array (issue #237), tuned by ``search_params.json``; OpenRouter
      runs it server-side and returns a grounded, cited answer. OpenRouter is *also* reachable
      through the ``openai`` SDK pointed at ``openrouter.ai`` — a permanent matrix cell — but
      **chat-only and server-built-in-free** (that cell ships no server tools), gated below with a
      clear error naming the fix.
    - Any other ``AI_SDK`` is a clear "no adapter yet" error.

    All read ``AI_MODEL`` and fall back to ``AI_API_KEY`` for the key. For the openai/openrouter
    SDKs, ``base_url`` is ``AI_BASE_URL`` if set, else the provider's canonical default
    (`_PROVIDER_BASE_URLS`); the native xai-sdk uses its own endpoint (``api.x.ai``).

    Optional SDK call parameters come from the operator's ``model_params.json``
    (`load_model_params`), read once here and threaded into the adapter as ``**default_params``.
    Harness-owned keys are stripped with a WARNING (`_split_model_params`) so tuning never
    overrides wiring; a malformed file raises here, failing the wake loudly at startup (the
    read-only introspection paths never build a provider, so they never touch it).
    """
    model = os.environ.get("AI_MODEL")
    if not model:
        raise ValueError("AI_MODEL is required — the model id to run (e.g. gpt-5.4-mini).")

    # ``model_params.json`` is read *after* each branch's config-shape guards (below), never here:
    # a config mismatch (e.g. AI_SDK=xai-sdk + AI_PROVIDER=openrouter) must surface its own
    # actionable error, not be masked by a malformed model_params file the wake would never even
    # use. So the read sits at the point of use in each branch, past its guard.

    if sdk == "xai-sdk":
        if provider != "xai":
            raise ValueError(
                f"AI_SDK=xai-sdk reaches xAI's native endpoint, so it requires AI_PROVIDER=xai "
                f"(got {provider!r}). Use AI_SDK=openai for a non-xAI provider."
            )
        params, extra_body = _split_model_params(
            load_model_params(), owned=_OWNED_XAI_SDK, sdk_label="the native xai-sdk"
        )
        if extra_body is not None:
            _log.warning(
                "model_params.json sets 'extra_body', which the native xai-sdk does not support "
                "(it is an openai-SDK concept) — ignoring it."
            )
        return XaiSdkProvider(model, builtin_tools=list(builtins), **params)

    if sdk == "openrouter":
        if provider != "openrouter":
            raise ValueError(
                f"AI_SDK=openrouter reaches OpenRouter's native endpoint, so it requires "
                f"AI_PROVIDER=openrouter (got {provider!r}). Use AI_SDK=openai for a non-OpenRouter "
                "provider, or set AI_PROVIDER=openrouter."
            )
        params, extra_body = _split_model_params(
            load_model_params(), owned=_OWNED_OPENROUTER, sdk_label="the openrouter SDK"
        )
        if extra_body is not None:
            _log.warning(
                "model_params.json sets 'extra_body', which the openrouter SDK does not support "
                "(its chat.send is typed with no extra_body) — ignoring it. On the openrouter SDK, "
                "pass only keys chat.send names; use the openai-SDK path for the extra_body escape "
                "hatch."
            )
        base_url = os.environ.get("AI_BASE_URL") or _PROVIDER_BASE_URLS.get(provider)
        # The opted-in web_search built-in rides the chat `tools` array as OpenRouter's
        # `openrouter:web_search` server tool (the adapter maps the name → wire type); its optional
        # `parameters` come from the operator's search_params.json. Read that file **only when web
        # search is actually active** — a malformed search_params.json must not fail the wake of a
        # default-riding agent that never opted the tool in (unlike model_params.json, which is
        # always relevant). Empty builtins/params → nothing extra is sent.
        builtin_list = list(builtins)
        web_search_params = (
            load_search_params() or None if WEB_SEARCH_BUILTIN in builtin_list else None
        )
        return OpenRouterProvider(
            model,
            base_url=base_url,
            builtin_tools=builtin_list,
            web_search_params=web_search_params,
            **params,
        )

    if sdk != "openai":
        raise ValueError(
            f"AI_SDK={sdk!r} has no adapter — the harness ships 'openai' (the OpenAI-wire SDK, "
            "also xAI over api.x.ai and OpenRouter over openrouter.ai), 'xai-sdk' (native xAI), "
            "and 'openrouter' (native OpenRouter). Set one of those."
        )
    if provider not in ("openai", "xai", "openrouter"):
        raise ValueError(
            f"AI_PROVIDER={provider!r} has no adapter via the openai SDK — 'openai', 'xai', and "
            "'openrouter' are wired (xAI over api.x.ai, OpenRouter over openrouter.ai)."
        )
    if provider == "openrouter" and surface != "chat":
        # OpenRouter's Responses API is beta upstream, so the openai-SDK-at-OpenRouter cell is
        # chat-only. The openai SDK's own default surface is `responses`, so this is the FIRST
        # thing an operator hits — the message carries the fix.
        raise ValueError(
            f"AI_PROVIDER=openrouter over the openai SDK is chat-only (its Responses API is beta "
            f"upstream), but AI_SDK_SURFACE resolved to {surface!r}. Set AI_SDK_SURFACE=chat (or "
            "use AI_SDK=openrouter for the native adapter, which is chat-only by design)."
        )
    base_url = os.environ.get("AI_BASE_URL") or _PROVIDER_BASE_URLS.get(provider)
    params, params_extra_body = _split_model_params(
        load_model_params(), owned=_OWNED_OPENAI, sdk_label="the openai SDK"
    )
    # `extra_headers` is now harness wiring on one of this adapter's three endpoints (the routing
    # header, below), so it is **lifted out** of the operator's tuning rather than splatted with it —
    # exactly as `extra_body` is, and for the same reason. Left in `params` it would arrive as a
    # *second* value for the same keyword (`TypeError: got multiple values for keyword argument`),
    # which is a raw crash, not the warned-and-dropped collision policy. Lifting keeps the operator's
    # headers working (they merge, below) instead of silently confiscating a key that works today.
    params_extra_headers = params.pop("extra_headers", None)
    if provider == "xai":
        # xAI's search built-ins ride `search_parameters`, not OpenAI tools entries — so they
        # go through `extra_body`, and nothing is offered as a `builtin_tools` tool here.
        search_parameters = _xai_search_parameters(builtins)
        harness_extra_body = {"search_parameters": search_parameters} if search_parameters else None
        extra_body = _merge_extra_body(params_extra_body, harness_extra_body)
        return OpenAIProvider(
            model,
            base_url=base_url,
            provider=provider,
            surface=surface,
            extra_body=extra_body,
            extra_headers=params_extra_headers,
            **params,
        )
    if provider == "openrouter":
        # OpenRouter over the openai SDK: chat wire only, no server-side built-ins and no code
        # bridge (`_maybe_code_bridge` already excludes it). The operator's `extra_body` (if any)
        # is the escape hatch and passes straight through.
        #
        # The routing-metadata header goes on for the same reason the native adapter sends it: a
        # router must be asked to say which endpoint it routed to, and unasked it says nothing
        # trustworthy (issue #280). Set *here* rather than in the adapter, because this one adapter
        # also serves OpenAI and xAI, where the header would be meaningless — which endpoint we are
        # aimed at is this layer's knowledge, and keeping it here is what keeps the adapter free of
        # a vendor branch.
        return OpenAIProvider(
            model,
            base_url=base_url,
            provider=provider,
            surface=surface,
            extra_body=params_extra_body,
            extra_headers=_merge_extra_headers(
                params_extra_headers, OPENROUTER_ROUTING_METADATA_HEADER
            ),
            **params,
        )
    # The code-execution Asset bridge supplies the live ``container`` for the code_interpreter
    # built-in per turn (`_code.py`); absent it, the built-in falls back to an auto container.
    code_container = code_bridge.container_spec if code_bridge is not None else None
    return OpenAIProvider(
        model,
        base_url=base_url,
        provider=provider,
        surface=surface,
        builtin_tools=list(builtins),
        code_container=code_container,
        extra_body=params_extra_body,
        extra_headers=params_extra_headers,
        **params,
    )


def _maybe_code_bridge(
    provider_name: str, surface: str, builtins: Sequence[str]
) -> CodeExecutionBridge | None:
    """The code-execution Asset bridge, when (and only when) it applies — else ``None``.

    The bridge is **OpenAI-only** (issue #172): it needs OpenAI's Code Interpreter container file
    API, which lives on the Responses surface. So it is built only for ``AI_PROVIDER=openai`` on
    the ``responses`` surface with the ``code_interpreter`` built-in opted in. On xAI (no
    input-file mechanism) or the chat surface (no Code Interpreter) there is no bridge — grok can
    still *run* code via its own built-in, it just cannot exchange files with the Asset system.
    The bridge reuses the same key/base_url the OpenAI provider uses.
    """
    if (
        provider_name != "openai"
        or surface != "responses"
        or CODE_EXECUTION_BUILTIN not in builtins
    ):
        return None
    base_url = os.environ.get("AI_BASE_URL") or None
    return CodeExecutionBridge(base_url=base_url)


def _resolve_tools_and_provider() -> tuple[
    Provider, ResolvedTools, MemoryProvider, CodeExecutionBridge | None
]:
    """Resolve the active tools, the model provider, and the memory provider from config + env.

    The single seam both `TimelineAgent.from_env` and `WakeAgent.from_env` use to wire their
    tools, replacing the old hardcoded list. It: reads the active ``(provider, sdk, surface)``
    config; loads the tool plugins (the ``tools/`` overlay, else packaged defaults — see
    `load_plugins`);
    resolves them against the active config (`resolve_plugins`), so an OpenAI-coupled tool or
    a Responses-only built-in self-excludes when its requirement isn't met; builds the
    **memory provider** (`memory_provider_from_env`) and folds *its* model-facing tools into
    the resolved set (`_merge_memory_tools`); then builds the model provider with the
    resolved built-ins.

    Memory is a provider now, not a hardcoded plugin — so the memory tool comes from
    ``memory.tools()`` (the default SQLite provider supplies the `MemoryTool`; the MemPalace
    adapter supplies its read-only `memory_search`; a provider that wants purely automatic
    memory supplies none). **MCP drop-ins** (Group 5) fold
    in next: every server configured under the config home's ``mcp/`` dir is connected, its
    tools proxied into the set, and its safe-by-default opt-out surfaced in ``.notices``
    (`_merge_mcp_tools`) — with ``mcp/`` empty (the default) this is a no-op. Finally the
    **locked policy** is applied here too (`_apply_safe_policy`): a drop-in tool that needs a
    forbidden capability is dropped and surfaced rather than crashing `Harness` construction,
    so the safe boundary degrades gracefully. Returns the model provider, the merged
    `ResolvedTools` (``.tools`` → `Harness`, where the policy gate still applies as
    defense-in-depth; ``.manifest``/``.notices`` → the persistent Turn-0 brief), the
    memory provider itself, which the wake holds to fire its `observe`/`context` hooks, and the
    code-execution Asset bridge (`_maybe_code_bridge`) when code execution is active — ``None``
    otherwise — which the hosting agent binds to its platform context and the engine turn hook.
    """
    # Parse + validate the config first, then drive both the upgrade reconcile and the load with
    # the *one* validated provider string — so an invalid AI_PROVIDER fails fast (before the
    # reconcile mutates anything) and the reconcile and the load can never disagree on the
    # provider (the divergence a second, unvalidated env read would risk).
    provider_name, sdk, surface = _config_from_env()
    # Reconcile a materialized config home that a `pip install -U` left behind *before* loading
    # the overlay, so a stale default plugin from the previous version is refreshed rather than
    # loaded broken (issue #160). A no-op for a never-installed deployment (@jt) and for an
    # already-current one; guarded so a reconcile hiccup never blocks startup.
    _reconcile_config_on_upgrade(provider_name)
    resolved, memory = _resolve_tools(provider_name, sdk, surface)
    bridge = _maybe_code_bridge(provider_name, surface, resolved.builtins)
    provider = _provider_from_config(
        provider_name, sdk, surface, builtins=resolved.builtins, code_bridge=bridge
    )
    return provider, resolved, memory, bridge


def _resolve_tools(
    provider_name: str, sdk: str, surface: str
) -> tuple[ResolvedTools, MemoryProvider]:
    """Settle the active tool set for a validated config triple — without building the provider.

    The shared tool-resolution core of `_resolve_tools_and_provider`, factored out so the
    read-only introspection path (`resolved_config`) can settle the **exact same** active tool
    set the running agent would — same plugin overlay, memory provider, MCP drop-ins, and locked
    policy — without constructing the model provider (which needs ``AI_API_KEY``). The caller
    owns the *reconcile* decision: a wake reconciles a `pip install -U` overlay first because it
    is about to act; the side-effect-free introspection deliberately does not (it must not write
    to the config home).
    """
    ctx = ActivationContext(
        provider=provider_name,
        sdk=sdk,
        surface=surface,
        model=os.environ.get("AI_MODEL", ""),
        env=os.environ,
    )
    # Provider-aware load (issue #160): a plugin file whose source declares affinity for another
    # provider is skipped before import, so an OpenAI agent neither loads nor risks importing the
    # grok/xAI plugins. The resolver still applies the full activation gate on what remains.
    loaded = load_plugins_report(provider=provider_name)
    resolved = resolve_plugins(loaded.plugins, ctx)
    memory = memory_provider_from_env()
    resolved = _merge_memory_tools(resolved, memory)
    resolved = _merge_mcp_tools(resolved, load_mcp_tools())
    # The deploy-selected profile (issue #256) gates this env-resolution filter, so the
    # resolved/skipped split matches the profile the registry is built with — one env read
    # (`_profile_from_env`) drives both. Absent/`locked`/unrecognized → locked, exactly as before.
    _, policy = _profile_from_env()
    resolved = _apply_safe_policy(resolved, policy)
    # Surface any broken *shipped-default* plugin loudly in the Turn-0 brief — a defect, never
    # a silent swallow (issue #160). Last, so it rides whatever the merges/policy produced.
    resolved = _surface_broken_defaults(resolved, loaded.broken_defaults)
    return resolved, memory


def _reconcile_config_on_upgrade(provider: str) -> None:
    """Run the upgrade reconcile for `provider`, degrading any failure to a log line.

    `provider` is the already-validated `AI_PROVIDER` from `_config_from_env`, threaded through so
    the reconcile filters/prunes by exactly the provider the load will use — never a second,
    independently-read env value that could diverge. The config-home refresh is a best-effort
    convenience (the operator can always run `basecradle-harness-install` by hand), so a
    permission/IO error reconciling it must not take the agent down — it degrades to the existing
    (possibly stale) overlay, exactly the graceful-degradation bar the dashboard fetch and the
    brief composition already hold to.
    """
    try:
        reconcile_on_upgrade(provider=provider)
    except Exception:  # noqa: BLE001 - a reconcile hiccup must never block the agent's startup
        _log.warning(
            "Config-home upgrade reconcile failed; continuing with the existing overlay.",
            exc_info=True,
        )


def _surface_broken_defaults(
    resolved: ResolvedTools, broken_defaults: list[tuple[str, str]]
) -> ResolvedTools:
    """Fold broken shipped-default plugins into the resolved set's `broken` defect lines.

    Each ``(filename, error)`` becomes one Turn-0 brief defect line (rendered by
    `_brief.render_defects` under its own loud heading) and rides ``.skipped`` too, for the
    "why isn't this tool here?" trail. A *defect*, kept apart from `notices` (an intentional
    opt-out): a shipped default that failed to load is a bug, not a choice. With nothing
    broken this returns `resolved` unchanged, so a healthy agent's brief is untouched.
    """
    if not broken_defaults:
        return resolved
    broken_lines = [f"{name} — failed to load: {exc}" for name, exc in broken_defaults]
    skipped = resolved.skipped + [
        (name, f"shipped default failed to load: {exc}") for name, exc in broken_defaults
    ]
    return ResolvedTools(
        tools=resolved.tools,
        builtins=resolved.builtins,
        skipped=skipped,
        manifest=resolved.manifest,
        notices=resolved.notices,
        broken=resolved.broken + broken_lines,
        opt_in_stems=resolved.opt_in_stems,
        mcp_images=resolved.mcp_images,
    )


def _merge_memory_tools(resolved: ResolvedTools, memory: MemoryProvider) -> ResolvedTools:
    """Fold the memory provider's tools into the resolved set, deduped by name.

    Memory graduated from a tool plugin to its own provider subsystem, so its tool is
    appended here rather than loaded from ``_defaults/tools/``. Dedup by name is a safety
    net for a config home that *predates* this change and still carries an orphaned
    ``tools/memory.py``: the plugin-loaded tool wins and the provider's duplicate is dropped,
    so the registry never sees two ``memory`` tools. The manifest is extended in lockstep so
    the persistent brief lists the memory tool exactly once.
    """
    existing = {tool.name for tool in resolved.tools}
    added = [tool for tool in memory.tools() if tool.name not in existing]
    if not added:
        return resolved
    return ResolvedTools(
        tools=resolved.tools + added,
        builtins=resolved.builtins,
        skipped=resolved.skipped,
        manifest=resolved.manifest + [(tool.name, None) for tool in added],
        notices=resolved.notices,
        broken=resolved.broken,
        opt_in_stems=resolved.opt_in_stems,
        mcp_images=resolved.mcp_images,
    )


def _merge_mcp_tools(resolved: ResolvedTools, mcp: McpResolution) -> ResolvedTools:
    """Fold an `McpResolution` into the resolved set: tools, manifest, skips, and notices.

    Each active MCP server's proxy tools join ``.tools`` (deduped by name as a safety net —
    MCP names are namespaced ``<server>__<tool>``, so a collision means an operator's
    ``tools/`` overlay already claimed the name, which wins). The per-tool manifest entries
    and the per-server safe-by-default opt-out `notices` extend the brief's inputs, and a
    failed server's ``(name, reason)`` rides ``.skipped`` like a Group-2 activation skip.
    With ``mcp/`` empty the `McpResolution` is empty and this returns ``resolved`` unchanged.
    """
    if not mcp.tools and not mcp.skipped and not mcp.notices:
        return resolved
    existing = {tool.name for tool in resolved.tools}
    added = [tool for tool in mcp.tools if tool.name not in existing]
    added_manifest = [entry for entry in mcp.manifest if entry[0] not in existing]
    return ResolvedTools(
        tools=resolved.tools + added,
        builtins=resolved.builtins,
        skipped=resolved.skipped + mcp.skipped,
        manifest=resolved.manifest + added_manifest,
        notices=resolved.notices + mcp.notices,
        broken=resolved.broken,
        opt_in_stems=resolved.opt_in_stems,
        # The per-wake image store (issue #318): carried so the assets ``post_image`` action can
        # reach it via the `PlatformContext`. ``None`` unless an MCP server's tools loaded.
        mcp_images=mcp.images,
    )


def _apply_safe_policy(resolved: ResolvedTools, policy: Policy | None = None) -> ResolvedTools:
    """Drop any resolved tool the policy forbids or that vetoes its runtime, surfacing it.

    `Harness` already gates tools through the policy at registration — but it does so by
    *raising* `PolicyError`, which on the env-resolution path would crash the whole wake the
    moment a drop-in ``tools/`` tool declared a forbidden capability (e.g. ``SHELL``). That
    is the wrong failure shape for a peer: one bad operator tool should self-exclude, not
    take the agent down. So the safe boundary is applied here as a *filter* — a forbidden
    tool is removed from ``.tools`` and its manifest entry, recorded in ``.skipped``, and
    surfaced as a `notices` line — exactly the Group-2 robustness bar, now extended to the
    policy gate (Part B). The policy is **not** bypassed: the tool never reaches the
    registry, and the locked `Harness` re-checks the survivors as defense-in-depth. Safe by
    default stays a policy property; activation never overrides it.

    The same filter also honors a tool's own runtime veto (`Tool.load_refusal` — e.g. the
    shell tool refusing to run as root, issue #253): registration would *raise* on it just
    as it does a policy refusal, so it is dropped and surfaced here in the identical shape,
    keeping the wake graceful whichever gate a tool trips.
    """
    policy = policy or Policy.locked()
    permitted: list[Tool] = []
    refused: dict[str, str] = {}
    notices = list(resolved.notices)
    skipped = list(resolved.skipped)
    for tool in resolved.tools:
        if not policy.permits(tool):
            blocked = ", ".join(sorted(tool.requires & policy.forbidden))
            reason = f"refused by the safe-by-default policy: needs {blocked}"
        elif runtime_refusal := tool.load_refusal():
            reason = f"refuses to load in this runtime: {runtime_refusal}"
        else:
            permitted.append(tool)
            continue
        refused[tool.name] = reason
        skipped.append((tool.name, reason))
        notices.append(f"Tool {tool.name!r} {reason}; not loaded.")
        _log.warning("Tool %r %s; not loaded.", tool.name, reason)
    if not refused:
        return resolved
    manifest = [entry for entry in resolved.manifest if entry[0] not in refused]
    return ResolvedTools(
        tools=permitted,
        builtins=resolved.builtins,
        skipped=skipped,
        manifest=manifest,
        notices=notices,
        broken=resolved.broken,
        opt_in_stems=resolved.opt_in_stems,
        mcp_images=resolved.mcp_images,
    )


# The env var that selects the deploy profile (issue #256). Delivered per-agent through
# ``agent.env`` — the channel every per-agent knob uses — so the router carries no profile
# logic; the NOC sets it (via `provision-config`) only after its unprivileged-account preflight
# passes. ``locked`` | ``unlocked``, extensible to future profiles.
HARNESS_PROFILE_ENV = "HARNESS_PROFILE"


def _profile_from_env() -> tuple[str, Policy]:
    """Resolve the active Harness profile from ``HARNESS_PROFILE`` — fail-closed to locked.

    The single deploy lever for the unlocked profile (issue #256): ``HARNESS_PROFILE=unlocked``
    (trimmed, case-insensitive) selects `Policy.unlocked()`; **anything else — unset, empty, or
    unrecognized — selects `Policy.locked()`**, so the shipped default is unchanged and a typo
    can never silently unlock a box. Returns ``(name, policy)`` — the name so ``--resolved-config``
    can report the active profile, the policy so the *same* decision drives both the registry
    (`Harness(policy=…)`) and the env-resolution filter (`_apply_safe_policy`); reading this one
    pure function of the env at each site is what keeps the two in lockstep.

    Selecting ``unlocked`` is *only* the deploy lever, and it never weakens the safety enforced
    around it: the NOC sets the var only after its unprivileged-account preflight passes
    (constitution Operational Baselines), the shell tool's own in-process root-refusal backstop
    still fires (issue #253), and a powerful tool like ``shell`` is still opt-in per persona.
    """
    raw = (os.environ.get(HARNESS_PROFILE_ENV) or "").strip().lower()
    if raw == "unlocked":
        return "unlocked", Policy.unlocked()
    return "locked", Policy.locked()


def _onboard_from_env() -> bool:
    """Read ``HARNESS_ONBOARD`` into the `onboard` flag — on unless explicitly off.

    Onboarding is the default (a peer waking on its Dashboard is the point); set
    ``HARNESS_ONBOARD`` to an explicit off token (``0``/``false``/``no``/``off``)
    to wake with only the operator's charter. Unset — or any other value, blank
    included — leaves it on: it is off only when explicitly turned off.
    """
    raw = os.environ.get("HARNESS_ONBOARD")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _context_messages_from_env() -> int | None:
    """Read ``HARNESS_CONTEXT_MESSAGES`` into a `context_messages` value.

    Unset → the default cap. The case-insensitive sentinel ``all`` → `None`
    (seed the whole backlog). Anything else is parsed as a non-negative int.
    """
    raw = os.environ.get("HARNESS_CONTEXT_MESSAGES")
    if raw is None:
        return DEFAULT_CONTEXT_MESSAGES
    if raw.strip().lower() == "all":
        return None
    return int(raw)


def _max_steps_from_env() -> int:
    """Read ``HARNESS_MAX_STEPS`` into the engine's per-turn step budget.

    Unset or blank → `DEFAULT_MAX_STEPS` (the shipped 24 — a deliberate research-lab
    over-provision, see `basecradle_harness._engine`). Set → the operator's per-persona
    override, parsed as an int; a positive value is the budget, and a non-positive value
    fails loudly here rather than producing an engine that can never make a call. This is
    the single-integer operational knob (like ``HARNESS_WAKE_BREAKER_MAX``), not a
    ``model_params.json`` tuning key — the model id is env, and so is its step budget.
    """
    raw = os.environ.get("HARNESS_MAX_STEPS")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_STEPS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"HARNESS_MAX_STEPS must be a positive integer, got {value}.")
    return value


def _response_retries_from_env() -> int:
    """Read ``HARNESS_RESPONSE_RETRIES`` into the engine's unparseable-response retry bound.

    Unset or blank → `DEFAULT_RESPONSE_RETRIES` (the shipped 2 — up to 3 total attempts on the
    truncated / EOF-mid-JSON class, issue #259). Set → the operator's per-persona override, parsed
    as an int; **zero is valid** (disable the retry — a single attempt), so the floor is 0, not 1,
    and a negative value fails loudly here rather than silently meaning "no attempts". This is a
    single-integer operational knob (like ``HARNESS_MAX_STEPS``), not a ``model_params.json`` key.
    """
    raw = os.environ.get("HARNESS_RESPONSE_RETRIES")
    if raw is None or not raw.strip():
        return DEFAULT_RESPONSE_RETRIES
    value = int(raw)
    if value < 0:
        raise ValueError(f"HARNESS_RESPONSE_RETRIES must be a non-negative integer, got {value}.")
    return value


def _max_context_tokens_from_env() -> int | None:
    """Read ``HARNESS_MAX_CONTEXT_TOKENS`` into the context budget's operator override (issue #276).

    Unset or blank → ``None``: the budget resolves the ceiling itself (the adapter's `context_limit`
    capability, else the conservative floor). Set → the operator's number, which **always wins** —
    it is the 2 a.m. escape hatch, the only correct answer for a model whose window is below the
    floor or whose routing an operator has pinned, and the knob for anyone who wants a *tighter*
    budget than the ceiling for cost reasons. **Zero is valid** and disables compaction outright —
    the pre-#276 behavior, and that includes the over-length self-heal: "off" means off, and an
    escape hatch that rewrites the operator's transcript anyway, at exactly the moment they would
    least expect it, is not an escape hatch. So the floor is 0; a negative value fails loudly rather
    than quietly meaning something no one intended.
    """
    raw = os.environ.get("HARNESS_MAX_CONTEXT_TOKENS")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(
            f"HARNESS_MAX_CONTEXT_TOKENS must be a non-negative integer (0 disables "
            f"compaction), got {value}."
        )
    return value


def _compactor_from_env(provider: Provider) -> Compactor:
    """The agent's context budget + compactor, wired to the model it will run against.

    Built for every deployed agent — this is the invariant, not a per-persona feature: *nothing
    replayed per wake may be unbounded*. It costs a quiet agent nothing (no extra API call is ever
    made until a call's reported usage is large enough that some ceiling could be in play), and it
    is what keeps a standing agent from walking into its context wall.

    The **step budget** is read here too, and handed to the budget for one purpose: the 50%
    compaction threshold is only safe while one turn's worst-case growth fits in the headroom above
    it, and that growth scales with the steps a turn may take (issue #287). Both terms of that
    inequality are operator-tunable — `HARNESS_MAX_CONTEXT_TOKENS` shrinks the headroom,
    `HARNESS_MAX_STEPS` grows the turn — so the budget is given the *effective* step count rather
    than the shipped default, and warns when the pair of them no longer clears the bar.
    """
    return Compactor(
        provider,
        ContextBudget(
            provider,
            override=_max_context_tokens_from_env(),
            max_steps=_max_steps_from_env(),
        ),
    )


def _log_level_from_env() -> int:
    """Read ``HARNESS_LOG_LEVEL`` into a logging level; unset/blank/garbage → ``INFO``.

    Accepts a level name (case-insensitive — ``DEBUG``/``info``/``WARNING``) or a numeric
    level (``10``). Anything unrecognized degrades to ``INFO`` rather than raising: a
    misconfigured verbosity knob must never take down a wake. ``INFO`` is the deliberate
    default — the per-step ledger (issue #248) exists to be seen.
    """
    raw = os.environ.get("HARNESS_LOG_LEVEL")
    if raw is None or not raw.strip():
        return logging.INFO
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    level = logging.getLevelName(raw.upper())
    return level if isinstance(level, int) else logging.INFO


def _configure_logging() -> None:
    """Install a stderr log handler at ``HARNESS_LOG_LEVEL`` (default ``INFO``) for a CLI entrypoint.

    A console-script process that never configures logging runs on Python's last-resort
    handler (``WARNING`` and above only), so every ``INFO`` breadcrumb — the per-step ledger,
    ``wake used X/N steps``, the reconcile/tool notes — is dropped before it reaches stderr
    (issue #248: the ledger shipped in #244 was invisible in production for exactly this
    reason). Mirror ``_cleanup.py``: configure a stderr handler at ``INFO``, but **only when
    nothing else has** (``root.handlers`` empty), so an embedding application's own logging
    setup always wins and a library import never hijacks logging. ``HARNESS_LOG_LEVEL`` tunes
    the level for a noisier or quieter run; the ``INFO`` default is the point.

    ``httpx`` is demoted to ``WARNING`` on the way (`_quiet_transport_logs`) — at ``INFO`` it
    narrates every HTTP call the platform SDK and the model SDKs make, which was the single
    loudest thing in the journal and said nothing the harness's own lines don't say better.
    """
    if logging.getLogger().handlers:
        return
    level = _log_level_from_env()
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    _quiet_transport_logs(level)


def _quiet_transport_logs(level: int) -> None:
    """Demote ``httpx``'s per-request chatter to ``WARNING`` — the harness's own lines replace it.

    ``INFO HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 200 OK"`` fired once
    per platform read, model call, and asset fetch: the dominant noise in Live Tail, and pure
    duplication now that the wake logs its own LLM, tool, and posted-message lines with the
    context (provider, model, tokens, duration) the transport line never had.

    A run explicitly turned up to ``DEBUG`` is the one exception — an operator debugging a wake
    at ``DEBUG`` is asking for the wire, so ``httpx`` is left alone there. Only *this* CLI's own
    logging setup does the demotion: it is skipped entirely when an embedding application already
    configured logging (see `_configure_logging`), so importing the harness as a library never
    silences a host application's transport logs.
    """
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
