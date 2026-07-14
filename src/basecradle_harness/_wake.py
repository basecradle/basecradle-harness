"""One-shot, per-event wake: the entrypoint a router invokes once per platform event.

Poll mode (`TimelineAgent.run`) is a long-lived loop that holds its high-water
mark in memory. Under a router ([basecradle-router](https://github.com/basecradle/basecradle-router))
the model inverts: the router runs a *command* once per platform event, the
process answers the timeline's unseen messages, and exits. Two things the poll
loop never needed become load-bearing:

1. **The high-water mark must persist across processes.** Each wake is a fresh
   process, so the mark cannot live in memory — a router retry or two events
   arriving close together would re-answer the same message. It is stored under
   the agent's `home`, one small file per timeline (`MarkStore`), and advanced
   after every reply so a crash mid-batch resumes without duplicating.
2. **The conversation must persist across processes too** — otherwise the agent
   would re-seed the whole backlog on every wake. That is exactly what a persisted
   `Session` already gives us: with `Harness(home=...)` and a `timeline:<uuid>`
   source, each wake reloads the prior transcript instead of starting blank.

A wake reconciles **every** kind of unseen actionable item on the timeline, not just
messages. Three cases the message scan would otherwise miss:

- A peer's posted **asset**: a file (image, doc, audio) shared on the timeline is an
  item like a message, so it rides the same high-water mark — but the message scan
  reads only messages, so the wake also scans assets and *perceives* a peer's file. An
  image is fetched and shown to the model inline, so a vision-capable agent actually
  sees the picture on wake; media it cannot yet fully perceive (a doc, audio, video)
  degrades to a description naming the file and its type, with the `view`/`read`/`listen`
  tools available to engage further on demand.
- An inbound **webhook delivery**: a received `webhook_event` is not a timeline item,
  so the wake fetches unseen ones through the SDK's webhook-events read surface, under
  their own high-water mark, and acts on them.
- A newly-**activated task**: a `task.activated` wake fires when a scheduled task comes
  due, but the activation is not a fresh timeline item the scan surfaces — so the wake
  lists the timeline's *activated* tasks and carries out the instructions of any it has
  not handled yet. Activated tasks are tracked by a persisted seen-set rather than a
  high-water mark, because activation order does not track creation order (a task
  scheduled earlier can come due later) and a task has no terminal status to mark done.

So a peer woken by `webhook_event.received`, `task.activated`, or an asset post
perceives and acts on the trigger, with the same idempotency across processes the
message path has. **The actor self-filter is the safety property running through all of
it:** messages and assets are skipped when the agent authored them, so it never reacts
to — or wake-loops on — its own posts, most importantly an image it just generated.

**A wake posts nothing of its own** (the Unspoken Channel, issue #293). It engages the
model on what it found; whether anything reaches the timeline is the model's decision,
taken by calling the `messages` tool. The turn's final text is *unspoken* — written to
the agent's log (`log_unspoken`), handed to its memory, shown to its own next turn, and
never posted.
The only message this file still creates is the NOC probe ack, which is a machine
contract (a signed nonce, model-free) rather than the agent talking. So a wake in which
the agent chose silence posts nothing and says `posted=0` — and that is a *decision on
the record*, not a failure: the reasoning behind it is in the journal.

Everything else — identity, Dashboard onboarding, turning messages into turns,
the newest-first scan up to the mark — is shared with `TimelineAgent`.

The command is `basecradle-harness-wake --timeline <uuid>` (also runnable as
`python -m basecradle_harness`); see `main`. **In production it works from the timeline
uuid alone** — the router wakes a harness agent with `--timeline <uuid>` and nothing
else (see basecradle-router `wake_command`), never naming the item that fired the wake.
So the reconcile cannot lean on a trigger: each kind surfaces its own newest unseen item
on a first wake and everything past its mark thereafter, with no router help. The optional
`--message`, `--event`, and `--asset` uuids still name a triggering item when a manual or
future-router invocation passes one — each sharpens its own first-wake bootstrap (see
`_bootstrap_split` and `_bootstrap_stream`) — but they are not required for a delivery,
asset, or task to be acted on.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from basecradle import BaseCradle, BaseCradleError

from basecradle_harness._assets import _describe, _is_image, image_input
from basecradle_harness._basecradle import (
    DEFAULT_CONTEXT_MESSAGES,
    _as_turn,
    _client_from_env,
    _compactor_from_env,
    _config_from_env,
    _configure_logging,
    _context_messages_from_env,
    _incoming_text,
    _max_context_tokens_from_env,
    _max_steps_from_env,
    _messages_since,
    _onboard_from_env,
    _parse_created_at,
    _profile_from_env,
    _recent,
    _resolve_tools,
    _resolve_tools_and_provider,
    _response_retries_from_env,
    resolved_model_params,
)
from basecradle_harness._brief import (
    compose_brief,
    fetch_dashboard_md,
    render_budget,
    render_defects,
    render_manifest,
    render_safety,
)
from basecradle_harness._code import CodeExecutionBridge
from basecradle_harness._engine import compose_hooks
from basecradle_harness._exceptions import (
    EngineError,
    HarnessError,
    ProviderContextLengthError,
    ProviderError,
)
from basecradle_harness._harness import Harness
from basecradle_harness._idempotency import IdempotencyKeys, interrupted
from basecradle_harness._install import charter_from_env, prompt_text, system_prompt_text
from basecradle_harness._mcp import load_mcp_configs
from basecradle_harness._memory_provider import (
    MemoryExchange,
    MemoryProvider,
    MemoryScope,
    describe_memory_provider,
)
from basecradle_harness._messages import ImageContent, Message
from basecradle_harness._observability import delivery_id, describe_provider, kv, log_unspoken
from basecradle_harness._platform import PlatformContext, bind_platform_tools, explain
from basecradle_harness._probe import ack_line, verify_probe
from basecradle_harness._session import INTERRUPTED, Session, turn_work
from basecradle_harness._unspoken import MentionInformer, SpeechLedger
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

# The kinds of timeline item the wake reconciles. Messages and webhook events are
# creation-ordered streams tracked by a high-water mark (below); activated tasks are
# not (a task scheduled earlier can activate later, and a task has no "done" status),
# so they are tracked by a persisted seen-set instead. `messages` keeps the original
# on-disk mark location so a deployed agent's existing marks still resolve.
_MESSAGES = "messages"
_EVENTS = "webhook_events"
_TASKS = "tasks"
_ASSETS = "assets"


class MarkStore:
    """Per-timeline, per-kind high-water marks, persisted under the agent's home.

    One file per (kind, timeline) — `<root>/marks/<timeline>.txt` for messages (the
    original location, kept for backward compatibility) and
    `<root>/marks/<kind>/<timeline>.txt` for any other kind, the uuid percent-encoded
    into a safe filename — holding the uuid of the newest item of that kind the agent
    has handled. This is what makes wake mode idempotent across separate processes:
    the next wake reads the mark and skips everything at or before it. Messages and
    webhook events advance their marks independently, so reconciling one never
    re-surfaces the other.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, timeline: str, *, kind: str = _MESSAGES) -> str | None:
        path = self._path(timeline, kind)
        if not path.exists():
            return None
        return path.read_text().strip() or None

    def set(self, timeline: str, uuid: str, *, kind: str = _MESSAGES) -> None:
        path = self._path(timeline, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(uuid)

    def _path(self, timeline: str, kind: str = _MESSAGES) -> Path:
        # Messages live directly under `marks/` (the original layout); every other
        # kind gets its own subdirectory, so the namespaces never collide.
        base = self.root / "marks"
        folder = base if kind == _MESSAGES else base / kind
        return folder / f"{quote(timeline, safe='')}.txt"


class SeenStore:
    """Per-timeline sets of handled item uuids, persisted under the agent's home.

    A high-water mark works for a creation-ordered stream (messages, webhook events):
    everything at or before the mark is seen. Activated **tasks** are not such a
    stream — a task scheduled earlier can activate later, so activation order does not
    track creation order, and a task carries no terminal "done" status the agent could
    set. So idempotency here is a *set*: the uuids already acted on, persisted one per
    line under `<root>/seen/<kind>/<timeline>.txt`. Appended to (not rewritten) after
    each item is handled, so a crash mid-batch resumes without re-acting on the rest.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def all(self, timeline: str, *, kind: str) -> set[str]:
        path = self._path(timeline, kind)
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def add(self, timeline: str, uuid: str, *, kind: str) -> None:
        path = self._path(timeline, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(uuid + "\n")

    def _path(self, timeline: str, kind: str) -> Path:
        return self.root / "seen" / kind / f"{quote(timeline, safe='')}.txt"


#: What this wake decided about one item it read — the input to the two things `_settle` does.
#:
#: - `_FINAL`  — its fate is already decided (settled claim, own post, acked probe). Mark may pass.
#: - `_OURS`   — this wake is answering it; `_settle` commits it, and *then* the mark may pass.
#: - `_PENDING`— **not** settled and **not** ours, so the mark may **never** pass it. Two things land
#:   here: an item a *live* concurrent wake is holding (it could still die holding it, and a mark
#:   past it would hide it forever), and a probe ack that failed to post (at-least-once: it must be
#:   re-acked on the next wake, so the cursor must not sail past it either). Both mean the same
#:   thing to the mark — *undecided* — which is the only thing the mark cares about.
_FINAL = "final"
_OURS = "ours"
_PENDING = "pending"

#: The three phases of a claim. `in-flight` is the *only* one that is not a final disposition:
#: it says a wake took the item and has not yet finished with it. The other two are terminal —
#: the item will never be acted on again, and the high-water mark may safely pass it.
_IN_FLIGHT = "in-flight"
_DONE = "done"
_ABANDONED = "abandoned"

#: How long an `in-flight` claim whose recorded pid *still resolves to a live process* may sit
#: before it is treated as orphaned anyway. It exists for exactly one case: the box rebooted (or
#: the process died long ago) and the kernel handed that pid to an unrelated process, so the
#: liveness check answers "alive" about a wake that is long gone. Six hours is orders of magnitude
#: beyond any real wake — a wake is bounded by `max_steps` model calls, each bounded by the SDK's
#: own timeout — so this can never fire on a wake that is genuinely still working, which is the
#: one thing it must never do (that would re-drive an item a live wake is mid-way through
#: answering, and post twice).
_CLAIM_STALE_AFTER = 6 * 60 * 60


@dataclass(frozen=True)
class Claim:
    """What a wake wrote down about an item it took: which phase it reached, and who took it.

    The owner is identified by **both** the process (`pid`) and the *wake* (`wake`, unique per
    `WakeAgent.wake()` call). The pid alone is not enough: a process can run several wakes in
    sequence (the poll loop does), and a claim left `in-flight` by an *earlier* wake of the
    **same** process is unambiguously orphaned — that wake is over, whatever happened to it —
    even though its pid is very much alive. Keying liveness on the pid alone would read that as
    "the owner is still running" and never recover it.
    """

    phase: str
    pid: int | None = None
    wake: str | None = None
    at: float | None = None
    reason: str | None = None

    @property
    def settled(self) -> bool:
        """Is this item's disposition **final** — never to be acted on again?

        The load-bearing question for the high-water mark: the mark is a *cursor*, so it may only
        pass an item whose fate is decided. Passing an `in-flight` item would hide it forever if
        the wake holding it then died — which is the exact bug this whole mechanism exists to fix.
        """
        return self.phase in (_DONE, _ABANDONED)


def _process_alive(pid: int | None) -> bool:
    """Is `pid` a live process on this box? Signal 0 tests existence without delivering anything.

    A `PermissionError` means the process exists but belongs to another user — **alive**. Reading
    it as dead would orphan a claim a live wake still owns. Claims are box-local, so the pid
    namespace is local too (if agents ever shared a claim dir across boxes, this assumption breaks
    — nothing does today).
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _orphaned(claim: Claim, *, pid: int, wake: str, now: float | None = None) -> bool:
    """Did the wake holding this `in-flight` claim die without finishing?

    **The safe direction is "no".** A false *no* costs a delay — the item is re-driven on some
    later wake. A false *yes* re-drives an item a live wake is still answering, and posts it twice.
    So the tests run from most certain to least:

    1. **This very wake owns it** → obviously not orphaned.
    2. **An earlier wake of *this* process owns it** → orphaned, certainly. A process runs its
       wakes one at a time (`wake()` is synchronous), so a claim from a different wake of the same
       pid belongs to one that is already over. Judging this by pid alone would call it "alive"
       and never recover it — which is exactly what a single-process poll loop would hit.
    3. **Another process owns it** → ask the OS whether that process still exists.
    4. **It says alive, but the claim is ancient** → the pid was reused (the box rebooted and the
       kernel handed the number to something unrelated). `_CLAIM_STALE_AFTER` catches it.

    Note pid reuse can only ever make step 3 answer "alive" about a *dead* wake, so it delays a
    recovery rather than causing a duplicate — the safe direction, with step 4 as the backstop.
    """
    if claim.phase != _IN_FLIGHT:
        return False
    if claim.wake == wake:
        return False
    if claim.pid == pid:
        return True  # an earlier wake of this same process: it is over, whatever became of it
    if not _process_alive(claim.pid):
        return True
    age = (now if now is not None else time.time()) - (claim.at or 0.0)
    return age > _CLAIM_STALE_AFTER


#: How far `ClaimStore.effective_owner` will follow a chain of take-overs before giving up. A
#: recovery that dies can itself be recovered, so the chain is real — but it is one hop per crashed
#: wake, so any depth beyond a handful means the records are corrupt, and spinning on them is worse
#: than the stalled item they describe.
_TAKEOVER_HOPS = 16


def _read_claim(path: Path) -> Claim | None:
    """Read a claim record from `path` — the one parser, shared by the claim and its take-over token.

    Two shapes are tolerated rather than trusted, and **both degrade to `done`** — the conservative
    reading, because `done` only ever costs a *skip*, while a wrong `in-flight` could re-drive an
    item that was already answered.
    """
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not raw:
        return Claim(phase=_DONE)  # legacy: the pre-#285 code wrote an empty claim
    try:
        data = json.loads(raw)
        return Claim(
            phase=str(data["phase"]),
            pid=data.get("pid"),
            wake=data.get("wake"),
            at=data.get("at"),
            reason=data.get("reason"),
        )
    except (ValueError, KeyError, TypeError):
        _log.warning(
            "Unreadable claim record at %s; treating it as done rather than re-driving an item "
            "we cannot reason about.",
            path,
        )
        return Claim(phase=_DONE)


def _payload(claim: Claim) -> dict[str, object]:
    """A claim as the JSON it persists as. One shape, so a `claim()` and a `commit()` cannot drift."""
    payload: dict[str, object] = {
        "phase": claim.phase,
        "pid": claim.pid,
        "wake": claim.wake,
        "at": claim.at,
    }
    if claim.reason is not None:
        payload["reason"] = claim.reason
    return payload


class ClaimStore:
    """Per-item atomic claims, so concurrent wakes handle each item exactly once — and so a
    **crashed** wake's item can be re-driven instead of silently dropped (issue #285).

    A high-water mark (`MarkStore`) and a seen-set (`SeenStore`) make a wake idempotent
    across *sequential* processes — a later wake reads the record and skips. Neither is
    safe across *concurrent* ones: two wakes firing at once (an upload posts
    `asset.created` and `message.created` together, so the router spawns two) both read the
    same record, both find the same message unseen, and both reply — the live duplicate, or
    worse a duplicated tool action. A claim closes that race with the one operation a POSIX
    filesystem makes atomic: an exclusive create. The first wake to create
    `<root>/claims/<kind>/<timeline>/<uuid>` wins and acts; a concurrent create raises
    `FileExistsError`, so the loser knows the item is already owned and skips it.

    **The claim is two-phase, and that is what makes a dead wake recoverable.** The exclusive
    create still decides the *race* — unchanged, and it is still the whole synchronization. What
    is new is that the file has *contents*: the phase the owning wake reached (`in-flight` →
    `done` / `abandoned`), and the pid that took it. A wake that dies mid-turn leaves an
    `in-flight` claim whose owner is gone, and a later wake can *see* that and decide, from
    evidence, whether re-driving the item is safe (`WakeAgent._recover_orphans`).

    Before this, claiming happened *before* acting and the item was marked seen at the same
    instant — so any hard failure between the claim and the reply (the provider is down, the
    process is killed) meant the peer's message was marked seen, never answered, and **never
    looked at again by any future wake**. It was gone, silently. That was a deliberate
    at-most-once trade (a dropped reply beats a backlog re-answered with side effects on every
    later wake), and the trade is now unnecessary: the phase says which of the two happened.

    **Legacy claims read as `done`.** A claim file written by the pre-#285 code is *empty*. The
    old code only ever created a claim for an item it was about to mark seen, so an empty claim
    means "the old code took this item" — final, by construction. Reading it as `done` reproduces
    the old behavior exactly and can never re-drive history on the upgrade wake. This is
    load-bearing for the fleet rollout: @jt and @glm-5.2 have live claim dirs.

    **Known bound — claims are not pruned.** One tiny file accrues per handled item, the same
    unbounded-growth shape the task `SeenStore` already has. If it ever matters, claims at or
    below a kind's high-water mark are dead (that item is never re-scanned) and prunable by
    UUIDv7 order. Out of scope here.
    """

    def __init__(self, root: str | Path, *, wake: str | None = None) -> None:
        self.root = Path(root)
        #: This wake's identity, stamped into every claim it takes (see `Claim`). Defaults to a
        #: fresh one so a store built ad hoc (a test, a script) still writes a well-formed record.
        self.wake = wake or uuid4().hex

    def claim(self, timeline: str, uuid: str, *, kind: str) -> bool:
        """Atomically claim `(kind, timeline, uuid)` as **in-flight**. True if this wake won it.

        Wins by atomically **linking the claim file into place already carrying its record**. The
        link is the whole synchronization — it fails `EEXIST` if anyone already holds the claim —
        and because the file is written *before* it is published, the claim path is never
        observable in an empty state.

        That last property is load-bearing, not tidiness. The obvious form — `O_CREAT|O_EXCL` to
        win the race, *then* write the record — leaves a window in which the path exists and the
        file is zero bytes. An empty claim is this store's **legacy sentinel** and reads as `done`
        (see `read`), so a concurrent wake landing in that window would read a live, `in-flight`
        claim as **settled**, let the high-water mark pass it, and — if its owner then died — lose
        the peer's message forever. The bug this file exists to fix, reintroduced by the mechanism
        meant to fix it.
        """
        path = self._path(timeline, kind, uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = Claim(phase=_IN_FLIGHT, pid=os.getpid(), wake=self.wake, at=time.time())
        temp = path.parent / f".{quote(uuid, safe='')}.{self.wake}.new"
        temp.write_text(json.dumps(_payload(record)))
        try:
            os.link(temp, path)  # atomic create-if-absent, published already populated
        except FileExistsError:
            return False
        finally:
            temp.unlink(missing_ok=True)
        return True

    def orphaned(self, claim: Claim) -> bool:
        """Is `claim` held by a wake that died? (`_orphaned`, bound to this store's identity.)"""
        return _orphaned(claim, pid=os.getpid(), wake=self.wake)

    def reclaim(self, timeline: str, uuid: str, *, kind: str, owner: str | None) -> bool:
        """Take over the orphaned claim **held by `owner`**. True if this wake won the take-over.

        A take-over is a **compare-and-swap**, not an overwrite: it must succeed only if the claim
        is still the very one the caller judged orphaned. The swap is decided by an exclusive
        create on a token path that names the dead owner — `.<uuid>.takeover.<owner>` — so exactly
        one wake can ever take this message *from this owner*, and the winner then writes the claim.

        Two simpler forms are **both broken**, and it is worth naming them, because each looks
        obviously fine:

        - **`unlink()` then exclusive-`claim()`.** Safe only if *both* unlinks happen before
          *either* create. Interleaved they do not: A unlinks, A creates (wins, writing a live
          claim) — then B unlinks, **destroying A's brand-new live claim**, and B creates and
          "wins" too. Both drive the message through the model and post it twice.
        - **`rename()` as a mutex.** Fails the same way for the same reason: after A's take-over a
          file *is* present at the path — A's fresh, live claim — so B's rename happily moves *that*
          aside and B wins as well. A rename can tell "something is there" from "nothing is there";
          it cannot tell the stale orphan from the live claim that replaced it. That distinction is
          the entire job, which is why the token names the owner it is taking over *from*.

        **The token carries its record, exactly as `claim()`'s does, and for a sharper reason**
        (issue #297). Winning the token and *then* writing the claim is two steps, and a wake that
        dies between them — or simply hits `ENOSPC`, which needs no race at all — leaves the token
        **taken** and the claim still naming the **dead** owner. Every future wake then judges that
        stale owner, tries to reclaim from it, loses the token it can never win, and returns
        `_PENDING`; `_settle` stops at the first `_PENDING`, so **the high-water mark is pinned
        forever and the peer's message is never answered by anyone**. A stall is not better than a
        drop — it *is* a drop, with the mark stuck behind it.

        So the token is published already populated (link-from-temp, never observable empty), and it
        *is* the record of who took over. `effective_owner` reads it, so a take-over that died
        before it could write the claim is itself recoverable: the next wake sees that the real
        owner is the wake named in the token, judges *that* wake, and competes on a fresh token path
        keyed to it. That is what makes recovery **repeatable** — a single non-generational token
        would let a message be recovered exactly once, ever, and strand it afterwards.
        """
        path = self._path(timeline, kind, uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = Claim(phase=_IN_FLIGHT, pid=os.getpid(), wake=self.wake, at=time.time())
        token = self._token(path, uuid, owner)
        temp = path.parent / f".{quote(uuid, safe='')}.{self.wake}.takeover.new"
        temp.write_text(json.dumps(_payload(record)))
        try:
            os.link(temp, token)  # atomic create-if-absent, published already carrying the record
        except FileExistsError:
            return False  # another wake already took this orphan over — it owns it now
        finally:
            temp.unlink(missing_ok=True)
        self._write(path, record)
        return True

    def effective_owner(self, timeline: str, uuid: str, *, kind: str, claim: Claim) -> Claim:
        """`claim`, unless a take-over won it and died before it could say so (issue #297).

        `reclaim` wins a token and then writes the claim. Between those two the claim file still
        names the *previous* owner — so a wake reading it would judge a wake that no longer holds
        the item, and would then lose a token that is already spoken for, forever. Following the
        token chain resolves the owner the claim file has not caught up with yet.

        The chain is a chain because a recovery can itself be recovered: A takes over from D, dies;
        B takes over from A, dies; C takes over from B. Each hop is a distinct wake id, so it
        terminates — but it is bounded anyway, because the one thing worse than a stalled item is a
        wake spinning on a corrupt one.
        """
        seen: set[str] = set()
        while claim.wake is not None and claim.wake not in seen and len(seen) < _TAKEOVER_HOPS:
            seen.add(claim.wake)
            token = self._token(self._path(timeline, kind, uuid), uuid, claim.wake)
            successor = _read_claim(token)
            if successor is None or successor.wake is None:
                break  # nobody took this owner over — the claim file is the truth
            claim = successor
        return claim

    def _token(self, path: Path, uuid: str, owner: str | None) -> Path:
        """The take-over token for `owner`'s claim: exclusive-create decides who takes it over."""
        return path.parent / f".{quote(uuid, safe='')}.takeover.{quote(str(owner), safe='')}"

    def read(self, timeline: str, uuid: str, *, kind: str) -> Claim | None:
        """The claim on `(kind, timeline, uuid)`, or ``None`` if the item was never claimed.

        Two shapes are tolerated rather than trusted, and **both degrade to `done`** — the
        conservative reading, because `done` only ever costs a *skip*, while a wrong `in-flight`
        could re-drive an item that was already answered:

        - an **empty** file — a pre-#285 claim (see the class docstring);
        - **unparseable** contents — which the atomic write below should make impossible, so it
          means something we do not understand touched the file. Never re-drive on a record we
          cannot read.
        """
        return _read_claim(self._path(timeline, kind, uuid))

    def commit(self, timeline: str, uuid: str, *, kind: str) -> None:
        """Mark the item **done** — acted on, disposition final. The mark may now pass it."""
        self._write(self._path(timeline, kind, uuid), Claim(phase=_DONE, at=time.time()))

    def abandon(self, timeline: str, uuid: str, *, kind: str, reason: str) -> None:
        """Mark the item **abandoned** — it will never be acted on, and we know why.

        This is the residual at-most-once case, and the whole point is that it is now *loud and
        rare* rather than silent and routine: it is reached only when a dead wake left behind
        side effects that make re-driving unsafe. The caller logs an ERROR naming the item; this
        records the same fact durably, so the mark can pass the item without hiding the story.
        """
        self._write(
            self._path(timeline, kind, uuid),
            Claim(phase=_ABANDONED, at=time.time(), reason=reason),
        )

    def _write(self, path: Path, claim: Claim) -> None:
        """Overwrite an existing claim record **atomically** — a torn record cannot be read.

        Write-to-temp + `os.replace` (atomic on POSIX), so a crash mid-write can only ever leave
        the *previous* record intact, never a half-written one. Without this, the crash window this
        whole file is about would gain a failure mode of its own: a wake killed while committing,
        leaving a claim that parses as neither phase.

        This is for *transitions* (`commit` / `abandon`), where the path already exists. The first
        publication of a claim is `claim()`'s `os.link`, which must also win a race — a plain
        replace would happily clobber a rival's live claim.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(_payload(claim)))
        os.replace(temp, path)

    def _path(self, timeline: str, kind: str, uuid: str) -> Path:
        folder = self.root / "claims" / kind / quote(timeline, safe="")
        return folder / f"{quote(uuid, safe='')}.claim"


# The wake-breaker's generous safe defaults (Phase 2 · Group 6). A genuine cross-wake
# runaway fires continuously — many wakes per second — so over a 60 s window it racks up
# far more than this cap, while a human-paced multi-peer conversation almost never reaches
# 10 *inbound* items to one timeline in a minute (the agent's own replies are self-filtered
# and never wake it, so only peer items count). Tunable via env for the rare firehose
# timeline; the router's cross-agent breaker (basecradle-router) is the complementary layer.
_DEFAULT_BREAKER_MAX = 10
_DEFAULT_BREAKER_WINDOW = 60.0

#: The "user" side of the exchange a compaction summary is `observe`d as, on a memory provider with
#: no store of its own (MemPalace). Its miner reads dialogue, so the summary needs a prompt to be the
#: answer *to* — and this states plainly what it is, so a later search hit reads as the agent's own
#: notes-to-self rather than as something a peer said (issue #276).
_COMPACTION_OBSERVE_NOTE = (
    "[Context compaction] Summarize the work and conversation about to be dropped from my "
    "transcript, so it survives in memory."
)


@dataclass(frozen=True)
class BreakerDecision:
    """The wake-breaker's verdict for one wake: what it decided, and why.

    `short_circuit` is the load-bearing field — when True this wake **self-declines**: it
    makes no provider call and acts on nothing. `tripped` and `reset` flag the one-time
    state *transitions* (a trip or an auto-reset happened on *this* wake), so the caller
    alerts exactly once per cycle rather than on every tripped wake. `count` is the number
    of wakes counted in the rolling window, for the alert and the log line.
    """

    short_circuit: bool
    tripped: bool
    reset: bool
    count: int


class WakeBreaker:
    """Per-timeline cross-wake circuit-breaker — the backstop for an *unknown* runaway loop.

    The runaway this defends against is a **cross-wake loop**: the agent is woken, it posts,
    the post fires a platform event, the router wakes it again → a tight cycle burning
    provider tokens and box resources. The in-wake `max_steps` cap, the actor self-filter,
    and the known B3/B8 fixes each stop a *specific* loop; this is the generic backstop for a
    *novel* one — most plausibly introduced by a custom `tools/` plugin (Group 2) or a
    drop-in MCP server (Group 5).

    It is a rolling-window rate limiter on **wakes per timeline**, persisted beside the
    `marks/`/`seen/`/`claims/` stores under the agent's home so it survives the
    process-per-wake model:

    - `breaker/<timeline>.wakes` — the timestamps of recent wakes, pruned to the window on
      every wake (so the file stays bounded even under a fast runaway).
    - `breaker/<timeline>.tripped` — the durable **trip marker**: present iff the timeline is
      currently tripped, holding the trip timestamp.

    On each wake `record_and_check` records the wake and returns a `BreakerDecision`:

    - Over the cap within the window → **TRIP**: write the marker, return `short_circuit`
      (the wake self-declines, **no provider call** — the whole point is to stop the burn)
      with `tripped=True` so the caller alerts once.
    - Already tripped → keep short-circuiting (every wake is still *counted*, so a runaway
      that keeps firing keeps the window saturated and stays tripped).
    - **Auto-reset** (the preferred reset, stated in CLAUDE.md): once the burst subsides —
      the window clears back under the cap *and* the cooldown has elapsed since the trip —
      clear the marker, restart the window from now, and return `reset=True` (normal
      operation resumes; the caller logs the recovery alert). A transient burst self-heals
      while the loud WARNING still leaves the operator a breadcrumb. Clearing the marker by
      hand is the equivalent operator reset.

    Disabled by setting the cap to 0 (or below) — an operator escape hatch; the default is a
    generous always-on sanity cap.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_wakes: int = _DEFAULT_BREAKER_MAX,
        window: float = _DEFAULT_BREAKER_WINDOW,
        cooldown: float | None = None,
        now=None,
    ) -> None:
        self.root = Path(root)
        self.max_wakes = max_wakes
        self.window = float(window)
        # The cooldown defaults to the window: hysteresis so a trip cannot reset until at
        # least one clear window has passed, which prevents flapping at the threshold.
        self.cooldown = float(cooldown) if cooldown is not None else float(window)
        # An injectable clock keeps the breaker deterministically testable; production uses
        # the wall clock (a synthetic burst in a test drives `now` directly).
        self._now = now or time.time

    @classmethod
    def from_env(cls, root: str | Path, *, now=None) -> WakeBreaker:
        """Build a breaker from `HARNESS_WAKE_BREAKER_MAX`/`_WINDOW`/`_COOLDOWN` (generous defaults)."""
        return cls(
            root,
            max_wakes=_breaker_max_from_env(),
            window=_breaker_window_from_env(),
            cooldown=_breaker_cooldown_from_env(),
            now=now,
        )

    @property
    def enabled(self) -> bool:
        """Off when the cap is 0 or below — the operator escape hatch."""
        return self.max_wakes > 0

    def tripped(self, timeline: str) -> bool:
        """Whether this timeline currently holds a durable trip marker."""
        return self._read_trip(timeline) is not None

    def record_and_check(self, timeline: str) -> BreakerDecision:
        """Record this wake for `timeline` and decide whether it must self-decline.

        Always appends the wake to the rolling window first (a tripped wake is still counted,
        so a continuing runaway keeps the window saturated and stays tripped), then evaluates
        trip/reset state. See the class docstring for the state machine.
        """
        if not self.enabled:
            return BreakerDecision(short_circuit=False, tripped=False, reset=False, count=0)
        now = self._now()
        recent = [t for t in self._read_window(timeline) if t > now - self.window]
        recent.append(now)
        self._write_window(timeline, recent)
        count = len(recent)

        trip_at = self._read_trip(timeline)
        if trip_at is not None:
            # Currently tripped. Auto-reset only once the burst has genuinely subsided — the
            # window cleared back under the cap — *and* the cooldown has elapsed since the
            # trip, so a runaway still firing every few seconds keeps it tripped.
            if count <= self.max_wakes and now - trip_at >= self.cooldown:
                self._clear_trip(timeline)
                self._write_window(timeline, [now])  # fresh window: re-measure from here
                return BreakerDecision(short_circuit=False, tripped=False, reset=True, count=count)
            return BreakerDecision(short_circuit=True, tripped=False, reset=False, count=count)

        if count > self.max_wakes:
            self._write_trip(timeline, now)
            return BreakerDecision(short_circuit=True, tripped=True, reset=False, count=count)
        return BreakerDecision(short_circuit=False, tripped=False, reset=False, count=count)

    # --- storage -------------------------------------------------------------

    def _window_path(self, timeline: str) -> Path:
        return self.root / "breaker" / f"{quote(timeline, safe='')}.wakes"

    def _trip_path(self, timeline: str) -> Path:
        return self.root / "breaker" / f"{quote(timeline, safe='')}.tripped"

    def _read_window(self, timeline: str) -> list[float]:
        path = self._window_path(timeline)
        if not path.exists():
            return []
        out: list[float] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(line))
            except ValueError:
                continue  # a corrupt line is dropped, never crashes the wake
        return out

    def _write_window(self, timeline: str, times: list[float]) -> None:
        path = self._window_path(timeline)
        path.parent.mkdir(parents=True, exist_ok=True)
        # `repr` round-trips a float exactly, so a re-read window is byte-faithful.
        path.write_text("".join(f"{t!r}\n" for t in times))

    def _read_trip(self, timeline: str) -> float | None:
        path = self._trip_path(timeline)
        if not path.exists():
            return None
        try:
            return float(path.read_text().strip())
        except ValueError:
            return None

    def _write_trip(self, timeline: str, when: float) -> None:
        path = self._trip_path(timeline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repr(when))

    def _clear_trip(self, timeline: str) -> None:
        self._trip_path(timeline).unlink(missing_ok=True)


def _breaker_max_from_env() -> int:
    """`HARNESS_WAKE_BREAKER_MAX` → the wake cap; unset/blank → the generous default."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_MAX")
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_MAX
    return int(raw)


def _breaker_window_from_env() -> float:
    """`HARNESS_WAKE_BREAKER_WINDOW` → the rolling-window seconds; unset/blank → the default."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_WINDOW")
    if raw is None or not raw.strip():
        return _DEFAULT_BREAKER_WINDOW
    return float(raw)


def _breaker_cooldown_from_env() -> float | None:
    """`HARNESS_WAKE_BREAKER_COOLDOWN` → the reset cooldown seconds; unset/blank → the window."""
    raw = os.environ.get("HARNESS_WAKE_BREAKER_COOLDOWN")
    if raw is None or not raw.strip():
        return None  # default: tie the cooldown to the window length
    return float(raw)


# Read-speed pacing (issue #224, reworked in #226; tracks basecradle#334). Simulate a human
# reading a peer AI's message before replying, so an AI↔AI exchange is watchable and stays
# well under the wake-breaker's trip line instead of slamming into it. ~1,020 chars/min ≈ 17
# chars/s is an unhurried silent-reading pace; the 20 s floor keeps even a one-word "ok" from
# replying in a blink. All env-tunable; the defaults are the real production values.
#
# The #226 rework tuned these *slower* (chars/s 20→17, floor 15→20) after a live Pinky × The
# Brain run read too fast, and added `MAX_BUILDS` — the Loop-2 mid-generation staleness cap:
# a reply is generated against a snapshot, and if messages land *during* generation the batch
# is rebuilt (at most `MAX_BUILDS` model calls; the Nth build posts unconditionally). See
# `WakeAgent._pace_and_settle` (Loop 1) and `WakeAgent._generate_settled` (Loop 2).
_DEFAULT_PACE_CHARS_PER_SEC = 17.0
_DEFAULT_PACE_FLOOR_SECONDS = 20.0
_DEFAULT_PACE_MAX_BUILDS = 3


class ReadPacer:
    """Receiver-side read-speed pacing for AI↔AI conversations — the pacing layer, not a backstop.

    The fleet's runaway guards (this repo's `WakeBreaker`, the router's `WakeRateBreaker`, the
    engine's `max_steps`) *trip and halt*; none of them **pace**. Two AIs in a timeline can
    cross-wake each other into a runaway (the 2026-06-18 Pinky × The Brain run: ~16 messages in
    ~16 s). This is the missing pacing layer: before a wake answers a **peer AI's** message it
    sleeps to *simulate a human reading that message*, which makes the exchange watchable and
    keeps it well under the breaker's trip line. It is entirely receiver-side and *derived* — no
    platform change, no per-timeline flag; the behavior falls out of data the wake already
    fetches (the newest message's author `kind`, its `body` length, and its `created_at`).

    **Human messages are unaffected** — the ``kind == "ai"`` gate is the whole opt-in, so a human
    peer gets an instant reply exactly as before. **Own messages never reach here** (the actor
    self-filter excludes them upstream). **A wake with no message to answer** (asset/task/webhook
    only) never calls `pace`.

    The delay for a peer-AI message is::

        target = max(FLOOR_SECONDS, len(body) / CHARS_PER_SEC)   # a human's read-time for it
        delay  = max(0.0, target - age)                         # only wait the *remainder*

    The ``- age`` subtraction is load-bearing, not optional: it makes the delay a true "time
    since the message appeared" simulation (the message kept aging while this process did other
    work), smooths what would otherwise be a lumpy cadence, and gives the "quicker across
    timelines" behavior — time spent handling *another* timeline counts against what is owed
    here. Without it, half the intended behavior is gone.

    Mirrors `WakeBreaker`'s injectable-seams shape: an injectable `clock` (default UTC now) and
    `sleep` (default `time.sleep`), so a test asserts the *computed* delay against a fake clock
    with a recording no-op sleep and never actually waits.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        chars_per_sec: float = _DEFAULT_PACE_CHARS_PER_SEC,
        floor_seconds: float = _DEFAULT_PACE_FLOOR_SECONDS,
        clock=None,
        sleep=None,
    ) -> None:
        self.enabled = enabled
        self.chars_per_sec = float(chars_per_sec)
        self.floor_seconds = float(floor_seconds)
        # Injectable seams (mirror `WakeBreaker.now`): production uses the wall clock and a real
        # sleep; a test drives `clock` directly and records `sleep` so it asserts without waiting.
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep

    @classmethod
    def from_env(cls, *, clock=None, sleep=None) -> ReadPacer:
        """Build a pacer from `HARNESS_PACE_ENABLED`/`_CHARS_PER_SEC`/`_FLOOR_SECONDS` (real defaults)."""
        return cls(
            enabled=_pace_enabled_from_env(),
            chars_per_sec=_pace_chars_per_sec_from_env(),
            floor_seconds=_pace_floor_seconds_from_env(),
            clock=clock,
            sleep=sleep,
        )

    def pace(self, message: object | None) -> float:
        """Simulate reading `message` (the newest peer message this wake will answer); return seconds slept.

        A no-op returning ``0.0`` — no sleep — when: pacing is disabled (the kill switch); there
        is no message (asset/task/webhook-only wake); or the author is **not** an AI peer (a
        human gets an instant reply). The caller passes the newest *non-self* message and has
        already excluded a recognized NOC probe (which must stay a sub-second token-free ack), so
        `pace` need only apply the ``kind == "ai"`` gate and the read-time math.

        Otherwise it sleeps the *remainder* of the message's human read-time (``target - age``,
        clamped at 0) and returns the seconds slept, so a message already older than its read-time
        adds no delay.
        """
        if not self.enabled or message is None:
            return 0.0
        if getattr(message.user, "kind", None) != "ai":
            return 0.0
        chars = len(message.content.body or "")
        # A non-positive rate (an operator setting the env to 0) can't divide; fall back to the
        # floor rather than raising, so a misconfigured rate degrades to "always the floor".
        read_time = chars / self.chars_per_sec if self.chars_per_sec > 0 else 0.0
        target = max(self.floor_seconds, read_time)
        # `age` is clamped non-negative before it is subtracted, so the delay is bounded to
        # `[0, target]`. A future-dated `created_at` or a lagging box clock yields a *negative*
        # age, and an unclamped `target - age` would then *inflate* the wait past `target`
        # (e.g. a 5-min clock skew → a 5-min sleep holding the router lock). Clamping treats a
        # not-yet-aged message as "just appeared" — it owes the full read-time, never more.
        age = (self._clock() - _parse_created_at(message.created_at)).total_seconds()
        delay = max(0.0, target - max(0.0, age))
        if delay > 0:
            self._sleep(delay)
        return delay


def _pace_enabled_from_env() -> bool:
    """`HARNESS_PACE_ENABLED` → the read-pacing kill switch — on unless explicitly off.

    On by default (pacing an AI↔AI exchange is the point); set an explicit off token
    (`0`/`false`/`no`/`off`) to disable it. Unset — or any other value, blank included — leaves
    it on: off only when explicitly turned off (mirrors `_onboard_from_env`).
    """
    raw = os.environ.get("HARNESS_PACE_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _pace_chars_per_sec_from_env() -> float:
    """`HARNESS_PACE_CHARS_PER_SEC` → the simulated reading rate; unset/blank → the default."""
    raw = os.environ.get("HARNESS_PACE_CHARS_PER_SEC")
    if raw is None or not raw.strip():
        return _DEFAULT_PACE_CHARS_PER_SEC
    return float(raw)


def _pace_floor_seconds_from_env() -> float:
    """`HARNESS_PACE_FLOOR_SECONDS` → the minimum read-delay seconds; unset/blank → the default."""
    raw = os.environ.get("HARNESS_PACE_FLOOR_SECONDS")
    if raw is None or not raw.strip():
        return _DEFAULT_PACE_FLOOR_SECONDS
    return float(raw)


def _pace_max_builds_from_env() -> int:
    """`HARNESS_PACE_MAX_BUILDS` → the Loop-2 rebuild cap; unset/blank → the default (3).

    The most times a batch reply is regenerated when messages land mid-generation (issue #226).
    The Nth build is posted unconditionally (no staleness check after it), so a value of 1
    means "never rebuild — generate once and post" (the pre-#226 single-shot behavior). A
    non-positive value is floored to 1 so the generate loop always runs at least once.
    """
    raw = os.environ.get("HARNESS_PACE_MAX_BUILDS")
    if raw is None or not raw.strip():
        return _DEFAULT_PACE_MAX_BUILDS
    return max(1, int(raw))


class WakeAgent:
    """Answers one timeline's unseen messages in a single process, then is done.

    A wake runs the agent's `Harness` against the `timeline:<uuid>` session,
    replying to every message newer than the persisted high-water mark and
    advancing the mark as it goes. Unlike `TimelineAgent` it does not loop and does
    not hold state between invocations — durability lives entirely in the persisted
    transcript (the session) and the `MarkStore`.

    Args:
        harness: The agent brain + tools. It **must** have a `home` (or pass an
            explicit `marks`) — wake mode needs somewhere to persist the mark and
            the transcript across processes.
        timeline: The uuid of the timeline this wake is scoped to.
        client: A `basecradle.BaseCradle`. Defaults to one built from the
            environment (`BASECRADLE_TOKEN`).
        marks: Where high-water marks persist. Defaults to a `MarkStore` under the
            harness's `home`.
        context_messages: How many backlog messages to seed as context on the
            first wake (see `_bootstrap`). The default bounds token cost; `None`
            seeds the whole backlog.
        onboard: Show the persistent operating brief on every wake (see
            `_wake_brief`). On by default. When on, the brief — `initialize.md`
            + the live tool manifest + the live `dashboard.md` + `system-prompt.md`
            — supersedes a static turn-0 charter, so the agent's standing context
            stays recent in a long transcript rather than aging out at turn 1. Off
            wakes with only the operator's charter, seeded once at turn 0.
        tool_manifest: ``(name, note)`` for the agent's active tools, rendered into
            the brief so it names exactly what the model can call. Defaults to the
            harness's registered function tools (no notes) when not supplied;
            `from_env` threads the precise `ResolvedTools.manifest` (built-ins and
            notes included).
        memory_provider: The agent's pluggable memory backend (see
            `basecradle_harness._memory_provider`). Its `observe` hook fires after every
            exchange and its `context` hook injects recalled memory into the persistent
            brief. ``None`` (the default for a directly-constructed wake) disables both
            hooks — the memory tool, if any, still comes from the harness's registry.
            `from_env` passes the env-selected provider (default SQLite, whose hooks are
            no-ops, so behavior is unchanged).
        breaker: The cross-wake circuit-breaker (see `WakeBreaker`). ``None`` (the default)
            constructs one over the harness's home with the generous default cap; `from_env`
            threads the env-tuned breaker (`HARNESS_WAKE_BREAKER_MAX`/`_WINDOW`/`_COOLDOWN`).
        pacer: The AI↔AI read-speed pacer (see `ReadPacer`). ``None`` (the default) constructs
            one with the real defaults; `from_env` threads the env-tuned pacer
            (`HARNESS_PACE_ENABLED`/`_CHARS_PER_SEC`/`_FLOOR_SECONDS`). Only the message path
            uses it (`_respond`); a human peer and a non-message wake are unaffected.
    """

    def __init__(
        self,
        harness: Harness,
        *,
        timeline: str,
        client: BaseCradle | None = None,
        marks: MarkStore | None = None,
        context_messages: int | None = DEFAULT_CONTEXT_MESSAGES,
        onboard: bool = True,
        probe_secret: str | None = None,
        tool_manifest: list[tuple[str, str | None]] | None = None,
        memory_provider: MemoryProvider | None = None,
        safety_notices: list[str] | None = None,
        defect_notices: list[str] | None = None,
        breaker: WakeBreaker | None = None,
        pacer: ReadPacer | None = None,
        max_builds: int = _DEFAULT_PACE_MAX_BUILDS,
        code_bridge: CodeExecutionBridge | None = None,
    ) -> None:
        if context_messages is not None and context_messages < 0:
            raise ValueError("context_messages must be non-negative or None")
        if marks is None and harness.home is None:
            raise ValueError(
                "Wake mode needs somewhere to persist its high-water mark across "
                "processes. Set HARNESS_HOME (or pass home= to Harness, or marks= here)."
            )
        self.harness = harness
        self.client = client or BaseCradle()
        self.timeline_uuid = timeline
        self.source = f"timeline:{timeline}"
        self.context_messages = context_messages
        self.onboard = onboard
        self.tool_manifest = tool_manifest
        self.memory_provider = memory_provider
        # Route every compaction summary into durable memory (issue #276, requirement 7). The
        # compactor is built with the provider, before there is a memory provider to hand it, so
        # the two are joined here — the one place that holds both. Without this the agent's
        # tool-driven work would vanish with the turns compaction drops (see `_remember_compaction`).
        if harness.compactor is not None:
            harness.compactor.on_summary = self._remember_compaction
        # Safe-by-default opt-out notices (active MCP servers, policy-refused drop-in tools)
        # surfaced into the persistent brief, so "all bets off" is stated and auditable —
        # empty for a pure-Harness config.
        self.safety_notices = safety_notices
        # Broken-shipped-default defect notices surfaced into the brief under their own loud
        # heading (issue #160), so a capability silently disabled by a stale overlay or a
        # packaging bug is impossible to miss — empty when every shipped default loaded.
        self.defect_notices = defect_notices
        # The persistent brief is composed at most once per `wake()` — lazily, right before the
        # first time the model is actually engaged — so an idle or probe-only wake never fetches
        # the live dashboard. It is then handed to every model call this wake makes as *ephemeral*
        # context (`Session.send(brief=…)`): the model reads it, and nothing brief-shaped is
        # written to the transcript (issue #275). Reset each wake.
        self._brief: str | None = None
        self._brief_composed = False
        # The shared HMAC key for the NOC synthetic-probe marker (see `_probe`). Set → the
        # message, webhook, and task reconciles each recognize a signed probe in their own
        # carrier field and ack it token-free, before the model. Unset → the short-circuit
        # is off and every item goes to the model.
        self.probe_secret = probe_secret
        self.marks = marks or MarkStore(harness.home)  # type: ignore[arg-type]
        # Activated tasks are tracked by a seen-set, not a high-water mark (see
        # `SeenStore`); it lives beside the marks, under the same home root.
        self.seen = SeenStore(self.marks.root)
        # Per-item atomic claims (see `ClaimStore`): the exactly-once guard that makes
        # `_act_on` safe across concurrent wakes and crash-resumable without reprocessing.
        # Lives beside the marks and the seen-set, under the same home root.
        self.claims = ClaimStore(self.marks.root)
        # Cross-wake circuit-breaker (see `WakeBreaker`): the generic backstop for an unknown
        # runaway wake loop. Records every wake and self-declines (no provider call) over the
        # cap. Lives beside the other stores, under the same home root. A directly-constructed
        # wake gets the generous defaults; `from_env` threads the env-tuned breaker.
        self.breaker = breaker or WakeBreaker(self.marks.root)
        # Read-speed pacing (see `ReadPacer`): before answering a peer AI's message, sleep to
        # simulate a human reading it, so an AI↔AI exchange is watchable and stays under the
        # breaker's trip line. A directly-constructed wake gets the real defaults; `from_env`
        # threads the env-tuned pacer. It holds no state, so nothing persists under home.
        self.pacer = pacer or ReadPacer()
        # Loop-2 (mid-generation staleness) rebuild cap (issue #226): the most times a batch
        # reply is regenerated when messages land during generation; the Nth build is posted
        # unconditionally. Floored to 1 so the generate loop always runs. It is a wake property
        # (Loop 2 re-reads the timeline through `self.client`), distinct from the pacer's
        # Loop-1 read-time seams — but shares the `HARNESS_PACE_ENABLED` kill switch: with
        # pacing off, Loop 2 does a single build and posts (see `_generate_settled`).
        self.max_builds = max(1, max_builds)
        self.timeline = self.client.timelines.get(timeline)

        # What this wake actually put on a timeline (issue #293). Now that the final text is never
        # auto-posted, the *only* messages that exist are the ones the model chose to create through
        # a tool — so the tools record here as they act, and this ledger is what the bookend line
        # counts (`posted=`) and what the mention informer reads to know whether a turn was silent.
        # One object, bound into the tools once; its contents cycle per wake (`reset`).
        self.speech = SpeechLedger()

        # Deterministic `Idempotency-Key`s for the four platform creates (issue #297). Bound into
        # the tools once, like the ledger above, and armed per *turn* with the item that turn is
        # answering. It is what makes a killed wake's create safe to re-issue: the recovering wake
        # mints the same key, so the platform returns the original record instead of a second one.
        self.keys = IdempotencyKeys()

        # Bind the live platform handle into every platform-aware tool — the same
        # seam the poll loop uses, so a router-woken peer can act on the timeline
        # exactly as a polling one. One wake serves one timeline; bind once. The
        # code-execution bridge (when active) is bound the same way, and rides the
        # context so the `code_attach` tool can reach it.
        self.code_bridge = code_bridge
        context = PlatformContext(
            client=self.client,
            timeline=self.timeline_uuid,
            home=self.harness.home,
            code_bridge=code_bridge,
            speech=self.speech,
            keys=self.keys,
        )
        bind_platform_tools(self.harness.tools, context)
        if code_bridge is not None:
            code_bridge.bind(context)

        # One Dashboard read answers "who am I?" — the identity uuid the actor self-filter
        # tests every item against, and the **handle** the mention informer looks for. (The
        # brief's orientation is the *live* `dashboard.md` primer, fetched per wake in
        # `_wake_brief`, not this structured read.) Read once: `me` is uncached, so touching
        # `self.client.me` twice would be a second HTTP round-trip on every wake.
        identity = self.client.me.identity
        self.me_uuid = identity.uuid
        self.me_handle = getattr(identity, "handle", None)

        # The deterministic mention informer (issue #293): when the agent is addressed by its own
        # @handle and its turn is about to end having done nothing on the timeline, it is *told* —
        # once, and never forced. It rides the engine's `TurnHook`, the seam that already exists for
        # exactly this ("the turn is about to end; does anything want to extend it?"), so it is
        # **composed** onto whatever hook is already wired (the code-execution Asset bridge) rather
        # than replacing it — the bridge runs first, and while it is still extending the turn the
        # informer stays quiet, because a turn being extended is not a turn that is ending.
        #
        # Composed onto the engine's **base** hook, never onto its live one: chaining would
        # *accrete*. Two agents built over one `Harness` would stack two informers, each holding the
        # other's dead `SpeechLedger` — and a stale-armed one would then nudge on a turn it knows
        # nothing about, reading "did it act?" off a ledger nobody writes to any more.
        self.informer = MentionInformer(handle=self.me_handle, speech=self.speech)
        engine = self.harness.engine
        engine.turn_hook = compose_hooks(engine.base_turn_hook, self.informer.on_turn)
        # Whether *this* turn degraded to the canned note (the step budget spent **and** the reserve
        # call failed). It labels the turn's unspoken text `kind=stuck`, so the journal tells the
        # three endings apart: an ordinary one, a step-capped self-authored report, and a give-up.
        self._degraded = False
        if onboard:
            # The persistent brief rides every model call (see `_wake_brief`) and carries the
            # personality charter (`system-prompt.md`) itself, so a static turn-0 seed would
            # only duplicate it — and unlike the brief, a seeded charter *persists*. Clear it;
            # the brief is the charter now. With onboarding off the operator's turn-0 charter
            # stands as before.
            self.harness.system_prompt = None

    @classmethod
    def from_env(cls, *, timeline: str | None = None) -> WakeAgent:
        """Build a fully wired wake agent from env vars — `TimelineAgent.from_env`'s twin.

        Reads the same provider/credential/charter vars, plus `HARNESS_HOME`, which
        wake mode requires: it is where the transcript and the high-water mark
        persist across the separate processes a router spawns. `timeline` overrides
        `BASECRADLE_TIMELINE` (the router passes it on the command line).

        Token reuse matters most here: each wake is a fresh process, so without
        persistence a credential-only agent would mint a new token (a new platform
        Session) on every wake. `_client_from_env` mints only when the token is missing
        or dead and writes it back to `BASECRADLE_ENV_FILE`, so the next wake — which
        sources that same file — reuses it. See `_token` for the full lifecycle.

        `NOC_PROBE_SECRET` (optional) is the shared HMAC key for the NOC's synthetic-probe
        marker. Set → a recognized probe is acked token-free before the model (see
        `_probe`), across the message, webhook, and task seams alike; unset → the
        short-circuit is off. The same var name the NOC box uses, so one provisioned value
        serves the harness half of all three seam contracts.
        """
        home = os.environ.get("HARNESS_HOME")
        if not home:
            raise ValueError(
                "Wake mode requires HARNESS_HOME — the directory where the agent's "
                "transcript and high-water mark persist across wakes."
            )
        provider, resolved, memory, bridge = _resolve_tools_and_provider()
        # The deploy-selected profile (issue #256): the same `_profile_from_env` read that gated
        # tool resolution now builds the registry, so an `unlocked` deploy admits its opted-in
        # shell-class tool rather than the locked default filtering it straight back out.
        _, policy = _profile_from_env()
        harness = Harness(
            provider,
            system_prompt=charter_from_env(),
            tools=resolved.tools,
            policy=policy,
            # The per-turn step budget (default 24, `HARNESS_MAX_STEPS` overrides). The engine
            # injects a live counter against it and, when spent with tools still pending, makes
            # the reserve summary call rather than cutting off with a canned string (issue #243).
            max_steps=_max_steps_from_env(),
            # How many extra times a truncated/unparseable provider response is re-requested
            # before the wake gives up (default 2, `HARNESS_RESPONSE_RETRIES` overrides). Without
            # it, a single EOF-mid-JSON flake aborted the wake and silently dropped the peer's
            # message — the item is marked seen before the model runs, so no later wake retried it
            # (issue #259).
            response_retries=_response_retries_from_env(),
            # The active server-side built-ins (e.g. web_search), so a model that calls one as a
            # function gets targeted guidance instead of the generic error (issue #245).
            server_builtins=resolved.builtins,
            home=home,
            # The code-execution Asset bridge (when active) harvests a run's output files +
            # source into Assets after each code-exec turn, then feeds their uuids back. None
            # when code execution isn't opted in → the engine loop is unchanged.
            turn_hook=bridge.on_reply if bridge is not None else None,
            # The context budget (issue #276). A wake replays the *whole* persisted transcript, so
            # without this a standing agent grows monotonically into its model's context ceiling and
            # bricks. `HARNESS_MAX_CONTEXT_TOKENS` overrides the ceiling (0 disables compaction);
            # otherwise the adapter is asked and, failing that, a conservative floor is assumed. The
            # summary's route into durable memory is wired below, once the memory provider is bound.
            compactor=_compactor_from_env(provider),
        )
        return cls(
            harness,
            timeline=timeline or os.environ["BASECRADLE_TIMELINE"],
            client=_client_from_env(),
            context_messages=_context_messages_from_env(),
            onboard=_onboard_from_env(),
            code_bridge=bridge,
            # `or None` so a set-but-blank NOC_PROBE_SECRET (an exported-but-unfilled
            # secret) reads as *off*, not as "enabled with an empty HMAC key".
            probe_secret=os.environ.get("NOC_PROBE_SECRET") or None,
            # The active tool manifest (built-ins + notes) for the persistent brief, so it
            # names exactly the tools this config resolved — never a present-but-broken one.
            tool_manifest=resolved.manifest,
            # The env-selected memory provider, whose observe/context hooks fire in the wake
            # loop. Default SQLite → no-op hooks → behavior unchanged for @jt.
            memory_provider=memory,
            # Safe-by-default opt-out notices from tool resolution (active MCP servers,
            # policy-refused drop-ins). Empty by default → no safety section in the brief.
            safety_notices=resolved.notices,
            # Broken-shipped-default defects from tool resolution (issue #160). Empty when
            # every shipped default loaded → no defect section in the brief.
            defect_notices=resolved.broken,
            # The env-tuned cross-wake circuit-breaker, persisted under HARNESS_HOME beside
            # the marks/seen/claims stores. Generous defaults; off only if explicitly capped
            # to 0. The router's cross-agent breaker is the complementary layer.
            breaker=WakeBreaker.from_env(home),
            # The env-tuned AI↔AI read-speed pacer (issue #224, reworked #226). On by default
            # with the real reading-rate/floor defaults; a human peer is never paced, so replies
            # to humans are instant exactly as before.
            pacer=ReadPacer.from_env(),
            # The Loop-2 mid-generation staleness rebuild cap (issue #226). Env-tuned; the Nth
            # build posts unconditionally. Shares the pacer's `HARNESS_PACE_ENABLED` kill switch.
            max_builds=_pace_max_builds_from_env(),
        )

    def wake(
        self,
        *,
        trigger: str | None = None,
        event_trigger: str | None = None,
        asset_trigger: str | None = None,
    ) -> list[object]:
        """Reconcile this timeline once — unseen messages, posted assets, inbound webhook
        deliveries, *and* newly-activated tasks — and return every message that reached the
        timeline (the agent's own, through its tools, plus any probe ack the harness sent).

        A wake surfaces every kind of unseen actionable item on the timeline, not just
        new messages, because each kind arrives by a different route: a message and an
        asset are read from the timeline, a webhook delivery and a task activation are not
        items the timeline scan would surface on their own. Each kind has its own
        idempotency record so reconciling one never re-surfaces another, and all engage
        through the one session, so the agent perceives them in a single coherent
        conversation. Messages and assets pass through the **actor self-filter**, so the
        agent never reacts to its own posts. If nothing is unseen, no provider call is
        made and nothing is posted.

        **Engaging is not answering** (issue #293). The wake puts what it found in front of the
        model; the model decides what — if anything — to say, by calling the `messages` tool. A
        wake that returns `[]` may mean "nothing was unseen" *or* "the agent read it and chose
        silence", and the two are told apart by the journal, not by this return value: the second
        leaves an `unspoken` line carrying the reasoning.

        **The cross-wake circuit-breaker runs first.** Before any reconcile, this wake is
        recorded on the per-timeline `WakeBreaker`; if this timeline is in a runaway wake
        loop (too many wakes in the rolling window), the wake **self-declines** — it makes no
        provider call, logs a single loud WARNING on the trip transition, and returns nothing.
        It auto-resets once the burst clears. This is the backstop the in-wake `max_steps`
        cap and the actor self-filter don't cover: an *unknown* cross-wake loop.

        `trigger`, `event_trigger`, and `asset_trigger` are the optional uuids of the
        message, webhook event, or asset that fired the wake. **The router never passes
        them** — it wakes a harness agent with the timeline uuid alone — so each kind
        finds its own unseen items unaided: the newest on a first wake, everything past its
        mark thereafter. The triggers remain accepted for a manual or future-router
        invocation that does name an item (each only sharpens its own first-wake bootstrap
        and is ignored once that kind's record exists), but nothing depends on them. Task
        activations never used a trigger — the reconcile finds every activated-but-unhandled
        task, keeping the router thin.

        **Bookended in the log.** A wake is the harness's unit of work, so it opens with one
        INFO line naming what it is about to run with (timeline, trigger, provider, model, and
        the router's delivery id when it exported one) and closes with one naming what came of
        it (outcome, steps spent against the budget, messages posted, wall-clock). Between them
        sit the LLM, tool, and posted-message lines — so a wake reads end-to-end in the journal
        without the transcript. The end line rides a ``finally``: a wake that *crashes* is the
        one whose outcome matters most, and it still reports what it had done by then.
        """
        started = time.monotonic()
        delivery = delivery_id()
        provider, model = describe_provider(self.harness.provider)
        _log.info(
            "wake start %s",
            kv(
                timeline=self.timeline_uuid,
                trigger=_trigger_label(trigger, event_trigger, asset_trigger),
                provider=provider,
                model=model,
                delivery=delivery,
            ),
        )
        # `posted` is what the *harness* put on the timeline (today: NOC probe acks). What the
        # *agent* said is in `self.speech` — it speaks only through its tools now — so the two are
        # summed at the end rather than merged as they go, which is what keeps the bookend's count
        # honest without double-counting. Reset before the breaker check, so a second wake in one
        # process can never report the first one's posts.
        posted: list[object] = []
        self.speech.reset()
        outcome = "error"  # only a clean return past the reconciles earns another verdict
        try:
            if self._breaker_short_circuits():
                outcome = "declined"
                return []  # a tripped timeline self-declines: no session, no provider call
            session = self.harness.session(self.source)
            self._brief = None  # compose the brief once this wake, lazily, before the model
            self._brief_composed = False
            # The message path's per-wake bookkeeping (issue #285). `_ledger` is every message this
            # wake looked at, in timeline order, with what it decided — `_settle` reads it to move
            # the mark, and may never move it past an item that is not settled. `_cursor` is the
            # in-wake read cursor (distinct from the persisted mark, which no longer advances at
            # claim time).
            self._ledger: list[tuple[str, str]] = []
            self._cursor: str | None = None
            # What `_recover` decided about each orphaned message, before this wake called the
            # model at all (issue #297). `_readmit` reads it rather than recomputing: recovery
            # reasons from the transcript, and the transcript stops being pristine evidence the
            # moment this wake writes to it — which it does from inside `_absorb`'s own re-read
            # loops. An orphan with no entry here is one recovery declined; it stays `_PENDING`.
            self._recovered: dict[str, str] = {}
            # Turns this wake has already resumed, and how that went. A dead wake's turn carried a
            # *batch*, so several of its messages arrive here as orphans of the same turn — and a
            # turn is finished **once**. On the happy path the transcript says so by itself (the
            # resumed turn now has a narration, so the rest commit), but a resume that failed leaves
            # the turn looking exactly as unfinished as it did before, and without this the next
            # message of the same batch would resume it *again* — a second full step budget, and a
            # second post under the next ordinal, which no key dedupes.
            self._resumed: list[tuple[Message, str]] = []
            # A fresh identity per *wake*, not per process: an agent that wakes twice in one
            # process must see its own earlier wake's abandoned claims as orphaned, not as "mine".
            self.claims.wake = uuid4().hex
            posted += self._wake_messages(session, trigger)
            posted += self._wake_assets(session, asset_trigger)
            posted += self._wake_events(session, event_trigger)
            posted += self._wake_tasks(session)
            outcome = "ok"
            return posted + self.speech.posts
        finally:
            # `max_steps` is a **per-turn** budget and a wake may take several turns — one per
            # activated task, posted asset, or webhook delivery (unseen messages batch into a
            # single turn), *plus* one for every mid-generation rebuild the staleness guard makes
            # (`_generate_settled`). So the turn count rides alongside the step total: without it
            # a legitimate 3-turn wake reading `steps=30/24` would look like a blown budget rather
            # than three turns of ten.
            engine = self.harness.engine
            _log.info(
                "wake end %s",
                kv(
                    timeline=self.timeline_uuid,
                    outcome=outcome,
                    turns=engine.turns_run,
                    steps=f"{engine.steps_used}/{engine.max_steps}",
                    # Everything that reached a timeline: the harness's own posts (probe acks) plus
                    # every message the agent *chose* to send through its tools. `posted=0` is now a
                    # real and legitimate outcome — the agent read, thought, and decided not to speak
                    # — so this count is what makes a silent wake **visibly** silent rather than
                    # indistinguishable from a broken one. Its reasoning is on the `unspoken` line.
                    posted=len(posted) + len(self.speech.posts),
                    duration=f"{time.monotonic() - started:.2f}s",
                    delivery=delivery,
                ),
            )

    # --- the cross-wake circuit-breaker --------------------------------------

    def _breaker_short_circuits(self) -> bool:
        """Record this wake on the breaker; trip/reset loudly, and report whether to decline.

        The first thing a wake does — *before* the session is loaded or the model is ever
        engaged — so a tripped timeline self-declines token-free, exactly like the NOC probe
        ack short-circuit. Logs the loud WARNING **once** on each state transition (the
        durable trip marker is the one-time guard), then returns the breaker's verdict: True →
        this wake makes no provider call. A reset transition does *not* short-circuit — it
        alerts that the burst cleared, then the wake proceeds normally (`record_and_check`
        already restarted the window).

        **The alert is a log line, not a timeline post** (issue #293). It used to be both: a
        WARNING *and* a message posted in the agent's voice ("I appear to be in a wake loop
        here…"), which the agent never wrote and never chose to send. Under the Unspoken Channel
        the harness does not speak for the agent — every timeline message is an intentional tool
        call — and this was the last place it did. Nothing is lost: the breadcrumb was always for
        the **operator**, who reads the journal, and the NOC alerts on this exact WARNING string
        (`Wake breaker TRIPPED`), never on the posted message. The peers, meanwhile, now see what
        the mechanism actually means: an agent that has gone quiet.

        **Known bound — a tripped timeline also stops acking NOC probes.** The probe ack
        lives per-item inside `_act_on`, downstream of this early return, so a tripped wake
        skips it along with everything else. That is acceptable: a probe timeline is quiet
        and low-cadence, so it never reaches the cap in practice, and a timeline genuinely in
        a runaway *failing* its heartbeat is honest signal, not a false FAIL — the loud trip
        alert is itself the louder out-of-band notice. We do **not** read the timeline to
        rescue a probe before deciding, because that would forfeit the whole point: a
        token-free decline before any platform work.
        """
        decision = self.breaker.record_and_check(self.timeline_uuid)
        if decision.tripped:
            _log.warning(
                "Wake breaker TRIPPED for timeline %s: %d wakes within %ss exceeds the cap "
                "of %d. Self-declining (no provider call) until the burst clears; an operator "
                "can reset by clearing the trip marker under HARNESS_HOME.",
                self.timeline_uuid,
                decision.count,
                self.breaker.window,
                self.breaker.max_wakes,
            )
        elif decision.reset:
            _log.warning(
                "Wake breaker RESET for timeline %s: the wake burst cleared; resuming normal "
                "operation.",
                self.timeline_uuid,
            )
        return decision.short_circuit

    # --- messages ------------------------------------------------------------

    def _wake_messages(self, session: Session, trigger: str | None) -> list[object]:
        """Answer unseen messages: bootstrap on the first wake, else reply incrementally."""
        mark = self.marks.get(self.timeline_uuid)
        if mark is None:
            return self._bootstrap(session, trigger)
        unseen = _messages_since(self.client.messages.filter(timeline=self.timeline_uuid), mark)
        return self._respond(session, unseen)

    # --- posted assets -------------------------------------------------------

    def _wake_assets(self, session: Session, asset_trigger: str | None) -> list[object]:
        """Surface a peer's posted files (assets) and let the agent engage with them.

        An asset is a timeline item, like a message, so it rides the same high-water
        mark — but the wake's *message* scan reads only messages, so a file a peer
        shares would otherwise go unseen. This is the founder's minimum wake set: a
        peer posts an image or a doc, and the agent perceives it (and can `view`,
        `read`, or `listen` to it via the assets/audio tools). The router wakes with the
        timeline uuid alone, so on the first wake (no mark, no `asset_trigger`) the agent
        acts on the **newest** unseen asset — the one that almost certainly woke it —
        bounding a fresh agent to a single action rather than replaying every file posted
        before it arrived. A named `asset_trigger`, when a manual/future invocation passes
        one, acts on that asset and everything newer.

        The **actor self-filter** is load-bearing here: an asset the agent itself posted
        — most importantly an image it just generated with `generate_image` — is skipped
        (never acted on), while its mark still advances, so the agent cannot react to its
        own output and spiral into generating image after image. `_respond_assets`
        applies that filter through the shared `_act_on` loop, so it holds on the trigger
        path too (a wake somehow fired on the agent's own asset is still a no-op).
        """
        mark = self.marks.get(self.timeline_uuid, kind=_ASSETS)
        if mark is None:
            return self._bootstrap_stream(
                session,
                self.client.assets.filter(timeline=self.timeline_uuid),
                asset_trigger,
                _ASSETS,
                self.client.assets.get,
                self._respond_assets,
            )
        unseen = _assets_since(self.client.assets.filter(timeline=self.timeline_uuid), mark)
        return self._respond_assets(session, unseen)

    def _respond_assets(self, session: Session, assets: list[object]) -> list[object]:
        """Perceive each newly-posted asset, skipping the agent's own (the self-filter).

        `_perceive_asset` fetches the file and presents an image **inline** so the agent
        actually sees a peer's picture (not just a description of it); media it cannot yet
        fully perceive degrades to a description rather than erroring. A recognized NOC
        synthetic-probe — read from the asset's **description**, the asset analog of a
        message body / task instructions / webhook payload (see `_asset_marker_carrier`) —
        is acked token-free before the model (`probe=`), the 4th seam's short-circuit. The
        probe check runs *before* the fetch, so a probe asset is acked without any download.
        """
        return self._act_on(
            session,
            assets,
            self._perceive_asset,
            lambda asset: self.marks.set(self.timeline_uuid, asset.content.uuid, kind=_ASSETS),
            kind=_ASSETS,
            skip=self._is_own,
            probe=lambda asset: self._probe_nonce(_asset_marker_carrier(asset)),
        )

    def _perceive_asset(self, asset: object) -> tuple[str, list[ImageContent]]:
        """A peer's posted asset as the agent perceives it: an image shown, else described.

        The asset-wake's perception step, and the reason the seam is more than a
        notification: for a viewable image the bytes are fetched and handed to the model as
        input (via `image_input`, the same gate the `view` tool uses), so a vision-capable
        agent *sees* the shared picture on wake. Anything it cannot yet fully perceive —
        a non-image file, an unviewable/oversized image, or an image whose download fails —
        degrades to the text description (`_incoming_asset_text`), which names the file and
        its type and points at the tools, so the seam is graceful, never an error. Audio/
        video perception *depth* is out of scope here (it rides the perception done-bar
        thread); this handles their seam by acknowledging the file rather than choking on it.
        """
        file = asset.content.file
        if _is_image(file.content_type):
            try:
                shown = image_input(file)
            except httpx.HTTPError:
                shown = None  # fetch failed: degrade to a description, never an error
            if isinstance(shown, ImageContent):
                intro = (
                    f"[{asset.created_at}] {asset.user.handle} posted a file to this "
                    f"timeline: {_describe(asset)}."
                )
                return f"{intro} Looking at it now.", [shown]
        # Non-image, unviewable/oversized image, or a failed fetch: describe, don't show.
        return _incoming_asset_text(asset), []

    # --- webhook events ------------------------------------------------------

    def _wake_events(self, session: Session, event_trigger: str | None) -> list[object]:
        """Surface unseen inbound webhook deliveries and let the agent act on them.

        A received webhook event is *not* a timeline item the way a message or an
        activated task is — so a woken agent never sees one unless it goes looking.
        This is that lookup: the same idempotent high-water-mark discipline used for
        messages, over the SDK's webhook-events read surface. The router wakes with the
        timeline uuid alone, so on the first wake (no mark, no `event_trigger`) the agent
        acts on the **newest** unseen delivery — the one that almost certainly woke it.
        This is the fix for the live bug where a `webhook_event.received` wake surfaced
        nothing: the old first-wake baselined silently when no trigger was passed, and the
        router never passes one, so every first delivery was dropped. A named
        `event_trigger`, when a manual/future invocation passes one, acts on that delivery
        and everything newer.
        """
        mark = self.marks.get(self.timeline_uuid, kind=_EVENTS)
        if mark is None:
            return self._bootstrap_stream(
                session,
                self.client.webhook_events.filter(timeline=self.timeline_uuid),
                event_trigger,
                _EVENTS,
                self.client.webhook_events.get,
                self._respond_events,
            )
        unseen = _events_since(self.client.webhook_events.filter(timeline=self.timeline_uuid), mark)
        return self._respond_events(session, unseen)

    # --- the first wake for a creation-ordered stream (events, assets) --------

    def _bootstrap_stream(self, session, items_filter, trigger, kind, fetch_one, respond):
        """First wake for a creation-ordered, mark-tracked stream (webhook events, assets).

        There is no persisted mark yet. **The router wakes a harness agent with the
        timeline uuid alone — it never names the triggering item** (see basecradle-router
        `wake_command`), so in production `trigger` is always `None`. A no-trigger first
        wake therefore acts on the **newest** unseen item — the one that almost certainly
        woke us — exactly as the message bootstrap replies to the newest message on a fresh
        join; `respond` advances the mark to it (and applies the asset self-filter, so the
        agent's own newest post is marked-but-not-acted, never a wake-loop). Older items
        are left behind, not replayed, so a fresh agent is bounded to a single action
        rather than flooded by a backlog it was never woken for. Acting on the newest is
        what makes a webhook delivery (and a peer's posted asset) actually surface under
        the real router contract; baselining silently was the live bug where the first
        delivery of each kind was dropped.

        With a trigger (a manual or future-router invocation that does name the item), act
        on that item and everything newer, then baseline to the true newest so the next
        wake is incremental. `fetch_one` retrieves the trigger by uuid if a burst pushed it
        past the fetched window, so a named item is never silently dropped.

        **Known bound — newest-only acts on one item on a *cold* first wake.** This is the
        first reconcile of this kind *ever* (no mark yet), so "newest only" is intentional:
        it bounds a fresh agent to a single action instead of replaying a backlog. Two
        edges fall out of it, both strictly better than the old baseline-silently behavior
        (which acted on nothing) and both confined to the cold first wake — every wake
        after the mark exists acts on **all** unseen items past it:
        - *Cold-start burst:* if several items of this kind predate the very first wake,
          only the newest is acted on and the rest are marked seen. In the live router flow
          each new item triggers its own wake, so this only bites items delivered before the
          agent ever reconciled this kind.
        - *Self-authored newest:* if the newest item is the agent's own (a just-generated
          asset), the self-filter makes the wake a no-op and the mark still advances past an
          older unseen peer item. Acceptable for the same reason — in the per-event flow the
          peer item had its own earlier wake.
        If a future need makes dropping a cold-start burst unacceptable (e.g. webhook
        deliveries that must each be processed), act on all of `recent` here for that kind
        rather than `recent[0]`, accepting the join-time replay that trades against.
        """
        recent = _recent(items_filter, self.context_messages)
        if trigger is None:
            if not recent:
                return []
            # Act on the newest unseen item; `respond` advances the mark to it. See the
            # "Known bound" note above for why a cold first wake acts on only the newest.
            return respond(session, [recent[0]])
        to_act = self._from_trigger(recent, trigger, fetch_one)
        if not to_act:
            return []  # no such item and an empty stream: nothing to do, nothing to mark
        posted = respond(session, to_act)
        newest = recent[0].content.uuid if recent else to_act[-1].content.uuid
        self.marks.set(self.timeline_uuid, newest, kind=kind)
        return posted

    def _from_trigger(self, recent: list[object], trigger: str, fetch_one) -> list[object]:
        """The items to act on for a named trigger, oldest-first.

        Normally the trigger is in the fetched window: act on it and everything newer. If
        a burst pushed it *past* the window (more than `context_messages` newer items
        arrived before the wake fired), the windowed scan would miss it — so fetch the
        named item directly and act on it together with the window, rather than silently
        dropping the one item the router explicitly woke us for.
        """
        for i, item in enumerate(recent):
            if item.content.uuid == trigger:
                return list(reversed(recent[: i + 1]))  # trigger and newer, chronological
        # Trigger not in the window: fetch it directly so it is never dropped, and act on
        # it before the window (it is older than everything fetched).
        try:
            triggered = fetch_one(trigger)
        except BaseCradleError:
            triggered = None  # gone or unreadable — act on what we can still see
        window = list(reversed(recent))  # all newer than the trigger, chronological
        return [triggered, *window] if triggered is not None else window

    def _respond_events(self, session: Session, events: list[object]) -> list[object]:
        """Act on each webhook delivery in order, advancing the event mark per delivery.

        A recognized NOC probe — read from the delivery's **payload** — is acked token-free
        before the model (`probe=`), at-least-once (post the ack, then record): the probe
        seam is never claimed, so a refused ack re-acks next wake rather than going silent.
        The short-circuit runs *inside* `_act_on`, after `_bootstrap_stream` has already
        selected the item, so the #100 cold-first-wake bootstrap (newest unseen delivery
        only — which on a quiet probe timeline is the probe) is preserved unchanged.
        """
        return self._act_on(
            session,
            events,
            _incoming_event_text,
            lambda event: self.marks.set(self.timeline_uuid, event.content.uuid, kind=_EVENTS),
            kind=_EVENTS,
            probe=lambda event: self._probe_nonce(event.content.payload),
        )

    # --- activated tasks -----------------------------------------------------

    def _wake_tasks(self, session: Session) -> list[object]:
        """Carry out every activated-but-unhandled task on the timeline.

        A task activation is not a timeline item the wake's message scan would surface,
        so — like a webhook delivery — the agent must go looking. Unlike messages and
        events, an activated task is not a creation-ordered stream a high-water mark can
        track (a task scheduled earlier can activate later) and carries no terminal
        status the agent could set, so idempotency is a persisted *seen-set* (see
        `SeenStore`), now backed by an atomic per-task claim (`ClaimStore`) for safety across
        concurrent wakes too: act on each activated task whose uuid is not yet recorded. The
        task is recorded and claimed **before** it is acted on (claim-first), not after — a
        task is at-most-once, so an action that re-wakes the agent (a generated image posts
        an `asset.created`) cannot re-surface the still-`activated` task and re-run it. That
        re-fire was the live monkey pile-up: the task stayed `activated`, its image-post
        re-woke the agent, and an act-then-record order let the unrecorded task fire again
        and again. An activated-but-unhandled task is genuinely undone work — not stale
        history — so acting on all of them is correct, and needs no router-passed
        trigger, which keeps the router thin.

        The full activated list is drained (no `context_messages` window): a task is
        acted on one at a time, never seeded as history, so there is no token-cost
        argument for a cap — and a cap would silently drop tasks beyond it, since a
        newly-due task can sit anywhere in the (creation-ordered) list and no high-water
        mark guarantees a re-scan. The per-wake cost is the count of activated tasks on
        the timeline, the same order as the seen-set read this already does.
        """
        activated = list(self.client.tasks.filter(timeline=self.timeline_uuid, status="activated"))
        if not activated:
            return []
        seen = self.seen.all(self.timeline_uuid, kind=_TASKS)
        # Oldest-first, so the agent works the backlog in the order it was scheduled.
        unhandled = [task for task in reversed(activated) if task.content.uuid not in seen]
        return self._act_on(
            session,
            unhandled,
            _activated_task_text,
            lambda task: self.seen.add(self.timeline_uuid, task.content.uuid, kind=_TASKS),
            kind=_TASKS,
            # A NOC probe task is recognized from its **instructions** and acked token-free.
            # `_act_on` checks `probe` *before* claiming, so the probe is acked at-least-once
            # (post, then record) and never claimed. That is correct and must be preserved: a
            # probe's only side-effect is @jt's own ack (self-filtered, and router wakes are
            # serialized), so there is no re-fire hazard to guard against, and at-least-once
            # is the safe failure direction for a monitor — a crash before recording re-acks
            # (harmless) rather than leaving the task claimed-but-unacked, which would
            # manufacture a false FAIL. See task-seam.md §4.
            probe=lambda task: self._probe_nonce(task.content.instructions),
        )

    # --- the shared act-on loop ----------------------------------------------

    def _act_on(
        self,
        session: Session,
        items,
        render,
        record,
        kind: str,
        skip=None,
        probe=None,
    ) -> list[object]:
        """Engage the model on each item in order, recording it as handled.

        The one loop behind all four reconcilers — messages, webhook events, activated
        tasks, and posted assets. `render` turns an item into the text the model reads, or
        a `(text, images)` pair when the item carries pictures to *show* the model (the
        asset-perception path); `record` marks it handled (a mark advance, or a seen-set
        add); `kind` namespaces its atomic claim (see `ClaimStore`). **`kind` must match the
        namespace `record` writes to** — they are the same per-reconciler constant
        (`_MESSAGES`/`_ASSETS`/`_EVENTS`/`_TASKS`); if they ever diverge, claims land in one
        namespace while marks land in another and the exactly-once guard silently goes dead.

        **It posts nothing** (issue #293). The turn's final text is the agent's *unspoken*
        narration: journaled, never posted. If the agent wants to say something about the file
        it was shown or the task that came due, it says it the one way anything is ever said
        here — by calling the `messages` tool, during the turn. So an activated task that ran
        quietly and told nobody is a *normal* outcome, not a lost reply.

        **Exactly-once, claim-first (`ClaimStore`).** Before acting on an item, the loop
        *atomically claims* it. The first wake to claim wins and acts; a concurrent wake
        loses the claim and skips — so two near-simultaneous wakes (an upload firing
        `asset.created` + `message.created` spawns two) handle the same message exactly
        once instead of double-replying. Because the claim lands *before* the action, it
        also makes a crashed or partial wake non-reprocessing: the claim persists, so the
        retry skips the item rather than re-running its turn and re-firing its tool actions.
        The `record` (mark/seen advance) likewise lands before the action. The **claim is the
        authoritative exactly-once guard**; the high-water mark is a best-effort scan bound
        that may briefly rewind under truly concurrent wakes (two wakes claiming adjacent
        items in either order) — harmless, because a rewound mark only re-scans an item whose
        claim then blocks the duplicate. This is at-most-once by design —
        a one-time dropped reply on a hard crash beats a backlog re-answered, with side
        effects, on every later wake (the live reprocess loop). Tasks gain the same
        cross-process exclusivity their image-post re-fire never had.

        `probe(item)` is the **NOC synthetic-probe short-circuit**, passed by all four
        reconcilers — messages, webhook events, activated tasks, and posted assets — each
        reading the marker from its own carrier field (a message body, a webhook payload, a
        task's instructions, an asset's description). When it returns a nonce (the item is a
        valid signed probe — see `_probe`), the loop posts the deterministic ack and records
        the item **without calling the model** — no provider request, no tokens, nothing
        into the transcript — so the message-, webhook-, task-, and asset-seam heartbeats
        all run token-free at rest (an asset probe is acked before its file is ever
        fetched). It is checked *after* the self-filter (a probe is from a distinct peer,
        never the agent's own) and **before the claim**, and follows an at-least-once order
        (post the ack, *then* record), so a crash mid-batch re-acks. This probe-before-claim
        order is **load-bearing for the task seam**: a probe is never claimed, so a crash
        before recording re-acks (harmless — the prober matches the first ack carrying its
        nonce) rather than leaving the item claimed-but-unacked, which would manufacture a
        false monitor FAIL (task-seam.md §4).

        **Never crash the wake (B2).** Both platform-touching steps degrade instead of
        raising: a refused post (`self._post` — the probe ack) is logged and the loop carries
        on; an engine that hits its step cap (`self._engage`) becomes a short unspoken note. So
        a wake that hits a locked timeline or the step cap exits 0 — no unhandled exception
        reaches the entrypoint, and the item is still recorded (no reprocess). A locked timeline
        now surfaces to the *model*, as the refusal its `messages` tool call returns, which is
        strictly better: the agent learns it cannot speak here and can act on that.

        `skip(item)` is the **actor self-filter** — the safety property. When it returns
        true (the item is the agent's *own* authored post — a message it sent, an image it
        generated), the item is *not* acted on, but its record **still advances** (no claim
        needed — there is no action to make exclusive), so the agent never reacts to its own
        output and never wake-loops on it, while the skipped item is still marked seen so it
        is not re-scanned forever.
        """
        posted = []
        for item in items:
            if skip is not None and skip(item):
                record(item)  # own post: never acted on, but marked so it is not re-scanned
                continue
            if probe is not None and (nonce := probe(item)) is not None:
                # A verified NOC probe: ack token-free (no model call), and record only
                # *after* a successful ack — at-least-once, never claimed. A refused ack
                # degrades *silently* (`note=False` keeps the probe seam trace-free) and
                # leaves the item unrecorded, so the next wake re-acks rather than marking it
                # seen with no ack ever posted (a false monitor FAIL — task-seam.md §4).
                ack = self._post(ack_line(nonce), kind="probe-ack")
                if ack is not None:
                    posted.append(ack)
                    record(item)
                continue
            if not self.claims.claim(self.timeline_uuid, item.content.uuid, kind=kind):
                continue  # a concurrent (or crashed prior) wake already owns this item
            record(item)  # claim-first: mark seen before acting, so a failed wake won't reprocess
            # `render` returns the text the model reads, or a (text, images) pair when the
            # item carries pictures to *show* the model (the asset-perception path). Images
            # ride into the model's input on this turn and are evicted after (see `Session`).
            rendered = render(item)
            text, images = rendered if isinstance(rendered, tuple) else (rendered, [])
            narration = self._engage(session, text, images, item)
            self._unspoken(narration)
            # **Memory observes every engaged turn — posted or silent** (issue #293). It used to
            # fire only when a reply posted, which under silence-default would quietly drop facts
            # arriving in items the agent chose not to answer: a peer's birthday mentioned in a
            # message the agent had no reason to reply to must still be recallable, from any
            # timeline, later. The exchange is real whether or not anyone else heard it.
            self._observe(text, narration)
        return posted

    def _engage(
        self,
        session: Session,
        text: str,
        images: list[ImageContent],
        item: object | None = None,
    ) -> str:
        """Run the model on one item, degrading the engine's step-cap to a graceful note.

        Returns the turn's **unspoken narration** — the model's final text, which the caller
        journals and never posts (issue #293). Anything the agent wanted the timeline to see, it
        already sent itself, with the `messages` tool, during the turn.

        The think→act loop bounds runaway tool use with `max_steps` (`_engine`), raising
        `EngineError` when the model never settles. A wake must not crash on that — so it
        degrades to a short, honest note (unspoken — for the record, not for the peer) and the
        batch carries on.
        Other failures still propagate (the entrypoint reports them cleanly); this catches only
        the step-cap, the one the issue names.

        The mention informer is armed here, on the text the model is about to read: this is the
        one place that knows *what* the agent is being shown, and it is re-armed per model call
        because "was I addressed, and did I act?" is a question about a turn, not about a wake.

        The persistent brief rides *with* this call — spliced in just ahead of the user turn,
        so it is present and recent for the item it governs, and never persisted (see
        `_wake_brief`). The item's text is the retrieval query the memory provider's `context`
        hook ranks against, so recalled memory is relevant to this turn.

        `item` is the platform item this turn is answering — an asset, a webhook delivery, an
        activated task. It anchors the turn's idempotency keys and is recorded on the turn, exactly
        as a message batch's uuids are (issue #297). These kinds do not yet *recover* (issue #289
        owns that), but a create they make is keyed all the same: it costs nothing, and the day the
        recovery reaches them the keys will already be right.
        """
        self.informer.arm(text)
        self._degraded = False
        uuid = _uuid_of(item) if item is not None else None
        if uuid is not None:
            self.keys.begin(session, timeline=self.timeline_uuid, anchor=uuid)
        else:
            # No item to anchor on → **disarm**. Leaving the minter armed would key this turn's
            # creates to the *previous* turn's anchor while counting their ordinals off this turn's
            # work — reproducing a key the previous turn already used, so the platform hands back
            # the old record and the new message is never posted, silently, with the tool reporting
            # success. Unkeyed is the honest fallback: the create behaves as it always did.
            self.keys.clear()
        try:
            return session.send(
                text,
                images=images,
                brief=self._wake_brief(query=text),
                items=[uuid] if uuid is not None else None,
            )
        except EngineError as error:
            return self._stuck_note(error)

    def _unspoken(self, text: str) -> None:
        """Journal a turn's final text — the Unspoken Channel's one delivery point (issue #293).

        This is where the auto-post used to be, and the swap is the whole design: what was
        broadcast to every viewer of a timeline is now written to the agent's own log and its
        memory. Nothing here reaches a peer, and **nothing is watching it** — it is a flight
        recorder. The agent's *speech* left earlier, through a tool, because it decided to speak.

        The three endings a turn can have are labelled, because they read nothing alike to whoever
        (or whatever) reconstructs this later: an ordinary `narration`, the step-cap's self-authored
        `reserve` report, and the `stuck` note when even that failed.
        """
        log_unspoken(text, timeline=self.timeline_uuid, kind=self._narration_kind())

    def _narration_kind(self) -> str:
        """Which of the three unspoken endings this turn produced (see `_unspoken`)."""
        if self._degraded:
            return "stuck"
        return "reserve" if self.harness.engine.reserve_used else "narration"

    def _wake_brief(self, *, query: str | None = None) -> str | None:
        """This wake's persistent operating brief — composed once, handed to every model call.

        This is what makes Turn 0 *persistent*: rather than a one-time onboarding seed that ages
        into the distant past of a long transcript, the brief rides ahead of the newest user turn
        on **every** model call, so the agent's standing context (how to operate, what tools it
        has, where it is, who it is) is always recent in the conversation.

        It is **ephemeral** (issue #275). The brief is a snapshot of a moment — the current time,
        this turn's step budget, the live dashboard — so writing it into the transcript stored a
        stale copy per wake: an agent read dozens of obsolete "current" times and spent step
        budgets as context, and paid for all of them on every later turn (47% of one agent's
        754 K-token context was ~66 near-identical briefs). So it is composed here and handed to
        `Session.send(brief=…)`, which shows it to the model and persists none of it.

        Composed lazily and at most once per wake — so a probe-only or idle wake never pays the
        live dashboard fetch — and `None` when onboarding is off.
        """
        if not self.onboard:
            return None
        if self._brief_composed:
            return self._brief
        self._brief_composed = True  # set first: a failed compose still won't retry-loop
        self._brief = self._compose_brief(query)
        if not self._brief:
            # Onboarding is on yet the brief composed empty (every part absent — deleted
            # prompts, a failed dashboard, no tools). Not an error, but log it: an agent
            # waking with no standing context is worth a breadcrumb, not a silent gap.
            _log.info("Persistent brief composed empty this wake; proceeding without it.")
        return self._brief

    def _compose_brief(self, query: str | None = None) -> str | None:
        """Compose the persistent brief: now + initialize + manifest + defects + safety + dashboard + memory + charter.

        The parts, in order (see `basecradle_harness._brief`): the **current-time anchor**
        (`_now_line` — the absolute "now" the model reasons every item's age against, fresh
        each wake since the brief is re-composed per wake), the **step-budget statement**
        (`render_budget` — the engine's per-turn budget N, stated once so the live per-step
        counter can stay terse), the provider-independent `initialize.md` operating guidance, the generated manifest of the agent's *active*
        tools, the **safe-by-default opt-out notice** (active MCP servers / policy-refused
        drop-ins — omitted when there are none), the live `dashboard.md` primer (fetched
        fresh each wake; a fetch failure degrades to omitting it, never breaking the wake),
        the memory provider's recalled
        **context** for this turn (its `context` hook — omitted when there is none or the
        provider's hook is a no-op), and the operator's `system-prompt.md` personality
        charter. Any part may be absent and the brief is composed from the rest.

        **Never break the wake.** `fetch_dashboard_md` already swallows its (network) failures
        and `_memory_context` swallows the provider's, but the prompt-file reads can also raise
        (a permission/IO error on `prompts/*.md` mid-wake) — so the whole composition is
        guarded too: any failure degrades to *no brief* and the wake carries on, the same
        invariant the dashboard fetch is held to.
        """
        try:
            return compose_brief(
                now=_now_line(),
                budget=render_budget(self.harness.engine.max_steps),
                initialize=prompt_text("initialize.md"),
                manifest=render_manifest(self._manifest_entries()),
                defects=render_defects(self.defect_notices),
                safety=render_safety(self.safety_notices),
                dashboard=fetch_dashboard_md(self.client),
                memory=self._memory_context(query),
                system_prompt=system_prompt_text(),
            )
        except Exception:  # noqa: BLE001 - the brief must never break the wake; degrade to none
            _log.warning(
                "Failed to compose the persistent brief; proceeding without it.", exc_info=True
            )
            return None

    def _memory_context(self, query: str | None) -> str | None:
        """The memory provider's recalled context for this turn, guarded — never breaks the wake.

        Calls the provider's `context` hook with the agent-scoped `MemoryScope` (the query is
        the incoming turn's text, so a relevance-ranked provider retrieves against it). A no
        provider, or a provider whose hook is a no-op, yields ``None`` and the brief omits the
        section. A hook that *raises* degrades to ``None`` — a memory backend hiccup must not
        drop the whole brief, let alone the wake — distinct from `_compose_brief`'s outer guard
        so a memory failure costs only the memory section, not the rest of the brief.
        """
        if self.memory_provider is None:
            return None
        try:
            recalled = self.memory_provider.context(
                MemoryScope(agent=self.me_uuid, timeline=self.timeline_uuid, query=query)
            )
        except Exception:  # noqa: BLE001 - a memory hook must never break the brief; degrade to none
            _log.warning(
                "Memory provider context() failed; omitting recalled memory.", exc_info=True
            )
            return None
        # DEBUG, never INFO: recall runs on every engaged wake, and a routine line per wake per
        # memory op would drown the signal the rest of this stream exists to carry. It is here
        # for the operator who turns HARNESS_LOG_LEVEL up to chase a memory question.
        _log.debug("memory %s", kv(op="recall", chars=len(recalled or "")))
        return recalled

    def _observe(self, user: str, assistant: str) -> None:
        """Hand a completed exchange to the memory provider's `observe` hook, guarded.

        Fires after each real exchange (never on a probe ack or a self-skip — those never
        reach here). A no provider, or one whose hook is a no-op (the default SQLite
        provider), does nothing. A hook that *raises* is swallowed: auto-capture is a
        best-effort side channel and must never break the wake or drop the reply that already
        posted.
        """
        if self.memory_provider is None:
            return
        try:
            self.memory_provider.observe(
                MemoryExchange(
                    user=user,
                    assistant=assistant,
                    scope=MemoryScope(agent=self.me_uuid, timeline=self.timeline_uuid),
                )
            )
        except Exception:  # noqa: BLE001 - a memory hook must never break the wake; swallow it
            _log.warning("Memory provider observe() failed; continuing.", exc_info=True)
            return
        _log.debug("memory %s", kv(op="observe", chars=len(user) + len(assistant)))

    def _remember_compaction(self, summary: str) -> None:
        """Write a compaction summary to durable memory — the answer to issue #276's requirement 7.

        **The gap this closes.** The memory seam's `observe` hook is handed the *dialogue* only
        (user text + the agent's reply — no briefs, no tool dumps), which is right: it is what keeps
        MemPalace's palace worth searching. But it means tool-driven work leaves no durable trace
        unless the agent happened to narrate it. Before compaction that is harmless — the tool
        results are still in the live transcript. At compaction it stops being harmless: the turns
        go, and with them any record that the work ever happened.

        So the boundary is where it is captured, because the boundary is where it would be lost. The
        summarizer is instructed to record the work first (`_context._SUMMARIZE_INSTRUCTION`), and
        that summary is written **here**, to whichever durable surface the bound provider has:

        - a provider with a `store` (the default SQLite one, whose `observe` is a no-op) → a write
          under an append-only, timestamped key, readable by the agent's own memory tool;
        - a provider without one (MemPalace, a pure middleware whose `store` is `None` by design) →
          its `observe` hook, which is what it durably keeps.

        What is *not* written is raw tool output: the point is a record of what was done, not a
        second copy of the bytes being dropped — that would re-create this very bloat inside memory.

        Its caller (`Compactor._remember`) guards it, like every memory hook: a failure here logs
        and the compaction still stands. Memory is best-effort; the transcript bound is not.
        """
        if self.memory_provider is None:
            return
        scope = MemoryScope(agent=self.me_uuid, timeline=self.timeline_uuid)
        store = getattr(self.memory_provider, "store", None)
        write = getattr(store, "write", None)
        if callable(write):
            key = f"compaction/{self.source}/{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            write(key, summary)
            _log.info("memory %s", kv(op="write", key=key, chars=len(summary)))
            return
        self.memory_provider.observe(
            MemoryExchange(user=_COMPACTION_OBSERVE_NOTE, assistant=summary, scope=scope)
        )
        _log.info("memory %s", kv(op="observe", source="compaction", chars=len(summary)))

    def _manifest_entries(self) -> list[tuple[str, str | None]]:
        """The ``(name, note)`` pairs for the tool manifest — the resolved set, else the registry.

        `from_env` threads the precise `ResolvedTools.manifest` (built-ins and notes included);
        a `WakeAgent` built directly (a test, or an embedder wiring its own `Harness`) falls
        back to the registered function tools by name, so the brief still names what the model
        can call even without the resolver's metadata.
        """
        if self.tool_manifest is not None:
            return list(self.tool_manifest)
        return [(tool.name, None) for tool in self.harness.tools]

    def _post(self, body: str, *, kind: str) -> object | None:
        """The **harness's own** post — degrading any SDK refusal instead of crashing.

        Since the Unspoken Channel (issue #293) there is exactly one caller: the NOC probe ack.
        That is not the agent talking — it is a machine contract (a signed nonce, acked
        model-free, so the fleet's seam heartbeats run token-free at rest), which is precisely
        why it survived an inversion that removed every other harness-authored post. **The agent's
        speech goes through the `messages` tool**, and nowhere else.

        Refusals still degrade rather than raise: a locked timeline raises `TimelineLockedError`
        from the SDK and, unguarded, would kill the whole wake (`exit 1`) — the live failure this
        guard was born from. Any `basecradle` error is caught, `None` is returned, and the caller
        carries on; the wake exits 0.

        **Both outcomes are logged**, because degrading gracefully is exactly what once made this
        failure invisible. A refusal is an **ERROR**; a success is an INFO naming the message
        created. `kind` says what it was, so a heartbeat ack never reads as the agent talking.
        """
        try:
            sent = self.timeline.messages.create(body=body)
        except BaseCradleError as error:
            _log.error(
                "post failed %s",
                kv(timeline=self.timeline_uuid, kind=kind, error=explain(error)),
            )
            return None
        _log.info(
            "posted %s",
            kv(
                message=_uuid_of(sent),
                timeline=self.timeline_uuid,
                kind=kind,
                chars=len(body),
            ),
        )
        return sent

    def _is_own(self, item: object) -> bool:
        """Whether an item was authored by this agent — the actor self-filter's test."""
        return item.user.uuid == self.me_uuid

    def _probe_nonce(self, carrier: str) -> str | None:
        """The nonce if `carrier` holds a valid signed NOC probe, else `None` (the short-circuit).

        Off unless `NOC_PROBE_SECRET` was provisioned (`probe_secret` set and non-empty):
        with no secret there is nothing to verify against, so every item falls through to
        the model, exactly as before this capability existed. A falsy secret (`None` *or*
        an empty string) keeps it off, so a blank env var never enables verification against
        an empty key. `carrier` is the field each reconciler reads the marker from — a
        message body, a task's instructions, a webhook payload, or an asset's description —
        so one signed-marker check serves all four seams (the NOC's message, task, webhook,
        and asset contracts).
        """
        if not self.probe_secret:
            return None
        return verify_probe(carrier, self.probe_secret)

    # --- the first wake: infer a high-water mark from the timeline ------------

    def _bootstrap(self, session: Session, trigger: str | None) -> list[object]:
        """The first wake for a timeline: there is no persisted mark yet.

        Read the recent backlog (newest-first, bounded by `context_messages`), then
        split it into the messages to *reply* to and the older ones to seed as
        *context* — so the agent answers the right thing while knowing what came
        before, the way a human scrolls up before speaking. The context is seeded
        only into a fresh transcript, so a restart that kept the transcript but lost
        the mark does not re-seed. Finally, record the true newest message as the
        mark, so the next wake is a normal incremental one.
        """
        recent = _recent(
            self.client.messages.filter(timeline=self.timeline_uuid), self.context_messages
        )
        if not recent:
            return []  # empty timeline: nothing to answer, nothing to mark
        split = self._bootstrap_split(recent, trigger)
        split = self._extend_split_over_unfinished(recent, split)
        to_reply = list(reversed(recent[: split + 1]))  # chronological; empty when split is -1
        context = recent[split + 1 :]  # older than the reply set, still newest-first
        if not _has_conversation(session):
            for message in reversed(context):  # oldest-first into history
                session.history.append(_as_turn(message, self.me_uuid))
        posted = self._respond(session, to_reply)
        # Close out the first wake by baselining the mark to `recent[0]` — but only when `_respond`
        # did *not* already advance it past `recent[0]` via a mid-wake arrival (issue #226). The
        # signal is exact and order-free: after `_respond`, the mark is either
        #   - a message that was in the initial `recent` read (a `to_reply` item, or None when every
        #     claim was lost to a concurrent/crashed wake or `to_reply` was empty), OR
        #   - a *later* arrival Loop 1/Loop 2 folded in and marked (never in `recent`).
        # So `mark is None or mark in recent` means "no arrival advanced it" → baseline to
        # `recent[0]`. This restores the old unconditional baseline for the empty/partial-claim
        # cases (without it a first wake whose only message's claim is orphaned by a crashed prior
        # wake would re-bootstrap forever, never advancing) while never regressing past a genuine
        # mid-wake arrival.
        # **The baseline may not sail past an undecided message** (issue #285). `_settle` has just
        # held the mark back from any `_PENDING` item — one a *live* concurrent wake is holding, or
        # a probe ack that failed to post — and baselining to `recent[0]` here would step straight
        # over it, hiding it from every future wake if that wake then died. That is precisely the
        # bug this issue fixes, and the bootstrap path is the one place it can sneak back in.
        #
        # Declining to baseline cannot livelock: the item is undecided *now*, but its owner either
        # settles it (the next bootstrap reads a settled claim, and the mark advances) or dies (the
        # next bootstrap finds it orphaned, recovers it, and the mark advances). Either way it
        # converges, and the cost of waiting is one more timeline read — never a model call.
        if any(disposition == _PENDING for _, disposition in self._ledger):
            return posted
        recent_uuids = {message.content.uuid for message in recent}
        mark = self.marks.get(self.timeline_uuid)
        if mark is None or mark in recent_uuids:
            self.marks.set(self.timeline_uuid, recent[0].content.uuid)
        return posted

    def _extend_split_over_unfinished(self, recent: list[object], split: int) -> int:
        """Pull the bootstrap's reply set back to cover any message an earlier wake never settled.

        **An unsettled claim is unfinished work, not history** (issue #285), and the bootstrap is
        the one path that would otherwise walk right over it. `_bootstrap_split` answers "what
        counts as unseen?" by *recency* — with no mark and nothing said yet, that means the newest
        message only, and everything older is seeded as context and baselined past. But a message a
        previous wake **claimed and never finished** is not old news: it is a peer waiting for an
        answer that a dead wake was supposed to give.

        Concretely, and this is reachable rather than theoretical: a wake claims M0 and dies; a new
        message M1 lands; the next wake has no mark, so it bootstraps, replies to M1 alone, and
        baselines the mark to M1 — putting M0 **permanently behind the cursor**, unanswered and now
        unreachable. Every other path is protected by the mark never passing an unsettled item, but
        the bootstrap's baseline is a *jump*, not a step, and it can leap the guard.

        So the reply set is extended to the oldest message in the window carrying an unsettled
        claim. `_absorb` then puts it through `_readmit` like any other — recovered if its owner is
        gone, skipped (and holding the mark) if its owner is alive. `recent` is newest-first, so a
        *larger* index is older and `max` widens the window. Settled claims are ignored, so an
        agent's answered history is never re-opened.
        """
        for index, message in enumerate(recent):
            claim = self.claims.read(self.timeline_uuid, message.content.uuid, kind=_MESSAGES)
            if claim is not None and not claim.settled:
                split = max(split, index)
        return split

    def _bootstrap_split(self, recent: list[object], trigger: str | None) -> int:
        """The newest-first index of the *oldest* backlog message to reply to.

        Resolves "what counts as unseen?" on the first wake, in priority order:

        1. **The triggering message**, if named and present — reply to it and
           anything newer. The precise answer; the router can pass it.
        2. **Everything since the agent last spoke here** — its own newest post is
           the natural high-water mark, which makes the poll→wake cutover lossless
           (it picks up exactly where polling left off) and answers a burst that
           arrived at once.
        3. **The newest message only** — when the agent has never spoken here (a
           fresh join), reply to the message that woke it without flooding history.

        Returns -1 when there is nothing newer than the agent's own latest post
        (e.g. it just spoke and nothing followed): reply to nothing, but still mark.
        """
        if trigger is not None:
            for i, message in enumerate(recent):
                if message.content.uuid == trigger:
                    return i
            # Trigger older than the fetched window — fall through to the defaults.
        for i, message in enumerate(recent):  # newest-first → first hit is the latest post
            if message.user.uuid == self.me_uuid:
                return i - 1  # reply to everything newer than our own last message
        return 0  # never spoken here → reply to the newest message only

    # --- replying ------------------------------------------------------------

    def _respond(self, session: Session, messages: list[object]) -> list[object]:
        """Engage the model on a wake's unseen messages as **one batched turn** — the #226 path.

        This is the single choke point for the message path — both the incremental
        (`_messages_since`) and bootstrap (`_bootstrap`) branches funnel their unseen set
        through here. Where the pre-#226 path looped `_act_on` per message (N unseen → N
        turns), it now gathers **all** unseen peer messages and seeds them as one turn (issue
        #226). Three coupled behaviors:

        1. **Many-to-one batch turn (`_absorb`).** Own posts are self-filtered (marked, not
           acted on); a recognized NOC probe — read from the message **body** — is acked
           token-free before the model; every remaining peer message is atomically claimed,
           marked, and collected into the batch. So the exactly-once machinery
           (`ClaimStore`/`MarkStore`, `kind=_MESSAGES`) moves to batch semantics: claim every
           message in the batch, advance the mark past the newest — but a *single* model turn
           reads them all.
        2. **Loop 1 — pace + settle (`_pace_and_settle`, AI-sender only).** Simulate a human
           reading the newest peer *AI* message; if a newer peer-AI message lands during that
           read, restart the wait on it and fold it in, so the reply reacts to the settled
           newest rather than a stale snapshot (the pre-#226 doublet defect). A human newest is
           never paced — an instant reply, exactly as before.
        3. **Loop 2 — mid-generation staleness (`_generate_settled`, all senders).** Generate
           against the batch; if any message (human or AI) arrives *during* generation, fold it
           in and rebuild, up to `max_builds` times — so the agent never posts a reply that was
           made stale by a message it hadn't yet seen.

        Only the message reconcile batches + paces; the asset/task/webhook reconcilers call
        `_act_on` directly and are deliberately out of scope (rare, naturally time-separated,
        or externally-sourced) — issue #289 tracks extending this to them.

        **Nothing here is marked seen until `_settle`** (issue #285). A wake that dies anywhere in
        this method leaves its messages `in-flight` and the mark unmoved, so the next wake finds
        them and recovers them. `_settle` runs on *every* return path, including the early one:
        it is what turns "this wake handled it" into a durable fact, and skipping it on any path
        would leave a healthy wake's own messages looking orphaned to the next one.

        **Recovery runs first, once, and finishes before the model is called at all** (issue #297).
        It reads the transcript as evidence, and the transcript is only trustworthy evidence while
        this wake has not yet written to it — so it cannot be something `_absorb` does *in passing*,
        because `_absorb` is re-entered from the read-pacer and from the mid-generation staleness
        loop, both of which run after a build. There it would be reasoning about a transcript its
        own wake had compacted, rolled back, or extended, and the answer it got would be about some
        other turn. One pass, up front, against a pristine transcript; `_absorb` then merely *reads*
        the verdicts it reached.
        """
        posted: list[object] = []
        self._cursor = self.marks.get(self.timeline_uuid)
        self._recover(session, messages)
        batch, probe_seen = self._absorb(session, messages, posted)
        if not batch:
            # Self-only / probe-only / empty: any probe acked, nothing to engage on. The model is
            # never engaged (and the brief never fetched) — but the mark still advances, so these
            # are not re-read forever.
            self._settle()
            return posted
        self._pace_and_settle(session, batch, posted, probe_seen)
        narration = self._generate_settled(session, batch, posted)
        # The turn is over. Whatever the agent wanted to say, it has already said — through the
        # `messages` tool, mid-turn, because it chose to (issue #293). What is left is its final
        # text, and that is **unspoken**: written to its log and its memory, never posted.
        self._unspoken(narration)
        # **Memory observes the turn unconditionally** — spoken or silent. The old guard
        # (`if reply.strip()`) is not merely unnecessary now, it would be *wrong*: under
        # silence-default the most valuable exchange to remember can be exactly the one that
        # produced no reply — a peer states a fact in passing, the agent rightly says nothing, and
        # the fact must still be recallable from another timeline a month later.
        self._observe(_render_batch(batch), narration)
        # The turn ran to completion, and its narration is journaled — which *is* the commit
        # record (issue #293). Commit the claims, then move the mark. A crash *before* this line
        # is the recoverable case, and the transcript says how far the turn got.
        self._settle()
        return posted

    def _absorb(
        self, session: Session, items: list[object], posted: list[object]
    ) -> tuple[list[object], bool]:
        """Fold a freshly-read chronological set into the reply batch, exactly once.

        Walks the items oldest→newest, applying the same per-item disposition `_act_on` does —
        just deferring the *model* engagement to a single batched reply:

        - **Own post** (the actor self-filter) → never acted on; its disposition is final at once
          (no claim needed — there is nothing to make exclusive, and nothing to lose).
        - **NOC probe** (a valid signed marker in the body) → ack token-free (no model call),
          final only on a successful ack (at-least-once, never claimed), and set `probe_seen`.
          The ack posts *here*, before any pacing sleep, so a probe's heartbeat stays sub-second
          even when a real peer message is paced alongside it.
        - **Peer message** → atomically claim it **in-flight** and collect it into the batch. A
          message that is *already* claimed goes to `_readmit`, which decides whether a live wake
          owns it, whether it is already settled, or whether the wake that owned it **died** — in
          which case it is recovered rather than silently dropped (issue #285).

        **What changed in #285, and why it is not just "move one line down".** This used to
        advance the persisted high-water mark here, at claim time — before the model was ever
        called. Any hard failure in between (provider down, process killed) therefore left the
        message marked seen and never answered, and *no future wake would ever look at it again*.
        The mark now advances only at `_settle`, once each item's fate is decided.

        That reordering has a trap in it, and it is why every item lands in `self._ledger` with a
        disposition instead of being marked in place: the mark is a **cursor**, not a set
        (`_messages_since` walks newest-first and stops at it). So marking a *newer* own post
        while an *older* peer message is still in flight would move the cursor **past** that
        message and hide it forever — the very bug this fixes, reintroduced by the fix. The rule
        is therefore: *the mark may never pass an item whose disposition is not final*, and
        `_settle` is the only place it moves.

        Returns `(peers, probe_seen)`. Reused by the initial gather and by every re-read
        (`_fetch_fresh`) during Loop 1 settle and Loop 2 rebuild, so the claiming/acking/ledger
        invariants are identical on every read.
        """
        peers: list[object] = []
        probe_seen = False
        for item in items:
            uuid = item.content.uuid
            # The in-wake read cursor. It advances over *everything we have looked at*, final or
            # not, so a re-read (`_fetch_fresh`) returns only what has since arrived. It is
            # deliberately separate from the persisted mark, which may not move yet.
            self._cursor = uuid
            if self._is_own(item):
                self._ledger.append((uuid, _FINAL))
                continue
            nonce = self._probe_nonce(item.content.body)
            if nonce is not None:
                probe_seen = True
                ack = self._post(ack_line(nonce), kind="probe-ack")
                if ack is not None:
                    posted.append(ack)
                self._ledger.append((uuid, _FINAL if ack is not None else _PENDING))
                continue
            if self.claims.claim(self.timeline_uuid, uuid, kind=_MESSAGES):
                self._ledger.append((uuid, _OURS))
                peers.append(item)
                continue
            disposition = self._readmit(session, item)
            self._ledger.append((uuid, disposition))
            if disposition == _OURS:
                peers.append(item)
        return peers, probe_seen

    def _fetch_fresh(self) -> list[object]:
        """Re-read the messages past this wake's cursor, chronological (may include self/probes).

        **The cursor is the in-wake one (`self._cursor`), not the persisted mark.** `_absorb`
        advances it past every message it looks at, so a re-read after absorbing returns only what
        has *since* arrived — which is how Loop 1 detects a message that landed during the
        read-pace and Loop 2 detects one that landed during generation.

        Before #285 this used the persisted mark, because `_absorb` advanced that mark in place.
        It no longer does (the mark cannot move past an in-flight item), so reading against the
        mark here would re-return the batch this wake is already holding on every re-read. That
        would still *appear* to work — the second `claim()` on our own message fails, so it would
        be skipped — but only by accident, and it would send every re-read down the recovery path.
        The two cursors are different things and now say so.
        """
        return _messages_since(
            self.client.messages.filter(timeline=self.timeline_uuid), self._cursor
        )

    def _readmit(self, session: Session, item: object) -> str:
        """An already-claimed message: is it owned, settled, or held by a wake that **died**?

        The claim's *phase* answers the first two, and it is why the record exists at all:

        - **settled** (`done` / `abandoned`, or a legacy empty claim) → this message's fate is
          decided. Final: skip it, and let the mark pass it.
        - **in-flight, owner alive** → a genuinely concurrent wake owns it. Skip — exactly the
          pre-#285 behavior — but **`_PENDING`, not final**: the mark must not pass a message
          another wake could still die holding.
        - **in-flight, owner gone** → the wake that took this message died mid-turn. It was
          recovered before this wake called the model (`_recover`), and this simply reports the
          verdict that pass reached. An orphan with **no** verdict was one recovery deliberately
          declined — it stays `_PENDING`, so the mark holds and the next wake tries again.

        This method used to do the recovering itself, and that was the bug (issue #297): `_absorb`
        calls it from the read-pacer and from the mid-generation staleness loop, both of which run
        *after* a build — so recovery could find itself reasoning about a transcript its own wake
        had just compacted or rolled back. It now looks up an answer instead of computing one, and
        the "is the evidence still pristine?" guard the old code needed is gone with the hazard.
        """
        uuid = item.content.uuid
        recovered = self._recovered.get(uuid)
        if recovered is not None:
            # **This wake's own verdict, and it is read before anything else.** `_recover` may have
            # *taken the claim over* — after which the claim file says `in-flight` owned by a wake
            # that is very much alive (us), and the orphan test below would correctly answer "not
            # orphaned" and hand back `_PENDING`, quietly dropping a message we had already decided
            # to answer. The claim is evidence about *other* wakes; it has nothing to tell us about
            # our own.
            return recovered
        claim = self.claims.read(self.timeline_uuid, uuid, kind=_MESSAGES)
        if claim is None:
            # The record vanished between our failed create and this read. Race for it; loser skips.
            return (
                _OURS if self.claims.claim(self.timeline_uuid, uuid, kind=_MESSAGES) else _PENDING
            )
        if claim.settled:
            return _FINAL
        if not self.claims.orphaned(claim):
            return _PENDING
        # An orphan with no verdict: recovery declined it (a live wake is recovering it, or we lost
        # the take-over race). Undecided, so the mark may not pass it; the next wake tries again.
        return _PENDING

    # --- recovery: one pass, before the model, against a pristine transcript ---

    def _recover(self, session: Session, items: list[object]) -> None:
        """Decide the fate of every orphaned message in this scan, **before the model runs.**

        The guarantee (issues #285, #293, #297): *at-least-once for the read, at-most-once for
        every side effect. The turn is the unit of commit.* A dead wake's message is re-driven only
        when the transcript **proves** nothing ran, and that proof is durable because the turn is
        persisted incrementally — most of all, the assistant turn naming a tool call reaches disk
        *before* that tool is dispatched (`_engine.run`). So:

        > a tool call absent from the transcript is a tool call that never ran.

        Four outcomes, tested in this order:

        1. **No turn carries this message** → the wake was killed after claiming it but before the
           model ever saw it. Nothing ran. **Re-drive.**
        2. **The turn reached its terminal narration** → the model *finished*. Everything it decided
           to say, it said itself, with its tools; everything it decided not to say was a decision.
           **Commit.** Tested before the tool check on purpose: a turn that ran tools *and* settled
           is a completed turn, not an interrupted one.
        3. **The turn issued tool calls** → the wake died mid-chain. Its calls may have fired, so a
           re-drive would fire them again — and it does not need one, because their results are on
           disk. **Resume** it: let the model finish the turn it started (`_resume_orphan`). The
           evidence is the *call*, not the result: a wake killed between the write and the POST is
           resumed, not re-driven, because the alternative is a message posted twice.
        4. **A turn, no calls, no narration** → it died inside the model call (the provider was
           down, the box was killed). Nothing ran. **Re-drive**, after excising the dead turn's
           inert residue, so the transcript does not accumulate a duplicate user turn per crash.

        Walking oldest-first is what makes a *batch* recover coherently: the dead wake's turn
        carried several messages, so the first of them resumes the turn, and the rest then find its
        narration and commit. The `_resumed` set is belt-and-braces on that — a turn is finished
        once.
        """
        for item in items:
            uuid = _uuid_of(item)
            if uuid is None:
                continue
            claim = self.claims.read(self.timeline_uuid, uuid, kind=_MESSAGES)
            if claim is None or claim.settled or not self.claims.orphaned(claim):
                continue  # never claimed (a peer's own post, a probe), already decided, or alive
            # A take-over that won its token and died before it could rewrite the claim leaves the
            # claim naming a wake that no longer holds the item. Judge the wake that really does.
            claim = self.claims.effective_owner(
                self.timeline_uuid, uuid, kind=_MESSAGES, claim=claim
            )
            if not self.claims.orphaned(claim):
                continue  # a live wake is already recovering this one; leave it to them
            self._recovered[uuid] = self._recover_orphan(session, item, claim)

    def _recover_orphan(self, session: Session, item: object, claim: Claim) -> str:
        """One orphan, classified from the transcript. See `_recover` for the four outcomes."""
        turn = _turn_of(session.history, item)
        if turn is None:
            return self._redrive(item, claim, "the wake died before the model ever saw it")

        work = _turn_work(session.history, turn)
        if _turn_narration(work) is not None:
            return self._committed(item)

        if any(message.tool_calls for message in work):
            already = self._resumed_outcome(turn)
            if already is not None:
                # This wake already finished (or failed to finish) this very turn for an older
                # message of the same batch. Never run it twice.
                return self._committed(item) if already == _FINAL else _PENDING
            return self._resume_orphan(session, item, turn, claim)

        # A turn with nothing in it: the model was called and never answered. Excise the residue
        # before re-driving — the user turn is on disk now (it was not, before incremental
        # persistence), so leaving it would stack a duplicate copy of the peer's message into the
        # transcript on every failed wake, forever. Safe precisely because we just proved the turn
        # issued no tool calls: there is nothing in it but a step note and a failure marker.
        disposition = self._redrive(item, claim, "the wake died inside the model call")
        if disposition == _OURS:
            session.excise(turn)
        return disposition

    def _resume_orphan(self, session: Session, item: object, turn: Message, claim: Claim) -> str:
        """The wake died mid-tool-chain. **Finish its turn** rather than re-running or dropping it.

        This is what replaced "abandon, loudly" (issue #297). The old code was right that a
        re-drive would re-fire the dead turn's tools, and right that dropping the peer was the
        least-bad remaining option — but only because it had no third one. It has one now: the
        turn's tool results are **on disk**, so the turn does not need re-running or abandoning. It
        needs *continuing*. Zero tools re-fire, because their results are already there.

        Two things happen before the model is handed the transcript:

        - **The dangling calls are settled.** `Session._load` has already given each one a result
          saying its outcome is unknown. For a **platform create** — and only for those four — that
          is replaced by *actually re-issuing the call*, under the deterministic idempotency key the
          dead wake minted for it (`_idempotency`): if it landed, the platform returns the original
          record and nothing duplicates; if it did not, it lands now. Either way the model gets a
          real answer instead of a shrug.
        - **Everything else keeps the shrug**, and that is not laziness — it is the only correct
          answer. No idempotency key can un-spend money at fal.ai, so a `generate_image` whose
          outcome is unknown is **not re-run**. The model is told plainly and left to decide, which
          is the Unspoken Channel's own stance applied to recovery: full visibility, never forcing.
          It can read the timeline and see for itself.

        **Only one wake may finish a turn.** The take-over CAS on this message is what says which —
        and because the dead wake's turn may have carried a *batch*, the rest of that batch is
        recognized here (`_resumed`) and committed when `_recover` reaches it, rather than resumed
        a second time.
        """
        uuid = item.content.uuid
        if not self.claims.reclaim(self.timeline_uuid, uuid, kind=_MESSAGES, owner=claim.wake):
            return _PENDING  # another recovering wake won the take-over; let it finish the turn
        _log.warning(
            "resuming %s",
            kv(
                message=uuid,
                timeline=self.timeline_uuid,
                handle=_handle_of(item),
                reason="the wake died mid-tool-chain; finishing the turn it started",
            ),
        )
        rendered = turn.content or _incoming_text(item)
        disposition = _FINAL
        try:
            self.keys.begin(session, timeline=self.timeline_uuid, anchor=_anchor_of(turn, item))
            self._reissue_interrupted_creates(session, turn)
            self.informer.arm(rendered)
            self._degraded = False
            try:
                narration = session.resume(turn, brief=self._wake_brief(query=rendered))
            except EngineError as error:
                narration = self._stuck_note(error)
            self._unspoken(narration)
            self._observe(rendered, narration)
            self.claims.commit(self.timeline_uuid, uuid, kind=_MESSAGES)
        except ProviderContextLengthError as error:
            # The transcript has outgrown the model's window and **cannot be compacted** — the
            # session already tried, and declined. Retrying next wake would fail identically, and
            # forever: a resume adds no new user turn, so it never gives the compactor the cut point
            # it lacked. That is not a delay, it is a permanent stall, with the mark pinned behind
            # this message and the peer never answered. So this is the one place the residual
            # at-most-once drop survives — rare, bounded, and **loud**, which is categorically
            # better than an invisible one.
            reason = f"the interrupted turn is over the model's context ceiling and cannot be compacted: {error}"
            self.claims.abandon(self.timeline_uuid, uuid, kind=_MESSAGES, reason=reason)
            _log.error(
                "dropped %s",
                kv(
                    message=uuid,
                    timeline=self.timeline_uuid,
                    handle=_handle_of(item),
                    reason=reason,
                ),
            )
            disposition = _FINAL  # abandoned is *settled*: the mark may pass, so nothing stalls
        except Exception:
            # Anything else — the provider is down, the box is unhappy. Not a message lost: this
            # wake's claim on it is in-flight, so the next wake finds it orphaned again and resumes
            # against a transcript carrying whatever this attempt managed. Do not take the whole
            # wake down over it; the other messages on this timeline still deserve an answer.
            _log.exception("Resuming the interrupted turn failed; the next wake will try again.")
            disposition = _PENDING
        self._resumed.append((turn, disposition))
        return disposition

    def _resumed_outcome(self, turn: Message) -> str | None:
        """How this wake's resume of `turn` went, or ``None`` if it has not resumed it.

        Keyed by **identity**, not equality: two turns carrying the same text are `==`, and
        resuming the wrong one twice is a second post the keys do not dedupe.
        """
        for resumed, outcome in self._resumed:
            if resumed is turn:
                return outcome
        return None

    def _reissue_interrupted_creates(self, session: Session, turn: Message) -> None:
        """Re-run each interrupted **platform create** under the key its dead wake minted for it.

        The one class of interrupted call that can be safely re-executed, because the platform
        recognizes the key and returns the original record rather than making a second one. The
        ordinal comes from `_idempotency.creates` — the *same* walk the live mint counts off, so the
        two cannot disagree, and disagreeing would mean minting a key the platform has never seen
        and posting the message twice.

        Everything else keeps its "outcome unknown" result. A failure here is fed back to the model
        as the tool's result, exactly as a live tool failure is: the turn continues either way.
        """
        # Name the turn *before* anything mints: the minter counts its ordinal off this turn's work,
        # and an unnamed turn reads as no work at all — every create in it minting ordinal 1.
        session.working_on(turn)
        for create in interrupted(_turn_work(session.history, turn), INTERRUPTED):
            with self.keys.reissue(create.kind, create.ordinal):
                text = self.harness.engine.run_tool(create.call.name, create.call.arguments)
            assert create.result is not None  # `interrupted` only yields calls that have one
            create.result.content = text
            _log.warning(
                "re-issued %s",
                kv(
                    tool=create.call.name,
                    kind=create.kind,
                    ordinal=create.ordinal,
                    timeline=self.timeline_uuid,
                ),
            )
        session.persist()

    def _redrive(self, item: object, claim: Claim, why: str) -> str:
        """Take over an orphaned claim so this wake answers the message. `_PENDING` if we lose it.

        The take-over is a compare-and-swap against the **owner we judged orphaned** (`claim.wake`),
        so a wake that lost the race — because a rival recovered it first, or because the "dead"
        owner was not the one still holding it — steps aside instead of answering in parallel.
        """
        uuid = item.content.uuid
        if not self.claims.reclaim(self.timeline_uuid, uuid, kind=_MESSAGES, owner=claim.wake):
            return _PENDING  # another recovering wake won the take-over; let it answer
        _log.warning(
            "re-driving %s",
            kv(message=uuid, timeline=self.timeline_uuid, handle=_handle_of(item), reason=why),
        )
        return _OURS

    def _committed(self, item: object) -> str:
        """The dead wake's turn **finished**. Commit its claim; there is nothing else to do.

        This is the Unspoken Channel's payoff in the recovery path (issue #293). The predecessor of
        this method held a saved reply and had to get it onto the timeline exactly once — which
        meant asking the platform whether it was already there, which meant comparing message
        bodies byte for byte, which meant a residual (small, real) false-match between two
        coincidentally identical replies, and a re-post path that could double-speak if the
        platform ever normalized a body.

        None of that survives the inversion, because the premise it rested on is gone: **the
        harness no longer holds a reply, because the harness no longer posts one.** A turn that
        reached its terminal narration is a turn the model finished — every message it decided to
        send, it sent itself, mid-turn, through its tools, and the platform has them. Every message
        it decided *not* to send is a decision, not a dropped write. So "did the reply land?" is not
        a question with an answer to hunt for; it is not a question at all.

        **A turn that settled on text having called no tools is a turn that chose silence**, and
        that is a decision too — including the rare one a mid-generation rebuild was about to
        replace (`_generate_settled` rolls a stale, speechless build back; a wake killed in the
        window between writing it and unmaking it leaves it standing). Committing it is right: the
        agent read the message and answered it with silence, and the newer message that triggered
        the rebuild drives its own turn, with this whole exchange in front of the model. What must
        never happen is the *reverse* — treating it as unfinished and running it again.
        """
        uuid = item.content.uuid
        _log.info(
            "recovered %s",
            kv(
                message=uuid,
                timeline=self.timeline_uuid,
                handle=_handle_of(item),
                action="turn-completed",
            ),
        )
        self.claims.commit(self.timeline_uuid, uuid, kind=_MESSAGES)
        return _FINAL

    def _settle(self) -> None:
        """Commit this wake's claims, then advance the mark — **the only place the mark moves.**

        Two steps, in this order, and the order is the guarantee:

        1. **Commit every message this wake owns.** Until now they were `in-flight`; a crash before
           this point leaves them that way, which is precisely what lets the next wake recover them
           rather than lose them.
        2. **Advance the high-water mark — but never past an item whose fate is undecided.** The
           mark is a *cursor* (`_messages_since` stops at it), so passing an item hides it from
           every future wake. Everything this wake handled is final by now, but an item owned by a
           *concurrent* wake (`_PENDING`) is not: if that wake then died, a mark past it would
           re-create the exact bug this fixes. So the mark stops at the newest item with an
           unbroken run of settled items behind it, and a foreign item simply holds it back until
           its owner settles it — after which a later wake sails past both. It converges, and the
           re-read it costs in the meantime is a timeline read, not a model call.
        """
        for uuid, disposition in self._ledger:
            if disposition == _OURS:
                self.claims.commit(self.timeline_uuid, uuid, kind=_MESSAGES)
        newest = None
        for uuid, disposition in self._ledger:
            if disposition == _PENDING:
                break  # a live wake still owns this one; the mark may not pass it
            newest = uuid
        if newest is not None:
            self.marks.set(self.timeline_uuid, newest)

    def _pace_and_settle(
        self, session: Session, batch: list[object], posted: list[object], probe_seen: bool
    ) -> None:
        """Loop 1 — read-pace the newest peer AI message, settling if a newer one lands mid-read.

        Simulate a human reading the newest peer *AI* message before replying, so an AI↔AI
        exchange is watchable and stays under the wake-breaker's trip line. Then re-read: if a
        newer peer message arrived *during* the sleep, fold it into the batch; if that newest
        arrival is itself a peer AI, restart the wait on it (a settling loop) — otherwise break
        (a human arrival means "respond now"). This is what stops the pre-#226 doublet: the
        reply reacts to the settled newest, not a snapshot taken before the sleep.

        Skipped entirely — an instant reply, exactly as before — when: pacing is disabled (the
        kill switch); a NOC probe was in the batch (its heartbeat must stay sub-second, and it
        was already acked in `_absorb`); or the newest peer message is a **human** (the
        ``kind == "ai"`` gate is the whole opt-in). New peer messages folded in here are marked
        and claimed by `_absorb`, so they are answered by this wake, not dropped.

        **Bounded by `max_builds` restarts.** In a 1-on-1 the settle converges in a step or two
        (the peer waits for this agent's reply). But with 3+ AI peers — or a peer whose own pacing
        is disabled — a new peer-AI message can land during *every* read window, so an uncapped
        settle would hold the wake (and the router's per-agent lock) indefinitely. The restart
        count is capped at `max_builds` (the same worst-case bound Loop 2 uses); once hit, the
        wake stops settling and proceeds to generate against the batch it has, folding any later
        arrivals through Loop 2 instead (and the rest drive the next wake). A WARNING is logged so
        a genuinely runaway room is visible.

        **Never crash the wake (B2).** A bad `created_at` or any hiccup in the pacer degrades to
        *no further delay* and proceeds with the current batch, never propagating — the same
        invariant the brief/dashboard/memory hooks are held to.
        """
        if probe_seen or not self.pacer.enabled:
            return
        newest = batch[-1]
        if getattr(newest.user, "kind", None) != "ai":
            return  # a human (or non-AI) newest → respond now, no read-pace
        try:
            restarts = 0
            while True:
                self.pacer.pace(newest)
                new_peers, _ = self._absorb(session, self._fetch_fresh(), posted)
                batch.extend(new_peers)
                if not (new_peers and getattr(new_peers[-1].user, "kind", None) == "ai"):
                    break  # settled: the newest is stable (or the latest arrival is a human)
                restarts += 1
                if restarts >= self.max_builds:
                    _log.warning(
                        "Read-pace settle hit the %d-restart cap for timeline %s (a runaway "
                        "multi-peer room?); proceeding to generate against the current batch.",
                        self.max_builds,
                        self.timeline_uuid,
                    )
                    break
                newest = new_peers[-1]  # a newer peer AI landed: restart the read on it
        except Exception:  # noqa: BLE001 - pacing must never break the wake; degrade to no delay
            _log.warning("Read-pacing failed; answering without further delay.", exc_info=True)

    def _generate_settled(self, session: Session, batch: list[object], posted: list[object]) -> str:
        """Loop 2 — run the turn against the batch, rebuilding if a message lands mid-generation.

        Optimistic concurrency (compare-and-swap) around the model call: generate against the
        batch snapshot, then re-read; if a peer message (human *or* AI — all senders count) has
        arrived *since* the batch's newest, fold it in and regenerate, up to `max_builds` times.
        The `max_builds`-th build stands **unconditionally** (no staleness check after it);
        messages that land during that final build are left **unseen** — not marked or claimed —
        so they drive the *next* wake rather than being lost.

        Returns the turn's **unspoken narration** — never a reply to post (issue #293).

        **A build that ran tools is never rolled back, and that is what keeps the agent from
        speaking twice.** The rollback (`del session.history[...]`) erases the transcript, not the
        world: a tool call's effects are real and irreversible — an image generated, a task
        scheduled, and now, most importantly, **a message posted**. Since speech is a tool call,
        rebuilding a build that spoke would say it again. So a build whose transcript span contains
        any tool turn is **committed**: it stands as-is and never rebuilds. Only a pure-text build —
        an agent that has said nothing yet — is eligible for a compare-and-swap rebuild, and it is
        free to rebuild precisely *because* it has said nothing. The guard predates the inversion
        (it was written for images) and inherits its most important job from it.

        The brief rides ephemerally with every build (composed once — see `_wake_brief`), so it
        never lands in the transcript and never needs rolling back. Intermediate (stale) pure-text
        builds *are* rolled back out of the transcript so only the surviving turn persists. Loop 2
        does **not** re-pace — Loop 1 already simulated the read, and re-pacing could stall a reply
        indefinitely in a chatty room. With pacing disabled it collapses to a single build (the
        pre-#226 single-shot behavior).
        """
        brief = self._wake_brief(query=_incoming_text(batch[-1]))
        base_len = len(session.history)  # rollback point: before any build
        # The anchor every key this turn mints is derived from: the batch's **oldest** message, and
        # oldest because it is the one thing a rebuild cannot change (a rebuild only ever appends
        # newer messages). Fixed here, before the first build, so every build of this turn — and
        # any later wake that has to finish it — mints the same keys.
        anchor = _uuid_of(batch[0])
        builds = 0
        while True:
            rendered = _render_batch(batch)
            # Re-arm per build, not per wake: a rebuild reads a *different* batch (a message landed
            # mid-generation), so "was I addressed, and have I acted?" has to be re-asked of the
            # text the model is actually about to read. Arming also resets the one-shot nudge, so a
            # rolled-back build's nudge does not silence the informer on the build that replaces it.
            self.informer.arm(rendered)
            narration = self._send_batch(session, rendered, brief, batch, anchor)
            builds += 1
            # **A compaction during the build invalidates `base_len`, so the build is committed.**
            # `Session.send` may compact (`_compact_if_needed`, and the over-length rescue), and a
            # compaction rewrites `history` *in place and shorter* — after which `base_len` no
            # longer points at this build's start, and both lines below quietly lie:
            # `history[base_len:]` comes back **empty**, so `used_tools` reads False even when the
            # build ran tools, and `del history[base_len:]` becomes a **no-op**, so the "rolled
            # back" build is still in the transcript when the next one is appended. Together they
            # rebuild a tool-using turn and **re-fire its side effects** — a second image generated
            # and posted for one request, or the same answer said twice — which is the one thing
            # this loop promises never to do. There is no index into a list that a rewrite of that
            # list preserves, so the honest move is not to compute a cleverer index: it is to stop
            # rebuilding. The build stands as-is (never rolled back, never re-run), and anything
            # that landed during it simply drives the next wake, exactly as `max_builds` already does.
            compacted = len(session.history) < base_len
            # A build that engaged tools has committed irreversible side effects — including,
            # since #293, anything it *said*. Standing on it (never rebuilding) is the only safe
            # move: a rollback+rebuild would re-fire them.
            used_tools = any(turn.role == "tool" for turn in session.history[base_len:])
            if not self.pacer.enabled or builds >= self.max_builds or used_tools or compacted:
                return narration  # pacing off → one build; the cap, tools, or a compaction → stand
            # Absorb the fresh read regardless — this marks any self post and, crucially, keeps a
            # NOC probe's ack sub-second even when it lands mid-generation (it is acked here, not
            # deferred to the next wake). Only peer messages fold in and trigger a rebuild.
            new_peers, _ = self._absorb(session, self._fetch_fresh(), posted)
            if not new_peers:
                return narration  # snapshot still current (any probe/self just handled) — stand
            # Stale: a peer message landed during generation. Fold it in, roll the stale build out
            # of the transcript, and regenerate against the current batch. Safe precisely because
            # this build ran no tools: it has said nothing, so there is nothing to unsay.
            #
            # The rollback **rewrites the file**, and since #297 it has to: the build being
            # discarded is already on disk, so an in-memory-only `del` would leave it there for a
            # wake killed before the replacement build persisted.
            batch.extend(new_peers)
            session.rollback(base_len)

    def _send_batch(
        self,
        session: Session,
        text: str,
        brief: str | None,
        batch: list[object],
        anchor: str | None,
    ) -> str:
        """Send the batch as one user turn, degrading the engine's step-cap to a graceful note.

        The batch counterpart of `_engage`'s model call; Loop 2 composes the brief once and passes
        it down, so a rebuild re-shows the same brief rather than re-composing (and re-fetching the
        live dashboard) per build. Messages carry no images, so this is text-only. `EngineError`
        (the `max_steps` cap) degrades to a short, honest note; other failures
        propagate to the entrypoint, which reports them cleanly.

        The turn is stamped with the **uuids it carries** and the minter is armed on the batch's
        anchor (issue #297): together they are how a wake that is killed here is recoverable — the
        uuids say which messages the model was shown, and the anchor is what any create in the turn
        keys against, identically on the wake that dies and the wake that finishes it.
        """
        self._degraded = False
        items = [uuid for uuid in (_uuid_of(message) for message in batch) if uuid is not None]
        if anchor is not None:
            self.keys.begin(session, timeline=self.timeline_uuid, anchor=anchor)
        else:
            self.keys.clear()  # nothing to anchor on — see `_engage`; never key to a stale anchor
        try:
            return session.send(text, brief=brief, items=items)
        except EngineError as error:
            return self._stuck_note(error)

    def _stuck_note(self, error: EngineError) -> str:
        """The canned narration for a degraded turn — and the WARNING that says one happened.

        `EngineError` reaches a wake only as the fallback-of-the-fallback: the step budget was
        spent *and* the engine's reserve summary (the self-authored progress report) failed or
        came back empty. The note is now **unspoken** like every other turn ending (issue #293):
        it goes to the journal and to the agent's own next turn, not to a peer — which is the
        honest destination for "the harness gave up on this turn", a fact about the machinery
        rather than an answer to anybody. To the operator this once looked like a perfectly
        ordinary wake, so the degradation is logged at WARNING, carrying the engine's own reason,
        wherever a wake catches it.
        """
        self._degraded = True  # labels this turn's unspoken text `kind=stuck`
        _log.warning("degraded %s", kv(timeline=self.timeline_uuid, reason=str(error)))
        return "I got stuck working through that and stopped before reaching an answer."


def _handle_of(item: object) -> str | None:
    """The speaker's handle, for a log line or a note. ``None`` when the object carries none."""
    return getattr(getattr(item, "user", None), "handle", None)


def _turn_of(history: list[Message], item: object) -> Message | None:
    """The user turn that carried `item` to the model, or ``None`` if none did.

    The recovery classifier's first question (issue #285): *did the dead wake ever actually put
    this message in front of the model?* The turn **carries the uuids it rendered** (`Message.items`,
    issue #297), so the answer is an exact lookup, and a message is only ever batched once, so at
    most one turn genuinely holds it.

    **It used to be a text match, and that was broken in two ways** — badly enough that the double
    post this whole issue is about was reachable in the field without any crash exotica. It compared
    the message's rendered line against the turn's content *split on newlines*, so a body containing
    a newline — a second paragraph, a list, a code block, the ordinary shape of a real message —
    produced a multi-line needle that can never be an element of a list of single lines. It matched
    nothing, the classifier concluded "the model never saw this", and it re-drove a turn that had
    already posted. And the needle was **peer-controlled**: a peer could paste another message's
    rendered line into their own body and steer the classifier at will. Neither is a thing you fix
    by escaping harder; the turn simply has to record what it carried.

    **The whole history is searched, not just the tail.** Two dead wakes in a row would leave the
    *older* one's message behind an intervening user turn, and a tail-only test would report "never
    sent" for a message that was in fact sent, generated for, and possibly answered.

    A transcript written before this version carries no uuids, so the old text match remains as the
    **legacy fallback** — repaired to compare whole blocks rather than lines, so it is at least
    right about multi-line bodies too. It heals itself: the next turn this session writes carries
    uuids.
    """
    uuid = _uuid_of(item)
    for message in history:
        if message.role == "user" and uuid is not None and uuid in message.items:
            return message
    rendered = _incoming_text(item)
    for message in history:
        if message.role == "user" and not message.items and _carries(message.content, rendered):
            return message
    return None


def _carries(content: str | None, rendered: str) -> bool:
    """Does this pre-#297 turn's rendered content contain `rendered` as a whole block?

    The legacy path only. "Whole block, never a substring" is the property the line-split was
    reaching for — a peer can quote an earlier message inside their own body, and a substring test
    would match the quoting turn — but a block may itself span lines, which is what the split got
    wrong. Anchoring on the newline boundaries keeps the guard and drops the bug.
    """
    if not content:
        return False
    if content == rendered:
        return True
    return (
        content.startswith(f"{rendered}\n")
        or content.endswith(f"\n{rendered}")
        or f"\n{rendered}\n" in content
    )


def _anchor_of(turn: Message, item: object) -> str:
    """The item a turn's idempotency keys are minted against — the **oldest** message it carried.

    The oldest, because it is the one thing about a batch that recovery cannot change: a rebuild or
    a re-drive only ever *appends* newer messages to it. Key the batch on its newest and the dead
    wake and the wake that recovers it would compute different keys for the same post, and the
    platform would dedupe nothing.

    Falls back to the item itself for a legacy turn that carries no uuids — which is exactly right,
    because such a turn is a single-message turn or nothing we can reason about anyway.
    """
    return turn.items[0] if turn.items else str(_uuid_of(item))


def _turn_work(history: list[Message], turn: Message) -> list[Message]:
    """What one turn produced — `_session.turn_work`, which is the *only* definition of it.

    Aliased rather than re-implemented, and that is load-bearing: the idempotency ordinal is counted
    once here, after a crash, and once in the live session while the turn runs, and the two numbers
    must be equal. Two functions that obviously mean the same thing is how they stop being equal.
    """
    return turn_work(history, turn)


def _turn_narration(work: list[Message]) -> str | None:
    """The terminal narration a turn settled on, or ``None`` if it never got that far.

    **This is the commit record** (issue #293): a turn that reached its final text *finished*, and a
    finished turn is never re-run — everything it meant to do, including everything it meant to say,
    it did.

    "Got that far" is **the last thing in the turn's work, and nothing weaker** (issue #297). The
    obvious form — scan backwards for any assistant turn with text and no tool calls — is wrong,
    because that shape is not the same as "the loop returned". `Engine.run` returns on
    ``not reply.tool_calls and not extend``, and **both** shipped turn hooks extend on exactly that
    shape: the mention informer nudges an agent that was addressed and did nothing, and the
    code-execution bridge harvests a run's output files and feeds their uuids back. A turn that was
    still being extended has an assistant text sitting *mid-work* — and reading it as the commit
    record would settle the claim, advance the mark, and never look at the message again. The
    `system` failure marker a raised run leaves behind (issue #244) is the same trap from the other
    side: it sits *after* the last assistant turn, and a turn that failed is one to finish, not one
    to file as done.

    So: the work must **end** on an assistant turn with real content and no tool calls. That is what
    the engine leaves behind when — and only when — it actually returned.
    """
    if not work:
        return None
    last = work[-1]
    if last.role == "assistant" and not last.tool_calls and (last.content or "").strip():
        return last.content
    return None


def _uuid_of(item: object) -> str | None:
    """A posted item's uuid for its log line, or ``None`` if the object carries none.

    Reached for through `getattr` rather than `item.content.uuid` because this is *only* a log
    line: an SDK whose create-response ever changed shape must not take the wake down over a
    breadcrumb. Missing → the field is simply omitted (`kv` drops it).
    """
    return getattr(getattr(item, "content", None), "uuid", None)


def _trigger_label(trigger: str | None, event: str | None, asset: str | None) -> str | None:
    """What fired this wake, as ``<kind>:<uuid>`` — or ``None`` when the router named nothing.

    The router wakes an agent with the timeline alone (each reconcile finds its own unseen
    items), so the usual answer *is* ``None`` and the field is simply omitted from the start
    line. When an invocation does name an item, the log says which kind it was — the same
    distinction the three optional args carry.
    """
    for kind, uuid in (("message", trigger), ("event", event), ("asset", asset)):
        if uuid:
            return f"{kind}:{uuid}"
    return None


def _render_batch(messages: list[object]) -> str:
    """Render a batch of unseen messages as one turn's text — the many-to-one reply input.

    Each message keeps its own `[created_at] handle: body` line (the same `_incoming_text`
    shape a single message got pre-#226), joined newest-last by newlines, so the model reads
    the batch as one contiguous stretch of conversation and answers all of it in one reply.
    A single-message batch renders identically to the pre-#226 per-message text.
    """
    return "\n".join(_incoming_text(message) for message in messages)


def _has_conversation(session: Session) -> bool:
    """Has this session said or heard anything yet (beyond a seeded system charter)?"""
    return any(turn.role in ("user", "assistant") for turn in session.history)


# A webhook event and an asset each share a message's newest-first ordering and
# `.content.uuid` shape, so the same high-water-mark scan walks all three. Aliased
# (not re-implemented) to keep them in lockstep.
_events_since = _messages_since
_assets_since = _messages_since


def _asset_marker_carrier(asset: object) -> str:
    """The asset field the NOC synthetic-probe marker rides in — the 4th seam's carrier.

    The asset analog of a message body / task instructions / webhook payload. The chosen
    field is the asset's **description**: the one free-text field a peer (or the NOC
    prober) controls on an upload, so a synthetic asset probe is minted by creating a tiny
    asset whose `description` is the `BCNOC1 …` marker. This is the contract the NOC's
    asset probe must agree with byte-for-byte. The field is optional, so an absent (or
    `None`) description reads as empty — an ordinary file with no description simply falls
    through to perception, never mistaken for a probe.
    """
    return getattr(asset.content, "description", None) or ""


def _now_line() -> str:
    """The current-time anchor injected at the head of every wake's brief.

    Renders the anchor line ``Current Time: 2026-06-21 17:09:49 UTC (+00:00, Sunday)`` —
    Title Case label, absolute UTC with an explicit ``+00:00`` offset, and the day-of-week,
    with no trailing period (it's a label, not a sentence) — **followed by** a one-line
    conversion instruction. The brief is re-composed and re-injected on every wake, so this is
    always current; the anchor is the reference every inbound item's ``[created_at]`` stamp is
    reasoned against. Accuracy rides on the host clock's NTP sync.

    The clock is UTC by design (every agent runs UTC on the box), but a bare UTC day/date was
    being parroted as if it were local — wrong whenever UTC has rolled to the next day but the
    asked-about locale hasn't (issue #180, live-confirmed on @jt). So the offset is now
    explicit and a conversion instruction rides with the anchor: when a peer names a locale,
    convert from UTC to that timezone first, because the local day can differ from the UTC day.
    """
    n = datetime.now(timezone.utc)
    anchor = f"Current Time: {n:%Y-%m-%d %H:%M:%S} UTC (+00:00, {n:%A})"
    instruction = (
        "This clock is UTC. For a question about a specific locale's date or time, convert "
        "from UTC to that timezone first — the local day can differ from the UTC day (e.g. "
        "US Central is UTC-5 in summer / UTC-6 in winter)."
    )
    return f"{anchor}\n{instruction}"


def _incoming_asset_text(asset: object) -> str:
    """A peer's posted file as the agent hears it: who shared what, and how to open it.

    The description fallback for the asset-perception path: used for media the wake cannot
    show inline (a non-image file, an unviewable/oversized image, or one whose download
    failed). Surfaces the asset's metadata (the shared `_describe` rendering) and points
    the agent at the tools that actually open it, rather than inlining the bytes — looking
    is a deliberate, on-demand step, the same discipline `view`/`read`/`listen` follow.

    The leading ``[created_at]`` stamp is the asset item's own timeline timestamp, read
    against the brief's `Current Time:` anchor so the model can reason about its age.
    """
    return (
        f"[{asset.created_at}] {asset.user.handle} posted a file to this timeline: "
        f"{_describe(asset)}. Use the assets tool to 'read' it (or 'view' an image / "
        "'listen' to audio) if you want to engage with it."
    )


# How much of a webhook payload to put in front of the model directly. A large body
# is truncated with a pointer to the webhook_events tool, which reads it in full —
# the same describe-don't-dump discipline the assets tool uses.
_MAX_EVENT_PAYLOAD = 8 * 1024


def _incoming_event_text(event: object) -> str:
    """An inbound webhook delivery as the agent hears it: what arrived, and its payload.

    A large payload is truncated with a pointer to the `webhook_events` tool (which
    reads the full headers and body by uuid), so a firehose delivery cannot blow up
    the model's context.

    The leading ``[created_at]`` stamp is the event item's own timeline timestamp, read
    against the brief's `Current Time:` anchor so the model can reason about its age.
    """
    content = event.content
    payload = content.payload
    if len(payload) > _MAX_EVENT_PAYLOAD:
        payload = (
            payload[:_MAX_EVENT_PAYLOAD].rstrip()
            + f"\n… (payload truncated — use the webhook_events tool to read {content.uuid} in full)"
        )
    return (
        f"[{event.created_at}] An inbound webhook was delivered to this timeline "
        f"(event {content.uuid}, endpoint {event.webhook_endpoint.uuid}, "
        f"content_type {content.content_type}). Decide whether and how to act on it. "
        f"Its payload:\n{payload}"
    )


def _activated_task_text(task: object) -> str:
    """A newly-activated task as the agent hears it: its instructions, to carry out now.

    A task is work the agent (or a peer) scheduled for this moment, so it is phrased as
    a directive to act, not a notification to consider — the activation *is* the cue.

    The leading ``[created_at]`` stamp is the task **item**'s own timeline timestamp. A task's
    Item is created when it *activates* (the platform's `Task::ActivationJob` inserts the Item
    in the same transaction it flips the task to activated), not when it was scheduled — so
    this reads ≈ now, consistent with every other inbound item, never "days ago." The
    complementary ``scheduled for {activate_at}`` text stays.
    """
    content = task.content
    return (
        f"[{task.created_at}] A task you scheduled has activated and is due now "
        f"(task {content.uuid}, scheduled for {content.activate_at}). "
        f"Carry out its instructions:\n{content.instructions}"
    )


# The PyPI distribution name for each AI_SDK value, so `--version` can report the *installed*
# vendor-SDK version. Only `openai` ships an adapter in Milestone 1; the others are listed so a
# later milestone's drift report names them without code change.
_SDK_DISTRIBUTIONS = {"openai": "openai", "xai-sdk": "xai-sdk", "anthropic": "anthropic"}


def _sdk_version(sdk: str) -> str | None:
    """The installed version of the vendor SDK named by ``AI_SDK``, or ``None`` if absent."""
    dist = _SDK_DISTRIBUTIONS.get(sdk, sdk)
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _version_string() -> str:
    """``basecradle-harness-wake <harness> · <sdk> SDK <ver>`` — both versions an upgrade tracks.

    Reports the harness version *and* the configured vendor SDK's installed version, so the
    fleet drift alarm catches a stale SDK as well as a stale harness (released ≠ deployed
    applies to the SDK pin too). With the SDK not installed it says so plainly — the same
    "no LLM, by design" signal the provider build gives.
    """
    sdk = (os.environ.get("AI_SDK") or "openai").strip().lower()
    version = _sdk_version(sdk)
    sdk_note = f"{sdk} SDK {version}" if version else f"{sdk} SDK not installed"
    return f"basecradle-harness-wake {__version__} · {sdk_note}"


def resolved_config() -> dict[str, object]:
    """The agent's live, *resolved* configuration + active tool set, as machine-readable ground truth.

    The introspection the fleet deployer (the NOC) reads to verify a deploy converged — by
    **ground truth, never self-report** (the basecradle#307 failure class, where a capability is a
    corpse while every version/health signal still reads green). It resolves through the **same
    code paths the running agent uses**: the validated ``(provider, sdk, surface)`` triple
    (`_config_from_env`, which hard-fails an unknown provider or an SDK-mismatched surface) and the
    active tool set after the full plugin/memory/MCP/locked-policy resolution (`_resolve_tools`) —
    so the output is what the agent would actually do, not a declared list.

    **Side-effect-free**, so it is safe to run repeatedly over SSH against a live agent home:

    - It does **not** build the model provider (no ``AI_API_KEY`` required, no client constructed),
      so ``ai_model`` is reported as the raw ``AI_MODEL`` env value (``None`` if unset) rather than
      raising the "AI_MODEL is required" the provider build would.
    - It does **not** run the config-home upgrade reconcile (which *writes* refreshed defaults), so
      it reports the overlay **as it is on disk**. The deploy order is install (which reconciles)
      then verify, so a verifier reads the post-install state.
    - Loading MCP drop-ins briefly starts each *configured* server to ``tools/list`` it (read-only
      at the protocol level — no ``tools/call`` — the same connection a wake makes, with failed
      servers self-excluding); with the default empty ``mcp/`` dir this is a no-op.

    The returned field set is an **additive contract** a downstream tool can depend on:

    - ``harness_version`` — the installed ``basecradle-harness`` version.
    - ``ai_provider`` / ``ai_sdk`` / ``ai_sdk_surface`` — the validated config triple (``surface``
      is ``""`` for a single-surface SDK that declares none).
    - ``ai_sdk_version`` — the installed version of the vendor SDK named by ``AI_SDK``, or ``None``
      if that SDK is not installed.
    - ``ai_model`` — the ``AI_MODEL`` env value, or ``None`` if unset.
    - ``active_profile`` — the deploy-selected policy profile, ``"locked"`` or ``"unlocked"``
      (`HARNESS_PROFILE`, fail-closed to ``"locked"``; issue #256). It governs the tool set below:
      under ``"unlocked"`` a policy-forbidden opted-in tool (e.g. ``shell``) appears in ``tools``;
      under ``"locked"`` the same tool appears in ``skipped``. Without it no automated surface
      could confirm a shell-class enablement's profile actually landed.
    - ``memory_provider`` — the **bound** memory backend (issue #269): ``sqlite`` (the default),
      ``mempalace``, or the ``module:Class`` path of a custom provider — read off the provider
      object `memory_provider_from_env` actually returned, **not** a re-read of
      ``HARNESS_MEMORY_PROVIDER`` (`describe_memory_provider`). Only the harness knows which store
      it binds (installed ≠ bound), and without this field the memory axis is invisible off-box: a
      MemPalace agent whose ``HARNESS_MEMORY_PROVIDER`` fell out of its ``agent.env`` would
      silently fall back to the default SQLite store — losing its palace — while every drift check
      still read green. The field makes the fallback *visible*.
    - ``memory_provider_version`` — the installed version of the package the harness *pins* for
      that provider (the ``mempalace`` extra today), else ``None``. ``None`` for the built-in
      ``sqlite`` store, which ships *inside* the harness (stdlib ``sqlite3``) and so has no
      separate pin — its version is ``harness_version`` above — and ``None`` for a custom
      provider, whose distribution the harness cannot honestly name (see
      `describe_memory_provider`). ``mempalace`` with ``None`` is a **defect signal**, not a
      shrug: bound (binding is lazy) while its extra is not installed.
    - ``tools`` — the resolved active **function** tool names, sorted.
    - ``builtins`` — the resolved active server-side **built-in** wire names, sorted (e.g. the
      Responses ``web_search``); a live capability the tool-set axis must count.
    - ``skipped`` — the names of plugins that did **not** activate, sorted — the auditable "why
      isn't this tool here?" trail (a diagnostic, not part of the active set).
    - ``opt_in_tools`` — the active **opt-in** (powerful) tools' source-file **stems**, sorted
      (issue #181). The stem is the unit the fleet inventory keys a powerful tool on, and is
      **not** 1:1 with the resolved ``tools``/``builtins`` names (one stem can fan out — e.g.
      ``code_execution`` → the ``code_interpreter`` built-in **+** the ``code_attach`` tool —
      and a name can differ from its stem — ``hear_audio`` → ``listen``). Reporting the stems
      lets the NOC's fleet-drift audit compare declared-vs-active inventory like-for-like,
      holding no stem→name map of its own. ``[]`` for a safe default config (no opt-in tool).
    - ``mcp_servers`` — the sorted **names** of the **configured** MCP servers, one per
      ``mcp/<name>.json`` drop-in (`load_mcp_configs`), independent of whether each one loaded
      this run (issue #261). The MCP-overlay analogue of ``opt_in_tools`` / ``active_profile``:
      the NOC's fleet-drift audit compares inventory-declared-vs-configured on this axis, both
      directions, holding **no** model of the harness's ``<server>__<tool>`` naming internals —
      the parallel-model anti-pattern the opt-in manifest retired. It reports the **configured**
      (on-disk) set, *not* the loaded set folded into ``tools``, so a transient upstream blip that
      self-excludes a server into ``skipped`` this run never reads as desired-state drift. Names
      only, never a server's ``env``/``headers`` (non-secret by contract, like the opt-in stems).
      ``[]`` for the default empty ``mcp/`` dir.
    - ``model_params`` — the operator's ``model_params.json`` object **verbatim** (``{}`` when the
      file is absent), the optional SDK call tuning (``reasoning``, ``temperature``, …) that
      `_provider_from_config` threads into every model call but nothing else introspects (issue
      #236). Non-secret by contract (secrets live in ``agent.env``), so emitting it is safe — it
      gives the NOC's drift audit and the capital's live-verify the wire-level proof that a tuning
      like ``reasoning: {effort: high}`` is actually loaded, which the other fields never showed.
    - ``model_params_stripped`` — the sorted keys in ``model_params`` that the active SDK's build
      **drops** as harness-owned collisions (plus ``extra_body`` on the SDKs that do not support
      it): the "warn and win" set (`resolved_model_params`). ``[]`` when nothing collides; the
      effective tuning the SDK receives is ``model_params`` minus these.
    - ``max_context_tokens`` — the operator's context-budget override (`HARNESS_MAX_CONTEXT_TOKENS`;
      issue #276), or ``null`` when unset. ``0`` means compaction is **disabled** on this agent, and
      that is the state worth being able to see from outside. The *resolved* ceiling is deliberately
      **not** reported: below the override it comes from the adapter's live `context_limit` capability
      (an API call against a provider this path never builds and holds no key for), so a number here
      would be a guess — and a guessed field in the file the drift audit trusts is worse than an
      absent one. The wake logs the resolved limit and its source (``context limit limit=… source=…``)
      on the run that actually resolves it.

    A malformed ``model_params.json`` makes this raise `ValueError` — the same failure a wake would
    hit, surfaced here at verify time (the caller turns it into a clean non-zero exit).
    """
    provider_name, sdk, surface = _config_from_env()
    profile_name, _policy = _profile_from_env()
    resolved, memory = _resolve_tools(provider_name, sdk, surface)
    memory_name, memory_version = describe_memory_provider(memory)
    model_params, stripped = resolved_model_params(sdk)
    return {
        "harness_version": __version__,
        "ai_provider": provider_name,
        "ai_sdk": sdk,
        "ai_sdk_surface": surface,
        "ai_sdk_version": _sdk_version(sdk),
        "ai_model": os.environ.get("AI_MODEL") or None,
        "active_profile": profile_name,
        "max_context_tokens": _max_context_tokens_from_env(),
        "memory_provider": memory_name,
        "memory_provider_version": memory_version,
        "tools": sorted(tool.name for tool in resolved.tools),
        "builtins": sorted(resolved.builtins),
        "skipped": sorted(name for name, _reason in resolved.skipped),
        "opt_in_tools": list(resolved.opt_in_stems),
        "mcp_servers": sorted({config.name for config in load_mcp_configs()}),
        "model_params": model_params,
        "model_params_stripped": stripped,
    }


def main(argv: list[str] | None = None) -> int:
    """The `basecradle-harness-wake` entrypoint: one wake, then exit.

    Exit code 0 on success including "nothing to do"; non-zero on a hard
    config/auth/credential failure, so the router can surface it.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-wake",
        description="Answer one BaseCradle timeline's unseen messages, then exit (router wake mode).",
    )
    # A token-free, model-free, timeline-free way to ask a deployed box "what version
    # are you actually running?" — the cheap probe a fleet drift-guard runs on-box to
    # catch a published-but-not-deployed release before it goes silent. The harness reaches
    # an LLM only through a vendor SDK, so an upgrade tracks **harness + vendor-SDK version
    # together** (issue #158): this reports both, e.g.
    # "basecradle-harness-wake 0.33.0 · openai SDK 2.43.0", so the drift alarm catches a stale
    # SDK as well as a stale harness. Exits 0 (argparse's built-in action).
    parser.add_argument(
        "--version",
        action="version",
        version=_version_string(),
        help="print the installed basecradle-harness and vendor-SDK versions, then exit.",
    )
    # The deploy verifier's ground-truth probe (issue #174): print the live, *resolved* config +
    # active tool set as machine-readable JSON, then exit — no timeline, no model call, no writes.
    # The NOC's `fleet-drift` check reads this to verify a converged deploy by ground truth rather
    # than self-report (the basecradle#307 failure class). JSON (pretty-printed, stable key order)
    # is both what a verifier parses and human-readable enough on its own, so it is the one format.
    parser.add_argument(
        "--resolved-config",
        action="store_true",
        help=(
            "print the resolved config + active tool set as JSON (ground truth for fleet drift), "
            "then exit. Read-only and timeline-free."
        ),
    )
    parser.add_argument(
        "--timeline",
        default=os.environ.get("BASECRADLE_TIMELINE"),
        help="uuid of the timeline to process (or set BASECRADLE_TIMELINE).",
    )
    parser.add_argument(
        "--message",
        default=os.environ.get("BASECRADLE_MESSAGE"),
        help="optional uuid of the triggering message; sharpens the first-wake bootstrap.",
    )
    parser.add_argument(
        "--event",
        default=os.environ.get("BASECRADLE_EVENT"),
        help=(
            "optional uuid of the triggering webhook event; the router passes it on a "
            "webhook_event.received wake so the first wake acts on that delivery."
        ),
    )
    parser.add_argument(
        "--asset",
        default=os.environ.get("BASECRADLE_ASSET"),
        help=(
            "optional uuid of the triggering asset; the router passes it on an "
            "asset.created wake so the first wake perceives that posted file."
        ),
    )
    args = parser.parse_args(argv)

    if args.resolved_config:
        # A read-only introspection — resolve the config + tool set (no model call, no writes) and
        # emit it as stable, pretty-printed JSON. A resolution error (an unknown AI_PROVIDER, an
        # SDK-mismatched AI_SDK_SURFACE) is the verifier's honest signal that the agent is
        # misconfigured, so it surfaces as a clean non-zero exit, never a raw traceback.
        try:
            print(json.dumps(resolved_config(), indent=2, sort_keys=True))
        except (HarnessError, ProviderError, BaseCradleError, ValueError, KeyError) as error:
            print(f"basecradle-harness-wake: {error}", file=sys.stderr)
            return 1
        return 0

    if not args.timeline:
        parser.error("a timeline uuid is required (--timeline or BASECRADLE_TIMELINE)")

    # Configure logging before the engine runs so the per-step ledger and the other INFO
    # breadcrumbs reach stderr (issue #248). Kept off the --version/--resolved-config paths
    # above: those exit before here, so their machine-readable stdout stays uncontaminated.
    _configure_logging()

    try:
        agent = WakeAgent.from_env(timeline=args.timeline)
        agent.wake(trigger=args.message, event_trigger=args.event, asset_trigger=args.asset)
    except (HarnessError, ProviderError, BaseCradleError, ValueError, KeyError) as error:
        # `_act_on` degrades the per-item failures (a locked-timeline post, the engine's
        # step cap) in flight, so a wake reaching a locked timeline still exits 0. A
        # BaseCradleError caught here is a harder failure — setup, an unreadable timeline —
        # which the router should see as a clean non-zero exit, never a raw traceback.
        #
        # It goes through the logger as an **ERROR** as well as to stderr: the bare print was
        # unleveled and unfilterable, so the harness's hardest failure — the wake that never ran
        # at all — was the one line a journald/Live-Tail severity filter could not find. The
        # print stays for a terminal run (where logging may be quieter than the operator's eyes).
        _log.error(
            "wake failed %s",
            kv(timeline=args.timeline, error=str(error), delivery=delivery_id()),
        )
        print(f"basecradle-harness-wake: {error}", file=sys.stderr)
        return 1
    return 0
