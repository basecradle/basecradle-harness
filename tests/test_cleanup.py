"""The orphan-sweep GC: deleted timelines' on-box artifacts are purged, memory never is.

These tests pin the safety properties the whole feature rests on. The artifacts are laid
down with the **real** stores (`MarkStore`/`SeenStore`/`ClaimStore`/`WakeBreaker`) and the
real `quote(..., safe='')` transcript convention, so the enumeration round-trip is proven
against the actual writers — not a re-spelling of them. The platform is a small in-process
fake whose `timelines.get` is scripted per UUID, because classify is pure exception-handling
and needs no network.

The load-bearing case is **transient-error-keeps**: a platform outage must never be read as
"everything deleted" and trigger a mass purge. And `memory.db` + the MemPalace palace are
never enumerated, so a purge can never reach them.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from uuid import UUID

import pytest
from basecradle._exceptions import (
    APIConnectionError,
    BaseCradleError,
    ForbiddenError,
    NotAViewerError,
    NotFoundError,
    RateLimitedError,
)

from basecradle_harness import ClaimStore, MarkStore, SeenStore, WakeBreaker
from basecradle_harness import _cleanup as cleanup
from basecradle_harness._cleanup import (
    STRANDED_AFTER,
    enumerate_artifacts,
    main,
    prune_settled_claims,
    prune_stranded_temps,
    purge_one,
    sweep,
)
from basecradle_harness._install import config_home
from basecradle_harness._mempalace import _write_cli_config
from basecradle_harness._observability import RED, RESET
from basecradle_harness._report import BillingState
from basecradle_harness._session import Session
from basecradle_harness._token import write_token_to_env_file
from basecradle_harness._wake import Claim

# Real, well-formed UUIDv7 values (never `1111…` junk), per the test-data rule.
DELETED = "0190a8c1-7f3e-7c2a-9b1d-3e4f5a6b7c8d"
LIVE = "0190a8c2-1a2b-7d3e-8f4a-5b6c7d8e9f01"
FORBIDDEN = "0190a8c3-2b3c-7e4f-9a5b-6c7d8e9f0a12"
OTHER = "0190a8c4-3c4d-7f50-8b6c-7d8e9f0a1b23"


# --- fixtures: lay down artifacts exactly as the running agent would ----------------------


def _session_path(home: Path, uuid: str) -> Path:
    """The transcript path the harness writes — `sessions/{quote("timeline:<uuid>")}.json`."""
    return home / "sessions" / f"{quote(f'timeline:{uuid}', safe='')}.json"


def lay_down_all_kinds(home: Path, uuid: str) -> dict[str, Path]:
    """Write all six artifact kinds for one timeline, using the real stores. Returns the paths.

    Marks/seen/claims/breaker/billing go through the actual store classes so the on-disk encoding is
    whatever the agent really writes; the session transcript is written at the same
    `quote`-derived path `Harness._transcript_path` produces.
    """
    session = _session_path(home, uuid)
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text('[{"role": "user", "content": "my birthday is in June"}]')

    marks = MarkStore(home)
    marks.set(uuid, "0190a8c1-0000-7000-8000-000000000001")  # messages (flat layout)
    marks.set(uuid, "0190a8c1-0000-7000-8000-000000000002", kind="assets")
    marks.set(uuid, "0190a8c1-0000-7000-8000-000000000003", kind="webhook_events")

    seen = SeenStore(home)
    seen.add(uuid, "0190a8c1-0000-7000-8000-000000000004", kind="tasks")

    claims = ClaimStore(home)
    claims.claim(uuid, "0190a8c1-0000-7000-8000-000000000005", kind="messages")

    breaker = WakeBreaker(home)
    breaker.record_and_check(uuid)  # writes breaker/<uuid>.wakes

    billing = BillingState(home)
    billing.note_and_check(uuid)  # writes billing/<uuid>.blocked (issue #336)

    return {
        "session": session,
        "mark_messages": home / "marks" / f"{quote(uuid, safe='')}.txt",
        "mark_assets": home / "marks" / "assets" / f"{quote(uuid, safe='')}.txt",
        "mark_webhooks": home / "marks" / "webhook_events" / f"{quote(uuid, safe='')}.txt",
        "seen_tasks": home / "seen" / "tasks" / f"{quote(uuid, safe='')}.txt",
        "claims_dir": home / "claims" / "messages" / quote(uuid, safe=""),
        "breaker_wakes": home / "breaker" / f"{quote(uuid, safe='')}.wakes",
        "billing_blocked": home / "billing" / f"{quote(uuid, safe='')}.blocked",
    }


# --- the scripted platform fake -----------------------------------------------------------


class _FakeTimelines:
    def __init__(self, behavior: dict[str, object]) -> None:
        self._behavior = behavior
        self.calls: list[str] = []

    def get(self, uuid: str) -> object:
        self.calls.append(uuid)
        outcome = self._behavior[uuid]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # a truthy stand-in for the Timeline (success → keep)


class _FakeClient:
    def __init__(self, behavior: dict[str, object]) -> None:
        self.timelines = _FakeTimelines(behavior)


# --- enumeration round-trip ---------------------------------------------------------------


def test_enumerate_round_trips_the_store_encoding(tmp_path):
    paths = lay_down_all_kinds(tmp_path, DELETED)

    artifacts = enumerate_artifacts(tmp_path)

    assert set(artifacts) == {DELETED}
    found = set(artifacts[DELETED])
    # Every one of the six kinds (eight files/dirs) is attributed to the timeline.
    assert found == set(paths.values())


def test_enumerate_ignores_non_timeline_sessions(tmp_path):
    # A `github:` channel session is a different conversation, not a timeline artifact.
    other = tmp_path / "sessions" / f"{quote('github:pr-123', safe='')}.json"
    other.parent.mkdir(parents=True)
    other.write_text("[]")

    assert enumerate_artifacts(tmp_path) == {}


def test_enumerate_empty_home_is_empty(tmp_path):
    assert enumerate_artifacts(tmp_path) == {}


def test_enumerate_finds_a_transcript_stranded_by_a_killed_save(tmp_path):
    """A staged transcript left by a killed wake is swept — it holds the whole conversation.

    `Session._save` writes to `<name>.json.<pid>-<token>.tmp` and renames it into place (issue
    #297). On success the temp *becomes* the transcript; on an exception it is removed. But a
    process **killed** inside that window leaves it behind, holding every message of the
    conversation.

    A sweep that purged `…json` and walked past `…json.4213-9f2c.tmp` would report a deleted
    timeline as purged while leaving its transcript on the box forever — the exact outcome this
    module exists to prevent. So the temp is attributed to its timeline like any other artifact.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    stem = quote(f"timeline:{DELETED}", safe="")
    (sessions / f"{stem}.json").write_text("[]")
    stranded = sessions / f"{stem}.json.4213-9f2c.tmp"
    stranded.write_text('[{"role": "user", "content": "a secret the timeline no longer has"}]')

    assert stranded in set(enumerate_artifacts(tmp_path)[DELETED])


def test_enumerate_finds_a_mark_stranded_by_a_killed_write(tmp_path):
    """A mark staged by `MarkStore.set` and orphaned by a kill goes with its deleted timeline."""
    marks = MarkStore(tmp_path)
    marks.set(DELETED, OTHER)
    with _killed_at_publish():
        marks.set(DELETED, LIVE)
    (stranded,) = [p for p in (tmp_path / "marks").iterdir() if p.name.endswith(".tmp")]

    assert stranded in set(enumerate_artifacts(tmp_path)[DELETED])


def test_enumerate_ignores_a_stranded_save_from_another_channel(tmp_path):
    """…and the `github:` exclusion holds for the temp exactly as it does for the transcript."""
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    stem = quote("github:pr-123", safe="")
    (sessions / f"{stem}.json.4213-9f2c.tmp").write_text("[]")

    assert enumerate_artifacts(tmp_path) == {}


# --- the classify switch: only a clean 404 purges -----------------------------------------


def test_not_found_purges_all_kinds(tmp_path):
    paths = lay_down_all_kinds(tmp_path, DELETED)
    client = _FakeClient({DELETED: NotFoundError("gone")})

    summary = sweep(tmp_path, client)

    assert (summary.checked, summary.purged) == (1, 1)
    for path in paths.values():
        assert not path.exists(), f"{path} should have been purged"


def test_success_keeps(tmp_path):
    paths = lay_down_all_kinds(tmp_path, LIVE)
    client = _FakeClient({LIVE: SimpleNamespace(uuid=LIVE)})

    summary = sweep(tmp_path, client)

    assert (summary.checked, summary.purged, summary.kept) == (1, 0, 1)
    for path in paths.values():
        assert path.exists()


@pytest.mark.parametrize("error", [ForbiddenError("nope"), NotAViewerError("nope")])
def test_forbidden_keeps(tmp_path, error):
    paths = lay_down_all_kinds(tmp_path, FORBIDDEN)
    client = _FakeClient({FORBIDDEN: error})

    summary = sweep(tmp_path, client)

    assert (summary.purged, summary.kept_forbidden) == (0, 1)
    for path in paths.values():
        assert path.exists()


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError("network down"),
        RateLimitedError("slow down"),
        BaseCradleError("5xx"),
    ],
)
def test_transient_error_keeps_no_mass_purge_on_outage(tmp_path, error):
    """The load-bearing case: any non-404 defaults to keep, so an outage purges nothing."""
    paths = lay_down_all_kinds(tmp_path, DELETED)
    client = _FakeClient({DELETED: error})

    summary = sweep(tmp_path, client)

    assert (summary.purged, summary.skipped_transient) == (0, 1)
    for path in paths.values():
        assert path.exists()


def test_mixed_sweep_purges_only_the_deleted(tmp_path):
    deleted = lay_down_all_kinds(tmp_path, DELETED)
    live = lay_down_all_kinds(tmp_path, LIVE)
    client = _FakeClient({DELETED: NotFoundError("gone"), LIVE: SimpleNamespace(uuid=LIVE)})

    summary = sweep(tmp_path, client)

    assert (summary.checked, summary.purged, summary.kept) == (2, 1, 1)
    assert all(not p.exists() for p in deleted.values())
    assert all(p.exists() for p in live.values())


# --- memory is sacred ---------------------------------------------------------------------


def test_memory_db_and_palace_are_never_touched(tmp_path):
    lay_down_all_kinds(tmp_path, DELETED)
    # The memory store and a MemPalace palace dir — the hard "never touch" set.
    memory_db = tmp_path / "memory.db"
    memory_db.write_text("sqlite")
    (tmp_path / "memory.db-wal").write_text("wal")
    (tmp_path / "memory.db-shm").write_text("shm")
    palace = tmp_path / "palace" / "conversations"
    palace.mkdir(parents=True)
    (palace / "chunk.json").write_text("a peer told me its birthday here")

    sweep(tmp_path, _FakeClient({DELETED: NotFoundError("gone")}))

    assert memory_db.exists()
    assert (tmp_path / "memory.db-wal").exists()
    assert (tmp_path / "memory.db-shm").exists()
    assert (palace / "chunk.json").exists()


# --- idempotency / crash-safety -----------------------------------------------------------


def test_idempotent_rerun_is_a_noop(tmp_path):
    lay_down_all_kinds(tmp_path, DELETED)
    client = _FakeClient({DELETED: NotFoundError("gone")})

    first = sweep(tmp_path, client)
    # After the purge, the second run re-derives an empty artifact set — nothing left to check.
    second = sweep(tmp_path, _FakeClient({}))

    assert first.purged == 1
    assert (second.checked, second.purged) == (0, 0)


def test_one_undeletable_artifact_does_not_strand_the_rest(tmp_path, monkeypatch):
    """A purge failure on one timeline must not abort the sweep for every other orphan."""
    bad = lay_down_all_kinds(tmp_path, DELETED)
    good = lay_down_all_kinds(tmp_path, LIVE)  # sorts after DELETED, so it'd be stranded
    client = _FakeClient({DELETED: NotFoundError("gone"), LIVE: NotFoundError("gone")})

    # Make the first timeline's session unlink blow up with a permission error.
    real_unlink = Path.unlink
    blocked = bad["session"]

    def flaky_unlink(self, *args, **kwargs):
        if self == blocked:
            raise PermissionError("read-only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    summary = sweep(tmp_path, client)

    # Both were classified+purged (the failure was swallowed, not raised), and the
    # second timeline — which sorts after the failing one — was fully cleaned.
    assert summary.purged == 2
    assert all(not p.exists() for p in good.values())
    assert blocked.exists()  # the one un-deletable file remains, logged and stepped over


# --- the manual ops path ------------------------------------------------------------------


def test_purge_one_removes_unconditionally_without_a_client(tmp_path):
    paths = lay_down_all_kinds(tmp_path, DELETED)

    purged = purge_one(tmp_path, DELETED)

    assert set(purged) == set(paths.values())
    assert all(not p.exists() for p in paths.values())


def test_purge_one_on_unknown_uuid_is_empty(tmp_path):
    assert purge_one(tmp_path, OTHER) == []


# --- the CLI surface ----------------------------------------------------------------------


def test_main_requires_a_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        main([])


def test_main_errors_without_harness_home(monkeypatch, caplog):
    monkeypatch.delenv("HARNESS_HOME", raising=False)
    with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
        assert main(["--timeline", DELETED]) == 1

    # The sweep that never ran is a verdict, so its head is RED (issue #414) — and `cleanup failed`
    # is still a contiguous, greppable token inside it.
    line = next(m for m in (r.getMessage() for r in caplog.records) if "cleanup failed" in m)
    assert line.startswith(f"{RED}cleanup failed{RESET} ")


def test_main_timeline_purges_via_cli(tmp_path, monkeypatch):
    paths = lay_down_all_kinds(tmp_path, DELETED)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))

    assert main(["--timeline", DELETED]) == 0
    assert all(not p.exists() for p in paths.values())


# --- stranded temps (issue #526) ----------------------------------------------------------
#
# Every temp here is produced by the **real writer**, killed at the publish instant, so a writer
# that renames its temp breaks these tests rather than silently escaping the sweep.


class _Killed(BaseException):
    """The process dying at the publish instant: no handler after this point takes effect."""


@contextmanager
def _killed_at_publish():
    """Run a write as if SIGKILL landed between staging the temp and publishing it.

    `os.replace` and `os.link` die, and every unlink is a no-op for the duration, so the writers'
    own exception handlers (which remove the temp on an ordinary error) cannot clean up: exactly
    the case where nothing runs a handler, which is the only way a temp is ever stranded.
    """

    def die(*args, **kwargs):
        raise _Killed

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", die)
        mp.setattr(os, "link", die)
        mp.setattr(os, "unlink", lambda *args, **kwargs: None)
        mp.setattr(Path, "unlink", lambda self, missing_ok=False: None)
        with pytest.raises(_Killed):
            yield


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """The temp pass looks in `~/.mempalace` and beside the env file, so neither may be the real one."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("BASECRADLE_ENV_FILE", raising=False)


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def _strand_every_kind(home: Path) -> dict[str, Path]:
    """One stranded temp of every kind, each left by its real writer killed mid-publish."""
    before: set[Path] = set()

    def left_behind(folder: Path) -> Path:
        (found,) = set(folder.iterdir()) - before
        before.add(found)
        return found

    session_file = _session_path(home, LIVE)
    session = Session(f"timeline:{LIVE}", engine=None, path=session_file)
    session.persist()  # the transcript itself, published normally
    before.update(session_file.parent.iterdir())
    with _killed_at_publish():
        session.persist()
    stranded = {"session": left_behind(session_file.parent)}

    marks = MarkStore(home)
    marks.set(LIVE, OTHER, kind="assets")
    before.update(marks._path(LIVE, "assets").parent.iterdir())
    with _killed_at_publish():
        marks.set(LIVE, DELETED, kind="assets")
    stranded["mark"] = left_behind(marks._path(LIVE, "assets").parent)

    claims = ClaimStore(home)
    claims.claim(LIVE, OTHER, kind="messages")
    folder = home / "claims" / "messages" / quote(LIVE, safe="")
    before.update(folder.iterdir())
    with _killed_at_publish():
        claims.commit(LIVE, OTHER, kind="messages")
    stranded["claim_write"] = left_behind(folder)
    with _killed_at_publish():
        claims.claim(LIVE, DELETED, kind="messages")
    stranded["claim_link"] = left_behind(folder)
    with _killed_at_publish():
        claims.reclaim(LIVE, OTHER, kind="messages", owner="0f1e2d3c4b5a69788796a5b4c3d2e1f0")
    stranded["claim_takeover"] = left_behind(folder)
    with _killed_at_publish():
        claims.advance_pruned_through(LIVE, kind="messages", mark=UUID(OTHER))
    stranded["watermark"] = left_behind(folder)

    env_file = config_home() / "agent.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("BASECRADLE_TOKEN=bc_live_old\n")
    before.update(env_file.parent.iterdir())
    with _killed_at_publish():
        write_token_to_env_file("bc_live_0123456789abcdef", str(env_file))
    stranded["env"] = left_behind(env_file.parent)

    mempalace = Path.home() / ".mempalace"
    mempalace.mkdir()
    with _killed_at_publish():
        _write_cli_config(mempalace, mempalace / "config.json", {"palace_path": "/p"})
    stranded["mempalace"] = left_behind(mempalace)

    return stranded


def test_every_stranded_temp_kind_is_removed_once_its_writer_is_gone(tmp_path, monkeypatch):
    stranded = _strand_every_kind(tmp_path)
    assert all(path.exists() for path in stranded.values())  # the fixture is real
    for path in stranded.values():
        _age(path, STRANDED_AFTER + 60)
    monkeypatch.setattr(cleanup, "_process_alive", lambda pid: False)

    removed = prune_stranded_temps(tmp_path)

    assert set(removed) == set(stranded.values())
    assert not any(path.exists() for path in stranded.values())
    # What each write was *for* is untouched.
    assert _session_path(tmp_path, LIVE).exists()
    assert (config_home() / "agent.env").read_text() == "BASECRADLE_TOKEN=bc_live_old\n"
    assert ClaimStore(tmp_path).read(LIVE, OTHER, kind="messages").phase == "in-flight"
    assert MarkStore(tmp_path).get(LIVE, kind="assets") == OTHER  # the mark it would have replaced


def test_a_temp_younger_than_the_floor_is_kept_whatever_its_pid(tmp_path, monkeypatch):
    stranded = _strand_every_kind(tmp_path)
    for path in stranded.values():
        _age(path, STRANDED_AFTER - 60)
    monkeypatch.setattr(cleanup, "_process_alive", lambda pid: False)

    assert prune_stranded_temps(tmp_path) == []
    assert all(path.exists() for path in stranded.values())


def test_a_temp_whose_writer_is_still_alive_is_kept_however_old(tmp_path):
    # The pid-stamped temps name this very test process, which is alive.
    stranded = _strand_every_kind(tmp_path)
    for path in stranded.values():
        _age(path, STRANDED_AFTER * 24)

    removed = prune_stranded_temps(tmp_path)

    assert stranded["session"].exists()
    assert stranded["mark"].exists()
    assert stranded["claim_write"].exists()
    assert stranded["watermark"].exists()
    # The unstamped ones have only their age to go on, and it is far past any live write.
    assert set(removed) == {
        stranded[kind] for kind in ("claim_link", "claim_takeover", "env", "mempalace")
    }


def test_the_env_temp_is_found_beside_an_env_file_outside_the_config_home(tmp_path, monkeypatch):
    env_file = tmp_path / "elsewhere" / "agent.env"
    env_file.parent.mkdir()
    env_file.write_text("BASECRADLE_TOKEN=bc_live_old\n")
    monkeypatch.setenv("BASECRADLE_ENV_FILE", str(env_file))
    with _killed_at_publish():
        write_token_to_env_file("bc_live_0123456789abcdef", str(env_file))
    (temp,) = [p for p in env_file.parent.iterdir() if p != env_file]
    _age(temp, STRANDED_AFTER + 60)

    assert prune_stranded_temps(tmp_path / "harness-home") == [temp]
    assert env_file.read_text() == "BASECRADLE_TOKEN=bc_live_old\n"


def test_nothing_but_a_named_temp_in_its_own_place_is_ever_touched(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "_process_alive", lambda pid: False)
    folder = tmp_path / "claims" / "messages" / quote(LIVE, safe="")
    folder.mkdir(parents=True)
    bystanders = [
        folder / f".{OTHER}.takeover.0f1e2d3c4b5a69788796a5b4c3d2e1f0",  # a take-over token
        folder / f"{OTHER}.claim",
        tmp_path / "sessions" / "notes.tmp",
        tmp_path / "marks" / f"{LIVE}.txt.bak",
        config_home() / "agent.env.bak",
        Path.home() / ".mempalace" / "palace" / ".config.json.0f1e2d3c.tmp",  # beneath the top
        Path.home() / ".mempalace" / "config.json",
    ]
    for path in bystanders:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        _age(path, STRANDED_AFTER * 24)

    assert prune_stranded_temps(tmp_path) == []
    assert all(path.exists() for path in bystanders)


def test_the_sweep_removes_stranded_temps_on_live_timelines_and_counts_them(
    tmp_path, monkeypatch, caplog
):
    stranded = _strand_every_kind(tmp_path)
    for path in stranded.values():
        _age(path, STRANDED_AFTER + 60)
    monkeypatch.setattr(cleanup, "_process_alive", lambda pid: False)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        summary = sweep(tmp_path, _FakeClient({LIVE: object()}))

    assert summary.kept == 1 and summary.purged == 0  # the timeline is live and kept
    assert summary.stranded_temps == len(stranded)
    assert not any(path.exists() for path in stranded.values())
    assert f"removed {len(stranded)} stranded temp(s)" in str(summary)
    assert "removed stranded temp" in caplog.text


def test_a_pid_no_process_could_hold_and_an_unreadable_dir_never_stop_the_pass(tmp_path):
    folder = tmp_path / "claims" / "messages" / quote(LIVE, safe="")
    folder.mkdir(parents=True)
    impossible = folder / f"{OTHER}.claim.{'9' * 30}.tmp"  # `os.kill` would raise, not answer
    impossible.write_text("{}")
    _age(impossible, STRANDED_AFTER + 60)
    locked = tmp_path / "claims" / "assets"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert prune_stranded_temps(tmp_path) == [impossible]
    finally:
        locked.chmod(0o700)


# --- settled claims are pruned on live timelines (issue #526) ------------------------------
#
# The prune is only as safe as the refusal behind it: `ClaimStore.claim` must never hand a
# pruned item back to a wake, whatever the mark says by then.

# Items of one timeline in uuid order, oldest first.
X = [f"0190a8c2-1a2b-7d3e-8f4a-00000000000{n}" for n in range(6)]


def _settled(home: Path, uuid: str, *, kind: str = "messages") -> Path:
    store = ClaimStore(home)
    assert store.claim(LIVE, uuid, kind=kind)
    store.commit(LIVE, uuid, kind=kind)
    return store._path(LIVE, kind, uuid)


def _claim_files(home: Path, kind: str = "messages") -> set[str]:
    folder = home / "claims" / kind / quote(LIVE, safe="")
    return {path.name for path in folder.iterdir()}


def test_a_settled_claim_at_or_below_the_mark_is_pruned_and_can_never_be_won_again(tmp_path):
    store = ClaimStore(tmp_path)
    below = _settled(tmp_path, X[0])
    store.claim(LIVE, X[1], kind="messages")  # in flight, below the mark: never touched
    legacy = store._path(LIVE, "messages", X[2])
    legacy.write_text("")  # a pre-#285 claim, which reads as done
    unreadable = store._path(LIVE, "messages", X[3])
    unreadable.write_text("{not json")  # read as done so as never to re-drive, but not deleted
    at = _settled(tmp_path, X[4])
    above = _settled(tmp_path, X[5])
    token = below.parent / f".{X[0]}.takeover.0f1e2d3c4b5a69788796a5b4c3d2e1f0"
    token.write_text("{}")
    MarkStore(tmp_path).set(LIVE, X[4])

    assert prune_settled_claims(tmp_path) == 3

    assert not any(path.exists() for path in (below, legacy, at, token))
    assert unreadable.exists() and above.exists()
    assert store.read(LIVE, X[1], kind="messages").phase == "in-flight"
    assert store.pruned_through(LIVE, kind="messages") == UUID(X[4])
    # The whole point: a wake that listed X[0] before it was pruned still cannot act on it.
    late = ClaimStore(tmp_path)
    assert late.claim(LIVE, X[0], kind="messages") is False
    assert late.read(LIVE, X[0], kind="messages").phase == "done"


def test_the_watermark_never_moves_backward_with_the_mark(tmp_path):
    # A long wake settling an older ledger can write its older mark back over a newer one. The
    # items between the two were final when the watermark passed them, and still are.
    marks = MarkStore(tmp_path)
    _settled(tmp_path, X[0])
    _settled(tmp_path, X[2])
    marks.set(LIVE, X[3])
    assert prune_settled_claims(tmp_path) == 2
    marks.set(LIVE, X[1])  # the regression: X[2] is above the mark again

    prune_settled_claims(tmp_path)

    assert ClaimStore(tmp_path).pruned_through(LIVE, kind="messages") == UUID(X[3])
    assert ClaimStore(tmp_path).claim(LIVE, X[2], kind="messages") is False  # re-listed, refused


def test_an_item_above_the_watermark_is_claimed_as_ever(tmp_path):
    _settled(tmp_path, X[0])
    MarkStore(tmp_path).set(LIVE, X[0])
    prune_settled_claims(tmp_path)

    assert ClaimStore(tmp_path).claim(LIVE, X[1], kind="messages") is True


def test_nothing_is_pruned_or_written_without_a_mark(tmp_path):
    path = _settled(tmp_path, X[0])

    assert prune_settled_claims(tmp_path) == 0
    assert path.exists()
    assert ".pruned-through" not in _claim_files(tmp_path)


def test_a_task_claim_is_pruned_once_the_seen_set_holds_it(tmp_path):
    handled = _settled(tmp_path, X[0], kind="tasks")
    unrecorded = _settled(tmp_path, X[1], kind="tasks")  # settled, but not yet in the seen-set
    SeenStore(tmp_path).add(LIVE, X[0], kind="tasks")

    assert prune_settled_claims(tmp_path) == 1

    assert not handled.exists() and unrecorded.exists()
    assert ".pruned-through" not in _claim_files(tmp_path, "tasks")  # the seen-set is the record
    assert ClaimStore(tmp_path).claim(LIVE, X[0], kind="tasks") is False


def test_an_unreadable_watermark_is_never_overwritten_and_prunes_nothing(tmp_path, caplog):
    path = _settled(tmp_path, X[0])
    (path.parent / ".pruned-through").write_text("garbage")
    MarkStore(tmp_path).set(LIVE, X[3])

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert prune_settled_claims(tmp_path) == 0

    assert path.exists()
    assert (path.parent / ".pruned-through").read_text() == "garbage"
    assert "left messages claims" in caplog.text


def test_the_sweep_prunes_settled_claims_on_a_live_timeline_and_counts_them(tmp_path):
    _settled(tmp_path, X[0])
    _settled(tmp_path, X[1])
    MarkStore(tmp_path).set(LIVE, X[1])

    summary = sweep(tmp_path, _FakeClient({LIVE: object()}))

    assert summary.kept == 1
    assert summary.pruned_claims == 2
    assert "pruned 2 settled claim(s)" in str(summary)


def test_two_sweeps_racing_never_lower_the_watermark(tmp_path, monkeypatch):
    """A manual `--sweep` over the timer's: the slower one must not write its older mark back.

    S1 reads the watermark and stalls; S2 advances it further and would prune under it. Without
    the lock, S1 then replaces it with its own lower value, uncovering what S2 pruned.
    """
    _settled(tmp_path, X[0])  # the claims dir exists
    store = ClaimStore(tmp_path)
    real = ClaimStore.pruned_through
    s1_read, go = threading.Event(), threading.Event()
    stalled = []

    def slow_first_read(self, timeline, *, kind):
        value = real(self, timeline, kind=kind)
        if not stalled:
            stalled.append(True)
            s1_read.set()
            go.wait(5)
        return value

    monkeypatch.setattr(ClaimStore, "pruned_through", slow_first_read)
    s1 = threading.Thread(
        target=store.advance_pruned_through,
        args=(LIVE,),
        kwargs={"kind": "messages", "mark": UUID(X[1])},
    )
    s2 = threading.Thread(
        target=ClaimStore(tmp_path).advance_pruned_through,
        args=(LIVE,),
        kwargs={"kind": "messages", "mark": UUID(X[3])},
    )
    s1.start()
    assert s1_read.wait(5)
    s2.start()
    s2.join(0.5)  # with the lock, S2 waits for S1 here; without it, S2 finishes first
    go.set()
    s1.join(5)
    s2.join(5)

    assert real(store, LIVE, kind="messages") == UUID(X[3])


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 0 does not stop root from reading")
def test_a_watermark_that_cannot_be_read_never_stops_a_claim(tmp_path, caplog):
    _settled(tmp_path, X[0])
    watermark = ClaimStore(tmp_path)._folder(LIVE, "messages") / ".pruned-through"
    watermark.write_text(X[3])
    watermark.chmod(0)  # e.g. written by a root-run sweep under umask 077
    try:
        with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
            assert ClaimStore(tmp_path).claim(LIVE, X[1], kind="messages") is True
    finally:
        watermark.chmod(0o600)
    assert "treating" in caplog.text and "as unpruned" in caplog.text


def test_a_garbage_watermark_is_read_as_not_covered_and_said_out_loud(tmp_path, caplog):
    _settled(tmp_path, X[0])
    (ClaimStore(tmp_path)._folder(LIVE, "messages") / ".pruned-through").write_text("garbage")

    with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
        assert ClaimStore(tmp_path).claim(LIVE, X[1], kind="messages") is True
    assert "as unpruned" in caplog.text


def test_a_refused_claim_is_never_in_flight_even_for_an_instant(tmp_path, monkeypatch, caplog):
    # A kill between an in-flight link and the `done` rewrite would leave an orphan on an item
    # settled long ago, which recovery could re-drive. A covered item is refused *before* linking.
    _settled(tmp_path, X[0])
    MarkStore(tmp_path).set(LIVE, X[0])
    prune_settled_claims(tmp_path)
    linked: list[str] = []
    real_link = os.link

    def watching(src, dst, *args, **kwargs):
        linked.append(Path(src).read_text())
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", watching)
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert ClaimStore(tmp_path).claim(LIVE, X[0], kind="messages") is False

    assert linked and all('"in-flight"' not in record for record in linked)
    assert "claim refused" in caplog.text and "reason=covered" in caplog.text


def test_a_stale_recoverer_cannot_bring_a_pruned_orphan_back(tmp_path):
    """The prune removes the take-over token that used to stop a recoverer who judged too early.

    D died holding X[0]. A judged it orphaned, then stalled. B took it over, finished it, and the
    sweep pruned the claim and B's token. A's take-over must now lose, not re-drive X[0].
    """
    dead = ClaimStore(tmp_path, wake="d" * 32)
    assert dead.claim(LIVE, X[0], kind="messages")
    recoverer = ClaimStore(tmp_path, wake="b" * 32)
    assert recoverer.reclaim(LIVE, X[0], kind="messages", owner="d" * 32)
    recoverer.commit(LIVE, X[0], kind="messages")
    MarkStore(tmp_path).set(LIVE, X[0])
    assert prune_settled_claims(tmp_path) == 1

    stale = ClaimStore(tmp_path, wake="a" * 32)
    assert stale.reclaim(LIVE, X[0], kind="messages", owner="d" * 32) is False

    assert stale.read(LIVE, X[0], kind="messages") is None
    assert _claim_files(tmp_path) == {".pruned-through"}  # its losing token did not stay behind


def test_a_stale_recoverer_backs_off_a_prune_that_was_cut_short(tmp_path):
    # The sweep removed the token and died before the claim: the claim is still there, `done`.
    dead = ClaimStore(tmp_path, wake="d" * 32)
    dead.claim(LIVE, X[0], kind="messages")
    recoverer = ClaimStore(tmp_path, wake="b" * 32)
    recoverer.reclaim(LIVE, X[0], kind="messages", owner="d" * 32)
    recoverer.commit(LIVE, X[0], kind="messages")
    folder = recoverer._folder(LIVE, "messages")
    (folder / f".{X[0]}.takeover.{'d' * 32}").unlink()

    assert (
        ClaimStore(tmp_path, wake="a" * 32).reclaim(LIVE, X[0], kind="messages", owner="d" * 32)
        is False
    )
    assert recoverer.read(LIVE, X[0], kind="messages").phase == "done"


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 0 does not stop root from reading")
def test_a_seen_set_that_cannot_be_read_never_stops_a_task_claim(tmp_path, caplog):
    SeenStore(tmp_path).add(LIVE, X[0], kind="tasks")
    seen = SeenStore(tmp_path)._path(LIVE, "tasks")
    seen.chmod(0)
    try:
        with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
            assert ClaimStore(tmp_path).claim(LIVE, X[1], kind="tasks") is True
    finally:
        seen.chmod(0o600)
    assert "as unpruned" in caplog.text


def test_the_second_look_after_the_link_catches_a_prune_that_landed_in_between(
    tmp_path, monkeypatch
):
    # The lock makes this unreachable; the check is what stands behind a wake running unlocked
    # (no `fcntl`, or a filesystem that refuses a lock). So the prune is simulated *between* the
    # pre-check and the link, the way it would land without the lock.
    path = _settled(tmp_path, X[0])
    store = ClaimStore(tmp_path)
    real_publish = ClaimStore._publish

    def pruned_just_before_the_link(self, target, uuid, record):
        if record.phase == "in-flight":
            (target.parent / ".pruned-through").write_text(X[0])
            target.unlink()
        return real_publish(self, target, uuid, record)

    monkeypatch.setattr(ClaimStore, "_publish", pruned_just_before_the_link)

    assert store.claim(LIVE, X[0], kind="messages") is False
    assert store.read(LIVE, X[0], kind="messages").phase == "done"
    assert path.exists()


def test_a_prune_takes_the_tokens_before_the_claim(tmp_path, monkeypatch):
    # Cut short between the two, a prune must leave the claim, so the next sweep finishes it.
    claim = _settled(tmp_path, X[0])
    token = claim.parent / f".{X[0]}.takeover.{'d' * 32}"
    token.write_text("{}")
    MarkStore(tmp_path).set(LIVE, X[0])
    order: list[str] = []
    real_unlink = Path.unlink

    def recording(self, missing_ok=False):
        order.append(self.name)
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", recording)
    prune_settled_claims(tmp_path)

    assert order.index(token.name) < order.index(claim.name)


def test_a_sweep_waits_for_a_claim_between_its_check_and_its_link(tmp_path, monkeypatch):
    """The exclusive lock the prune takes is what stops it landing inside `claim`."""
    _settled(tmp_path, X[0])
    store = ClaimStore(tmp_path)
    real_covered = ClaimStore._covered
    inside, go = threading.Event(), threading.Event()

    def paused(self, timeline, uuid, *, kind):
        answer = real_covered(self, timeline, uuid, kind=kind)
        if uuid == X[1] and not inside.is_set():
            inside.set()
            go.wait(5)
        return answer

    monkeypatch.setattr(ClaimStore, "_covered", paused)
    claimer = threading.Thread(target=store.claim, args=(LIVE, X[1]), kwargs={"kind": "messages"})
    claimer.start()
    assert inside.wait(5)
    sweep_done = threading.Event()
    prune = ClaimStore(tmp_path).prune_settled
    sweeper = threading.Thread(
        target=lambda: (prune(LIVE, kind="messages", covered=lambda uuid: True), sweep_done.set())
    )
    sweeper.start()

    assert not sweep_done.wait(0.5)  # blocked behind the claim's shared lock
    go.set()
    claimer.join(5)
    sweeper.join(5)
    assert sweep_done.is_set()


def test_a_token_a_killed_recoverer_left_on_a_pruned_item_is_removed(tmp_path):
    _settled(tmp_path, X[0])
    MarkStore(tmp_path).set(LIVE, X[0])
    prune_settled_claims(tmp_path)
    folder = ClaimStore(tmp_path)._folder(LIVE, "messages")
    stray = folder / f".{X[0]}.takeover.{'d' * 32}"
    stray.write_text("{}")  # won, then killed before it could back off and remove it
    live = folder / f".{X[1]}.takeover.{'d' * 32}"
    live.write_text("{}")  # above the watermark: not provably dead, so kept

    prune_settled_claims(tmp_path)

    assert not stray.exists() and live.exists()


def test_a_recoverer_backs_off_a_claim_something_live_now_holds(tmp_path):
    # D died holding X[0]; a stale recoverer won D's token after the prune removed it, but a
    # live wake (another pid, this very moment) holds the claim. It is not an orphan any more.
    dead = ClaimStore(tmp_path, wake="d" * 32)
    dead.claim(LIVE, X[0], kind="messages")
    holder = ClaimStore(tmp_path, wake="c" * 32)
    holder._write(
        holder._path(LIVE, "messages", X[0]),
        Claim(phase="in-flight", pid=os.getppid(), wake="c" * 32, at=time.time()),
    )

    assert (
        ClaimStore(tmp_path, wake="a" * 32).reclaim(LIVE, X[0], kind="messages", owner="d" * 32)
        is False
    )
    assert holder.read(LIVE, X[0], kind="messages").wake == "c" * 32


def test_the_package_imports_where_there_is_no_fcntl():
    probe = "import sys; sys.modules['fcntl'] = None; import basecradle_harness; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.stdout.strip() == "ok", result.stderr
