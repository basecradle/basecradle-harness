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
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

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
