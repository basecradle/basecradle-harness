"""The token lifecycle: persist a minted token, and self-heal a dead one (issue #104).

Two behaviours, both offline (respx stands in for the platform, no network):

- `write_token_to_env_file` rewrites only the ``BASECRADLE_TOKEN`` line of the agent's
  env file, in place, leaving every other secret untouched — the persistence half.
- `SelfHealingBaseCradle` re-mints from email + password and retries once when a call
  comes back 401 — the recovery half. With no credentials to re-mint from, it fails
  loudly instead of silently spinning.

The fictional cast is the fixed fiction: Nova Digital (``nova``, AI) is the agent.
"""

import os
import stat

import httpx
import pytest
import respx
from basecradle import AuthenticationError

from basecradle_harness._token import (
    SelfHealingBaseCradle,
    write_token_to_env_file,
)

BC_URL = "https://basecradle.com"
OLD_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
NEW_TOKEN = "bc_uat_9mZ2pQ7rT4vW1xY6sLkN3bHcJ8dGfEa0"


def unauthorized():
    """A 401 problem+json the SDK maps to `UnauthorizedError` (a dead/invalid token)."""
    return httpx.Response(
        401,
        json={
            "code": "unauthorized",
            "status": 401,
            "title": "Unauthorized",
            "detail": "The access token is missing or invalid.",
        },
    )


def minted():
    """A 201 from `POST /session` — a freshly minted token."""
    return httpx.Response(201, json={"token": NEW_TOKEN, "start_here": None})


@pytest.fixture
def platform():
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        yield router


# --- write_token_to_env_file: surgical, in-place persistence -----------------


def test_replaces_the_token_line_in_place_leaving_other_secrets(tmp_path):
    """Only the BASECRADLE_TOKEN value changes; EMAIL/PASSWORD/comments are untouched."""
    env = tmp_path / "agent.env"
    env.write_text(
        "# the agent's secrets\n"
        "ANTHROPIC_API_KEY=sk-ant-keepme\n"
        f"BASECRADLE_TOKEN={OLD_TOKEN}\n"
        "BASECRADLE_EMAIL=nova@example.com\n"
        "BASECRADLE_PASSWORD=correct-horse\n"
    )

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == (
        "# the agent's secrets\n"
        "ANTHROPIC_API_KEY=sk-ant-keepme\n"
        f"BASECRADLE_TOKEN={NEW_TOKEN}\n"
        "BASECRADLE_EMAIL=nova@example.com\n"
        "BASECRADLE_PASSWORD=correct-horse\n"
    )


def test_preserves_an_export_prefix_and_indentation(tmp_path):
    """A shell-sourced `export BASECRADLE_TOKEN=…` keeps its prefix when rewritten."""
    env = tmp_path / "harness-test.env"
    env.write_text(f"  export BASECRADLE_TOKEN={OLD_TOKEN}\nexport OPENAI_API_KEY=sk-test\n")

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == (
        f"  export BASECRADLE_TOKEN={NEW_TOKEN}\nexport OPENAI_API_KEY=sk-test\n"
    )


def test_appends_a_token_line_when_absent_mirroring_export_style(tmp_path):
    """A file with only EMAIL+PASSWORD gains a token line in the file's own style."""
    env = tmp_path / "agent.env"
    env.write_text(
        "export BASECRADLE_EMAIL=nova@example.com\nexport BASECRADLE_PASSWORD=correct-horse\n"
    )

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == (
        "export BASECRADLE_EMAIL=nova@example.com\n"
        "export BASECRADLE_PASSWORD=correct-horse\n"
        f"export BASECRADLE_TOKEN={NEW_TOKEN}\n"
    )


def test_appends_bare_when_the_file_uses_no_export(tmp_path):
    """A bare-dotenv file gets a bare token line, not an `export` one."""
    env = tmp_path / "agent.env"
    env.write_text("BASECRADLE_EMAIL=nova@example.com\n")

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == (f"BASECRADLE_EMAIL=nova@example.com\nBASECRADLE_TOKEN={NEW_TOKEN}\n")


def test_creates_a_missing_file_at_mode_600(tmp_path):
    """No file yet → create it with just the token line, owner-only (it holds a credential)."""
    env = tmp_path / "agent.env"

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == f"BASECRADLE_TOKEN={NEW_TOKEN}\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_preserves_an_existing_files_permissions(tmp_path):
    """Rewriting an existing file keeps its mode (the atomic replace copies it)."""
    env = tmp_path / "agent.env"
    env.write_text(f"BASECRADLE_TOKEN={OLD_TOKEN}\n")
    env.chmod(0o640)

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert stat.S_IMODE(env.stat().st_mode) == 0o640


def test_only_the_first_token_line_is_rewritten(tmp_path):
    """Defensive: a stray second token line is left alone (we rewrite the first)."""
    env = tmp_path / "agent.env"
    env.write_text(f"BASECRADLE_TOKEN={OLD_TOKEN}\nBASECRADLE_TOKEN=second-stray-line\n")

    write_token_to_env_file(NEW_TOKEN, str(env))

    assert env.read_text() == (
        f"BASECRADLE_TOKEN={NEW_TOKEN}\nBASECRADLE_TOKEN=second-stray-line\n"
    )


def test_unset_env_file_warns_and_does_not_write(capsys):
    """No BASECRADLE_ENV_FILE → nothing is written, and a clear warning is logged."""
    write_token_to_env_file(NEW_TOKEN, None)

    warning = capsys.readouterr().err
    assert "BASECRADLE_ENV_FILE" in warning
    assert "next wake will mint again" in warning


# --- SelfHealingBaseCradle: re-mint and retry once on a 401 ------------------


def test_self_heals_on_401_then_remints_persists_and_retries(platform, tmp_path, monkeypatch):
    """A dead token → one re-mint (POST /session), token swapped + persisted, call retried."""
    monkeypatch.setenv("BASECRADLE_TOKEN", OLD_TOKEN)  # restored at teardown, not leaked
    env = tmp_path / "agent.env"
    env.write_text(f"BASECRADLE_TOKEN={OLD_TOKEN}\n")
    ping = platform.get("/ping").mock(
        side_effect=[unauthorized(), httpx.Response(200, json={"ok": True})]
    )
    login = platform.post("/session").mock(return_value=minted())

    client = SelfHealingBaseCradle(
        OLD_TOKEN,
        email="nova@example.com",
        password="correct-horse",
        session_name="nova-harness",
        env_file=str(env),
    )
    try:
        result = client.request("GET", "/ping")
    finally:
        client.close()

    assert result == {"ok": True}
    assert login.called and login.call_count == 1  # exactly one re-mint
    assert ping.call_count == 2  # original 401, then the retry
    # The new token is adopted on the live client, mirrored to the env, and persisted.
    assert client.token == NEW_TOKEN
    assert client._client.headers["Authorization"] == f"Bearer {NEW_TOKEN}"
    assert os.environ["BASECRADLE_TOKEN"] == NEW_TOKEN
    assert env.read_text() == f"BASECRADLE_TOKEN={NEW_TOKEN}\n"


def test_a_persistent_401_retries_only_once_then_raises(platform, tmp_path, monkeypatch):
    """If the re-minted token also 401s, it raises — one re-mint per call, never a loop."""
    monkeypatch.setenv("BASECRADLE_TOKEN", OLD_TOKEN)  # restored at teardown, not leaked
    env = tmp_path / "agent.env"
    env.write_text(f"BASECRADLE_TOKEN={OLD_TOKEN}\n")
    ping = platform.get("/ping").mock(side_effect=[unauthorized(), unauthorized()])
    login = platform.post("/session").mock(return_value=minted())

    client = SelfHealingBaseCradle(
        OLD_TOKEN, email="nova@example.com", password="correct-horse", env_file=str(env)
    )
    try:
        with pytest.raises(AuthenticationError):
            client.request("GET", "/ping")
    finally:
        client.close()

    assert login.call_count == 1  # re-minted exactly once
    assert ping.call_count == 2  # the original and the single retry, no more


def test_401_without_credentials_fails_loudly_and_never_mints(platform):
    """Token-only and dead, no email/password → a clear error, and login is never called."""
    platform.get("/ping").mock(return_value=unauthorized())
    login = platform.post("/session").mock(return_value=minted())

    client = SelfHealingBaseCradle(OLD_TOKEN)  # no credentials to re-mint from
    try:
        with pytest.raises(ValueError, match="BASECRADLE_EMAIL"):
            client.request("GET", "/ping")
    finally:
        client.close()

    assert not login.called  # nothing to mint from — never attempted


def test_a_rejected_remint_fails_loudly_not_with_a_raw_401(platform, tmp_path, monkeypatch):
    """A dead token whose credentials were rotated → a clear error, not a confusing 401."""
    monkeypatch.setenv("BASECRADLE_TOKEN", OLD_TOKEN)  # restored at teardown, not leaked
    env = tmp_path / "agent.env"
    env.write_text(f"BASECRADLE_TOKEN={OLD_TOKEN}\n")
    platform.get("/ping").mock(return_value=unauthorized())
    # The re-mint login itself is rejected (the stored password no longer works).
    platform.post("/session").mock(
        return_value=httpx.Response(
            401,
            json={"code": "invalid_credentials", "status": 401, "title": "Invalid credentials"},
        )
    )

    client = SelfHealingBaseCradle(
        OLD_TOKEN, email="nova@example.com", password="rotated-away", env_file=str(env)
    )
    try:
        with pytest.raises(ValueError, match="BASECRADLE_EMAIL / BASECRADLE_PASSWORD"):
            client.request("GET", "/ping")
    finally:
        client.close()

    # The dead token was not written over the env file by a failed mint.
    assert env.read_text() == f"BASECRADLE_TOKEN={OLD_TOKEN}\n"


def test_a_non_token_401_is_not_self_healed(platform):
    """A 401 that is not a dead-bearer-token signal (e.g. invalid_signature) is left alone."""
    platform.get("/ingest").mock(
        return_value=httpx.Response(
            401, json={"code": "invalid_signature", "status": 401, "title": "Invalid signature"}
        )
    )
    login = platform.post("/session").mock(return_value=minted())

    client = SelfHealingBaseCradle(OLD_TOKEN, email="nova@example.com", password="correct-horse")
    try:
        with pytest.raises(AuthenticationError):  # surfaced as-is, not turned into a re-mint
            client.request("GET", "/ingest")
    finally:
        client.close()

    assert not login.called  # a fresh token can't fix a signature problem — never minted


def test_a_healthy_call_never_remints(platform):
    """The common path: a live token just works, and login is never touched."""
    platform.get("/ping").mock(return_value=httpx.Response(200, json={"ok": True}))
    login = platform.post("/session").mock(return_value=minted())

    client = SelfHealingBaseCradle(OLD_TOKEN, email="nova@example.com", password="correct-horse")
    try:
        assert client.request("GET", "/ping") == {"ok": True}
    finally:
        client.close()

    assert not login.called
    assert client.token == OLD_TOKEN
