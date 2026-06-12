"""The agent's token lifecycle: reuse a live token, mint only when missing or dead.

The platform credential an agent holds is a *token*. There are two ways it gets one
(`_client_from_env`): a pre-minted ``BASECRADLE_TOKEN`` it is handed, or an email +
password it logs in with. The founder directive is one coherent loop on top of those:

    use the existing token for everything, and mint a new one ONLY when there is no
    token OR the token is dead.

Two pieces deliver it, both here so the lifecycle reads in one place:

1. `write_token_to_env_file` — **persistence.** A minted (or re-minted) token is written
   back to the ``BASECRADLE_TOKEN=`` line of the file the agent *sources its own env from*
   (its ``agent.env`` on the fleet, a test env file on a laptop), named by
   ``BASECRADLE_ENV_FILE``. That one env var **is** the persistence layer — the next wake
   sources the file, finds the token, and reuses it, so a credential-only agent mints
   exactly once instead of once per wake. We do not invent a parallel token store.

2. `SelfHealingBaseCradle` — **recovery.** A `basecradle.BaseCradle` subclass that, when a
   platform call comes back 401 (the token is dead/revoked), re-mints from the same
   email + password *in place*, re-persists, and retries the call once — transparently,
   with no human. Every SDK resource call and every platform tool routes through
   `BaseCradle.request`, so overriding that one method is the whole self-heal: it covers
   construction, the poll loop, the wake reconcile, and tool calls alike, with no
   per-call-site wrapping and no client swap (the token and auth header are mutated on the
   live client, so every resource and tool already holding it picks up the new token).

When the token is dead and there is **no** email + password to mint from, recovery is
impossible, so we fail loudly with a remediation message rather than silently spinning.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from basecradle import AuthenticationError, BaseCradle, UnauthorizedError

# The ``BASECRADLE_TOKEN`` assignment to rewrite, in either form an env file takes: a
# shell ``export BASECRADLE_TOKEN=…`` or a bare dotenv ``BASECRADLE_TOKEN=…``, with any
# leading indentation. The two capture groups are the indentation and the optional
# ``export `` keyword, so a rewrite preserves the line's exact prefix.
_TOKEN_LINE_RE = re.compile(r"^(\s*)(export\s+)?BASECRADLE_TOKEN\s*=")

# Any ``export NAME=…`` assignment — used only to decide whether a *new* token line
# (appended because the file had none) should carry ``export `` to match the file's style.
_EXPORT_RE = re.compile(r"^\s*export\s+\w+\s*=")

_UNSET_ENV_FILE_WARNING = (
    "basecradle-harness: minted a new BaseCradle token but BASECRADLE_ENV_FILE is not "
    "set, so it will not be persisted — the next wake will mint again. Set "
    "BASECRADLE_ENV_FILE to the env file the agent sources (e.g. its agent.env) to "
    "persist the token and reuse it across wakes."
)

_NO_REMINT_MESSAGE = (
    "BaseCradle rejected the token (401) and it cannot be re-minted: no BASECRADLE_EMAIL "
    "+ BASECRADLE_PASSWORD are set to mint a fresh one from. Set them so the agent can "
    "re-mint automatically, or supply a valid BASECRADLE_TOKEN."
)

_MINT_REJECTED_MESSAGE = (
    "BaseCradle rejected the BASECRADLE_EMAIL / BASECRADLE_PASSWORD while minting a token "
    "(the platform returned an authentication error). Update them to a valid credential "
    "pair so the agent can mint and re-mint its token."
)


def mint_token(
    *,
    email: str,
    password: str,
    session_name: str | None,
    env_file: str | None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> str:
    """Mint a fresh token via login, persist it, adopt it into the process env, return it.

    The one place a token is minted — used by both the startup mint (`_client_from_env`)
    and the mid-run re-mint (`SelfHealingBaseCradle._remint_in_place`), so the two never
    drift. Reuses the SDK's tested ``login`` (``POST /session``); the throwaway login
    client is closed once its token is read (we only wanted the token). The minted token
    is written to ``env_file`` (so the next wake reuses it) *and* mirrored into
    ``os.environ`` (so anything that re-reads it in this process — including a later bare
    ``BaseCradle()`` — sees it). A login that fails to authenticate (rotated password,
    disabled account) is surfaced as a clear `ValueError`, not a raw nested 401.
    """
    login_kwargs: dict[str, Any] = {}
    if base_url is not None:
        login_kwargs["base_url"] = base_url
    if timeout is not None:
        login_kwargs["timeout"] = timeout
    try:
        minted = BaseCradle.login(
            email_address=email, password=password, name=session_name, **login_kwargs
        )
    except AuthenticationError as err:
        raise ValueError(_MINT_REJECTED_MESSAGE) from err
    try:
        token = minted.token
    finally:
        minted.close()  # we only wanted the token; don't leak its connection pool

    os.environ["BASECRADLE_TOKEN"] = token
    write_token_to_env_file(token, env_file)
    return token


def write_token_to_env_file(token: str, env_file: str | None) -> None:
    """Persist ``token`` to the ``BASECRADLE_TOKEN=`` line of ``env_file``, in place.

    The single durable home for a minted token: the file the agent sources its env from.
    The write is surgical and atomic so it can never corrupt the agent's secrets —

    - Only the first ``BASECRADLE_TOKEN`` assignment is touched; its value is swapped and
      the line's exact prefix (indentation + optional ``export ``) is preserved. Every
      other line — ``EMAIL``, ``PASSWORD``, comments, blanks — keeps its content unchanged
      (the file is normalized to ``\\n`` line endings with a trailing newline, the form
      these sourced env files already take).
    - With no such line, one is appended, carrying ``export `` only if the file already
      uses that style, so it sources the same way the rest of the file does.
    - The file is rewritten via a temp file in the same directory and ``os.replace`` (an
      atomic rename on POSIX), with the original file's permissions preserved (a fresh
      file is created ``0600`` — it holds a credential).

    When ``env_file`` is falsy (``BASECRADLE_ENV_FILE`` unset), there is nowhere to
    persist: warn once on stderr and return. The token still works for the current
    process; it just will not survive into the next wake.
    """
    if not env_file:
        print(_UNSET_ENV_FILE_WARNING, file=sys.stderr)
        return

    path = Path(env_file).expanduser()
    if path.exists():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            match = _TOKEN_LINE_RE.match(line)
            if match:
                prefix = match.group(1) + (match.group(2) or "")  # indent + optional "export "
                lines[i] = f"{prefix}BASECRADLE_TOKEN={token}"
                break
        else:  # no existing token line — append one in the file's own style
            keyword = "export " if any(_EXPORT_RE.match(line) for line in lines) else ""
            lines.append(f"{keyword}BASECRADLE_TOKEN={token}")
        content = "\n".join(lines) + "\n"
        mode = stat.S_IMODE(path.stat().st_mode)
    else:
        content = f"BASECRADLE_TOKEN={token}\n"
        mode = 0o600

    _atomic_write(path, content, mode)


def _atomic_write(path: Path, content: str, mode: int) -> None:
    """Replace ``path`` with ``content`` atomically, at file mode ``mode``.

    Writes a sibling temp file (so the rename stays on one filesystem), sets its mode,
    then ``os.replace``s it over the target — a reader of ``path`` sees either the old
    file or the new one, never a half-written secret. The temp file is cleaned up if
    anything fails before the rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".basecradle-env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SelfHealingBaseCradle(BaseCradle):
    """A `BaseCradle` that re-mints and retries once when its token is rejected (401).

    Built by `_client_from_env` whenever email + password are available to re-mint from
    (so it carries them, plus the ``BASECRADLE_ENV_FILE`` path to re-persist to). Behaves
    exactly like `BaseCradle` until a call returns 401; then it logs in again with the
    same credentials, swaps the token onto the live client (so every resource and tool
    already holding this client uses the new token), persists it, and retries the call.

    The retry uses ``super().request`` directly, so a still-failing token cannot loop —
    one re-mint per call, then the error propagates. With no credentials to re-mint from,
    the 401 is re-raised as a clear, actionable error rather than silently retried.

    The catch is `UnauthorizedError` (the SDK's *dead/invalid bearer token* signal),
    deliberately narrower than `AuthenticationError`: a fresh token fixes a dead token,
    not a bad webhook signature or a sign-in credential rejection (the other 401s), so we
    do not re-mint on those.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        email: str | None = None,
        password: str | None = None,
        session_name: str | None = None,
        env_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(token, **kwargs)
        self._email = email
        self._password = password
        self._session_name = session_name
        self._env_file = env_file

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """`BaseCradle.request`, with a one-shot re-mint-and-retry on a dead token (401)."""
        try:
            return super().request(method, path, **kwargs)
        except UnauthorizedError as err:
            if not (self._email and self._password):
                raise ValueError(_NO_REMINT_MESSAGE) from err
            self._remint_in_place()
            return super().request(method, path, **kwargs)  # retry once; a second 401 propagates

    def _remint_in_place(self) -> None:
        """Mint a fresh token from the stored credentials and adopt it on this client.

        Delegates the mint + persistence to `mint_token` (the one mint path, shared with
        startup), then mutates the token and the live ``Authorization`` header in place so
        no client swap or tool re-bind is needed — every resource and tool already holding
        this client uses the new token on its next call. ``login`` is never routed through
        ``request``, so this cannot recurse.
        """
        token = mint_token(
            email=self._email,
            password=self._password,
            session_name=self._session_name,
            env_file=self._env_file,
            base_url=self.base_url,
            timeout=self._timeout,
        )
        self.token = token
        self._client.headers["Authorization"] = f"Bearer {token}"
