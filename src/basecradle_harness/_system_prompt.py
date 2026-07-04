"""Self-authorship: an agent reads and edits its OWN ``system-prompt.md``, nothing else.

The most powerful tool in the kit — a persona that can rewrite its own personality charter.
The whole design is about making that power *structurally* narrow rather than validated-narrow,
so a prompt-injected argument has nothing to grab:

- **Own prompt only, by construction (issue #241, invariant 1).** Neither tool takes a path or
  agent argument of any kind. The target is resolved internally, every time, from the agent's
  own config home the *same* way the wake brief locates it — ``config_home() / "prompts" /
  "system-prompt.md"`` (`_target`). There is nothing to traverse and nothing to point
  elsewhere; the OS-user boundary (each agent runs as its own user, mode-600/700 home) backstops
  it. "Scoped to self" holds even against an argument that *tries* to point elsewhere, because
  no such argument exists.
- **``system-prompt.md`` ONLY — never ``initialize.md`` (invariant 2).** The fleet-wide
  input-security floor lives in the default ``initialize.md`` (issue #239). Keeping it outside
  the editable surface — again, by construction, since there is no file selector — means a
  manipulated or misguided agent **cannot edit away its own injection hardening**. The floor
  stays above self-authorship, permanently.
- **Guarded-confirm on the edit (invariant 4).** ``system_prompt_edit`` writes only when its
  ``confirm`` equals a hash of the *current* file content (`_content_token`). A bare or
  mismatched confirm changes nothing and returns a **preview** plus the token to use — the same
  preview-on-refuse discipline the irreversible timeline actions use (`_confirmed.py`). The hash
  is content-derived, so it doubles as **compare-and-swap**: if the file changed since the agent
  last read it, the token no longer matches and a stale overwrite is refused, not silently
  applied.
- **Versioned history, every edit (invariant 5).** A successful edit first snapshots the current
  file beside it — ``system-prompt.md.<utc-timestamp>.bak`` — so an operator can audit and roll
  back. An agent quietly rewriting itself with no trail is the failure mode; the trail is not
  optional.
- **Takes effect next wake (invariant 6).** The Turn-0 brief is composed per wake, reading
  ``system-prompt.md`` fresh (`_install.system_prompt_text`), so an edit lands on the *next*
  wake, not the current turn. Both tool descriptions say so, so the agent has an accurate model
  of when its self-edit takes hold.

Opt-in, off by default on every provider (issue #168): the shipped plugin file declares
``opt_in=True``, so it is never auto-scaffolded and never loaded from the packaged defaults —
it activates only when an operator deliberately drops it into a persona's ``tools/`` overlay.
Enablement is a founder decision, per-agent; as of issue #241 no agent has it.

Plain `Tool`s (not `PlatformTool`s): they need no SDK client and no bound context — only the
config-home resolver, read from the environment exactly as the runtime reads it. Filesystem I/O
scoped to one file in the agent's own config home is the same discipline the memory tool uses;
the locked policy gates shell/exec, which this never touches.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from basecradle_harness._install import _prompts_installed, config_home
from basecradle_harness._tools import NO_PARAMETERS, Tool

# The one file these tools ever touch, relative to the config home. Named as a constant so the
# self-scoping is stated once and both tools share it — there is deliberately no way to vary it.
_PROMPT_REL = ("prompts", "system-prompt.md")


def _target() -> Path:
    """The agent's own ``system-prompt.md`` — resolved from its config home, never from an argument.

    Uses the same `config_home` resolver (``BASECRADLE_CONFIG_HOME`` → ``$HOME/.config/basecradle``)
    the wake brief uses when it reads the charter with ``home=None`` (`_wake.py` →
    `system_prompt_text`), so the file these tools edit is exactly the file the next wake will
    read. Takes no parameter, by construction: there is nothing here for a prompt-injected
    argument to redirect.
    """
    return config_home().joinpath(*_PROMPT_REL)


def _content_token(content: str) -> str:
    """A short content hash used as the edit's confirm token (compare-and-swap).

    Derived from the current file bytes, so passing it back proves the agent edited against the
    content it actually saw: if the file changed in between, the token no longer matches and the
    edit is refused (a preview) rather than clobbering the newer version.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _read_current(path: Path) -> str:
    """The current file text, or ``""`` when it does not exist (a not-yet-authored charter)."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


class SystemPromptReadTool(Tool):
    """Read this agent's OWN system prompt (personality charter), verbatim.

    Returns the raw ``system-prompt.md`` — including any HTML-comment operator notes and exact
    formatting the Turn-0 brief strips before showing it — plus a short *edit token* (a hash of
    the current content). Pass that token as ``confirm`` to ``system_prompt_edit`` to change the
    prompt; because it is derived from the content, it also guards against overwriting a version
    edited since you read it. This reads only your *own* prompt — it takes no path or agent
    argument and cannot read anyone else's.
    """

    name = "system_prompt_read"
    description = (
        "Read your OWN system prompt (your personality charter) verbatim, exactly as stored on "
        "disk — including operator HTML-comment notes and formatting the wake brief strips out. "
        "Also returns a short edit token (a hash of the current content) to pass as 'confirm' to "
        "system_prompt_edit. This is scoped to YOUR prompt by construction: it takes no path or "
        "agent argument and can read no one else's, and it can never read initialize.md. Reading "
        "does not change anything."
    )
    parameters = NO_PARAMETERS

    def run(self) -> str:
        if not _prompts_installed(config_home()):
            # Read only the file the wake actually reads. When prompts are not manifest-installed,
            # the wake's live charter comes from HARNESS_SYSTEM_PROMPT or the packaged default —
            # NOT this file (`system_prompt_text`). Reporting the file (or handing out an edit
            # token) would misrepresent the live persona and invite an edit that cannot land, so
            # decline honestly instead — symmetric with the edit tool's refusal.
            return (
                "This config home's prompts are not installed, so there is no editable "
                "system-prompt.md yet (your live charter, if any, comes from the environment or "
                "the packaged default, which this tool does not manage). Run "
                "basecradle-harness-install to install the config-home prompts first."
            )
        path = _target()
        if not path.exists():
            return (
                f"No system prompt is set: {path} does not exist. The edit token for an empty "
                f"(unset) prompt is {_content_token('')} — pass it as 'confirm' to "
                "system_prompt_edit to author one."
            )
        content = path.read_text(encoding="utf-8")
        token = _content_token(content)
        return (
            f"Your current system prompt (edit token: {token} — pass as 'confirm' to "
            f"system_prompt_edit; it changes if the file changes):\n\n{content}"
        )


class SystemPromptEditTool(Tool):
    """Rewrite this agent's OWN system prompt, behind a compare-and-swap confirm gate.

    Replaces ``system-prompt.md`` **in full** with ``content``, but only when ``confirm`` equals
    the current content's edit token (from ``system_prompt_read``, or from this tool's own
    preview). A bare or mismatched ``confirm`` changes nothing and returns a preview plus the
    token to use — so a stale or reflexive call cannot silently rewrite your persona. Every
    successful edit first snapshots the old file as a timestamped ``.bak`` beside it. The change
    takes effect on your **next wake**, when the brief is re-composed — not this turn. Scoped to
    your own prompt by construction (no path/agent argument); it can never touch ``initialize.md``.
    """

    name = "system_prompt_edit"
    description = (
        "Rewrite your OWN system prompt (personality charter) IN FULL. This is the most powerful "
        "action available: you are editing your own persona. It writes only when 'confirm' equals "
        "the current content's edit token (get it from system_prompt_read, or from this tool's "
        "preview when you call it without a matching confirm) — a compare-and-swap that refuses a "
        "stale overwrite and previews instead of acting. A bare or mismatched call changes "
        "NOTHING and returns a preview plus the exact token to use. Every successful edit first "
        "saves a timestamped .bak backup of the old prompt beside it, so an operator can audit "
        "and roll back. The new prompt takes effect on your NEXT WAKE (the brief is re-composed "
        "each wake), not this turn. Scoped to YOUR prompt by construction: no path or agent "
        "argument, and it can NEVER edit initialize.md (your input-security floor and operating "
        "guidance stay outside this surface)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The full new text of your system prompt. It REPLACES the file entirely — "
                    "this is not an append or a patch, so include everything you want to keep."
                ),
            },
            "confirm": {
                "type": "string",
                "description": (
                    "The current content's edit token, required to actually write. Get it from "
                    "system_prompt_read (or this tool's preview). A missing or mismatched token "
                    "changes nothing and returns a preview — which happens both when you have not "
                    "confirmed yet and when the file changed since you read it (a stale edit)."
                ),
            },
        },
        "required": ["content"],
    }

    def run(self, content: str, confirm: str | None = None) -> str:
        if not _prompts_installed(config_home()):
            # Invariant 6 (takes effect next wake) held by construction: the wake brief reads the
            # config-home system-prompt.md only when prompts are manifest-installed
            # (`system_prompt_text`). If they are not, an edit here would be silently ignored next
            # wake — so refuse rather than write an edit that never lands.
            return (
                "Cannot edit: this config home's prompts are not installed, so an edit here would "
                "not take effect on your next wake. Run basecradle-harness-install first."
            )
        path = _target()
        current = _read_current(path)
        token = _content_token(current)
        if confirm != token:
            return self._preview(path, current, token, content, confirm)
        if content == current:
            return (
                "No change: the new content is identical to the current system prompt. Nothing "
                "was written and no backup was made."
            )
        backup = self._snapshot(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # installed home whose prompts/ was deleted
        path.write_text(content, encoding="utf-8")
        note = f" The previous version was saved to {backup.name}." if backup else ""
        return (
            f"Rewrote your system prompt ({path}).{note} This takes effect on your NEXT WAKE, "
            "when the brief is re-composed — not this turn. The new edit token is "
            f"{_content_token(content)}."
        )

    def _preview(
        self, path: Path, current: str, token: str, proposed: str, confirm: str | None
    ) -> str:
        """Preview-on-refuse: name the current state and hand back the exact token to confirm with.

        Mirrors the irreversible timeline actions' gate (`_confirmed.py`): a refusal is never a
        dead end — it says why nothing happened, shows what is there now, and gives the token to
        re-call with on purpose.
        """
        if confirm is None:
            why = "No confirm was passed."
        else:
            why = (
                f"The confirm you passed ({confirm!r}) does not match the current edit token — "
                "either you have not confirmed yet, or the prompt changed since you read it."
            )
        if current:
            state = (
                f"The current system prompt is {len(current)} character(s) long. To replace it, "
                f"read it first if you have not, then call system_prompt_edit again with "
                f"confirm={token}."
            )
        else:
            state = (
                f"No system prompt is set yet ({path} is empty or absent). To author one, call "
                f"system_prompt_edit again with confirm={token}."
            )
        return (
            f"Refused to edit your system prompt: nothing was written. {why} Your proposed new "
            f"content is {len(proposed)} character(s). {state} (A successful edit snapshots the "
            "old prompt as a .bak first and takes effect next wake.)"
        )

    def _snapshot(self, path: Path) -> Path | None:
        """Save the current file as a timestamped ``.bak`` beside it, returning that path.

        Reads the file's bytes **fresh** here (not a value captured earlier in ``run``) so the
        backup is the true on-disk content immediately before the overwrite — the ``.bak`` cannot
        record a stale snapshot. Returns ``None`` when there is nothing to back up (authoring a
        prompt from scratch). The timestamp is UTC to the microsecond so successive edits never
        collide; on the vanishingly rare same-microsecond collision, a numeric suffix keeps the
        older backup intact.
        """
        if not path.exists():
            return None
        current = path.read_text(encoding="utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        suffix = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.{stamp}.{suffix}.bak")
            suffix += 1
        backup.write_text(current, encoding="utf-8")
        return backup
