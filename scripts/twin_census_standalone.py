"""Count what the repeat rule (issue #950) would have done to one agent's recorded exchanges.

A one-off measurement for basecradle-harness#950, run once by the capital on an agent's server, as
the agent, against the agent's own palace ``conversations/`` directory, before the release that
ships the rule reaches that server:

    python3 twin_census_standalone.py /home/<agent>/harness/mempalace/conversations

**Standard library only**, so it runs in the agent's own venv (or any Python 3.11+) with no install.
**It prints numbers only**: no drawer text, no file name, no path, and an error by its class alone.
It reads the directory and writes nothing.

**The rule is the package's, copied**: `twin_key`, `_OPENINGS`, `census`'s per-timeline replay,
`timeline_of_file` and `exchange_of` are `basecradle_harness._twins` and the
`basecradle_harness._mempalace` readers at the commit that carries this file, and `tests/test_twin_census_standalone.py` proves the two agree on every
renderer's output, so the count is the count the shipped rule would give. The file leaves the
repository once its count is posted on the issue; the lasting form is
``basecradle-harness-palace-check --twin-census``.
"""

import hashlib
import re
import sys
from pathlib import Path

TWIN_RULE = 1
_MOMENT = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_OPENINGS = (
    (
        re.compile(
            rf"\A\[{_MOMENT}\] (A task scheduled by @\S+ has activated and is now in front of every "
            rf"viewer of this timeline )\(task {_UUID}, scheduled for {_MOMENT}\)"
        ),
        r"[*] \1(task *, scheduled for *)",
    ),
    (
        re.compile(
            rf"\A\[{_MOMENT}\] (An inbound webhook was delivered to this timeline )"
            rf"\(event {_UUID}, endpoint "
        ),
        r"[*] \1(event *, endpoint ",
    ),
    (re.compile(rf"\A\[{_MOMENT}\] "), "[*] "),
)
_TIMELINE_FILE = re.compile(
    r"^(?P<timeline>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.[0-9a-f]{32}\.md$"
)
_QUOTE = "> "


def _opening_masked(user):
    for pattern, mask in _OPENINGS:
        masked, count = pattern.subn(mask, user, count=1)
        if count:
            return masked
    return user


def twin_key(user, assistant):
    user = _opening_masked(user.strip())
    assistant = assistant.strip()
    payload = f"{len(user)}:{user}{len(assistant)}:{assistant}"
    return f"v{TWIN_RULE}:{hashlib.sha256(payload.encode()).hexdigest()}"


def timeline_of_file(name):
    match = _TIMELINE_FILE.match(name)
    return match["timeline"] if match else None


def exchange_of(text):
    lines = text.splitlines()
    split = 0
    while split < len(lines) and lines[split].startswith(_QUOTE.rstrip()):
        split += 1
    user = "\n".join(
        line[len(_QUOTE) :] if line.startswith(_QUOTE) else "" for line in lines[:split]
    )
    return user, "\n".join(lines[split:])


def census(directory):
    """``(read, kept, skipped, collapsed)`` for the files in ``directory``, oldest first by mtime.

    An exchange is skipped when it repeats the newest exchange kept on its timeline (named in its
    file name); one whose file names no timeline is always kept.
    """
    files = sorted(
        (path for path in Path(directory).iterdir() if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    newest, repeated = {}, set()
    kept = skipped = 0
    for path in files:
        timeline = timeline_of_file(path.name)
        key = twin_key(*exchange_of(path.read_text(encoding="utf-8", errors="replace")))
        if timeline is not None and newest.get(timeline) == key:
            skipped += 1
            repeated.add((timeline, key))
            continue
        kept += 1
        if timeline is not None:
            newest[timeline] = key
    return kept + skipped, kept, skipped, len(repeated)


def main(argv):
    if len(argv) != 2:
        print("usage: twin_census_standalone.py <conversations directory>", file=sys.stderr)
        return 2
    try:
        read, kept, skipped, collapsed = census(argv[1])
    except OSError as error:
        print(f"error: {type(error).__name__}", file=sys.stderr)
        return 1
    print(f"exchanges read: {read}")
    print(f"would be kept: {kept}")
    print(f"would be skipped: {skipped}")
    print(f"distinct exchanges the skipped collapse onto: {collapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
