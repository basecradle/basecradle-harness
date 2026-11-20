"""Count what the twin rule (issue #950) would have done to one agent's recorded exchanges.

A one-off measurement for basecradle-harness#950, run once by the capital on an agent's server, as
the agent, against the agent's own palace ``conversations/`` directory, before the release that
ships the rule reaches that server:

    python3 twin_census_standalone.py /home/<agent>/harness/mempalace/conversations

**Standard library only**, so it runs in the agent's own venv (or any Python 3.11+) with no install.
**It prints numbers only**: no drawer text, no file name, no path, and an error by its class alone.
It reads the directory and writes nothing.

**The rule is the package's, copied**: `twin_key`, `_STAMPS` and `exchange_of` are
`basecradle_harness._twins` and `basecradle_harness._mempalace.exchange_of` at the commit that
carries this file, and `tests/test_twin_census_standalone.py` proves the two agree on every
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
_STAMPS = (
    (re.compile(rf"^\[{_MOMENT}\] ", re.MULTILINE), "[*] "),
    (re.compile(rf"\(task {_UUID}, scheduled for {_MOMENT}\)"), "(task *, scheduled for *)"),
    (re.compile(rf"\(event {_UUID}, endpoint "), "(event *, endpoint "),
)
_QUOTE = "> "


def _canonical(text):
    for pattern, mask in _STAMPS:
        text = pattern.sub(mask, text)
    return " ".join(text.split())


def twin_key(user, assistant):
    digest = hashlib.sha256(f"{_canonical(user)}\x00{_canonical(assistant)}".encode()).hexdigest()
    return f"v{TWIN_RULE}:{digest}"


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
    """``(read, kept, skipped, collapsed)`` for the files in ``directory``, oldest first by mtime."""
    files = sorted(
        (path for path in Path(directory).iterdir() if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    seen, repeated = set(), set()
    kept = skipped = 0
    for path in files:
        key = twin_key(*exchange_of(path.read_text(encoding="utf-8", errors="replace")))
        if key in seen:
            skipped += 1
            repeated.add(key)
        else:
            seen.add(key)
            kept += 1
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
