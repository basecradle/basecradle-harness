"""When two texts are near-copies: one measure, read by the prune and by the observe's door rule.

`basecradle-harness-prune-palace` groups an agent's routine check-ins by it (issue #973), and the
MemPalace adapter's door rule for recurring tasks declines to file a reply by it (issue #1011). Two
readers of one measure, so a reply the door lets through is judged by the same words the prune
would judge it by later, and neither can drift from the other.

Near-copy is symmetric Jaccard over distinct words at `NEAR_COPY`, after `normalize` turns uuids,
timestamps, dates, clock times and bare numbers into placeholders and strips no ordinary word.
"""

from __future__ import annotations

import re

#: Symmetric Jaccard on distinct normalised words at or above which two texts are near-copies.
#: Not lowered (issue #973): a lower bar starts grouping different jobs that share a vocabulary.
NEAR_COPY = 0.8

#: What `normalize` turns into placeholders, in order: uuids, timestamps, dates, clock times, then
#: bare numbers. The placeholders cannot be spelled by `\w`, so no ordinary word becomes one.
_NORMALIZERS = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), " ⟨uuid⟩ "),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[t ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?\b"
        ),
        " ⟨time⟩ ",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"), " ⟨date⟩ "),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?m\.?)?(?![\w:])"), " ⟨time⟩ "),
    (re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])"), " ⟨n⟩ "),
)
_WORD = re.compile(r"⟨(?:uuid|time|date|n)⟩|\w+")


def normalize(text: str) -> str:
    """``text`` lowercased, with uuids, timestamps, dates and bare numbers made placeholders."""
    text = text.lower()
    for pattern, placeholder in _NORMALIZERS:
        text = pattern.sub(placeholder, text)
    return text


def words(text: str) -> frozenset[str]:
    """The distinct words of ``text`` after `normalize`: placeholders and every ordinary word."""
    return frozenset(_WORD.findall(normalize(text)))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Symmetric overlap: shared words over all words. Two empty halves are identical."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def near_copy(left: str, right: str) -> bool:
    """Whether two texts are near-copies: `jaccard` of their `words` at or above `NEAR_COPY`."""
    return jaccard(words(left), words(right)) >= NEAR_COPY
