"""A capped call is never what runs, and the harness's elision marker is never the agent's words.

Issue #576. @briggs posted a message whose body held, verbatim, the marker the transcript cap leaves in
an elided argument: head, `[... elided from 2413 chars — … ...]`, tail. The first suspicion was the
cap reaching the executor. It does not, and the first half of this file pins that on the incident's
shape: a transcript that has been capped for a long time, reloaded by a fresh wake, then one more long
call. It watches **the disk while the tool is running**, so what it proves is exactly what was feared.
The capped copy can already be on disk, and the executor still gets the whole value. (The one path that
runs arguments *read back from* disk, the recovery's re-issue of an interrupted create, is pinned in
`test_resume.py`: `test_an_interrupted_creates_body_survives_the_cap_and_is_re_posted_whole`.)

What happened instead is that the model *copied* the marker. It reads its own past speech in the capped
shape on every wake. The second half pins the guard: the engine refuses a call carrying a harness
elision marker and tells the model why; the recovery's seam refuses it too, saying the outcome is
unknown (`test_resume.py`: `test_an_interrupted_create_carrying_a_copied_marker_is_not_re_issued`).
"""

import json
import logging
import re

import pytest

from basecradle_harness import Harness, Message, Session, Tool, ToolCall
from basecradle_harness._elision import (
    _LEGACY,
    archive_marker,
    argument_marker,
    gone,
    refusal,
    result_marker,
)
from basecradle_harness._session import (
    TOOL_ARGS_CAP,
    TOOL_RESULT_CAP,
    _cap_arguments,
    _elide,
    _elide_argument,
    _json_size,
)

#: Real, well-formed UUIDv7s, per the repo's test-data convention.
TIMELINE = "0198e3f1-0000-7000-8000-000000000001"
POSTED = "0198e3f1-0000-7000-8000-0000000000a1"

#: The length of the body @briggs's marker named. Any body over the step's argument budget would do.
LONG = 2413


class ScriptedProvider:
    """Replays prepared assistant turns, and keeps a copy of every tool call it was shown."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        #: Per chat call, `(role, content, tool-call arguments)` snapshotted at chat time — "what did
        #: the model actually read?" is the whole question this file turns on.
        self.seen: list[list[tuple[str, str | None, list[dict]]]] = []

    def chat(self, messages, tools=None):
        self.seen.append(
            [
                (m.role, m.content, [json.loads(json.dumps(c.arguments)) for c in m.tool_calls])
                for m in messages
            ]
        )
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        return self._replies.pop(0)


class Recording(Tool):
    """Records every set of arguments it is run with, **and what the disk held for that call then**.

    The second half is the point. The feared defect was a cap that had already rewritten the call by
    the time the tool read it, so the only honest witness is the transcript file as it stands at the
    moment `run` executes.
    """

    description = "records its arguments"

    def __init__(self, name: str, path) -> None:
        self.name = name
        self.path = path
        self.received: list[dict] = []
        self.on_disk: list[dict] = []

    def run(self, **kwargs):
        self.received.append(kwargs)
        if self.path is not None:
            stored = [c for m in json.loads(self.path.read_text()) for c in m.get("tool_calls", [])]
            self.on_disk.append(stored[-1]["arguments"])  # the call running now is the newest one
        return f"created {POSTED}"


def prose(length: int, seed: str = "") -> str:
    """A multi-paragraph body of exactly `length` characters, the shape of a real long message.

    Paragraph breaks matter: a newline costs two characters once serialized, and the cap measures the
    serialized size, so a body without them would exercise a gentler path than the incident's.
    """
    sentence = f"{seed}The plan holds; the next step is ours, and we take it together. "
    paragraph = sentence * 4 + "\n\n"
    return (paragraph * (length // len(paragraph) + 1))[:length]


def text(content: str) -> Message:
    return Message.assistant(content=content)


def call(call_id: str, name: str, **arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def arguments(action: str | None, argument: str, value: str) -> dict:
    return ({"action": action} if action else {}) | {argument: value}


def wake(path, tool: Recording, replies: list[Message], prompt: str) -> ScriptedProvider:
    """One wake: a fresh `Session` that loads the transcript from disk, as every real wake does."""
    provider = ScriptedProvider(*replies)
    Session(f"timeline:{TIMELINE}", Harness(provider, tools=[tool]).engine, path=path).send(prompt)
    return provider


# --- The cap never reaches the executor --------------------------------------------------------

#: Every tool family a cap can bound, in both shapes it treats differently. A platform **create** is
#: kept whole on disk until its result lands (`_replayable`: the recovery might re-issue it from
#: there). **Everything else** is capped from the pre-dispatch save on, so the capped copy is already
#: on disk while the tool runs. That second shape is what issue #576 feared.
FAMILIES = [
    pytest.param("messages", "create", "body", True, id="messages-create"),
    pytest.param("tasks", "create", "instructions", True, id="tasks-create"),
    pytest.param("assets", "create", "content", True, id="assets-create"),
    pytest.param("webhook_endpoints", "create", "description", True, id="webhook-create"),
    pytest.param("messages", "update", "body", False, id="messages-update"),
    pytest.param("memory", "store", "content", False, id="memory-store"),
    pytest.param("shell", None, "command", False, id="shell"),
]


@pytest.mark.parametrize(("name", "action", "argument", "is_create"), FAMILIES)
def test_a_long_capped_transcript_never_reaches_the_executor(
    tmp_path, name, action, argument, is_create
):
    """The DoD's pin, on the incident's shape.

    Forty-three wakes of long replies (~258 entries, each reply capped on disk once it settles), then a
    fresh wake reloads that transcript and makes one more long call. Every value the executor was ever
    handed is the whole one the model wrote. For a non-create, the disk already held the *capped* copy
    of that same call while it ran, which is the exact state #576 suspected and it did not leak. The
    disk keeps the excerpt afterward. And the model, reading the reloaded transcript, is shown its own
    past calls capped: the pattern it can copy, and why the second half of this file exists.
    """
    path = tmp_path / "t.json"
    tool = Recording(name, path)
    for turn in range(43):
        body = prose(LONG, seed=f"[{turn}] ")
        reply = call(f"call_{turn}", name, **arguments(action, argument, body))
        wake(path, tool, [reply, text("Sent.")], f"message {turn}")
    assert len(json.loads(path.read_text())) >= 257

    body = prose(LONG, seed="[new] ")
    reply = call("call_new", name, **arguments(action, argument, body))
    provider = wake(path, tool, [reply, text("Sent.")], "one more, please")

    wrote = [prose(LONG, seed=f"[{t}] ") for t in range(43)] + [body]
    assert [r[argument] for r in tool.received] == wrote
    assert all(archive_marker(r) is None for r in tool.received)

    # What the disk held for each call *while it ran*: whole for a create, already capped otherwise.
    if is_create:
        assert [d[argument] for d in tool.on_disk] == wrote
    else:
        assert all(archive_marker(d) is not None for d in tool.on_disk)
        assert all(_json_size(d) <= TOOL_ARGS_CAP for d in tool.on_disk)

    # Once it settled, the disk keeps the bounded excerpt of the new call.
    stored = [c for m in json.loads(path.read_text()) for c in m.get("tool_calls", [])][-1]
    assert _json_size(stored["arguments"]) <= TOOL_ARGS_CAP
    assert stored["arguments"][argument].startswith("[new] The plan holds")
    assert f"elided from {LONG} chars" in stored["arguments"][argument]

    # The model's view of the reloaded transcript: every past long call, capped.
    shown = [a for _, _, calls in provider.seen[0] for a in calls if argument in a]
    assert len(shown) == 43
    assert all(archive_marker(a) is not None for a in shown)


# --- A copied marker is refused, and the model is told why -------------------------------------


def test_a_call_carrying_the_argument_marker_is_refused_and_nothing_is_sent(tmp_path, caplog):
    """The incident, replayed at @briggs's geometry: 951 characters, the marker, ~180 after it.

    The copied call reaches no tool. The model reads the refusal as that call's result, on the very
    next step. A call without a marker then goes through, once. (That the model *chooses* to rewrite
    is scripted here and can only be observed live; what is pinned is that it is told, and that
    nothing got out first.) The refusal is a WARNING on the `tool` head, where a fleet filter sees it.
    """
    path = tmp_path / "t.json"
    tool = Recording("messages", path)
    whole = prose(LONG, seed="[whole] ")
    copied = whole[:951] + argument_marker(LONG) + whole[-179:]

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        provider = wake(
            path,
            tool,
            [
                call("call_0", "messages", action="create", timeline=TIMELINE, body=copied),
                call("call_1", "messages", action="create", timeline=TIMELINE, body=whole),
                text("Sent the whole thing."),
            ],
            "write it all up",
        )

    assert tool.received == [{"action": "create", "timeline": TIMELINE, "body": whole}]

    (answered,) = [c for r, c, _ in provider.seen[1] if r == "tool"]
    assert answered == refusal(archive_marker(copied))
    assert "nothing was sent" in answered

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        w.startswith("tool ") and "name=messages" in w and "elision marker" in w for w in warnings
    )


def test_the_recovery_seam_is_told_the_outcome_is_unknown_never_that_nothing_was_sent():
    """`Engine.run_tool` re-issues an interrupted create, which a dead wake **already attempted**.

    "Nothing was sent" would be a guess there, and the wrong guess invites a rewrite under the next
    ordinal: a second post, where the same-key re-issue would have been deduplicated. The end-to-end
    form is in `test_resume.py`.
    """
    tool = Recording("messages", None)
    engine = Harness(ScriptedProvider(), tools=[tool]).engine

    result = engine.run_tool("messages", {"action": "create", "body": "a" + gone(9_000) + "b"})

    assert result == refusal(gone(9_000), reissue=True)
    assert "not re-issued" in result and "unknown" in result
    assert "nothing was sent" not in result
    assert tool.received == []


@pytest.mark.parametrize(
    "rendered",
    [
        pytest.param(lambda: _elide_argument(prose(LONG), 1_200), id="argument-excerpt"),
        pytest.param(lambda: _elide_argument(prose(LONG), 150), id="argument-floor"),
        pytest.param(lambda: _elide_argument(list(range(2_000)), 300), id="argument-structure"),
        pytest.param(lambda: _elide(prose(20_000), TOOL_RESULT_CAP), id="result-excerpt"),
        pytest.param(lambda: _elide(prose(20_000), 100), id="result-floor"),
        pytest.param(
            lambda: _cap_arguments({f"field_{i}": "v" * 40 for i in range(400)}, TOOL_ARGS_CAP),
            id="arguments-stub",
        ),
    ],
)
def test_every_marker_the_cap_writes_is_one_the_guard_recognizes(rendered):
    """Renderer and matcher share one template, proven off the real cap rather than the templates.

    Each case runs a path the transcript actually takes (an excerpt, the floor a share squeezes it to,
    a structure, a result, the stub for a call too wide to cut) and asserts the guard sees what it
    wrote.
    """
    value = rendered()
    assert archive_marker(value) is not None, value


#: What 0.70.0 wrote (issue #301, commit f4af460), **spelled out as its source spelled it**, so the
#: `_LEGACY` templates are pinned against history rather than against themselves. Both were replaced in
#: 0.72.0 (issue #304). Never edit these to match `_LEGACY`; it is `_LEGACY` that must match them.
SIZE, COUNT = 12_345, 3
WRITTEN_BY_0_70 = [
    (
        f"[... {SIZE} chars elided — this argument was archived out of the transcript; the full "
        f"value was sent when the call ran. ...]"
    ),
    (
        f"[... the {COUNT} arguments of this call ({SIZE} chars) were "
        f"archived out of the transcript; they were sent in full when the call ran. ...]"
    ),
]


def test_the_legacy_wordings_are_what_0_70_actually_wrote():
    assert [t.format(size=SIZE, count=COUNT) for t in _LEGACY] == WRITTEN_BY_0_70


def _wrapped(marker: str) -> str:
    """The marker as a line-wrapping model copies it: every third space a newline."""
    words = marker.split(" ")
    return "".join(w + ("\n" if i % 3 == 2 else " ") for i, w in enumerate(words[:-1])) + words[-1]


def _grouped(marker: str) -> str:
    """The marker with its numbers written the way a person writes them: `12,345`."""
    return re.sub(r"\d{4,}", lambda n: f"{int(n.group()):,}", marker)


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param(argument_marker(SIZE), id="argument"),
        pytest.param(result_marker(137_412, 145_984), id="result"),
        pytest.param(gone(60_000), id="floor"),
        *(pytest.param(w, id=f"legacy-{i}") for i, w in enumerate(WRITTEN_BY_0_70)),
    ],
)
def test_each_marker_is_refused_wherever_it_sits_and_however_it_was_copied(marker):
    """Top-level, inside a list, inside an object; line-wrapped; with its numbers grouped."""
    assert _wrapped(marker) != marker and _grouped(marker) != marker  # each variant really varies
    for variant in (marker, _wrapped(marker), _grouped(marker)):
        assert archive_marker({"body": f"head{variant}tail"}) is not None
        assert archive_marker({"items": ["fine", {"note": variant}]}) is not None


@pytest.mark.parametrize(
    "value",
    [
        {"action": "create", "body": "Hello, John."},
        {"body": "The cap elided 2413 chars of my last message; this is an archived excerpt."},
        {"body": "I saw `[... elided from N chars — …]` in my transcript; what is it?"},
        # The template as it appears in source code: a placeholder, not a number.
        {"command": "grep -n '[... elided from {size} chars — this argument' _elision.py"},
        # One word of the prose changed is not the harness's marker.
        {"body": argument_marker(2413).replace("the full value was sent", "the value was posted")},
        {"count": 3, "flag": True, "nothing": None, "ratio": 0.5},
        {},
    ],
)
def test_ordinary_arguments_and_talk_about_markers_are_left_alone(value):
    """Exact on the prose: discussing elision, excerpts or the marker is not the marker."""
    assert archive_marker(value) is None


def test_no_nesting_a_provider_can_deliver_makes_the_guard_raise():
    """The walk is iterative: the guard runs outside `_run_tool`'s error net, so it must not raise."""
    deep: object = "the bottom" + gone(5_000)
    for _ in range(50_000):
        deep = [deep]
    assert archive_marker({"body": deep}) is not None


@pytest.mark.parametrize("reissue", [False, True])
def test_the_refusal_is_not_itself_a_marker_to_trip_on(reissue):
    """Quoted only up to its dash, so the error never becomes another copy of what it warns about."""
    for marker in (argument_marker(SIZE), result_marker(10, 20), gone(5), *WRITTEN_BY_0_70):
        found = archive_marker({"body": marker})
        assert found is not None
        assert archive_marker({"body": refusal(found, reissue=reissue)}) is None
