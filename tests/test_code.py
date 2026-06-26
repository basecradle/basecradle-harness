"""The code-execution Asset bridge (`_code.py`): IN/OUT file flow, the tool, the engine hook.

Code runs **server-side, in the vendor's sandbox** — the harness never executes model-authored
code (issue #172). These tests cover the *bridge* half: staging a BaseCradle Asset into OpenAI's
Code Interpreter container (IN), harvesting a run's output files + executed source back into
Assets (OUT), the `code_attach` tool, the generic engine turn-hook, and the gate that builds the
bridge only where it applies (OpenAI + responses). The OpenAI container API is a **fake client**
(no socket); the BaseCradle SDK transport is respx-mocked, exactly as the media-tool tests do.

The live ground-truth — a real gpt-5.4-mini run round-tripping a file through the Asset system —
is the capital's verification on @jt; it cannot run offline against a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    CodeAttachTool,
    CodeExecutionBridge,
    CodeExecutionFile,
    CodeExecutionTrace,
    Engine,
    Message,
    PlatformContext,
    ToolRegistry,
)

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
FAKE_KEY = "sk-test-0123456789abcdef"

JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"
NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
SOURCE_UUID = "019e7754-7d4e-7f50-8162-aaaabbbbcccc"
POSTED_UUID = "019e7754-7d4e-7f50-8162-4d5e6f708192"

CSV_BYTES = b"name,score\nalice,10\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n fake chart pixels"


# --- fakes: an injected OpenAI container client ------------------------------


class _FakeContent:
    def __init__(self, data_by_id):
        self._data = data_by_id

    def retrieve(self, file_id, *, container_id):
        return SimpleNamespace(content=self._data[file_id])


class _FakeFiles:
    def __init__(self, content, outputs):
        self.content = content
        self.created: list[tuple[str, str, bytes]] = []
        # Assistant-written output files the run produced: [(file_id, path)].
        self._outputs = list(outputs)

    def create(self, container_id, *, file):
        self.created.append((container_id, file.name, file.read()))
        return SimpleNamespace(path=f"/mnt/data/{file.name}")

    def list(self, container_id):
        # Inputs we staged come back as source 'user'; produced files as 'assistant'.
        inputs = [
            SimpleNamespace(id=f"in_{i}", path=f"/mnt/data/{name}", source="user")
            for i, (_cid, name, _data) in enumerate(self.created)
        ]
        outputs = [
            SimpleNamespace(id=fid, path=path, source="assistant") for fid, path in self._outputs
        ]
        return SimpleNamespace(data=inputs + outputs)


class _FakeContainers:
    def __init__(self, files):
        self.files = files
        self.created: list[str] = []

    def create(self, *, name):
        self.created.append(name)
        return SimpleNamespace(id=f"cntr_new_{len(self.created)}")


class _FakeOpenAI:
    """Stands in for the OpenAI client's container surface — no network."""

    def __init__(self, content_map=None, outputs=()):
        self.containers = _FakeContainers(_FakeFiles(_FakeContent(content_map or {}), outputs))


# --- BaseCradle SDK transport (respx), shaped like the real platform ---------


def _asset_response(*, uuid, filename, content_type, description, user_uuid=JOHN_UUID):
    return {
        "asset": {
            "type": "asset",
            "created_at": "2026-06-24T00:00:00.000Z",
            "user": {"uuid": user_uuid, "handle": "john", "name": "John Doe", "kind": "human"},
            "timeline": {"uuid": TIMELINE_UUID},
            "content": {
                "uuid": uuid,
                "description": description,
                "file": {
                    "filename": filename,
                    "byte_size": 99,
                    "content_type": content_type,
                    "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                    "url": f"{BC_URL}/blobs/{uuid}",
                },
            },
        }
    }


def _timeline_envelope():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Studio",
            "locked": False,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
            "participants": [
                {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
            ],
        },
        "items": [],
    }


def _mock_upload(mock, captured):
    """Mock the timeline asset-create endpoint, capturing every uploaded filename."""

    def upload(request):
        name = _multipart_filename(request.content)
        captured.append(name)
        return httpx.Response(
            201,
            json=_asset_response(
                uuid=POSTED_UUID,
                filename=name,
                content_type="application/octet-stream",
                description="x",
            ),
        )

    mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=_timeline_envelope())
    )
    mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)


def _multipart_filename(body: bytes) -> str:
    marker = b'filename="'
    start = body.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    return body[start : body.find(b'"', start)].decode("ascii", "replace")


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


def _bridge(client, *, content_map=None, outputs=()):
    bridge = CodeExecutionBridge(client=_FakeOpenAI(content_map, outputs), api_key=FAKE_KEY)
    bridge.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return bridge


# === IN: stage a BaseCradle Asset into the executor ==========================


def test_stage_asset_uploads_asset_bytes_into_a_container(client):
    bridge = _bridge(client)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(
            return_value=httpx.Response(
                200,
                json=_asset_response(
                    uuid=SOURCE_UUID,
                    filename="scores.csv",
                    content_type="text/csv",
                    description="data",
                ),
            )
        )
        mock.get(f"{BC_URL}/blobs/{SOURCE_UUID}").mock(
            return_value=httpx.Response(200, content=CSV_BYTES)
        )
        result = bridge.stage_asset(SOURCE_UUID)

    # A container was created and the bytes uploaded into it under the asset's filename.
    fake = bridge._client
    assert fake.containers.created == ["basecradle-harness"]
    (container_id, name, data) = fake.containers.files.created[0]
    assert name == "scores.csv" and data == CSV_BYTES
    # The model is told the sandbox path to read it from.
    assert "/mnt/data/scores.csv" in result
    # The container is now pinned, so the provider reuses it across the wake's turns.
    assert bridge.container_spec() == container_id


def test_stage_asset_reports_a_bad_uuid_to_the_model_without_raising(client):
    bridge = _bridge(client)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(return_value=httpx.Response(404))
        result = bridge.stage_asset(SOURCE_UUID)

    assert "Couldn't read asset" in result  # readable text, never an exception
    assert bridge._client.containers.created == []  # no container created on a read failure


def test_container_spec_is_auto_until_a_container_exists(client):
    bridge = _bridge(client)
    assert bridge.container_spec() == {"type": "auto"}  # no eager container — no cost on idle wakes


# === OUT: harvest a run's files + source back into Assets ====================


def _code_reply(*, code, files, container="cntr_x"):
    reply = Message.assistant(content="Done — see the outputs.")
    reply.code_execution = CodeExecutionTrace(
        container=container,
        code=list(code),
        output_files=[CodeExecutionFile(file_id=fid, filename=fn) for fid, fn in files],
    )
    return reply


def test_on_reply_stores_source_and_output_files_then_feeds_back_uuids(client):
    bridge = _bridge(
        client, content_map={"cfile_1": PNG_BYTES}, outputs=[("cfile_1", "/mnt/data/chart.png")]
    )
    reply = _code_reply(code=["import pandas"], files=[("cfile_1", "chart.png")])
    messages = [reply]
    captured: list[str] = []
    with respx.mock(assert_all_called=True) as mock:
        _mock_upload(mock, captured)
        cont = bridge.on_reply(reply, messages)

    assert cont is True  # the loop takes one more turn so the model can cite the new Assets
    # Both the executed source and the output file were posted as Assets.
    assert any(name.endswith(".py") for name in captured)
    assert any(name == "chart.png" for name in captured)
    # A continuation turn was appended naming the produced Asset uuid for the model to reference.
    assert messages[-1].role == "user"
    assert POSTED_UUID in messages[-1].content


def test_on_reply_dedups_so_a_resettled_run_surfaces_nothing_new(client):
    bridge = _bridge(
        client, content_map={"cfile_1": PNG_BYTES}, outputs=[("cfile_1", "/mnt/data/chart.png")]
    )
    reply = _code_reply(code=["x=1"], files=[("cfile_1", "chart.png")])
    captured: list[str] = []
    with respx.mock() as mock:
        _mock_upload(mock, captured)
        first = bridge.on_reply(reply, [reply])
        count_after_first = len(captured)
        # The same trace again (the transcript can re-present it) — nothing new is uploaded.
        second = bridge.on_reply(reply, [reply])

    assert first is True
    assert second is False
    assert len(captured) == count_after_first  # dedup by file id + source hash


def test_on_reply_harvests_an_uncited_output_file_via_listing(client):
    # The robustness win (verified live): the model writes a file but never cites it. Listing the
    # container (source='assistant') still finds it, where the cited annotations alone would miss it.
    bridge = _bridge(
        client, content_map={"cfile_x": PNG_BYTES}, outputs=[("cfile_x", "/mnt/data/silent.png")]
    )
    reply = _code_reply(code=["plt.savefig('silent.png')"], files=[])  # nothing cited
    captured: list[str] = []
    with respx.mock(assert_all_called=True) as mock:
        _mock_upload(mock, captured)
        cont = bridge.on_reply(reply, [reply])

    assert cont is True
    assert any(name == "silent.png" for name in captured)  # found despite not being cited


def test_on_reply_retries_a_transiently_failed_fetch_on_a_later_harvest(client):
    # A fetch that fails once must NOT mark the file seen — else a transient timeout drops it
    # forever. A later harvest (the model ran code again) retries and stores it.
    bridge = _bridge(
        client, content_map={"cfile_1": PNG_BYTES}, outputs=[("cfile_1", "/mnt/data/late.png")]
    )

    calls = {"n": 0}
    real_retrieve = bridge._client.containers.files.content.retrieve

    def flaky(file_id, *, container_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient timeout")
        return real_retrieve(file_id, container_id=container_id)

    bridge._client.containers.files.content.retrieve = flaky
    reply = _code_reply(code=[], files=[])  # no source to store, so only the file matters
    captured: list[str] = []
    with respx.mock() as mock:
        _mock_upload(mock, captured)
        first = bridge.on_reply(reply, [reply])  # fetch fails → nothing stored
        second = bridge.on_reply(reply, [reply])  # retried → stored

    assert first is False  # nothing harvested the first time
    assert second is True
    assert captured == ["late.png"]  # stored exactly once, on the retry


def test_on_reply_is_a_noop_without_a_trace(client):
    bridge = _bridge(client)
    reply = Message.assistant(content="just text")
    assert bridge.on_reply(reply, [reply]) is False


def test_on_reply_degrades_gracefully_when_the_platform_upload_fails(client):
    # A bridge failure must never break the wake (the memory/dashboard degradation bar).
    bridge = _bridge(client, content_map={"cfile_1": PNG_BYTES})
    reply = _code_reply(code=["x=1"], files=[("cfile_1", "chart.png")])
    with respx.mock() as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(
            return_value=httpx.Response(500)
        )
        cont = bridge.on_reply(reply, [reply])  # does not raise

    assert cont is False


# === The code_attach tool ====================================================


def test_code_attach_routes_to_the_bridge(client):
    bridge = _bridge(client)
    tool = CodeAttachTool()
    tool.bind(PlatformContext(client=client, timeline=TIMELINE_UUID, code_bridge=bridge))
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(
            return_value=httpx.Response(
                200,
                json=_asset_response(
                    uuid=SOURCE_UUID, filename="in.csv", content_type="text/csv", description="d"
                ),
            )
        )
        mock.get(f"{BC_URL}/blobs/{SOURCE_UUID}").mock(
            return_value=httpx.Response(200, content=CSV_BYTES)
        )
        result = tool.run(asset_uuid=SOURCE_UUID)

    assert "/mnt/data/in.csv" in result


def test_code_attach_explains_when_code_execution_is_inactive(client):
    tool = CodeAttachTool()
    tool.bind(PlatformContext(client=client, timeline=TIMELINE_UUID, code_bridge=None))
    result = tool.run(asset_uuid=SOURCE_UUID)
    assert "not active" in result.lower()


# === The engine turn-hook (generic) ==========================================


class _ScriptedProvider:
    """Returns a queued reply per chat() call — no network."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        return self._replies.pop(0)


def test_turn_hook_can_extend_the_loop_past_a_no_tool_call_turn():
    # A hook returning True takes one more turn even with no tool calls (the code-exec harvest
    # uses this to feed Asset uuids back); when it returns False the loop ends.
    provider = _ScriptedProvider(
        [Message.assistant(content="first"), Message.assistant(content="second")]
    )
    flags = iter([True, False])

    engine = Engine(provider, ToolRegistry(), turn_hook=lambda reply, msgs: next(flags))
    final = engine.run([Message.user("go")])

    assert provider.calls == 2  # extended once, then stopped
    assert final.content == "second"


def test_no_turn_hook_is_unchanged():
    provider = _ScriptedProvider([Message.assistant(content="only")])
    engine = Engine(provider, ToolRegistry())
    final = engine.run([Message.user("go")])
    assert provider.calls == 1 and final.content == "only"


# === The build gate: bridge only where it applies ============================


def test_maybe_code_bridge_builds_only_for_openai_responses_with_code_interpreter(monkeypatch):
    from basecradle_harness._basecradle import _maybe_code_bridge

    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    # OpenAI + responses + the code_interpreter built-in opted in → a bridge.
    assert _maybe_code_bridge("openai", "responses", ["code_interpreter"]) is not None
    # xAI (no input-file mechanism), the chat surface (no Code Interpreter), or not opted in → none.
    assert _maybe_code_bridge("xai", "native", ["code_execution"]) is None
    assert _maybe_code_bridge("openai", "chat", []) is None
    assert _maybe_code_bridge("openai", "responses", ["web_search"]) is None
