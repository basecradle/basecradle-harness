"""The doc-truth test: every Python example in the README runs, verbatim.

Truth in Documentation (the constitution): documentation that lies is worse than
none. This extracts every ``python`` block from README.md and executes it against
a mocked model and a mocked platform — if the README drifts from the code, CI fails.
"""

import re
from pathlib import Path

import httpx
import pytest
import respx

README = Path(__file__).parent.parent / "README.md"

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
BC_URL = "https://basecradle.com"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
MESSAGE_UUID = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"


def python_blocks() -> list[str]:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert blocks, "README.md has no ```python code blocks"
    return blocks


def _chat_completion(text="Sure thing, peer."):
    return {
        "id": "chatcmpl-readme",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


def _message():
    return {
        "type": "message",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {"uuid": MESSAGE_UUID, "body": "Hello, Nova."},
    }


def _dashboard():
    return {"identity": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}}


def _timeline():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Incident response",
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


class TestReadmeExamples:
    @pytest.fixture(autouse=True)
    def mocked_world(self, monkeypatch):
        """A mocked model and platform — enough for every README block to run."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-readme-key")
        monkeypatch.setenv("BASECRADLE_TOKEN", "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa")
        monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
        monkeypatch.setenv("HARNESS_MODEL", "gpt-4o")

        with respx.mock(assert_all_called=False) as router:
            # The model: always a plain text reply (no tool calls), so examples terminate.
            router.post(OPENAI_CHAT_URL).mock(
                return_value=httpx.Response(200, json=_chat_completion())
            )
            # The platform: identity, the timeline, and a steady message list (nothing new
            # after priming, so poll_once does nothing and never calls the model).
            router.get(f"{BC_URL}/users/dashboard").mock(
                return_value=httpx.Response(200, json=_dashboard())
            )
            router.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
                return_value=httpx.Response(200, json=_timeline())
            )
            router.get(f"{BC_URL}/messages").mock(
                return_value=httpx.Response(
                    200, json={"messages": [_message()], "next_cursor": None}
                )
            )
            router.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/messages").mock(
                return_value=httpx.Response(201, json={"message": _message()})
            )
            yield router

    @pytest.mark.parametrize("block_number", range(len(python_blocks())))
    def test_block_runs_verbatim(self, block_number):
        code = python_blocks()[block_number]
        exec(compile(code, f"{README}#block{block_number}", "exec"), {})

    def test_custom_provider_block_is_offline_and_echoes(self, capsys):
        """The add-a-provider example must work with no model at all."""
        block = next(b for b in python_blocks() if "class EchoProvider" in b)
        exec(compile(block, str(README), "exec"), {})
        assert "You said: Hello!" in capsys.readouterr().out

    def test_safety_block_prints_the_policy_error(self, capsys):
        block = next(b for b in python_blocks() if "DangerousTool" in b)
        exec(compile(block, str(README), "exec"), {})
        assert "PolicyError" in capsys.readouterr().out
