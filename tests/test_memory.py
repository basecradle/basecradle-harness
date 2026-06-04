"""The shipped memory tool: persistence and the tool contract.

Every test points the tool at a temp file (`tmp_path`) so nothing touches the
real default location.
"""

import pytest

from basecradle_harness import MemoryTool, Policy, ToolRegistry


@pytest.fixture
def memory(tmp_path):
    return MemoryTool(path=tmp_path / "memory.json")


# --- The three actions -------------------------------------------------------


def test_write_then_read_recalls_the_value(memory):
    assert memory.run(action="write", key="city", value="Dallas") == "Remembered 'city'."
    assert memory.run(action="read", key="city") == "Dallas"


def test_read_unknown_key_on_empty_store_is_a_clear_message(memory):
    result = memory.run(action="read", key="missing")
    assert "No memory stored under 'missing'" in result
    assert "no memories yet" in result


def test_read_miss_lists_the_keys_you_do_have(memory):
    """A wrong-key read surfaces the real keys, so a fresh agent can self-correct."""
    memory.run(action="write", key="favorite_language", value="Ruby")
    memory.run(action="write", key="city", value="Dallas")

    result = memory.run(action="read", key="language")  # close, but not the stored key

    assert "No memory stored under 'language'" in result
    assert "city, favorite_language" in result  # the keys it does have, sorted


def test_list_is_empty_then_names_the_keys(memory):
    assert memory.run(action="list") == "No memories stored yet."
    memory.run(action="write", key="city", value="Dallas")
    memory.run(action="write", key="role", value="developer")
    assert memory.run(action="list") == "city, role"


def test_write_overwrites_an_existing_key(memory):
    memory.run(action="write", key="city", value="Dallas")
    memory.run(action="write", key="city", value="Austin")
    assert memory.run(action="read", key="city") == "Austin"


# --- Input guards (model-friendly strings, not exceptions) -------------------


def test_write_without_value_is_rejected(memory):
    assert "needs both a key and a value" in memory.run(action="write", key="city")


def test_read_without_key_is_rejected(memory):
    assert "needs a key" in memory.run(action="read")


def test_unknown_action_is_rejected(memory):
    assert "unknown action" in memory.run(action="forget", key="city")


# --- Persistence -------------------------------------------------------------


def test_memory_survives_across_instances(tmp_path):
    """A new tool on the same file recalls what an earlier one wrote.

    This is the across-turns / across-runs guarantee: the engine builds a fresh
    tool each run and the fact is still there.
    """
    path = tmp_path / "memory.json"
    MemoryTool(path=path).run(action="write", key="city", value="Dallas")

    later = MemoryTool(path=path)
    assert later.run(action="read", key="city") == "Dallas"


def test_no_file_is_written_until_the_first_write(tmp_path):
    path = tmp_path / "memory.json"
    reader = MemoryTool(path=path)
    assert "No memory stored under 'city'" in reader.run(action="read", key="city")
    assert not path.exists()


# --- The tool contract & the safe profile ------------------------------------


def test_spec_advertises_the_action_enum(memory):
    spec = memory.to_spec()
    assert spec.name == "memory"
    assert spec.parameters["properties"]["action"]["enum"] == ["write", "read", "list"]


def test_memory_loads_under_the_locked_profile(tmp_path):
    """The shipped example must register on the safe default — it needs no capability."""
    assert MemoryTool().requires == frozenset()
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(MemoryTool(path=tmp_path / "memory.json"))
    assert "memory" in registry


def test_runs_through_the_registry(memory):
    registry = ToolRegistry()
    registry.register(memory)
    registry.run("memory", action="write", key="city", value="Dallas")
    assert registry.run("memory", action="read", key="city") == "Dallas"
