"""The shipped memory tool: full CRUD, keyword search, and schema migration.

Every test points the tool at a temp file (`tmp_path`) so nothing touches the
real default location. SQLite is in the standard library, so these never hit the
network — they are unit tests of the store itself.
"""

import sqlite3

import pytest

from basecradle_harness import MemoryTool, Policy, ToolRegistry
from basecradle_harness._memory import SCHEMA_VERSION


@pytest.fixture
def memory(tmp_path):
    tool = MemoryTool(path=tmp_path / "memory.db")
    yield tool
    tool.close()


# --- write / read / list -----------------------------------------------------


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


def test_overwrite_keeps_created_at_and_advances_updated_at(memory):
    """An upsert refreshes the value and updated_at but preserves created_at."""
    memory.run(action="write", key="city", value="Dallas")
    created, first_updated = _timestamps(memory, "city")

    memory.run(action="write", key="city", value="Austin")
    created_again, second_updated = _timestamps(memory, "city")

    assert created_again == created  # created_at is sticky
    assert second_updated >= first_updated  # updated_at moves forward (never back)


# --- delete ------------------------------------------------------------------


def test_delete_removes_a_key(memory):
    memory.run(action="write", key="city", value="Dallas")
    assert memory.run(action="delete", key="city") == "Forgot 'city'."
    assert "No memory stored under 'city'" in memory.run(action="read", key="city")
    assert memory.run(action="list") == "No memories stored yet."


def test_delete_missing_key_is_a_clean_message(memory):
    assert "nothing to delete" in memory.run(action="delete", key="ghost")


# --- search ------------------------------------------------------------------


def test_search_finds_by_value_keyword(memory):
    """Recall without remembering the key: a word from the value finds the fact."""
    memory.run(action="write", key="home_city", value="Dallas, Texas")
    memory.run(action="write", key="role", value="Ruby developer")

    result = memory.run(action="search", query="texas")

    assert "home_city: Dallas, Texas" in result
    assert "role: Ruby developer" not in result


def test_search_finds_by_key_keyword(memory):
    memory.run(action="write", key="favorite_language", value="Ruby")
    result = memory.run(action="search", query="language")
    assert "favorite_language: Ruby" in result


def test_search_no_match_is_a_clean_message(memory):
    memory.run(action="write", key="city", value="Dallas")
    assert "No memories match 'tokyo'" in memory.run(action="search", query="tokyo")


def test_search_tolerates_fts_operator_characters(memory):
    """User text is quoted into the MATCH expression, so punctuation never errors."""
    memory.run(action="write", key="note", value="meeting at 9am")
    # A bare '(' / 'OR' / '"' would be FTS syntax if not escaped — must not raise.
    assert "No memories match" in memory.run(action="search", query='( OR "')


def test_search_reflects_a_delete(memory):
    """The FTS index tracks deletes — a forgotten fact stops matching."""
    memory.run(action="write", key="city", value="Dallas")
    memory.run(action="delete", key="city")
    assert "No memories match 'dallas'" in memory.run(action="search", query="dallas")


def test_search_falls_back_to_substring_when_fts5_is_absent(tmp_path, monkeypatch):
    """On a SQLite build without FTS5, search degrades to a substring scan that still
    does per-term OR recall and treats LIKE wildcards as literal, not match-all."""
    monkeypatch.setattr("basecradle_harness._memory._fts5_available", lambda conn: False)
    tool = MemoryTool(path=tmp_path / "memory.db")
    tool.run(action="write", key="home_city", value="Dallas, Texas")
    tool.run(action="write", key="role", value="Ruby developer")

    assert tool.store._fts is False  # the fallback path is genuinely exercised
    # Per-term OR recall: order-independent, like the FTS path.
    assert "role: Ruby developer" in tool.run(action="search", query="developer Ruby")
    # A bare '%' is a literal, not a match-everything wildcard.
    assert "No memories match '%'" in tool.run(action="search", query="%")
    tool.close()


def test_search_reflects_an_overwrite(memory):
    """The FTS index tracks updates — the old value no longer matches, the new one does."""
    memory.run(action="write", key="city", value="Dallas")
    memory.run(action="write", key="city", value="Austin")
    assert "No memories match 'dallas'" in memory.run(action="search", query="dallas")
    assert "city: Austin" in memory.run(action="search", query="austin")


# --- input guards (model-friendly strings, not exceptions) -------------------


def test_write_without_value_is_rejected(memory):
    assert "needs both a key and a value" in memory.run(action="write", key="city")


def test_read_without_key_is_rejected(memory):
    assert "needs a key" in memory.run(action="read")


def test_delete_without_key_is_rejected(memory):
    assert "needs a key" in memory.run(action="delete")


def test_search_without_query_is_rejected(memory):
    assert "needs a query" in memory.run(action="search", query="   ")


def test_unknown_action_is_rejected(memory):
    assert "unknown action" in memory.run(action="forget", key="city")


# --- persistence -------------------------------------------------------------


def test_memory_survives_across_instances(tmp_path):
    """A new tool on the same file recalls what an earlier one wrote.

    This is the across-turns / across-runs guarantee: each wake is a fresh process
    that builds a fresh tool, and the fact is still there.
    """
    path = tmp_path / "memory.db"
    first = MemoryTool(path=path)
    first.run(action="write", key="city", value="Dallas")
    first.close()

    later = MemoryTool(path=path)
    assert later.run(action="read", key="city") == "Dallas"
    later.close()


def test_no_file_is_written_until_the_first_call(tmp_path):
    path = tmp_path / "memory.db"
    reader = MemoryTool(path=path)
    assert not path.exists()  # construction touches no disk
    assert "No memory stored under 'city'" in reader.run(action="read", key="city")
    reader.close()


def test_default_path_lives_under_harness_home(tmp_path, monkeypatch):
    """With HARNESS_HOME set, the store is the agent's private file in its own home."""
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    tool = MemoryTool()
    assert tool.path == tmp_path / "memory.db"
    tool.close()


# --- schema migration --------------------------------------------------------


def test_fresh_db_is_migrated_to_the_target_version(tmp_path):
    """A brand-new DB self-migrates to the code's target schema version on open."""
    path = tmp_path / "memory.db"
    tool = MemoryTool(path=path)
    tool.run(action="list")  # forces the lazy open + migration
    tool.close()

    assert _user_version(path) == SCHEMA_VERSION


def test_old_code_opens_a_newer_additive_db(tmp_path):
    """Additive proof: a DB written by *newer* code (higher version, an extra column)
    still opens and reads under this code, which targets a lower version.

    Simulates the uneven-rollout case the migration discipline exists for: server B
    upgraded and bumped the schema; server A is still on the old code and must keep
    working against the same DB.
    """
    path = tmp_path / "memory.db"
    # Build a normal DB at the current version, then fast-forward it as a future,
    # additive migration would: add a column and bump user_version past the target.
    seed = MemoryTool(path=path)
    seed.run(action="write", key="city", value="Dallas")
    seed.close()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE memories ADD COLUMN tags TEXT")  # additive: new column
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    # This code targets SCHEMA_VERSION; it must not try to re-run anything, and must
    # read the row written before the column existed.
    tool = MemoryTool(path=path)
    assert tool.run(action="read", key="city") == "Dallas"
    assert _user_version(path) == SCHEMA_VERSION + 1  # untouched — never downgraded
    tool.close()


# --- the tool contract & the safe profile ------------------------------------


def test_spec_advertises_the_full_action_enum(memory):
    spec = memory.to_spec()
    assert spec.name == "memory"
    assert spec.parameters["properties"]["action"]["enum"] == [
        "write",
        "read",
        "list",
        "delete",
        "search",
    ]


def test_memory_loads_under_the_locked_profile(tmp_path):
    """The shipped example must register on the safe default — it needs no capability."""
    assert MemoryTool().requires == frozenset()
    registry = ToolRegistry(policy=Policy.locked())
    registry.register(MemoryTool(path=tmp_path / "memory.db"))
    assert "memory" in registry


def test_runs_through_the_registry(memory):
    registry = ToolRegistry()
    registry.register(memory)
    registry.run("memory", action="write", key="city", value="Dallas")
    assert registry.run("memory", action="read", key="city") == "Dallas"


# --- helpers -----------------------------------------------------------------


def _timestamps(memory, key):
    """(created_at, updated_at) for a key, read straight from the DB."""
    conn = memory.store._connect()
    return conn.execute(
        "SELECT created_at, updated_at FROM memories WHERE key = ?", (key,)
    ).fetchone()


def _user_version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
