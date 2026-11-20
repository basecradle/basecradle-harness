"""The standalone census agrees with the shipped rule (issue #950).

`scripts/twin_census_standalone.py` is a copy of the rule that runs on an agent's server with no
install, so its count is only worth posting if it is the count the package's rule gives. These pin
the copy to the package: the same masks, the same keys on every renderer's output, the same reading
of an exchange file, and the same census of a directory.
"""

import ast
import importlib.util
import random
import sys
from pathlib import Path

import pytest

from basecradle_harness import MemoryExchange, MemoryScope, _twins
from basecradle_harness import _wake as wake_module
from basecradle_harness._mempalace import _exchange_markdown, exchange_of
from basecradle_harness._palace_check import twin_census

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "twin_census_standalone.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("twin_census_standalone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Item:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _corpus():
    """Every renderer's output across stamps and bodies, plus replies, as ``(user, assistant)``."""
    rng = random.Random(950)
    bodies = ["ping", "Use timeline 019e7750-66ee-7f53-829f-13a8a710b6da.", "two\n> lines", ""]
    replies = ["", "Here.", "All quiet.\n\n> quoted", "  spaced   out  "]
    out = []
    for _ in range(40):
        at = f"2026-11-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00.{rng.randint(0, 999):03d}Z"
        uuid = f"019e7761-2222-7333-8444-{rng.randint(0, 16**12 - 1):012x}"
        body = rng.choice(bodies)
        message = _Item(created_at=at, user=_Item(handle="john"), content=_Item(body=body))
        task = _Item(
            created_at=at,
            user=_Item(handle="nova"),
            content=_Item(uuid=uuid, activate_at=at.replace("Z", "+00:00"), instructions=body),
        )
        event = _Item(
            created_at=at,
            webhook_endpoint=_Item(user=_Item(handle="nova"), content=_Item(uuid=uuid)),
            content=_Item(
                uuid=uuid, content_type="application/json", payload=body, verified_at_receipt=True
            ),
        )
        for user in (
            wake_module._dialogue_of(message, wake_module._MESSAGES),
            wake_module._dialogue_of(task, wake_module._TASKS),
            wake_module._dialogue_of(event, wake_module._EVENTS),
            wake_module._render_batch([message, message]),
        ):
            out.append((user, rng.choice(replies)))
    return out


def test_the_copy_imports_only_the_standard_library():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= set(sys.stdlib_module_names)


def test_the_copy_carries_the_same_rule(script):
    assert script.TWIN_RULE == _twins.TWIN_RULE
    assert [(p.pattern, p.flags, m) for p, m in script._STAMPS] == [
        (p.pattern, p.flags, m) for p, m in _twins._STAMPS
    ]


def test_the_copy_keys_and_reads_every_exchange_as_the_package_does(script):
    for user, assistant in _corpus():
        assert script.twin_key(user, assistant) == _twins.twin_key(user, assistant)
        filed = _exchange_markdown(
            MemoryExchange(user=user, assistant=assistant, scope=MemoryScope(agent="nova"))
        )
        assert script.exchange_of(filed) == exchange_of(filed)


def test_the_copy_counts_a_directory_as_the_palace_check_does(script, tmp_path, capsys):
    convos = tmp_path / "mempalace" / "conversations"
    convos.mkdir(parents=True)
    for n, (user, assistant) in enumerate(_corpus()):
        exchange = MemoryExchange(user=user, assistant=assistant, scope=MemoryScope(agent="nova"))
        (convos / f"{n:032x}.md").write_text(_exchange_markdown(exchange), encoding="utf-8")

    found = twin_census(tmp_path / "mempalace")
    assert script.main(["twin_census_standalone.py", str(convos)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"exchanges read: {found.total}",
        f"would be kept: {found.kept}",
        f"would be skipped: {found.skipped}",
        f"distinct exchanges the skipped collapse onto: {found.collapsed}",
    ]
    assert found.skipped > 0  # the corpus repeats, so the comparison is not of two zeros


def test_the_copy_names_an_error_by_its_class_alone(script, tmp_path, capsys):
    assert script.main(["twin_census_standalone.py", str(tmp_path / "missing-palace-path")]) == 1
    assert capsys.readouterr().err == "error: FileNotFoundError\n"
