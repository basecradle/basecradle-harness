"""The palace binding is published where MemPalace reads it, judged by MemPalace (issue #1026).

`tests/test_mempalace.py` pins the publisher against a fake ``mempalace.config`` that says where
upstream reads. What a fake cannot say is where upstream *does* read, and since 3.10 that is a
choice of its own: ``MEMPALACE_CONFIG_DIR``, else ``~/.mempalace`` when it holds an install, else
``$XDG_CONFIG_HOME/mempalace``, else ``~/.config/mempalace``. The publisher used to assume
``~/.mempalace``; over an existing XDG ``config.json`` that write created an install upstream then
read instead, so the operator's settings silently stopped being read (measured on 3.11.0). Each case
here publishes, then asks upstream's own ``MempalaceConfig`` what it now reads.

Marked ``mempalace`` and excluded from the default run: run it with
``uv run --extra mempalace pytest -m mempalace``. It reads and writes only under a temporary
``HOME``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.mempalace

config = pytest.importorskip("mempalace.config")

from basecradle_harness._mempalace import cli_config_dir, publish_palace_binding  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A home of its own, with none of the variables that move upstream's config directory."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    for var in ("MEMPALACE_CONFIG_DIR", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    return fake


def _upstream():
    return config.MempalaceConfig()


def test_a_fresh_home_is_published_where_upstream_reads_and_makes_no_install_elsewhere(home):
    palace = home / "agent" / "mempalace"

    written = publish_palace_binding(palace)

    assert written == home / ".config" / "mempalace" / "config.json"
    assert _upstream().config_dir == home / ".config" / "mempalace"
    assert _upstream().palace_path == str(palace)
    assert not (home / ".mempalace").exists()  # no install upstream would switch to


def test_an_existing_xdg_config_keeps_being_read_with_every_setting_it_had(home):
    """The case the old publisher broke: its write made ``~/.mempalace`` an install, and
    upstream's ``collection_name`` fell back to its default."""
    xdg = home / ".config" / "mempalace"
    xdg.mkdir(parents=True)
    (xdg / "config.json").write_text(
        json.dumps({"collection_name": "operator_drawers", "backend": "chroma"}), encoding="utf-8"
    )
    palace = home / "agent" / "mempalace"

    publish_palace_binding(palace)

    upstream = _upstream()
    assert upstream.config_dir == xdg
    assert upstream.collection_name == "operator_drawers"
    assert upstream.palace_path == str(palace)
    assert not (home / ".mempalace").exists()


def test_a_config_dir_named_by_the_environment_is_the_one_published_and_read(home, monkeypatch):
    """Before, the publication landed in ``~/.mempalace`` and upstream never read it."""
    elsewhere = home / "elsewhere"
    monkeypatch.setenv("MEMPALACE_CONFIG_DIR", str(elsewhere))
    palace = home / "agent" / "mempalace"

    written = publish_palace_binding(palace)

    assert written == elsewhere / "config.json"
    assert _upstream().palace_path == str(palace)


def test_a_home_that_already_holds_an_install_is_published_in_place(home):
    """Every fleet box today: ``~/.mempalace/config.json`` exists, so upstream reads it, and the
    publication stays there, merged."""
    legacy = home / ".mempalace"
    legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps({"backend": "chroma"}), encoding="utf-8")
    palace = home / "agent" / "mempalace"

    written = publish_palace_binding(palace)

    assert written == legacy / "config.json" == cli_config_dir() / "config.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "backend": "chroma",
        "palace_path": str(palace),
    }
    assert _upstream().palace_path == str(palace)
    assert not (home / ".config" / "mempalace").exists()
