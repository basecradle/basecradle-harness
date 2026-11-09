"""A shell path that carries a secret prints none of it under ``set -x`` (issue #832).

The class is basecradle#772: a traced shell prints every assignment and every expanded test, so a
secret read into a variable lands in whatever captures stderr. These tests run the repository's
own bytes, the mint recipe out of the ``bot-auth-setup`` skill, the heartbeat step out of the
dependency-scan workflow and the git credential helper out of ``.claude/settings.json`` (issue
#837), under a trace. Each has a control that removes the guard and sees the secret appear, so a
green result means the guard held and not that the trace saw nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / ".claude" / "skills" / "bot-auth-setup" / "SKILL.md"
WORKFLOW = ROOT / ".github" / "workflows" / "dependency-security-scan.yml"
SETTINGS = ROOT / ".claude" / "settings.json"

FAKE_TOKEN = "ghs_Zm9vYmFyYmF6cXV4MDEyMzQ1Njc4OWFiY2Rl"
FAKE_URL = "https://uptime.example.com/api/v1/heartbeat/Zm9vYmFyYmF6cXV4"
MINTER = "~/Documents/claude-workspace/shared/fleet-identity/gh-app-token basecradle-harness-ai"
RECIPE_GUARD = "_x=$-; set +x; "
RECIPE_RESTORE = "; case $_x in *x*) set -x;; esac"
HEARTBEAT_STEP = "- name: Ping the basecradle-harness Dependency Scan heartbeat"
HELPER_GUARD = "set +x; "
HELPER_OPEN = "f() { "

SHELLS = {"bash": ["bash", "-xc"], "zsh": ["zsh", "-xc"]}


def recipes() -> list[str]:
    """Every line of the skill that is the mint recipe, guarded or not."""
    return [
        line
        for line in SKILL.read_text(encoding="utf-8").splitlines()
        if MINTER in line and "GH_TOKEN=" in line
    ]


def stub_minter(tmp_path: Path) -> Path:
    """A minter that prints a correctly-shaped fake token, so no real credential is ever read."""
    stub = tmp_path / "stub-minter"
    stub.write_text(f"#!/bin/sh\nprintf '%s\\n' '{FAKE_TOKEN}'\n", encoding="utf-8")
    stub.chmod(0o755)
    return stub


def run_traced(shell: str, script: str) -> subprocess.CompletedProcess[str]:
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} is not installed")
    return subprocess.run(
        [*SHELLS[shell], script], capture_output=True, text=True, timeout=30, check=False
    )


def heartbeat_script() -> str:
    """The heartbeat step's ``run:`` body, dedented, exactly as the runner receives it."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if HEARTBEAT_STEP in line)
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run]) - len(lines[run].lstrip())
    body: list[str] = []
    for line in lines[run + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    width = min(len(line) - len(line.lstrip()) for line in body if line.strip())
    return "\n".join(line[width:] for line in body) + "\n"


def run_heartbeat(tmp_path: Path, script: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the step under ``bash -x`` with ``curl`` and ``jq`` stubbed, as the runner would."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "curl-args"
    (bin_dir / "curl").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{called}"\n')
    (bin_dir / "jq").write_text("#!/bin/sh\necho clean\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    step = tmp_path / "step.sh"
    step.write_text(script, encoding="utf-8")
    bash = shutil.which("bash")
    assert bash is not None
    done = subprocess.run(
        [bash, "-x", str(step)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HEARTBEAT_URL": FAKE_URL,
            "RUNNER_TEMP": str(tmp_path),
        },
    )
    return done, called


def test_the_skill_spells_the_recipe_one_way_and_every_copy_is_guarded():
    found = recipes()
    assert len(found) == 3, found
    assert len(set(found)) == 1, "a recipe spelled differently from the others drifts"
    assert found[0].startswith(RECIPE_GUARD)
    assert found[0].endswith(RECIPE_RESTORE)


@pytest.mark.parametrize("shell", sorted(SHELLS))
def test_the_mint_recipe_prints_no_token_under_a_trace(shell, tmp_path):
    recipe = recipes()[0].replace(MINTER, str(stub_minter(tmp_path)))

    done = run_traced(shell, f"{recipe}\nprintenv GH_TOKEN\necho still-traced")

    assert done.returncode == 0, done.stderr
    assert FAKE_TOKEN not in done.stderr
    assert done.stdout.splitlines()[0] == FAKE_TOKEN, "the guard must not cost the export"
    assert "echo still-traced" in done.stderr, "the caller's xtrace must be restored"


@pytest.mark.parametrize("shell", sorted(SHELLS))
def test_the_recipe_without_its_guard_does_print_the_token(shell, tmp_path):
    guarded = recipes()[0].replace(MINTER, str(stub_minter(tmp_path)))
    bare = guarded.removeprefix(RECIPE_GUARD).removesuffix(RECIPE_RESTORE)

    assert FAKE_TOKEN in run_traced(shell, bare).stderr


def test_a_failed_mint_still_stops_the_call(tmp_path):
    recipe = recipes()[0].replace(MINTER, "/usr/bin/false")

    done = run_traced("bash", f"{recipe}\necho reached")

    assert done.returncode == 1
    assert "reached" not in done.stdout


def test_the_heartbeat_step_prints_no_url_under_a_trace(tmp_path):
    done, called = run_heartbeat(tmp_path, heartbeat_script())

    assert done.returncode == 0, done.stderr
    assert FAKE_URL not in done.stderr
    assert FAKE_URL in called.read_text(), "the guard must not cost the ping"


def test_the_heartbeat_step_without_its_guard_does_print_the_url(tmp_path):
    script = heartbeat_script()
    assert "set +x\n" in script

    done, _ = run_heartbeat(tmp_path, script.replace("set +x\n", "", 1))

    assert FAKE_URL in done.stderr


def fence() -> dict[str, str]:
    """The ``GIT_CONFIG_*`` pairs of the fail-closed fence, exactly as the session receives them."""
    env = json.loads(SETTINGS.read_text(encoding="utf-8"))["env"]
    return {key: value for key, value in env.items() if key.startswith("GIT_CONFIG_")}


def helper() -> str:
    """The helper's shell body: git strips the ``!`` and hands the rest to ``sh``."""
    return fence()["GIT_CONFIG_VALUE_1"].removeprefix("!")


def run_helper(body: str) -> subprocess.CompletedProcess[str]:
    """Run the helper under a parent's exported ``SHELLOPTS``, the one trace that reaches it.

    Named ``bash`` rather than ``sh``: dash, which is ``sh`` on CI, ignores ``SHELLOPTS``, so a
    control run there would see no token whether or not the guard exists.
    """
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(
        [bash, "-c", f"{body} get"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": "/usr/bin:/bin", "SHELLOPTS": "xtrace", "GH_TOKEN": FAKE_TOKEN},
    )


def git_credential_fill(tmp_path: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    """Ask real git for github.com's credential through the fence, and nothing but the fence."""
    return subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(tmp_path),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            **fence(),
            **extra,
        },
    )


def test_the_credential_helper_prints_no_token_under_an_exported_shellopts():
    done = run_helper(helper())

    assert done.returncode == 0, done.stderr
    assert FAKE_TOKEN not in done.stderr
    assert done.stdout == f"username=x-access-token\npassword={FAKE_TOKEN}\n"


def test_the_credential_helper_without_its_guard_does_print_the_token():
    body = helper()
    assert body.startswith(HELPER_OPEN + HELPER_GUARD), "the guard must be f's first statement"

    done = run_helper(body.replace(HELPER_GUARD, "", 1))

    assert FAKE_TOKEN in done.stderr


def test_git_still_gets_the_token_through_the_guarded_helper(tmp_path):
    done = git_credential_fill(tmp_path, GH_TOKEN=FAKE_TOKEN, SHELLOPTS="xtrace")

    assert done.returncode == 0, done.stderr
    assert f"password={FAKE_TOKEN}" in done.stdout
    # This one bites only where git's ``sh`` is bash; on CI the two ``run_helper`` tests carry it.
    assert FAKE_TOKEN not in done.stderr


def test_the_guarded_helper_still_quits_without_a_token(tmp_path):
    done = git_credential_fill(tmp_path)

    assert done.returncode != 0
    assert "told us to quit" in done.stderr
    assert "password=" not in done.stdout
