# Deploy Units

systemd units authored by the **harness captain** and deployed by the **NOC** (the fleet's
sole software deployer). The captain owns the unit *files*; the NOC owns the *install* — final
paths, hardening, and cadence are the NOC's to tune for each box.

## Orphan-Artifact Sweep (`basecradle-harness-cleanup`)

GCs the on-box artifacts of timelines that no longer exist on the platform. When a Timeline is
destroyed, nothing on the fleet server is cleaned up by itself; the harness persists
per-timeline state under `$HARNESS_HOME` (chiefly the session transcript, which holds the full
conversation). The sweep enumerates those artifacts, asks the platform about each timeline once
(one cheap `timelines.get`, **no model call**), and purges only those it 404s (confirmed
deleted). The **first run on a box is the backfill** — it clears artifacts that accumulated
before the sweep existed.

**Memory is never touched** — `memory.db` (+ `-wal`/`-shm`) and the MemPalace palace dir
persist across timeline deletion by design, so the agent keeps what a peer told it even after
the timeline is gone.

| File | Role |
|---|---|
| `basecradle-harness-cleanup@.service` | oneshot, runs `basecradle-harness-cleanup --sweep` as agent `%i` |
| `basecradle-harness-cleanup@.timer` | schedule (suggested every 30 min) |

`%i` is the agent slug — also its OS user and home (`/home/%i`), per the universal-identity
rule. One instance per agent because each agent's `BASECRADLE_TOKEN` scopes `timelines.get` to
its own visibility — exactly the timelines whose artifacts it holds.

### Install (per agent — NOC)

```bash
# Place (or symlink) the template units, then enable one instance per agent:
cp basecradle-harness-cleanup@.{service,timer} /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now basecradle-harness-cleanup@jt.timer

# One-off run / manual verify:
systemctl start basecradle-harness-cleanup@jt.service
journalctl -u basecradle-harness-cleanup@jt.service --no-pager | tail
```

The service reads `/home/%i/.config/basecradle/agent.env` for `BASECRADLE_TOKEN` and
`HARNESS_HOME` (same file the wake and installer use) and runs the script from the agent's venv
(`/home/%i/venv/bin/...`). Adjust those paths and the `--sweep` cadence to the box.

### Manual ops

A single timeline's artifacts can be purged unconditionally (no platform check) for one-off
cleanup:

```bash
HARNESS_HOME=/path/to/home basecradle-harness-cleanup --timeline <uuid>
```
