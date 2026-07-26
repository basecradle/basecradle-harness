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

## Paper-Trading Sweep (`basecradle-harness-polymarket-sweep`)

Only for an agent with the **`polymarket_paper`** stem opted in. It settles markets that public
state now reports resolved, fills resting limit orders the book has crossed, and refreshes
mark-to-market prices — appending to that agent's paper ledger under `$HARNESS_HOME/polymarket`
and touching nothing else.

**It makes no model call and sends no wake.** The module it runs imports no provider and no
BaseCradle client, so there is nothing in it that could; the agent finds out what happened on
its next `get_fills` / `get_positions` / `get_orders` (issue #347, NORMATIVE §A3). It therefore
needs **no `BASECRADLE_TOKEN` and no `AI_API_KEY`** — two public, read-only endpoints over
HTTPS GETs is the whole of its network access.

| File | Role |
|---|---|
| `basecradle-harness-polymarket-sweep@.service` | oneshot, runs the sweep as agent `%i` |
| `basecradle-harness-polymarket-sweep@.timer` | schedule — **hourly, and contractually so** |

The hourly cadence is not a suggestion: "hourly sweep plus on any agent tool call touching that
market" is the resting-limit re-check policy, frozen into each epoch's `epoch_open` ledger row.
Changing it changes the simulation's semantics, so move it at an epoch boundary.

```bash
cp basecradle-harness-polymarket-sweep@.{service,timer} /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now basecradle-harness-polymarket-sweep@jt.timer

# One-off run / manual verify:
systemctl start basecradle-harness-polymarket-sweep@jt.service
journalctl -u basecradle-harness-polymarket-sweep@jt.service --no-pager | tail
```

**Integrity — the part that matters most.** The ledger's rows are hash-chained and the on-box
JSONL is only a *spool*: every row is emitted in full as a `polymarket_ledger_row {...}` line
through the harness logger, so journald (which the NOC already ships) holds the authoritative
copy under a user the agent is not. Two things follow for operations:

- The sweep **exits non-zero and writes nothing** if the chain does not verify. Alarm on that
  exit code — it is the governance layer's scoreboard-tampering detector, and the agent-side
  symptom is only that every `polymarket_paper` call returns `ledger_tampered`.
- Verify a box on demand without writing anything:

```bash
basecradle-harness-polymarket-sweep --home /home/jt --verify
# epoch-20260726T183744Z: OK rows=14 head=1abaee5a…
```

Pin `head` + `rows` against the off-box copy: a chain that verifies on-box can still have been
rewritten forward wholesale, and only the shipped copy catches that.

Operator controls (all ledger-only, all append-only):

```bash
# Halt new orders on the current epoch — reads/cancels keep working, place_order returns `frozen`.
basecradle-harness-polymarket-sweep --home /home/jt --freeze "under review"
basecradle-harness-polymarket-sweep --home /home/jt --unfreeze

# Start a fresh epoch (a new bankroll and a new scorecard; the old ledger is never touched).
basecradle-harness-polymarket-sweep --home /home/jt --new-epoch
```
