# Deploy Units

systemd units authored by the **harness captain** and deployed by the **NOC** (the fleet's
sole software deployer). The captain owns the unit *files*; the NOC owns the *install* — final
paths, hardening, and cadence are the NOC's to tune for each box.

**These are reference copies; the NOC's are canonical.** The bytes the fleet actually stamps live
in [`basecradle-noc`](https://github.com/basecradle/basecradle-noc)'s own `deploy/`, and both files
here are byte-identical to them as of 0.127.0 (issue #536). Read that as the direction of the
arrow, not as a rule about who may improve a unit: an operator can install these as they stand, a
change worth making is made here and handed to the NOC, and the sandbox below is the NOC's
measurement rather than this repo's guess.

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

### What the sandbox grants, and why (issue #536)

The founder's boundary on the cleanup rule: *we clean what **we** created; an agent's own property
in its home — its workspace, the scripts it wrote, the repositories it cloned, its wallets and
vaults — is never ours to touch, whatever its age.* The sweep's **code** has always honored that
(it removes only the harness's own filenames, in the places the harness stages writes), but a
periodic root-launched unit whose one job is deleting files is where that argument bites hardest,
so the **OS** enforces it too. The write sandbox is exactly the three places
`basecradle_harness._cleanup` writes:

| Granted write access | Why |
|---|---|
| `/home/%i/harness` | `$HARNESS_HOME` — the six per-timeline artifact dirs the sweep purges (`sessions`, `marks`, `seen`, `claims`, `breaker`, `billing`), plus the settled claims and `.pruned-through` records it prunes |
| `/home/%i/.config/basecradle` | the stranded `.basecradle-env.*.tmp` beside `agent.env` (it holds a live `BASECRADLE_TOKEN`), and the in-place token re-mint on a 401 |
| `-/home/%i/.mempalace` | the stranded `.config.json.<hex>.tmp` at its top level. The *sweep* never walks beneath it; the `-` prefix makes a missing directory ignored rather than a unit that refuses to start |

Everything else in the home is **read-only** to this unit — a write there fails `EROFS`. **Reads
stay unrestricted, deliberately**: the sweep must read what it classifies, and full transparency is
the founder's rule — it is *touching* that is fenced, never looking.

**What this does and does not fence, stated precisely.** `ReadWritePaths=` grants a directory **and
everything beneath it**, so the three rows above are subtrees, not top levels. The sweep's own
restraint is narrower than its sandbox in two places worth naming: it touches only the top level of
`~/.mempalace`, and it never enumerates `memory.db` (+ `-wal`/`-shm`) or the MemPalace palace, both
of which live under `$HARNESS_HOME` in the default configuration and are therefore *inside* the
grant. **The code remains the only thing that keeps this unit off the agent's memory.** The sandbox
fences the agent's home — its workspace, its scratch, its repos, its wallets, its `.ssh`, the venv —
and it narrows the blast radius of a bug in the sweep; it does not make one impossible.

Three details are measurements, not preferences, and each was got wrong first:

- **`ProtectHome=read-only` is the directive that enforces all of it.** `ProtectSystem=strict` does
  *not* cover `/home` — it defers `/home`, `/root` and `/run/user` to `ProtectHome=` entirely — and
  `ReadWritePaths=` only ever *adds* write access. With `ProtectHome=false`, narrowing the
  `ReadWritePaths` list is byte-identical to granting the whole home: all six of the NOC's probe
  writes still succeeded. This one is **the NOC's measurement on systemd 255**
  ([basecradle-noc#736](https://github.com/basecradle/basecradle-noc/issues/736)), recorded here
  rather than re-derived — it reads against `systemd.exec(5)`'s summary of `strict`, which is why
  it was measured.
- **Never `ProtectHome=true`.** It makes `/home` inaccessible and empty, which hides the artifacts
  the sweep must classify and blinds the read-everything supervision the founder's rule requires.
- **The `-` prefix on `.mempalace` is load-bearing.** A `ReadWritePaths` entry naming a missing path
  fails the unit at namespace setup (exit 226; `ExecStart` never runs), and an agent on the default
  SQLite memory provider has no `~/.mempalace`.

A sandbox that is wrong can only fail in the safe direction — a sweep that cannot write, never one
that writes too much — but a cleanup that silently stops cleaning is worse than a loud one that
stops. So **every removal the sweep decides to make and cannot is logged at `ERROR` naming the
path, and the run exits non-zero**, which systemd turns into a failed unit:

```
cleanup blocked path=/home/jt/harness/breaker/<uuid>.wakes error="[Errno 30] Read-only file system: '/home/jt/harness/breaker/<uuid>.wakes'"
```

The honest limit: a run with nothing to remove attempts no write, so a green run is evidence about
that run and never a proof that the `ReadWritePaths` list is right.

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
