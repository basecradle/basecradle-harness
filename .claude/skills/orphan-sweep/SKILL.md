---
name: orphan-sweep
description: The reasoning behind the orphan sweep's OS fence and its error classification — the fleet unit's systemd sandbox (ProtectHome=read-only + ReadWritePaths) and the NOC's three measurements on systemd 255, why the fence is around the home and not around memory, why only a refused write fails the run and every other fault stays a WARNING, why _present exists, the Path.is_dir version split across 3.10–3.14, and shutil.rmtree's abandoned walk. Use before editing _cleanup.py (the purge, stranded-temp and claims passes, _note_blocked, _present, _locked, LockUnavailable), the sweep's exit status or log levels, or anything that proposes a change to the basecradle-harness-cleanup unit's sandbox. The invariants live in CLAUDE.md → Config Home → Security invariants (the memory-persists and OS-fence bullets); the sweep's mechanics are in docs/harness-internals.md; this skill carries the reasoning.
---

# Orphan Sweep — The OS Fence and the Refused-Write Rule

The invariants live in `CLAUDE.md` → "Config Home" → "Security invariants" (the *Memory persists across timeline deletion* and *The OS fences the sweep too* bullets) and govern at all times; the sweep's mechanics — the six artifact dirs, the classify switch, the claims watermark, the stranded-temp pass — are in `docs/harness-internals.md` → "Orphan-Artifact Sweep". This skill carries the reasoning behind the fence and the error classification (issue #536, founder-approved 2026-09-19). Read it in full before changing either.

## The Sandbox — Three Measurements, Each Got Wrong First

The fleet unit sandboxes the sweep to exactly the three directories this module writes — `ProtectHome=read-only` plus `ReadWritePaths` for `$HARNESS_HOME`, the config home, and `-/home/%i/.mempalace` — so "only what it staged or wrote" is enforced by the operating system and not only by an allow-list of filenames. Three facts about that are the NOC's measurements on systemd 255 (basecradle-noc#736), recorded rather than re-derived, and each was got wrong first:

- **`ProtectSystem=strict` does *not* cover `/home`.** It defers `/home` to `ProtectHome=` entirely, so narrowing `ReadWritePaths` under `ProtectHome=false` is byte-identical to granting the whole home. This reads against `systemd.exec(5)`'s own summary of `strict`, which is exactly why it was measured.
- **`ProtectHome=true` is never right.** It makes `/home` empty and blinds both the sweep and the founder's read-everything supervision.
- **A `ReadWritePaths` entry naming a *missing* path fails the unit at namespace setup** (exit 226), so `.mempalace` carries the `-` prefix: an agent on the SQLite provider has none.

## The Fence Is Around the Home

A grant is a whole subtree, so `memory.db` and (by default) the palace sit *inside* `$HARNESS_HOME`, and the code remains the only thing keeping the sweep off them. The unit narrows a bug's blast radius; it does not make one impossible, and a doc that says otherwise is the overclaim to strike.

## Being Wrong Must Be Loud

The sandbox can only be wrong in the safe direction, so what matters is that being wrong is **loud**. Reads are deliberately unrestricted, so a wrong sandbox still enumerates every artifact and still classifies it deleted, and *only the unlink* comes back `EROFS` — a `WARNING` and a zero exit would be a cleanup that reports success having deleted nothing, forever. So every removal the sweep decides to make and cannot is logged at **ERROR naming the path** (`_note_blocked`, one spelling for all three passes) and **exits the run non-zero**, and one refusal never aborts the rest of the sweep.

Where a removal is the unit of judgement the bound is **errno-agnostic** — every `OSError` counts, because enumerating the ways an OS can refuse is the disease a vendor cap table is, and the observable fact is simply that the artifact is still on the box. **That fact is asked by `_present`, because both stdlib spellings answer it wrongly**: `exists` is False for a dangling symlink, and `lexists` swallows every `OSError` and answers False for a path it was refused a look at — the exact shape a wrong sandbox makes — so only `ENOENT`/`ENOTDIR` count as an answer, and **a removal we cannot confirm is a removal we do not claim**.

## Only a Refused Write Is Judged

**The thing judged is a refused *write*, and everything else is a `WARNING` and a clean exit.** This is the half a later "consistency" pass will flatten, and each exclusion is load-bearing:

- **Reading a mark or a seen-set is a *read***, which `ProtectHome=read-only` never refuses, so reporting it would name a writable directory under a message blaming the sandbox.
- **An advisory lock the filesystem will not give** is the skip `_locked(required=True)` has always specified. Left indistinguishable from a refusal (it raised a bare `OSError`; it now raises `LockUnavailable`), an NFS home would report *every* claims directory on the box as un-cleanable, on every run, forever.
- **A path that is merely gone** is a concurrent purge, not a fault.
- **An unparseable watermark** stays a `WARNING`, because promoting it would fail the unit forever over one corrupt file.

## Two Traps a Refactor Will Undo

- **Looking can be refused too, and the version matrix splits on how.** Up to 3.12 `Path.is_dir` re-raises everything but `ENOENT`/`ENOTDIR`/`EBADF`/`ELOOP`, so an unguarded `stat` escapes the sweep and abandons every orphan behind it with nothing in `blocked`; from 3.13 it swallows the same error and the refusal only shows up at the `unlink`, where `_present` has to catch it. Miss either and the sweep is silently green on half the matrix — a local run is one Python, CI's matrix is five.
- **`shutil.rmtree` abandons its walk on the first error**, a vanished sibling included, so the best-effort salvage pass has to run for every error class, or a half-purged directory reads as gone.

## The Honest Limit

**A run with nothing to remove attempts no write**, so a green run is evidence about that run and never a proof that the `ReadWritePaths` list is right.
