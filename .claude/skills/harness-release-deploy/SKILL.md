---
name: harness-release-deploy
description: Step-by-step procedure for releasing and deploying basecradle-harness — the OIDC Trusted-Publishing pipeline (v* tag → TestPyPI rehearsal → capital-approved pypi env-gate → PyPI), the contractual workflow/environment names, the four-owner build→publish→deploy→verify flow, and the @jt verify (the token-free plumbing check, then the real model wake on a temporary timeline that proves the release). Use when cutting a release, bumping the version for a release, waiting on or reasoning about the pypi env-gate, or confirming a release reached and converged the fleet. The standing invariants (released ≠ deployed; no closing keyword on release PRs; the capital not @origin actuates publish) live in CLAUDE.md → Releasing.
---

# Harness Release + Deploy Procedure

The invariants live in `CLAUDE.md` → "Releasing" and govern at all times:
- **A release is not done at PyPI — not done until the fleet is deployed AND verified live** (the
  recurring *released ≠ deployed* failure class).
- **No closing keyword on a release PR** — close the release issue by hand, only after the package
  is verified live on PyPI, recording version + URL in the closing comment.
- **The capital, not @origin, actuates publish** (`constitution.md` → Earned Autonomy).

This skill is the step-by-step pipeline behind them.

## The pipeline (OIDC Trusted Publishing, zero stored credentials)

Mirror the Python SDK's pipeline — `../sdks/python/.github/workflows/release.yml` is the template:

1. Push a `v*` git tag →
2. **preflight** (the coherence gate — see below) →
3. build →
4. **TestPyPI** rehearsal →
5. the **capital** approves the `pypi` env-gate →
6. **PyPI**.

### Preflight — the tag cannot lie (issue #308)

Nothing is built or published until `scripts/check_release.py preflight <tag>` passes. It asserts
three things, each of which has actually gone wrong here:

- **The tag matches `_version.py`.** Hatchling reads the version from the *code*, never from the
  tag — so a `v0.72.0` tag on a commit that still says `0.71.0` publishes a **0.71.0 wheel**,
  silently and under the wrong number.
- **That version has a `## [x.y.z]` section in the CHANGELOG.** No undocumented releases.
- **`[Unreleased]` is empty.** It is a legitimate staging area — a PR may land content there
  without bumping, and a release commit promotes it — right up until a tag claims otherwise.
  Tagging over staged changes publishes a version whose own changelog calls its headline fix
  unreleased. That was issue #308's exact state, and it now cannot ship.

**If preflight fails, do not force the tag.** Fix `main` (promote `[Unreleased]` into the release's
section, bump `_version.py`), delete and re-push the tag. The same script runs on every PR as the
`changelog` CI job, in `repo` mode, so `main` should already be coherent when you tag.

**Contractual names — never rename:** the workflow filename and the environment names
`testpypi` / `pypi` match the Trusted Publisher registrations on PyPI/TestPyPI. Renaming any of
them breaks the trust relationship. The `pypi` environment's required reviewer is `drawkkwast` as
a *config* fact, but that credential is operated by the **capital** via local `gh` — @origin
is out of the publish loop.

## The four-owner flow (keep the owners separate)

Constitution baselines: **basecradle#362** (one deployer for the fleet's machines: the NOC) and
**basecradle#363** (a captain *builds* software but never *deploys* it).

1. **Build — the harness captain (you).** Implement the change, bump the version, update the
   changelog — and leave `[Unreleased]` **empty**, its content promoted into the version's own
   section (`## [x.y.z] - <date>`). Never fold a change into an already-dated section: that is the
   move that erased v0.43.0 and v0.47.0 from the changelog. `python3 scripts/check_release.py repo`
   answers whether `main` is releasable; preflight will refuse the tag if it is not.
   **Your release responsibility ends at the version bump** — you do not publish, deploy, or
   verify on a box. The *change* is still yours to prove before it ships: where its behavior lives
   in a model turn, drive a real wake of the local build on a temporary timeline (and delete it
   after), the same shape as step 4b. Where the laptop cannot reach the platform or the provider,
   say so in the completion comment so step 4b knows it carries the whole proof — never let green
   offline tests stand in for it silently.
2. **Publish to PyPI — the capital.** Owns the `pypi` env-gate.
3. **Deploy / converge the fleet (incl. @jt) — the NOC, the fleet's sole deployer.** The NOC reads
   each box's running version, compares it to the git-tracked desired state, and converges any
   off-target box via its `fleet-upgrade-campaign` (triggered by its release-drift detection).
   **No one hand-runs `pip install -U`** on `/home/jt/venv` — or any agent box — anymore. No
   long-running service to restart (the router spawns `basecradle-harness-wake` fresh per event);
   a wake self-migrates its own DB (SDK schema is forward-only/additive), so no manual migration.
4. **Verify live on @jt + close the handoff — the capital.** Two checks, in order, and only the
   second one proves the release.

   **a. Plumbing (token-free).** After the NOC converges, confirm on-box (not inferred from PyPI):

   ```bash
   /home/jt/venv/bin/basecradle-harness-wake --version   # reports the new version
   ```

   plus a token-free synthetic-probe wake still acking sub-second (the duration check from the box
   docs). `--version` is the cheap, model-free, credential-free probe added for exactly this — it
   is also what the NOC's standing release-drift detection runs on a cadence to fail loud when
   @jt's running version drifts from PyPI latest. **This proves the right bytes are installed and
   the wake path answers. It proves nothing a model turn does**, and a harness release has been
   called verified without a single one.

   **b. Proof (a real wake).** The capital drives **one real model wake** of @jt that exercises
   what the release changed, in the same session as the verify:
   1. Create a **temporary timeline** with @jt on it.
   2. Post the message (or activate the task, or post the asset) that makes @jt do the thing the
      release changed. The router wakes @jt on that event, exactly as it would for a peer.
   3. Read that wake's own log lines on the box: the `wake` bookend, the `llm` line(s), and every
      line the release added or changed. The outcome is judged from what the log and the timeline
      show, never from the absence of an error.
   4. **Delete the timeline.** The orphan sweep (`basecradle-harness-cleanup --sweep`) then GCs its
      on-box session artifacts on its next run; @jt's memory keeps the exchange by design.

   **Nobody waits for traffic to arrive.** A system being built, rolled or verified is not at
   rest, so "no token burn at rest" does not apply to it (`constitution.md` → How We Build,
   "building is not rest"; @origin, 2026-10-03). The verify is driven, never left to a wake that
   might happen by itself — a release whose changed path no peer exercises would otherwise sit
   "verified" forever.

   **Budget.** A few dollars of test tokens needs no permission. A test expected to use more than
   about **one million agent-side tokens** is put to @origin first (`constitution.md` → How We
   Build) — by the capital; a builder raises it to the capital with `needs-capital`, never to a
   founder directly.

The NOC's drift detection is the **backstop**; this documented flow is the primary fix. Neither
replaces the other — the flow keeps a release honest, the drift alarm catches the release whose
deploy step was skipped.
