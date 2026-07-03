---
name: harness-release-deploy
description: Step-by-step procedure for releasing and deploying basecradle-harness — the OIDC Trusted-Publishing pipeline (v* tag → TestPyPI rehearsal → capital-approved pypi env-gate → PyPI), the contractual workflow/environment names, the four-owner build→publish→deploy→verify flow, and the on-box @jt verify commands. Use when cutting a release, bumping the version for a release, waiting on or reasoning about the pypi env-gate, or confirming a release reached and converged the fleet. The standing invariants (released ≠ deployed; no closing keyword on release PRs; the capital not the founder actuates publish) live in CLAUDE.md → Releasing.
---

# Harness Release + Deploy Procedure

The invariants live in `CLAUDE.md` → "Releasing" and govern at all times:
- **A release is not done at PyPI — not done until the fleet is deployed AND verified live** (the
  recurring *released ≠ deployed* failure class).
- **No closing keyword on a release PR** — close the release issue by hand, only after the package
  is verified live on PyPI, recording version + URL in the closing comment.
- **The capital, not the founder, actuates publish** (`constitution.md` → Earned Autonomy).

This skill is the step-by-step pipeline behind them.

## The pipeline (OIDC Trusted Publishing, zero stored credentials)

Mirror the Python SDK's pipeline — `../sdks/python/.github/workflows/release.yml` is the template:

1. Push a `v*` git tag →
2. build →
3. **TestPyPI** rehearsal →
4. the **capital** approves the `pypi` env-gate →
5. **PyPI**.

**Contractual names — never rename:** the workflow filename and the environment names
`testpypi` / `pypi` match the Trusted Publisher registrations on PyPI/TestPyPI. Renaming any of
them breaks the trust relationship. The `pypi` environment's required reviewer is `drawkkwast` as
a *config* fact, but that credential is operated by the **capital** via local `gh` — the founder
is out of the publish loop.

## The four-owner flow (keep the owners separate)

Constitution baselines: **basecradle#362** (one deployer for the fleet's machines: the NOC) and
**basecradle#363** (a captain *builds* software but never *deploys* it).

1. **Build — the harness captain (you).** Implement the change, bump the version, update the
   changelog. **Your release responsibility ends at the version bump** — you do not publish,
   deploy, or verify on a box.
2. **Publish to PyPI — the capital.** Owns the `pypi` env-gate.
3. **Deploy / converge the fleet (incl. @jt) — the NOC, the fleet's sole deployer.** The NOC reads
   each box's running version, compares it to the git-tracked desired state, and converges any
   off-target box via its `fleet-upgrade-campaign` (triggered by its release-drift detection).
   **No one hand-runs `pip install -U`** on `/home/jt/venv` — or any agent box — anymore. No
   long-running service to restart (the router spawns `basecradle-harness-wake` fresh per event);
   a wake self-migrates its own DB (SDK schema is forward-only/additive), so no manual migration.
4. **Verify live on @jt + close the handoff — the capital.** After the NOC converges, the capital
   confirms on-box (not inferred from PyPI):

   ```bash
   /home/jt/venv/bin/basecradle-harness-wake --version   # reports the new version
   ```

   plus a token-free synthetic-probe wake still acking sub-second (the duration check from the box
   docs). `--version` is the cheap, model-free, credential-free probe added for exactly this — it
   is also what the NOC's standing release-drift detection runs on a cadence to fail loud when
   @jt's running version drifts from PyPI latest.

The NOC's drift detection is the **backstop**; this documented flow is the primary fix. Neither
replaces the other — the flow keeps a release honest, the drift alarm catches the release whose
deploy step was skipped.
