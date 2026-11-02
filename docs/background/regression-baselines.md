# Regression baselines and the CI coupling gate

Each test case is pinned to a regression baseline:
a recorded, known-good result that every pull request is checked against.
This stops a change to a diagnostic's results from landing unnoticed.

For the day-to-day workflow see
[Testing diagnostics](../how-to-guides/testing-diagnostics.md#regression-baselines).

## Why baselines?

A change to a provider, a dependency or the REF can shift a diagnostic's results
without any unit test failing.
A shifted result looks as plausible as a correct one.
Re-running every diagnostic on every pull request would catch this,
but the input data is too large to download each time
and some diagnostics require significant compute.
So CI checks recorded results instead.

## What a baseline holds

- The **committed bundle** is the small JSON output in `regression/`
  (`series.json`, `diagnostic.json`, `output.json`), tracked in git.
  Paths are replaced with placeholders and floats rounded to 7 significant figures,
  so it is the same on every machine.
  Reviewers see it change in the diff.
- The **native files** are the large outputs, such as netCDF and plots.
  They are stored by sha256 in an object store that anyone can read.
  Only a mint writes them.
- **`manifest.json`** ties the two together.
  It records the digests of both, a hash of the input `catalog.yaml`,
  and two version numbers that only go up:
  `test_case_version`, bumped by each mint that changes the bundle,
  and `diagnostic_version`, the `Diagnostic.version` at mint time.

A baseline can have no native files yet, e.g. one from a fork that cannot mint.
It is still checked through its committed bundle.

## Commands

| Command                 | Credentials | What it does                                                                                                    |
| ----------------------- | ----------- | --------------------------------------------------------------------------------------------------------------- |
| `ref test-cases run`    | none        | Runs the diagnostic and rebuilds the bundle. Writes `regression/` only the first time, or with `--force-regen`. |
| `ref test-cases replay` | none        | Downloads the native files, rebuilds the bundle from them without running the diagnostic, and compares.         |
| `ref test-cases mint`   | write       | Runs the diagnostic, uploads the native files and updates the manifest. Usually run by CI.                      |

Each command writes its output to `output/<label>/` in the test case directory, `latest` by default.
Named slots are kept, so `--label before` and `--label after` give two runs to compare.

Replay compares numbers with a relative tolerance of `1e-5`,
which is coarser than the rounding, so rounding cannot change a verdict.

## The CI coupling gate

`ref test-cases ci-gate` compares each test case's manifest with the base branch and picks one action:

- **`fail`**: the baseline does not match the code or its inputs.
  For example, `Diagnostic.version` is not the version in the manifest,
  `regression/` changed without a version bump,
  or `catalog.yaml` changed without a new baseline.
- **`replay`**: something that could change the results moved, such as code in the provider package,
  and native files exist to check against.
- **`skip`**: nothing relevant changed, or there are no native files to replay yet.
  Review of the diff is then the only check.

Any state the gate cannot verify is a `fail`, except a missing native baseline, which is a `skip`.
The full decision logic is in `climate_ref_core.regression.gate`.

## Test-case catalogs

- `catalog.yaml` lists the datasets of a test case. It is tracked in git.
- `catalog.paths.yaml` holds where each dataset is on this machine.
  It lives in the dataset cache, at `$REF_DATASET_CACHE_DIR/test-case-paths/`,
  or in the user cache directory when that is unset.
  `ref test-cases fetch` rewrites it each time.

CI runs `ref test-cases fetch --strict`, which fails if any case cannot be fetched.
Provider test jobs set `REF_TEST_CASES_STRICT=true`, so a missing baseline fails instead of being skipped.
Diagnostics listed in `REF_TEST_CASES_SKIP` are still skipped.

## Continuous integration

| Workflow | Trigger | Credentials | What it does |
| --- | --- | --- | --- |
| `regression-pr-gate.yaml` | every pull request | none | Runs the gate, then replays the cases it picked. |
| `regression-mint.yaml` | manual | R2 write | Mints baselines and commits them to the branch. |
| `regression-diff-report.yaml` | after a mint, or manual | R2 write | Builds an HTML diff of the changed baselines and links it on the PR. |
| `regression-drift.yaml` | nightly | none | Replays every baseline, to catch drift such as after a dependency upgrade. |

Run the PR gate locally with `make regression-gate`.

### Gated mint

A maintainer dispatches `regression-mint.yaml` on the pull request's branch.
It needs approval through the `native-baselines` GitHub Environment and refuses to run on `main`.
Its inputs are `provider`, `diagnostic` (one or more, blank for all), `test_case`,
`bump_version` and `dry_run`.
It commits the new `regression/` and `manifest.json` to the branch, then builds the diff report.
With `dry_run`, it only checks and previews, and uploads and commits nothing.
If one diagnostic fails, the rest are still committed.

/// warning | The mint commit does not re-run the PR gate

GitHub does not start workflows for a push made with `GITHUB_TOKEN`.
Push another commit, or re-run the gate from the Actions tab.

///

The `native-baselines` Environment holds the R2 secrets
`R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.

### Diff report

The report shows each changed baseline: images side by side, line diffs for text,
and per-variable statistics for netCDF.
It is published at `https://reports.baselines.climate-ref.org/<pr>/<sha>/index.html`
and linked in a comment on the pull request.

After minting by hand, dispatch the report yourself, or build it locally:

```bash
gh workflow run regression-diff-report.yaml --ref <branch>
uv run ref test-cases diff --base origin/main --html-dir out/
```

The report workflow needs the R2 secrets `R2_REPORTS_ACCESS_KEY_ID` and `R2_REPORTS_SECRET_ACCESS_KEY`.
