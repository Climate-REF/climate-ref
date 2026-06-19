# Regression baselines and the CI coupling gate

Climate REF pins each test case to a **regression baseline**:
a recorded, known-good output that a pull request is checked against
so that an unintended change in a diagnostic's results cannot land unnoticed.

This page explains the two-layer baseline model,
the lifecycle commands that produce and verify it,
and the CI coupling gate that decides *how* each test case is verified in a pull request.

## The two-layer baseline model

A baseline is split into two layers with very different trust and portability properties.

- The **committed bundle** is the small, text-only CMEC output
  (`series.json`, `diagnostic.json`, `output.json`)
  written into the test case's `regression/` directory and tracked in git.
  Absolute paths are rewritten to portable placeholders
  (`<OUTPUT_DIR>`, `<TEST_DATA_DIR>`) so the bytes are machine independent.
  This bundle is **the gate signal**: it is what review actually sees in the diff.

- The **native bundle** is the heavy binary output
  (`.nc`, `.png`, ...) that the committed bundle references.
  Native files are content-addressed by their sha256 digest in an object store,
  fetched anonymously, and are **never required to be present**.
  They are written only by the credentialed `mint` step.

The two layers are bound by a **`manifest.json`** alongside the bundle, which records:

- `schema` — integer schema version for the manifest format itself (currently `1`).
  The loader rejects manifests whose `schema` does not match the current `SCHEMA_VERSION`,
  so format migrations are detected immediately rather than silently misread.
- `test_case_version` — a monotonic, author-bumped integer that *authorises* a new baseline.
- `committed` — sha256 digests of the committed JSON artefacts, over the exact placeholder-substituted bytes on disk.
- `native` — sha256 + size of each curated native file (empty `{}` until minted).
- `catalog_hash` — the hash of the test case's input `catalog.yaml`, coupling the baseline to the inputs that produced it.

!!! note "An empty native set is a permanent, valid state"
    Fork contributors cannot mint
    (minting needs object-store write credentials that never run on untrusted pull-request code).
    A baseline with `native: {}` is fully gated by its committed JSON bundle;
    the native layer is **opt-in extra verification** that only runs when the blobs exist.

## Lifecycle commands

The `ref test-cases` verbs produce and verify baselines.
Only `mint` needs write credentials; everything else is anonymous and safe on untrusted code.

```mermaid
flowchart LR
    run["run<br/>(no creds)"] -->|"seed committed bundle<br/>+ manifest, native = {}"| repo[("regression/<br/>+ manifest.json")]
    repo --> mint["mint<br/>(write creds, CI)"]
    mint -->|"upload blobs to store<br/>populate manifest.native"| store[("native object<br/>store (sha256)")]
    store --> sync["sync<br/>(public read)"]
    store --> replay["replay<br/>(public read)"]
    sync -->|"fetch blobs locally"| local[(local native)]
    replay -->|"materialise native +<br/>re-run build_execution_result +<br/>tolerant compare"| verdict{matches<br/>committed?}
    verdict -->|yes| ok([pass])
    verdict -->|no| bad([fail])
```

| Verb | Credentials | What it does |
| --- | --- | --- |
| `run` | none | Execute the diagnostic, curate outputs, write the committed bundle and seed `manifest.json` (`native = {}`). |
| `mint` | write | Upload the curated native files to the object store and populate `manifest.native`. Generally run by CI. |
| `sync` | public read | Fetch the native blobs referenced by the manifest into the local tree. |
| `replay` | public read | Materialise the native baseline, re-run only `build_execution_result`, and tolerantly compare to the committed bundle. |
| `check-store` | write | Preflight the writable native store (credentials + bucket) without uploading anything. Run this before a slow `mint` to confirm credentials are correct. |

## CMIP7 test data

CMIP7 data is not yet available on ESGF,
so `CMIP7Request` (importable from `climate_ref_core.esgf`) bridges the gap:
it internally maps CMIP7 facets to their CMIP6 equivalents
(e.g. `variant_label` → `member_id`),
fetches the corresponding CMIP6 files,
and converts them to CMIP7 format on the fly.

Converted files are cached under the user cache directory at `climate_ref/cmip7-converted/`
(resolved by `platformdirs.user_cache_dir`),
so repeated fetches for the same request are cheap.
The cache is not version-controlled; clear it manually if a conversion produces stale output.

Use `CMIP7Request` in `test_data_spec` exactly like `CMIP6Request`:

```python
from climate_ref_core.esgf import CMIP7Request

CMIP7Request(
    slug="cmip7-tas",
    facets={
        "source_id": "ACCESS-ESM1-5",
        "experiment_id": "historical",
        "variable_id": "tas",
        "variant_label": "r1i1p1f1",   # CMIP7 name; maps to member_id internally
        "table_id": "Amon",
    },
)
```

## Committed-bundle float precision

Floats written into the committed bundle (`series.json`, `diagnostic.json`, `output.json`)
are rounded to **7 significant figures** at write time.
This keeps the committed bytes stable and human-reviewable across machines
(local developer run vs. CI mint),
where tiny floating-point differences would otherwise produce noisy diffs on every baseline update.

Seven significant figures is deliberately one digit finer than the `rtol=1e-6`
tolerance used by the coupling gate's tolerant comparison,
so rounding at write time can never flip a gate verdict:
a value that rounds identically on all machines will always compare within tolerance.

## Tolerant comparison

`replay` and the execute path do not require byte-equality of the regenerated JSON.
A byte-equal fast path short-circuits the common case;
otherwise values are compared with a small relative/absolute tolerance
(`math.isclose`), and volatile keys/values are sanitised to placeholders before comparison.
This absorbs platform-level floating-point noise without masking a real change in results.

## The CI coupling gate

For every test case, CI must decide *how* to verify the baseline in a pull request,
purely from what changed relative to the base branch.
`decide_coupling` is a pure function over the head and base manifests
plus three on-disk facts the caller supplies —
whether the diagnostic's extraction code changed,
whether the committed bundle still matches its manifest digests,
and whether the input catalog still matches its manifest hash.

The gate **fails closed**:
any state it cannot positively verify is a failure, not a silent skip.
The single exception is the native layer —
because an empty native set is legitimate, a missing native baseline downgrades to `skip` (with a warning), never `fail`.

### Actions

- **`skip`** — nothing relevant to this case changed, or the case is not under regression management.
- **`replay`** — cheap, anonymous re-check against the cached native baseline (only when native blobs exist). This also verifies a `test_case_version` bump that ships native blobs.
- **`execute`** — full end-to-end re-run with tolerant compare; required when `test_case_version` was bumped to authorise a new baseline *and* no native baseline exists to replay against.
- **`fail`** — an unauthorised or unverifiable change (committed bundle edited without a version bump, version moved backwards, bundle drifted from its manifest, or catalog changed without regenerating the baseline).

### Decision flow

```mermaid
flowchart TD
    start([test case]) --> hasManifest{manifest.json<br/>present?}

    hasManifest -->|no| baseHad{present on<br/>base branch?}
    baseHad -->|yes| failDeleted[["FAIL<br/>managed baseline removed"]]
    baseHad -->|no| skipUnmanaged[["SKIP<br/>not under management"]]

    hasManifest -->|yes| committedOk{committed bundle<br/>matches manifest<br/>digests?}
    committedOk -->|no| failDrift[["FAIL<br/>bundle drifted from manifest"]]

    committedOk -->|yes| catalogOk{catalog.yaml<br/>matches manifest<br/>catalog_hash?}
    catalogOk -->|no| failCatalog[["FAIL<br/>inputs changed without<br/>regenerating baseline"]]

    catalogOk -->|yes| seeding{newly added<br/>this branch?}
    seeding -->|yes| seedNative{native<br/>non-empty?}
    seedNative -->|yes| replaySeed[["REPLAY<br/>seed against committed"]]
    seedNative -->|no| skipSeed[["SKIP<br/>committed-only baseline"]]

    seeding -->|no| versionCmp{version vs base}
    versionCmp -->|decreased| failVersion[["FAIL<br/>version not monotonic"]]
    versionCmp -->|bumped| bumpNative{native<br/>present?}
    bumpNative -->|yes| replayBump[["REPLAY<br/>verify new baseline<br/>reproduces from native"]]
    bumpNative -->|no| execute[["EXECUTE<br/>full re-run<br/>(no native to replay)"]]
    versionCmp -->|unchanged| committedChanged{committed<br/>changed?}

    committedChanged -->|yes| failUnauthorised[["FAIL<br/>baseline changed without<br/>version bump"]]
    committedChanged -->|no| nativeChanged{native<br/>changed?}

    nativeChanged -->|yes| nativeEmpty{"head native<br/>empty?<br/>(de-mint)"}
    nativeEmpty -->|no| replayNative[["REPLAY<br/>confirm new native<br/>reproduces committed"]]
    nativeEmpty -->|yes| skipDemint[["SKIP + WARN<br/>native removed"]]

    nativeChanged -->|no| extractionChanged{extraction<br/>code changed?}
    extractionChanged -->|no| skipNothing[["SKIP<br/>nothing to verify"]]
    extractionChanged -->|yes| extractNative{native<br/>non-empty?}
    extractNative -->|yes| replayExtract[["REPLAY<br/>verify against cached native"]]
    extractNative -->|no| skipExtract[["SKIP<br/>no native to replay"]]
```

### How changed files map to the signals

The gate diffs `base...HEAD` (against the merge base, so unrelated base-branch churn is excluded)
and maps the changed-file list onto each case:

- **`extraction_changed`** is coarse and errs toward `replay` (cheap, credential-free):
  any change under the diagnostic's provider package,
  or under the core regression package, counts for every case in that provider.
- **`committed_integrity_ok`** recomputes the committed digests from the working tree
  and compares them to `manifest.committed`.
- **`catalog_integrity_ok`** recomputes the input catalog hash
  and compares it to `manifest.catalog_hash`,
  catching an input change that was not accompanied by a regenerated baseline.

The gate exits non-zero if any case is gated `fail`;
the `--json` output drives CI's dispatch of the `replay` and `execute` jobs.

!!! warning "Credentials never cross the trust boundary"
    `replay`, `sync`, `run`, and the gate itself are anonymous and safe to run on untrusted pull-request code.
    Only `mint` holds object-store write credentials,
    and it runs exclusively on the trusted-tier runner — never on fork pull-request code.

## Continuous integration

The lifecycle commands are wired into three GitHub Action workflows.
The minting process requires credentials to upload data.
Since this is a public project we have to be careful about when this is run to not leak these credentials.

| Workflow | Trigger | Credentials | What it does |
| --- | --- | --- | --- |
| `regression-pr-gate.yaml` | every pull request | none | Runs the coupling gate, then `replay`s every case it routes to `replay`. |
| `regression-mint.yaml` | manual dispatch | R2 write | `mint`s native baselines and commits the regenerated manifest back to the branch. |
| `regression-drift.yaml` | nightly + manual | none | `sync`s and `replay`s every baseline to catch silent drift. |

### PR gate (`regression-pr-gate.yaml`)

On every pull request, the gate runs `ref test-cases ci-gate --json` against the base
branch and acts on each decision:

- **`fail`** aborts the job with the offending cases and their reasons.
- **`replay`** is verified in place against the public native baseline.
- **`execute`** is surfaced as a warning: a `test_case_version` bump with no native baseline to replay against cannot be verified
   (it would need a full diagnostic run), so the committed bundle is gated by review here and and requires `minting`.

The job runs on the public `ubuntu-latest` runner with no secrets, so it is safe on fork pull requests.
The decision-to-replay fan-out lives in `scripts/ci/regression-pr-gate.sh`.

Neither stage downloads input datasets:
`ci-gate` reads only manifests and the git diff,
and `replay` rebuilds the committed bundle from the public native blobs plus the committed `catalog.yaml`/`manifest.json`
(`build_execution_result` reads only its output directory).
The gate therefore fetches no sample data or ESGF inputs — only the small native blobs for the cases it actually replays.

#### Running the gate locally

The same script is the local entry point, so you can reproduce a pull request's verdict before pushing:

```bash
make regression-gate                          # gate + replay against origin/main
bash scripts/ci/regression-pr-gate.sh main    # or pass any base ref
```

The base ref defaults to `origin/${GITHUB_BASE_REF:-main}`.
Under GitHub Actions the script emits log groups and `::error::`/`::warning::` annotations;
run locally it prints plain output instead.

### Gated mint (`regression-mint.yaml`)

Minting is the only step that writes to the object store,
so it is **manually dispatched** and gated behind the `native-baselines` GitHub Environment.
Dispatch it on the feature branch that should receive the new baseline:
the job runs `mint`, and commits the regenerated `manifest.json` (and committed bundle) back to that branch,
so the change is reviewed through its pull request and no developer ever needs write credentials.
A `dry_run` input previews without uploading or committing, and the job refuses to run on the default branch.

!!! warning "The mint commit does not re-trigger the PR gate"
    The mint job pushes with the default `GITHUB_TOKEN`, and GitHub deliberately does not start new workflow runs for such pushes.
    So the freshly minted manifest is *not* automatically `replay`-verified by `regression-pr-gate.yaml`.
    After minting, push a follow-up commit (or re-run the PR gate from the Actions tab) to verify the new native baseline reproduces the committed bundle.

!!! note "Required repository configuration"
    Create a `native-baselines` Environment (Settings -> Environments) with **required reviewers**,
    and add two secrets to it holding an object-scoped R2 token:

    - `R2_ACCESS_KEY_ID` -> `REF_NATIVE_STORE_ACCESS_KEY_ID`
    - `R2_SECRET_ACCESS_KEY` -> `REF_NATIVE_STORE_SECRET_ACCESS_KEY`

    The endpoint and bucket default to the production R2 account
    (`REF_NATIVE_STORE_S3_ENDPOINT_URL` / `REF_NATIVE_STORE_BUCKET` override them).

### Nightly drift (`regression-drift.yaml`)

A scheduled (and manually dispatchable) job `sync`s every referenced native blob and `replay`s it against the committed bundle.
This catches a baseline that no longer reproduce the committed results within tolerance — for example after a dependency upgrade. It is read-only and credential-free.
