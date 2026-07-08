# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## climate-ref 0.16.0 (2026-07-08)

### Features

- Metric values now carry a first-class `kind` of either `model` or `reference`,
  so model and reference (observation) values can be distinguished and filtered directly
  rather than inferred from provider-specific conventions.
  Reference series additionally record a stable `reference_id` content hash,
  so an identical observation produced by different executions is recognised as the same value.
  Series can also carry typed presentation metadata (`value_units`, `value_long_name`, `index_units`, and `calendar`). ([#772](https://github.com/Climate-REF/climate-ref/pull/772))
- Every PMP scalar metric value now keys its reference identity on `reference_source_id`
  and declares its role with the first-class `kind` of `model`.
  The ENSO diagnostics previously keyed reference identity on `reference_datasets`,
  so a consumer now reads reference identity the same way for every PMP diagnostic.
  ENSO references that the metrics package scores against two observation datasets jointly
  (for example `HadISST_GPCP-Monthly-3-2`) are kept as a single combined `reference_source_id` value,
  because the score is one value that depends on both references and cannot be split per reference. ([#775](https://github.com/Climate-REF/climate-ref/pull/775))
- ESMValTool series now carry an explicit `kind` of either `model` or `reference`,
  so an observation curve can be distinguished from a model curve directly
  rather than inferred from the presence of a reference source. ([#776](https://github.com/Climate-REF/climate-ref/pull/776))
- ILAMB metric values now carry the shared `kind` field, distinguishing reference (observation) series from model series without the previous `"Reference"` sentinel.
  Series units are presented in a clean CF form so a model series and its reference agree, the typed presentation fields (`value_units`, `value_long_name`, `index_units`, `calendar`) are populated, and comparison scalars record the reference they were scored against so they can be grouped by it. ([#777](https://github.com/Climate-REF/climate-ref/pull/777))
- The obs4REF reference-data registry gained 33 new files covering 12 CEDA
  `obs_for_ref_v2` observational datasets,
  including cloud and radiation observations (ESACCI-CLOUD),
  burnt-area (GFED-5-0), ozone (SAGE-CCI-OMPS), FLUXNET site fluxes,
  soil carbon (HWSD-2-0), CALIPSO ice cloud, WOA-23 ocean biogeochemistry,
  RAPID AMOC transport, snow cover (CCI-CryoClim), and ocean carbon fluxes (Hoffman).
  Older WECANN and WOA2023 entries were replaced by their newer versions. ([#778](https://github.com/Climate-REF/climate-ref/pull/778))
- Ten ILAMB standard diagnostics now evaluate models against the obs4REF reference datasets rather than the bundled ILAMB files.

  This updates the version of:

  - gross primary productivity (WECANN, FLUXNET2015)
  - runoff (LORA)
  - soil carbon (HWSD-2-0)
  - net biome productivity (Hoffman)
  - snow cover (CCI-CryoClim),
  - burnt area (GFED-5-0)
  - sea-surface temperature and salinity (WOA-23),
  - RAPID Atlantic overturning transport.

  ([#779](https://github.com/Climate-REF/climate-ref/pull/779))
- Added `climate_ref.results`, a typed read layer for querying results from notebooks and other tools.
  A single `Reader` exposes per-domain readers for metric values, executions, datasets, diagnostics, and output artifacts, each returning frozen objects that remain usable after the database session closes and can be turned into a pandas DataFrame.

  Added a `ref diagnostics list` command that lists the registered diagnostics with their promoted version and per-diagnostic successful, in-flight, and total execution-group counts.
  Metric values now expose a first-class `kind` (model or reference) field, and `ref datasets list` returns only the latest version of each dataset. ([#780](https://github.com/Climate-REF/climate-ref/pull/780))
- Added `REF_TEST_CASES_SKIP`,
  a comma-separated list of `diagnostic` slugs or `provider/diagnostic` pairs
  that are excluded from both `ref test-cases fetch` and the per-provider no-drift test.
  This keeps diagnostics whose full-resolution ESGF input cannot be fetched within CI limits
  (for example `ilamb/thetao-woa2023-surface`, a 3D ocean field)
  from being downloaded or executed.
  ESGF fetching is now confined to the `populate-cache` CI job. ([#797](https://github.com/Climate-REF/climate-ref/pull/797))
- The ESGF fetch script can now retrieve the obs4MIPs reference datasets used by the PMP diagnostics,
  namely `GPCP-Monthly-3-2`, `TropFlux-1-0`, `CERES-EBAF-4-2`, `HadISST-1-1` and `20CR-V2`.
  Previously these were only available through the obs4REF registry.

  Note that the psl modes of variability diagnostics request `20CR`,
  which ESGF publishes as `20CR-V2`,
  so those diagnostics continue to source their reference data from the obs4REF registry. ([#799](https://github.com/Climate-REF/climate-ref/pull/799))

### Improvements

- Added a `register` argument to `ProviderRegistry.build_from_config`.
  Registration is the only step that writes to the database, so read-only consumers such as the API can now build the registry with `register=False` and serve a database mounted read-only. ([#791](https://github.com/Climate-REF/climate-ref/pull/791))
- Removed the ERA-20C and ERA-Interim reference datasets from the obs4REF and sample data registries,
  reducing the volume of data fetched by `ref datasets fetch-data --registry obs4ref`.
  No diagnostic used either dataset,
  and neither is available from obs4MIPs.
  The unrelated ERA-5 dataset is unaffected.

  If you have already fetched the sample data,
  re-fetch it with `fetch_sample_data(force_cleanup=True)` so that the removed files are cleared from your local copy. ([#799](https://github.com/Climate-REF/climate-ref/pull/799))

### Bug Fixes

- Made ECS diagnostic work with CMIP7 data. ([#671](https://github.com/Climate-REF/climate-ref/pull/671))
- Made TCR diagnostic work with CMIP7 data. ([#686](https://github.com/Climate-REF/climate-ref/pull/686))
- Widened the contiguous-timerange tolerance so monthly datasets split across multiple files are no longer dropped as false-positive gaps. ([#774](https://github.com/Climate-REF/climate-ref/pull/774))
- Committed regression baselines no longer embed the machine-specific source-checkout path or the
  host operating system in provider command-line provenance.
  Both are now redacted to portable ``<SOURCE_DIR>`` and ``<OS>`` placeholders,
  so a baseline minted on one platform (for example macOS) reproduces one re-derived on another
  (for example Linux in CI) instead of drifting. ([#775](https://github.com/Climate-REF/climate-ref/pull/775))
- Fixed execution listings duplicating an execution group when two of its executions shared the same timestamp; each group now resolves to exactly one latest execution. ([#786](https://github.com/Climate-REF/climate-ref/pull/786))
- Decoupled fetching of the ignore datasets file from configuration loading;
  the file is now fetched lazily at solve time rather than while the configuration is loaded,
  so read-only commands no longer perform network I/O.
  Added an `ignore_datasets_url` setting (env `REF_IGNORE_DATASETS_URL`)
  whose empty-string value disables fetching entirely for offline and air-gapped deployments.
  A failed download no longer creates an empty placeholder file;
  an existing cached copy is reused unchanged,
  and a solve with no cached copy fails loudly rather than running without ignore-dataset protections. ([#792](https://github.com/Climate-REF/climate-ref/pull/792))
- Fixed the ``LocalExecutor`` per-task timeout counting time spent waiting in the process-pool queue as execution time. The budget is now measured from when a worker actually starts a task, so a large backlog of executions queued behind other work is no longer culled en masse once it ages past the timeout. ([#795](https://github.com/Climate-REF/climate-ref/pull/795))
- Fixed ``ref datasets ingest --n-jobs`` greater than 1 crashing (SIGSEGV/SIGBUS) with the default``complete`` CMIP6 parser.
  Parsing previously fanned out across threads, but the netCDF4/HDF5 wheels published on PyPI are not thread-safe.
  Parsing now runs using a process pool. ([#796](https://github.com/Climate-REF/climate-ref/pull/796))
- Fixed three integration-test failures caused by the self-hosted runner's shared cache
  and stable output directory leaking into tests that assumed a clean, default environment. ([#797](https://github.com/Climate-REF/climate-ref/pull/797))
- The ESGF fetch script now exits with a non-zero status if any request failed,
  rather than reporting `0 datasets` and exiting successfully.
  Transient ESGF access errors are retried with an exponential backoff,
  configurable via `--max-attempts` and `--retry-delay`.
  A request that matches no datasets is reported but is not treated as a failure. ([#799](https://github.com/Climate-REF/climate-ref/pull/799))

### Trivial/Internal Changes

- [#670](https://github.com/Climate-REF/climate-ref/pull/670), [#783](https://github.com/Climate-REF/climate-ref/pull/783), [#787](https://github.com/Climate-REF/climate-ref/pull/787), [#789](https://github.com/Climate-REF/climate-ref/pull/789), [#794](https://github.com/Climate-REF/climate-ref/pull/794)


## climate-ref 0.15.0 (2026-06-30)

### Breaking Changes

- The hash that identifies a diagnostic's input datasets is now computed deterministically across pandas versions and platforms.
  It previously relied on `pandas.util.hash_pandas_object`, whose output changes between pandas releases and platforms.

  Because the underlying hash values change, the committed regression `catalog_hash` values have been regenerated,
  and existing databases will re-run each execution once on first use after upgrading. ([#741](https://github.com/Climate-REF/climate-ref/pull/741))
- The minimum supported Python version is now 3.12; Python 3.11 is no longer supported.
  Update your environment to Python 3.12 or newer before upgrading. ([#745](https://github.com/Climate-REF/climate-ref/pull/745))
- Removed the legacy central regression-testing framework in favour of the native baselines managed by `ref test-cases`.
  The `DiagnosticValidator` and `ExecutionRegression` helpers, the `diagnostic_validation`, `execution_regression`, and `regression_data_dir` pytest fixtures (from `climate_ref.conftest_plugin`), the central `tests/test-data/regression/` baseline tree, and the per-provider `test_diagnostics` / `test_build_results` tests have all been removed.

  Provider regression baselines are now recorded and verified entirely through the `ref test-cases` CLI and the per-package committed bundles (see [Regression baselines](background/regression-baselines.md)). ([#757](https://github.com/Climate-REF/climate-ref/pull/757))

### Features

- Added `ref test-cases mint`, `replay`, and `sync` commands for managing native regression baselines.
  `mint` records a test case's native outputs into the baseline store,
  `replay` regenerates a bundle from the stored outputs and checks it against the committed copy,
  and `sync` downloads the native outputs referenced by committed baselines into the local cache. ([#724](https://github.com/Climate-REF/climate-ref/pull/724))
- Added the `ref test-cases ci-gate` command, which decides how CI should verify each
  test case's regression baseline against the base branch: skip it, replay the cached
  native baseline, re-run the diagnostic in full, or fail an unauthorised change.
  Baselines are now coupled to their input catalogs, so changing a test case's inputs
  without regenerating its baseline is reported as a failure rather than passing silently. ([#727](https://github.com/Climate-REF/climate-ref/pull/727))
- Added a Cloudflare R2 object-store backend for diagnostic regression baselines, so `ref test-cases mint` can publish a test case's native outputs to the shared baseline store rather than only a local directory.
  The store endpoint and bucket are configured through the `REF_NATIVE_STORE_*` settings (defaulting to the project's public baseline bucket), and write credentials are supplied via the environment or a named AWS/R2 profile, never the persisted configuration.

  Minting is also easier to get right.
  `ref test-cases mint` now checks the store credentials and bucket up front and stops with a clear message when they are misconfigured, the new `ref test-cases check-store` command verifies store connectivity without minting, `ref test-cases mint --dry-run` previews what would be minted, and `ref test-cases fetch` restores a missing local paths file on its own so a fresh checkout no longer needs `--force`. ([#732](https://github.com/Climate-REF/climate-ref/pull/732))
- Regression baselines are now verified automatically in CI.
  Every pull request runs the coupling gate and replays each test case's cached native baseline, a nightly job re-checks every baseline for silent drift, and a manually gated workflow mints new baselines to the shared object store behind required reviewers — so write credentials never run on untrusted pull-request code.

  The diagnostic testing guide also gained a "pull request workflow" section explaining what the gate decides for each change and how to publish a new baseline. ([#733](https://github.com/Climate-REF/climate-ref/pull/733))
- ILAMB regression baselines are migrated to the per-package Framework B layout, starting with the `mrsos-wangmao`, `gpp-fluxnet2015` and `lai-avh15c1` cmip6 test cases. Each carries a committed CMEC bundle and a manifest; native blobs are published separately through the gated mint workflow. ILAMB executions now also declare their scalar CSV and netCDF outputs in the CMEC output bundle so they are persisted with the results and can be replayed. ([#738](https://github.com/Climate-REF/climate-ref/pull/738))
- Extended the ILAMB cmip6 regression-baseline migration to the per-package Framework B layout with nine more test cases:
  `gpp-wecann`, `mrro-lora`, `csoil-hwsd2`, `nbp-hoffman`, `snc-esacci`, `burntfractionall-gfed`, `emp-gleamgpcp2.3`, `thetao-woa2023-surface` and `so-woa2023-surface`. ([#743](https://github.com/Climate-REF/climate-ref/pull/743))
- `ref test-cases` runs now materialise their outputs into an inspectable, gitignored output slot
  (`<case>/output/<label>/`) holding the curated native files, the rebuilt committed bundle, and a source stamp,
  so you can see and diff exactly what a run produced.
  Added a `build` verb that rebuilds the committed bundle from an existing slot without re-executing the diagnostic,
  a `mint --from-replay` option that re-authors a baseline from the stored native instead of re-running it,
  and a `--label` option that keeps several runs side by side for comparison. ([#748](https://github.com/Climate-REF/climate-ref/pull/748))
- Added a `--format json` option to `ref datasets list`, `ref providers list`, `ref executions list-groups`, and `ref config list` for machine-readable output with full, untruncated identifiers and paths. ([#765](https://github.com/Climate-REF/climate-ref/pull/765))
- Added `ref config init`, `get`, `set`, `unset`, `edit`, and `validate` commands to make configuration onboarding and management easier from the CLI. ([#768](https://github.com/Climate-REF/climate-ref/pull/768))
- Added a five-minute [Quick Start](getting-started/quickstart.md) guide
  and a new model-vs-observation diagnostic to the example provider,
  `global-sst-bias`,
  which compares modelled global-mean sea surface temperature (`tos`, area-weighted by `areacello`)
  against the HadISST-1-1 observations and emits scalar metrics, the model/reference/bias time series, and figures.
  A curated `quickstart` data registry provides the single reference dataset it needs,
  avoiding the multi-gigabyte obs4REF download. ([#769](https://github.com/Climate-REF/climate-ref/pull/769))
- The CI coupling gate now couples each regression baseline to its diagnostic's `Diagnostic.version`.
  A test case fails the gate when the diagnostic's in-code version no longer matches the version recorded
  when its baseline was minted, so an execution-affecting change can no longer slip through as a green replay
  against a stale committed bundle.
  The new author-declared `diagnostic_version` field lives in the test case `manifest.json` (schema version 2),
  and the new `ref test-cases migrate-manifests` command backfills it across existing baselines.
  A legitimate revert is still permitted: lower the diagnostic version and re-mint the baseline. ([#770](https://github.com/Climate-REF/climate-ref/pull/770))

### Improvements

- ILAMB reference (observational) series are now tagged with their source via the existing
  `reference_source_id` dimension (for example `reference_source_id="WangMao"`),
  so a reference series is self-identifying without adding a new schema field. ([#743](https://github.com/Climate-REF/climate-ref/pull/743))
- Committed regression baseline bundles (`series.json` and `diagnostic.json`) now round floating-point values to seven significant figures when written, giving stable, reviewable bytes that no longer churn between local and CI runs while staying well within the regression comparison tolerance. ([#744](https://github.com/Climate-REF/climate-ref/pull/744))
- Reduced database size by deduplicating the index of series metric values into a shared `index_axis` table, referenced by each series rather than stored inline on every row.
  A database migration backfills existing data losslessly; on SQLite, run `VACUUM` afterwards to release the freed space on disk. ([#747](https://github.com/Climate-REF/climate-ref/pull/747))
- Sped up `ref executions reingest` by eager-loading datasets and batching the database writes, so bulk reingests now scale with the number of executions rather than the number of data files. ([#750](https://github.com/Climate-REF/climate-ref/pull/750))

### Bug Fixes

- Regression `series.json` baselines are now written with a stable, dimension-sorted order. Previously a diagnostic could emit its series in a platform-dependent order, causing a committed bundle minted on one platform to falsely fail the regression gate on another even when every value matched within tolerance. ([#738](https://github.com/Climate-REF/climate-ref/pull/738))
- Selecting the latest dataset version now compares version numbers numerically, so versions like v10 are correctly preferred over v2 (previously the older version could be chosen). ([#739](https://github.com/Climate-REF/climate-ref/pull/739))
- Fixed a bug where a diagnostic whose results failed to ingest was incorrectly recorded as successful with no metric values; such executions are now marked failed and retried on the next solve. ([#740](https://github.com/Climate-REF/climate-ref/pull/740))
- ILAMB regression `output.json` bundles are now written with a stable, sorted block order
  (the plot and data sections are sorted by filename),
  so a baseline regenerated on one machine reproduces byte-for-byte on another
  instead of reordering and producing a spurious diff. ([#743](https://github.com/Climate-REF/climate-ref/pull/743))
- Fixed `ref executions reingest` reporting a reingest as successful when the result failed to re-ingest; a failed re-ingestion is now rolled back (leaving no orphaned execution row or output directory) and reported as a failure instead. ([#749](https://github.com/Climate-REF/climate-ref/pull/749))
- ILAMB metric bundles now always report the `units` attribute as a string (for example `"1"` for dimensionless metrics) instead of sometimes emitting the number `1`,
  keeping the metric schema consistent across diagnostics and stable for regression comparison. ([#755](https://github.com/Climate-REF/climate-ref/pull/755))
- PMP climatology reference datasets whose institution identifier contains a hyphen (for example the `GPCP-3-3` precipitation climatology from `NASA-GISS`) are now resolved by the registry instead of being silently skipped, so diagnostics that depend on them no longer fail with "No datasets found matching facets". ([#761](https://github.com/Climate-REF/climate-ref/pull/761))
- Committed regression baselines are now reproducible from their stored native blobs on any machine: native outputs are rewritten to portable placeholders (`<OUTPUT_DIR>` / `<TEST_DATA_DIR>` / `<SOFTWARE_ROOT_DIR>`) before snapshotting, so re-minting on a different host yields identical digests instead of churning the baseline store. Diagnostics can also declare `reconstruction_inputs` — extra output globs such as ESMValTool `diagnostic_provenance.yml` or the PMP driver `*_cmec.json` — that are persisted into the baseline, so `ref test-cases replay` rebuilds the bundle by re-running `build_execution_result` against the captured inputs. ([#761](https://github.com/Climate-REF/climate-ref/pull/761))
- Fixed minting of the ESMValTool ECS, TCR, TCRE and ZEC diagnostics, which failed with `can only concatenate str (not "datetime.timedelta") to str` when their datasets were loaded from a committed catalog. `get_child_and_parent_dataset` now parses the YAML-serialised string `start_time` back to a calendar-aware cftime datetime before deriving the parent timerange. ([#764](https://github.com/Climate-REF/climate-ref/pull/764))
- `ref --quiet --help` no longer prints debug log messages to stderr, and successful solves no longer log a spurious `ERROR` when an optional dataset source type (such as CMIP7) is not present in the data catalog. ([#765](https://github.com/Climate-REF/climate-ref/pull/765))
- `ref solve --dry-run` is now a read-only preview and no longer creates execution groups in the database.
  It still reports execution groups with stale, abandoned in-progress executions as runnable, matching what the next real solve would pick up.

  `ref solve` now exits with an error and remediation advice when no diagnostic providers are configured, or when a `--provider` filter matches none of the configured providers, instead of exiting successfully without doing anything. ([#766](https://github.com/Climate-REF/climate-ref/pull/766))

### Improved Documentation

- Corrected and expanded the regression-testing documentation, fixing inaccuracies in the diagnostic testing guide and filling gaps in the regression baselines reference. ([#744](https://github.com/Climate-REF/climate-ref/pull/744))
- Consolidated the *In a Nutshell*, *Basic Concepts*, and *Explanation* documentation into a single *Concepts* page, and corrected the documented default dataset parser (the complete parser is the default, not the DRS parser). ([#751](https://github.com/Climate-REF/climate-ref/pull/751))
- Corrected command examples in the documentation and package READMEs that used invalid flags (for example `---output-directory` and `--output directory`) or linked to a renamed helper script (`fetch-esgf.py`). ([#765](https://github.com/Climate-REF/climate-ref/pull/765))

### Trivial/Internal Changes

- [#742](https://github.com/Climate-REF/climate-ref/pull/742), [#744](https://github.com/Climate-REF/climate-ref/pull/744), [#745](https://github.com/Climate-REF/climate-ref/pull/745), [#748](https://github.com/Climate-REF/climate-ref/pull/748), [#755](https://github.com/Climate-REF/climate-ref/pull/755), [#761](https://github.com/Climate-REF/climate-ref/pull/761), [#762](https://github.com/Climate-REF/climate-ref/pull/762), [#764](https://github.com/Climate-REF/climate-ref/pull/764), [#767](https://github.com/Climate-REF/climate-ref/pull/767)


## climate-ref 0.14.7 (2026-06-18)

### Bug Fixes

- Fixed `ref solve --timeout 0` discarding completed work instead of waiting for it.
  A timeout of `0` (or any non-positive value) now means "wait with no time limit",
  so executions are collected, copied to the results directory, and ingested as expected;
  previously they ran but their outputs were left in the scratch directory.
  Use `--no-wait` to queue executions and exit immediately,
  and recover any orphaned scratch outputs from an earlier run with `ref executions reingest --include-failed`. ([#735](https://github.com/Climate-REF/climate-ref/pull/735))
- Fixed `ref executions reingest` failing for ESMValTool diagnostics.
  Reingest copies an execution's outputs to a new directory, but the absolute paths recorded in ESMValTool's provenance files still pointed at the original location,
  which caused an error that silently rolled back the reingest.
  These embedded paths are now re-pointed at the new output directory so the results are reingested as expected. ([#736](https://github.com/Climate-REF/climate-ref/pull/736))


## climate-ref 0.14.6 (2026-06-17)

### Bug Fixes

- Fixed loading the default configuration in environments where the user cache directory cannot be written (for example, read-only or restricted HPC systems).
  Previously, running a `ref` command without an existing configuration file could fail with `Error loading configuration` because the default ignore-datasets file could not be cached.
  The REF now degrades gracefully and continues without the cached ignore-datasets file. ([#734](https://github.com/Climate-REF/climate-ref/pull/734))


## climate-ref 0.14.5 (2026-06-17)

### Bug Fixes

- Made ESMValTool regression baselines deterministic by giving each captured run a stable execution directory instead of a timestamped one, so re-running a test case no longer churns the committed output. ([#714](https://github.com/Climate-REF/climate-ref/pull/714))
- Fixed fetching datasets from ESGF failing with `Must have equal len keys and value when setting with an iterable` under pandas 3.0,
  by running the intake-esgf catalogue build with the legacy object-string dtype. ([#726](https://github.com/Climate-REF/climate-ref/pull/726))

### Trivial/Internal Changes

- [#719](https://github.com/Climate-REF/climate-ref/pull/719), [#720](https://github.com/Climate-REF/climate-ref/pull/720), [#723](https://github.com/Climate-REF/climate-ref/pull/723), [#725](https://github.com/Climate-REF/climate-ref/pull/725)


## climate-ref 0.14.4 (2026-06-04)

### Improvements

- Added GPCP dataset version to 3.3 for the PMP precipitation climatology reference dataset. ([#672](https://github.com/Climate-REF/climate-ref/pull/672))
- Updated GPCP dataset version to 3.3 for the PMP precipitation climatology reference dataset. ([#684](https://github.com/Climate-REF/climate-ref/pull/684))
- Removed the pandas upper bound and fixed pandas 3 compatibility in catalog and ILAMB result handling. ([#708](https://github.com/Climate-REF/climate-ref/pull/708))

### Bug Fixes

- Updates the PMP provider data directory ([#685](https://github.com/Climate-REF/climate-ref/pull/685))
- Fixed a `KeyError: 'branded_variable'` when solving diagnostics against CMIP7 datasets.
  The `branded_variable` facet is now reconstructed when a data catalog is loaded from the database,
  so it is available to data requirement filters. ([#712](https://github.com/Climate-REF/climate-ref/pull/712))

### Improved Documentation

- Added a Zenodo DOI badge to the README and documentation landing page so users can easily find how to cite the REF. ([#634](https://github.com/Climate-REF/climate-ref/pull/634))
- Added a getting-started guide for modelling centres that explains how to evaluate local or pre-publication CMOR-compliant model output with Climate-REF, including deployment options, accessing results, and a suggested first run. ([#709](https://github.com/Climate-REF/climate-ref/pull/709))

### Trivial/Internal Changes

- [#706](https://github.com/Climate-REF/climate-ref/pull/706)


## climate-ref 0.14.3 (2026-05-18)

### Bug Fixes

- Fixed a race condition when trying to create the same `ExecutionGroup` concurrently.
  `Database.get_or_create` and `Database.update_or_create` now wrap the INSERT in a SAVEPOINT
  and re-fetch the winning row on conflict instead of aborting the transaction. ([#679](https://github.com/Climate-REF/climate-ref/pull/679))


## climate-ref 0.14.2 (2026-05-15)

### Bug Fixes

- Changed the behaviour for `register_dataset` treatment of files absent from the current ingest slice as kept-in-place with a warning, instead of raising NotImplementedError. ([#677](https://github.com/Climate-REF/climate-ref/pull/677))


## climate-ref 0.14.1 (2026-05-14)

### Features

- Added a `--chunk-size` option to `ref datasets ingest` (CMIP6 and CMIP7) that streams the catalog in directory-aligned batches
  instead of loading the whole archive into memory at once.
  Peak memory is now bounded by `chunk_size` rather than by the total number of files in the input tree. ([#674](https://github.com/Climate-REF/climate-ref/pull/674))

### Improvements

- Skip the redundant per-dataset `validate_data_catalog` call inside `register_dataset`.
  The production ingest path already validates the catalog (and each streamed chunk) once
  up-front, so the inner re-validation only duplicated work. Cuts ~20% off ingest wall time
  on a 50k-file / 500-dataset synthetic CMIP6 archive. A cheap per-slice guard (no groupby)
  remains so callers that bypass the upstream validation contract still get a clear error
  instead of silently registering inconsistent metadata. ([#675](https://github.com/Climate-REF/climate-ref/pull/675))

### Bug Fixes

- Stopped re-ingesting unchanged CMIP6 files on every run.
  Previously, `ref datasets ingest` reported every file as updated and emitted a
  "Updating file metadata" warning per file even when nothing on disk had changed,
  because the file-metadata comparison treated a `str` loaded from the database as
  unequal to a freshly parsed `cftime.datetime`. Re-ingesting an unchanged directory
  is now a no-op. ([#673](https://github.com/Climate-REF/climate-ref/pull/673))


## climate-ref 0.14.0 (2026-05-12)

### Features

- Write the diagnostic `version` to the database.

  Bumping a diagnostic's ``version`` now creates a fresh execution group
  (preserving the prior version's group and results) instead of overwriting it,
  and each execution is stamped with the provider version that produced it. ([#667](https://github.com/Climate-REF/climate-ref/pull/667))

### Improvements

- Reworked the layout of execution output directories.
  New executions now write to ``<provider>/<diagnostic>/<group_short>/<execution_id>/``
  instead of ``<provider>/<diagnostic>/<dataset_hash>/``,
  so reruns of the same diagnostic group no longer overwrite earlier outputs.
  Existing rows on disk continue to resolve through their stored ``Execution.output_fragment``. ([#655](https://github.com/Climate-REF/climate-ref/pull/655))
- Added diagnostic versioning read-path foundations.
  New database columns track the diagnostic version for each execution group,
  the promoted version for each diagnostic, and the provider version for each execution.
  Default queries now return only results at the promoted version, and the ``ref executions stats`` command reflects the same filter. ([#665](https://github.com/Climate-REF/climate-ref/pull/665))

### Bug Fixes

- `ref datasets ingest` now exits with a non-zero status when one or more input directories fail to ingest,
  so cron- and Kubernetes-based deployments can detect failures.
  The CMIP6, CMIP7, and obs4MIPs adapters also now tolerate individual files with missing DRS components:
  those files are logged and skipped instead of aborting the whole batch. ([#668](https://github.com/Climate-REF/climate-ref/pull/668))


## climate-ref 0.13.2 (2026-05-10)

### Features

- Added support for Python 3.14. ([#625](https://github.com/Climate-REF/climate-ref/pull/625))

### Improvements

- Add extra options for users to ingest options other than base to parsl functions ([#651](https://github.com/Climate-REF/climate-ref/pull/651))
- Updated ESMValCore to v2.14.0 and ESMValTool to 2.15.0.dev15+gdead90ca8. ([#652](https://github.com/Climate-REF/climate-ref/pull/652))

### Bug Fixes

- Fixed `ref test-cases fetch` aborting the entire run when a single test case could not be parsed (for example because of a `PermissionError` on a cached CMIP6 file); the failing test case is now logged as a warning and the loop continues. ([#639](https://github.com/Climate-REF/climate-ref/pull/639))
- Fixed `ref solve` periodically reporting executions as never finishing. Stuck executions left in the in-progress state by a crashed worker, walltime kill, or hung diagnostic are now reaped and retried on the next solve, the CLI `--timeout` default has been raised to 6 hours to match the worker time limit, and per-task timeouts in the local executor cancel hung diagnostics instead of blocking the whole pool. ([#641](https://github.com/Climate-REF/climate-ref/pull/641))

### Improved Documentation

- Updated CITATION.cff with extra author names and orcid ids. ([#636](https://github.com/Climate-REF/climate-ref/pull/636))

### Trivial/Internal Changes

- [#619](https://github.com/Climate-REF/climate-ref/pull/619), [#637](https://github.com/Climate-REF/climate-ref/pull/637), [#638](https://github.com/Climate-REF/climate-ref/pull/638)


## climate-ref 0.13.1 (2026-04-13)

### Features

- Added first-class read-only support and a migration-status helper to the `Database` API.

  `Database.from_config(..., read_only=True)` rewrites file-based SQLite URLs to read-only URI form and skips migrations,
  and `Database.migration_status(config)` reports the current/head revisions and state. ([#624](https://github.com/Climate-REF/climate-ref/pull/624))


## climate-ref 0.13.0 (2026-04-11)

### Features

- Added timeseries extraction for ESMValTool regional historical and ozone polar cap diagnostics, and updated file patterns to match the current ESMValTool output directory structure. ([#607](https://github.com/Climate-REF/climate-ref/pull/607))
- Added `ref executions reingest` command to re-ingest existing execution results without re-running diagnostics. Creates a new immutable execution record with a timestamped output fragment, leaving the original execution untouched. ([#610](https://github.com/Climate-REF/climate-ref/pull/610))
- Added `ref db` CLI subcommand group for database management. Includes commands for running migrations, checking schema status, viewing migration history, creating backups, executing SQL queries, and listing tables. ([#615](https://github.com/Climate-REF/climate-ref/pull/615))

### Improvements

- Fetch ESMValTool recipes when installing the provider. ([#582](https://github.com/Climate-REF/climate-ref/pull/582))
- Unify facet filter parsing across CLI commands. `--filter` in `executions list-groups` and `delete-groups` now supports multiple values for the same key with OR semantics (e.g., `--filter source_id=A --filter source_id=B`), consistent with `--dataset-filter` in `datasets list` and `solve`. ([#613](https://github.com/Climate-REF/climate-ref/pull/613))
- Improved solver performance by batching dataset finalisation before grouping, avoiding redundant file I/O when multiple groups share overlapping datasets. ([#616](https://github.com/Climate-REF/climate-ref/pull/616))

### Bug Fixes

- Fixed CLI test isolation by making the `invoke_cli` fixture depend on the `config` fixture, ensuring tests use an isolated database rather than the user's real one. Also marked `RLIMIT_AS` tests as expected failures on macOS where this resource limit is not supported. ([#611](https://github.com/Climate-REF/climate-ref/pull/611))
- Copied the scratch directory for the previous execution when reingesting ([#612](https://github.com/Climate-REF/climate-ref/pull/612))


## climate-ref 0.12.3 (2026-03-30)

### Features

- Added `ref datasets stats` and `ref executions stats` CLI commands for viewing summary statistics without listing individual records. ([#584](https://github.com/Climate-REF/climate-ref/pull/584))

### Bug Fixes

- Fixed zero-emission-commitment diagnostic failures caused by space-separated `activity_id` values creating path mismatches, incorrect parent timerange computation when datasets are split across multiple files, and models with `esm-1pctCO2` as parent experiment being incorrectly scheduled (see #586). ([#585](https://github.com/Climate-REF/climate-ref/pull/585))
- Fixed experiment selection for computing ZEC. ([#589](https://github.com/Climate-REF/climate-ref/pull/589))
- Solve regression tests now use the local `default_ignore_datasets.yaml` instead of downloading from the `main` branch on GitHub, ensuring tests reflect the current ignore list. ([#606](https://github.com/Climate-REF/climate-ref/pull/606))
- Use flexible time stamp for PMP annual cycle. ([#465](https://github.com/Climate-REF/climate-ref/pull/465))

### Improvements

- Enable multiple-file-input for PMP's variability modes diagnostics. ([#583](https://github.com/Climate-REF/climate-ref/pull/583))

### Improved Documentation

- Added a how-to guide on controlling memory use and parallism during diagnostic execution. ([#591](https://github.com/Climate-REF/climate-ref/pull/591))

## climate-ref 0.12.2 (2026-03-06)

No significant changes.

## climate-ref 0.12.1 (2026-03-06)

### Bug Fixes

- Improved conda error logging by capturing stderr in solve logs and avoiding unnecessary stacktraces. ([#580](https://github.com/Climate-REF/climate-ref/pull/580))

### Trivial/Internal Changes

- [#580](https://github.com/Climate-REF/climate-ref/pull/580)

## climate-ref 0.12.0 (2026-03-03)

### Features

- Added ESMValTool ozone diagnostics. ([#473](https://github.com/Climate-REF/climate-ref/pull/473))
- Metric for diagnostics of double ITCZ was added: Spatial corrleation of simulated DJF precipitation climatology against reference dataset over the area of 20S to 0 latitude and 100 to 210 longitude. ([#557](https://github.com/Climate-REF/climate-ref/pull/557))
- Added lazy loading and finalisation support for CMIP7 datasets via DRS and complete parsers, matching the existing CMIP6 pattern.

  This removes the realm filter for CMIP7 executions as it cannot be parsed from the DRS.
  We strongly recommend the use of branded variables when filtering CMIP7 datasets to properly constrain the expected variables. ([#571](https://github.com/Climate-REF/climate-ref/pull/571))
- Update `ilamb3`, remove the `ohc-noaa` diagnostic and add the `evspsbl-pr` diagnostic. ([#573](https://github.com/Climate-REF/climate-ref/pull/573))
- Added `time_units` and `calendar` metadata to CMIP6 and CMIP7 datasets, enabling proper handling of non-standard CF calendars such as `360_day` and `noleap`. Time values are now stored as `cftime.datetime` objects instead of `datetime.datetime`. ([#574](https://github.com/Climate-REF/climate-ref/pull/574))
- Added CMIP7 data catalog and ESMValTool recipe variants for CMIP7 diagnostics. ([#577](https://github.com/Climate-REF/climate-ref/pull/577))

### Improvements

- Implemented a memory constraint using the environment variable `MEMORY_LIMIT_PARSL_JOB_GB` to set the memory limit (units: GB) for a PARSL worker. ([#464](https://github.com/Climate-REF/climate-ref/pull/464))
- Extracted shared finalisation logic into `FinaliseableDatasetAdapterMixin`, reducing code duplication between CMIP6 and CMIP7 adapters. ([#571](https://github.com/Climate-REF/climate-ref/pull/571))
- Use the metadata from the data catalog instead of reading netCDF files when determining ESMValTool branch times ([#577](https://github.com/Climate-REF/climate-ref/pull/577))

### Bug Fixes

- Fixed fire diagnostic CMIP7 data selection and recipe writing. ([#540](https://github.com/Climate-REF/climate-ref/pull/540))
- Migrated to use a different post-processed JSON file for mapping CMIP6 compound names to CMIP7.
  This fixes some errors when dealing with tasmax/tasmin and the removes the out_name attribute which was correctly included in [#530](https://github.com/Climate-REF/climate-ref/pull/530). ([#547](https://github.com/Climate-REF/climate-ref/pull/547))
- Apply same version filtering logic to the regression tests as loading the data catalog ([#570](https://github.com/Climate-REF/climate-ref/pull/570))

## climate-ref 0.11.1 (2026-02-24)

### Bug Fixes

- Fixed DRS re-ingestion from crashing or regressing already-finalised datasets. Previously, re-ingesting the same directory with the DRS parser would either crash with a `TypeError` due to `pd.NA` comparisons, or overwrite finalised metadata with empty values. Finalised datasets are now skipped during DRS ingestion while still adding any new files.

  Reduced memory usage during dataset ingestion by releasing ORM objects from the SQLAlchemy session after each dataset commit, preventing unbounded memory growth on large archives. ([#567](https://github.com/Climate-REF/climate-ref/pull/567))

## climate-ref 0.11.0 (2026-02-24)

### Breaking Changes

- Changed `get_branding_suffix`, `get_realm`, and `get_cmip7_compound_name` to require a `table_id` parameter in addition to `variable_id`, enabling Data Request compound name lookups. ([#530](https://github.com/Climate-REF/climate-ref/pull/530))
- Failed diagnostic executions now clear the execution group's dirty flag,
  preventing automatic retry on subsequent solves.

  Previously, failed executions were retried indefinitely.
  Use ``ref solve --rerun-failed`` or ``ref executions flag-dirty`` to explicitly retry failed diagnostics.
  The solver also now skips duplicate submissions when an execution with the same dataset hash is already in progress. ([#552](https://github.com/Climate-REF/climate-ref/pull/552))

### Features

- Added a constraint to add the parent experiment. ([#214](https://github.com/Climate-REF/climate-ref/pull/214))
- Added lazy dataset ingestion with two-phase finalisation. Datasets are now bootstrapped from directory structure metadata only (no file I/O), with full metadata extracted lazily at solve time after filtering narrows candidates. This dramatically reduces ingestion time for large CMIP6 archives on HPC parallel file systems. ([#515](https://github.com/Climate-REF/climate-ref/pull/515))
- Added CMIP7 support to all ESMValTool diagnostics using OR-logic data requirements, enabling automatic evaluation of CMIP7 datasets alongside existing CMIP6 support. ([#519](https://github.com/Climate-REF/climate-ref/pull/519))
- Added CMIP7 data requirements and test data specifications for all PMP diagnostics (annual cycle, ENSO, and variability modes). ([#526](https://github.com/Climate-REF/climate-ref/pull/526))
- Added `esgf_data_catalog` test fixture and per-provider solver regression baselines using pre-generated parquet catalogs, enabling solver regression testing without requiring sample data downloads. ([#529](https://github.com/Climate-REF/climate-ref/pull/529))
- Added structured CMIP6-to-CMIP7 variable mappings from the CMIP7 Data Request, with a typed `DReqVariableMapping` class for reliable branding suffix, realm, and output name lookups. ([#530](https://github.com/Climate-REF/climate-ref/pull/530))
- TOML file for QAQC requirement from REF added. ([#532](https://github.com/Climate-REF/climate-ref/pull/532))
- Added dimensions to files produced by ESMValTool diagnostics. ([#534](https://github.com/Climate-REF/climate-ref/pull/534))
- Added CMIP7 data support to ILAMB diagnostics, enabling dual CMIP6/CMIP7 data requirements with branded variable name lookups and dynamic source type detection. ([#535](https://github.com/Climate-REF/climate-ref/pull/535))
- Distinguished system errors (OOM, disk full, worker crash) from diagnostic logic errors when handling execution failures.
  System errors leave the execution group dirty so they are automatically retried on the next solve,
  while diagnostic errors clear the dirty flag to prevent retrying indefinitely with the same data.

  The solver also now skips duplicate submissions when an execution is already in progress for the same dataset hash.

  Added ``--rerun-failed`` and ``--no-wait`` flags to ``ref solve``, and a new ``ref executions fail-running`` command for marking stuck executions as failed. ([#552](https://github.com/Climate-REF/climate-ref/pull/552))
- Added validation that prevents DataRequirements from filtering or grouping on columns that require dataset finalisation, raising a clear error instead of silently producing empty results. ([#561](https://github.com/Climate-REF/climate-ref/pull/561))
- Added `--limit` flag to the `solve` command to cap the number of executions, and `--dataset-filter` option to both `solve` and `datasets list` commands to filter input datasets by facet values before solving. ([#562](https://github.com/Climate-REF/climate-ref/pull/562))

### Improvements

- Improved solver performance by caching slug column lookups and avoiding expensive DataFrame string representation in debug logging. ([#533](https://github.com/Climate-REF/climate-ref/pull/533))
- Removed the `ecgtools` dependency and 16 transitive dependencies (intake, intake-esm, joblib, zarr, etc.) by replacing it with a focused internal catalog builder module. This also eliminates the pydantic v1 deprecation warning that ecgtools was causing. File parsing now shows a tqdm progress bar. ([#558](https://github.com/Climate-REF/climate-ref/pull/558))
- Replaced xarray with netCDF4 for metadata-only reads during dataset ingestion, significantly reducing per-file parsing overhead. ([#559](https://github.com/Climate-REF/climate-ref/pull/559))
- Parallelised the `finalise_datasets` operation for CMIP6 datasets, mirroring the threaded approach used during ingest.
  The number of worker threads is controlled by the existing `n_jobs` parameter on `CMIP6DatasetAdapter`. ([#564](https://github.com/Climate-REF/climate-ref/pull/564))

### Bug Fixes

- Removed deprecated `mix_stderr` parameter from `CliRunner` in the test fixture, fixing compatibility with Click 8.3+. ([#528](https://github.com/Climate-REF/climate-ref/pull/528))
- Fixed a `ValueError` in `AddSupplementaryDataset` when the data catalog contained duplicate index labels for supplementary datasets. ([#537](https://github.com/Climate-REF/climate-ref/pull/537))
- Fixed confusion between variable_id and out_name in fake CMIP7 data. ([#539](https://github.com/Climate-REF/climate-ref/pull/539))
- Improved the resiliance of the celery worker configuration to failures ([#550](https://github.com/Climate-REF/climate-ref/pull/550))
- Mount a ``celeryconfig.py`` via ConfigMap for Flower so ``accept_content`` is read correctly by Celery's config loader. The ``CELERY_ACCEPT_CONTENT`` env var is not picked up by Flower/Kombu directly. The config is user-configurable via ``flower.celeryConfig`` in Helm values. ([#556](https://github.com/Climate-REF/climate-ref/pull/556))
- Fixed `REF_CMIP6_PARSER` and `REF_LOG_FORMAT` environment variables not being applied because the `Config` class was missing the post-init hook for environment variable overrides. ([#561](https://github.com/Climate-REF/climate-ref/pull/561))
- Resolved pandas FutureWarnings to support both pandas 2 and 3, including fixes for DataFrame concatenation with empty or all-NA entries and null-type mismatches in parquet round-trips. ([#565](https://github.com/Climate-REF/climate-ref/pull/565))

### Trivial/Internal Changes

- [#538](https://github.com/Climate-REF/climate-ref/pull/538), [#555](https://github.com/Climate-REF/climate-ref/pull/555)

## climate-ref 0.10.0 (2026-02-10)

### Features

- Added database support for CMIP7 datasets based on the CMIP7 Global Attributes v1.0 specification. ([#503](https://github.com/Climate-REF/climate-ref/pull/503))
- Added CMIP7 data requirements support, enabling providers to fetch CMIP6 data from ESGF and translate it to CMIP7 format using the CMIP7 CV converter. ([#510](https://github.com/Climate-REF/climate-ref/pull/510))
- Added diagnostic summary introspection and auto-generated documentation for all providers. The `ref providers show` command now defaults to detailed list format and supports `--columns` for filtering table output. ([#518](https://github.com/Climate-REF/climate-ref/pull/518))

### Improvements

- Separated model from observation runs for regional historical diagnostics. ([#460](https://github.com/Climate-REF/climate-ref/pull/460))
- Made listing the changed files from a regression test faster. ([#514](https://github.com/Climate-REF/climate-ref/pull/514))
- Prepared the monorepo for splitting diagnostic provider packages into independent repositories. Extracted shared test fixtures into a `climate-ref[test]` pytest plugin, decoupled `climate-ref-core` from application-level types, and added API surface documentation, versioning strategy, provider compatibility CI, and a copier template for bootstrapping new provider repositories. ([#520](https://github.com/Climate-REF/climate-ref/pull/520))

### Bug Fixes

- Fixed CMEC bundle dimension validation to use a subset check instead of exact equality, allowing diagnostics with multiple data requirements to have varying output dimensions. ([#523](https://github.com/Climate-REF/climate-ref/pull/523))

### Trivial/Internal Changes

- [#516](https://github.com/Climate-REF/climate-ref/pull/516)

## climate-ref 0.9.1 (2026-02-05)

### Features

- Added `ingest_data()` lifecycle hook to providers, enabling automatic dataset ingestion during `ref providers setup`. PMP climatology data is now ingested automatically, eliminating the need for a separate manual ingestion step. ([#508](https://github.com/Climate-REF/climate-ref/pull/508))

### Improvements

- Implemented coupled versioning for Helm chart: chart version, appVersion, and default image tag now stay in sync with the application version and are updated automatically by bump-my-version. ([#507](https://github.com/Climate-REF/climate-ref/pull/507))
- Improved CLI performance by skipping database backup for read-only commands like `config list` and `datasets list`. ([#511](https://github.com/Climate-REF/climate-ref/pull/511))
- Improved CLI startup time by deferring heavy imports until needed. ([#512](https://github.com/Climate-REF/climate-ref/pull/512))

### Bug Fixes

- Fixed Helm chart CI to use correct image tag override path (`defaults.image.tag` instead of invalid `climate-ref.image.tag`). ([#507](https://github.com/Climate-REF/climate-ref/pull/507))

### Improved Documentation

- Updated getting started documentation with clearer configuration and dataset download instructions. ([#508](https://github.com/Climate-REF/climate-ref/pull/508))

## climate-ref 0.9.0 (2026-02-03)

### Features

- Added test data management infrastructure for diagnostic development:

  - New `ref test-cases` CLI commands (`fetch`, `list`, `run`) for managing and running diagnostic test cases.
  - ESGF data fetching utilities with support for CMIP6 and obs4MIPs datasets.
  - `TestDataSpecification` and `TestCase` classes for defining reproducible test scenarios.

  ([#475](https://github.com/Climate-REF/climate-ref/pull/475))
- Add functionality to translate a CMIP6 dataset to the new CMIP7 conventions ([#484](https://github.com/Climate-REF/climate-ref/pull/484))
- Added test automation infrastructure for diagnostic testing using test-cases ([#485](https://github.com/Climate-REF/climate-ref/pull/485))
- Added CMIP6 to CMIP7 format conversion command-line script to translate CMIP6 datasets into CMIP7-compatible format. ([#489](https://github.com/Climate-REF/climate-ref/pull/489))
- Added `RegistryRequest` class for fetching datasets from pooch registries (pmp-climatology, obs4ref) instead of ESGF. ([#490](https://github.com/Climate-REF/climate-ref/pull/490))
- Added Helm chart for Kubernetes deployment with automated CI/CD pipeline for building and publishing the chart to GitHub Container Registry, including deployment templates for provider workloads (ESMValTool, PMP, ILAMB), Flower monitoring UI, Dragonfly Redis dependency, and comprehensive integration testing in minikube. ([#492](https://github.com/Climate-REF/climate-ref/pull/492))
- Add CI workflow to verify solve works without network access (ci-offline-solve.yaml).
  This test uses Docker with --network none to block all network access including
  subprocesses. Runs every other day and can be triggered manually. ([#497](https://github.com/Climate-REF/climate-ref/pull/497))
- Added provider lifecycle hooks for offline execution setup. Providers can now implement `setup_environment()`, `fetch_data()`, and `post_setup()` methods to prepare for execution on HPC compute nodes without internet access. A new `ref providers setup` CLI command runs all provider setup hooks, fetching required reference data to the local cache before offline solving. ([#498](https://github.com/Climate-REF/climate-ref/pull/498))

### Improvements

- Clean up the open database connections in the test suite ([#482](https://github.com/Climate-REF/climate-ref/pull/482))
- Improved catalog handling with hash-based change detection and multi-file dataset support. Enhanced CLI `test-cases` commands with new flags: `--only-missing`, `--force`, `--dry-run`, `--if-changed`, and `--clean`. ([#490](https://github.com/Climate-REF/climate-ref/pull/490))
- Updated ESMValTool to v2.13.0 ([#500](https://github.com/Climate-REF/climate-ref/pull/500))

### Trivial/Internal Changes

- [#480](https://github.com/Climate-REF/climate-ref/pull/480), [#506](https://github.com/Climate-REF/climate-ref/pull/506)

## climate-ref 0.8.1 (2026-01-06)

### Bug Fixes

- Add a pin for fastprogress (dependency of intake-esm) to work around bug in newer versions ([#476](https://github.com/Climate-REF/climate-ref/pull/476))

### Trivial/Internal Changes

- [#477](https://github.com/Climate-REF/climate-ref/pull/477)

## climate-ref 0.8.0 (2026-01-05)

### Features

- Enable spatial 3-d variables that has levels for PMP annual cycle. ([#411](https://github.com/Climate-REF/climate-ref/pull/411))
- Added ignore datasets constraint and configuration file.

  If no path is configured for the `ignore_datasets_file` in the configuration file,
  the default ignore datasets file is downloaded from the Climate-REF GitHub repository
  if it does not exist or is older than 6 hours. ([#447](https://github.com/Climate-REF/climate-ref/pull/447))

### Improvements

- Validated the slurm configurations of the HPCExecutor using pydantic ([#375](https://github.com/Climate-REF/climate-ref/pull/375))
- Added reference values to ESMValTool series output. ([#452](https://github.com/Climate-REF/climate-ref/pull/452))

### Bug Fixes

- General fixes found when rerunning, including handling an edge case where no log output is written,
  ignoring empty input directories and increased logging of the number of executions. ([#444](https://github.com/Climate-REF/climate-ref/pull/444))
- Worked around [pydata/xarray#2742](https://github.com/pydata/xarray/issues/2742)
  by always replacing the default fillvalue for the data type with NaN in arrays
  read with Xarray. ([#454](https://github.com/Climate-REF/climate-ref/pull/454))
- Excluded piControl from PMP annual cycle and variability metrics owing to non-overlapping periods with observations. ([#463](https://github.com/Climate-REF/climate-ref/pull/463))

### Improved Documentation

- Add a Jupyter notebook showing how to use the CMIP7 Assessment Fast Track website OpenAPI. ([#466](https://github.com/Climate-REF/climate-ref/pull/466))

## climate-ref 0.7.0 (2025-10-01)

### Breaking Changes

- Use the directory structure template from the obs4MIPs specification to define
  the instance_id. ([#383](https://github.com/Climate-REF/climate-ref/pull/383))
- Used logical or instead of logical and to combine multiple facet filters.

  This means that the selected input datasets will include items that match any of
  the specified `FacetFilter`s, rather than only those that match all of them.
  This change was made to improve the usability of the option to use multiple facet
  filters, allowing users to be more specific with their filtering criteria. ([#414](https://github.com/Climate-REF/climate-ref/pull/414))
- Changed all constraints so they return a filtered dataframe instead of a boolean.

  This allows using the constraints in multi-model diagnostics. ([#416](https://github.com/Climate-REF/climate-ref/pull/416))

### Features

- Add a timerange constraint to ensure the required data is available. ([#399](https://github.com/Climate-REF/climate-ref/pull/399))
- Add a human readable representation for `climate_ref_core.datasets.ExecutionDatasetCollection`. ([#401](https://github.com/Climate-REF/climate-ref/pull/401))
- Allow execution outputs to have dimensions.
  These dimensions allow the outputs to be filtered in the same manner as the metric values ([#434](https://github.com/Climate-REF/climate-ref/pull/434))
- Adds filter options to the `ref executions list-groups` command.
  This allows users to filter by facet, provider and diagnostic,
  as well as whether the latest executions was successful
  or if the exection group was has been marked as dirty ([#438](https://github.com/Climate-REF/climate-ref/pull/438))
- Added a `ref executions delete-groups` command for removing unneeded results from the database.
  This will delete the execution group and any executions and outputs associated with the group. ([#441](https://github.com/Climate-REF/climate-ref/pull/441))

### Improvements

- Updated the sample data to v0.7.3. ([#402](https://github.com/Climate-REF/climate-ref/pull/402))
- PMP version updated to v3.9.2 ([#404](https://github.com/Climate-REF/climate-ref/pull/404))
- Specified the "table_id" for all ESMValTool diagnostics to improve data selection. ([#407](https://github.com/Climate-REF/climate-ref/pull/407))
- Added a groupby operation to the RequireFacets constraint to reduce failed runs because of missing data. ([#408](https://github.com/Climate-REF/climate-ref/pull/408))
- Adds extra ESMValTool dimensions to the output series ([#410](https://github.com/Climate-REF/climate-ref/pull/410))
- The dataset ingestion process now supports updating datasets if new files become available ([#412](https://github.com/Climate-REF/climate-ref/pull/412))
- Added series output for ESMValTool diagnostics. ([#413](https://github.com/Climate-REF/climate-ref/pull/413))
- Gracefully handles empty directories when ingesting.
  The default behaviour is now to skip any invalid datasets as that is a sane default when ingesting many datasets. ([#419](https://github.com/Climate-REF/climate-ref/pull/419))
- Include obs4MIPs and thetao in the ESGF download script and turn into a CLI tool.
  This script was renamed to `fetch-esgf.py` to `fetch-esgf.py` as it now includes more than just CMIP6 data. ([#420](https://github.com/Climate-REF/climate-ref/pull/420))
- Improved support for concatenating historical and future experiments in the ESMValTool recipe ([#431](https://github.com/Climate-REF/climate-ref/pull/431))
- Log the ESMValTool recipe and configuration for easier debugging ([#432](https://github.com/Climate-REF/climate-ref/pull/432))
- Adds output dimensions to ILAMB outputs.
  This adds additional dimensions and attributes to the series values ([#435](https://github.com/Climate-REF/climate-ref/pull/435))
- Replaced instance_id by individual facets in group_by of ESMValTool diagnostics. ([#439](https://github.com/Climate-REF/climate-ref/pull/439))

### Bug Fixes

- Resolves missing index and index names for series ([#410](https://github.com/Climate-REF/climate-ref/pull/410))
- Added missing column to the PMP dataset model ([#418](https://github.com/Climate-REF/climate-ref/pull/418))
- Removes the ERA-5 ta data Obs4REF in preference for Obs4MIPs ([#421](https://github.com/Climate-REF/climate-ref/pull/421))
- Avoid selecting the areacella variable when filling the recipe for cloud scatterplots. ([#427](https://github.com/Climate-REF/climate-ref/pull/427))
- Fixes to the ocean heat content IOMB diagnostic ([#433](https://github.com/Climate-REF/climate-ref/pull/433))

### Trivial/Internal Changes

- [#440](https://github.com/Climate-REF/climate-ref/pull/440)

## climate-ref 0.6.6 (2025-09-10)

### Features

- Added a diagnostic to create scatterplots of two cloud-relevant variables. ([#261](https://github.com/Climate-REF/climate-ref/pull/261))
- Added support for serialising series metric values from an execution.
  These series metric values are also ingested into the database for later retrieval.

  Currently, only the ESMValTool example diagnostic supports this feature,
  but this will be extended to other diagnostics in the future. ([#374](https://github.com/Climate-REF/climate-ref/pull/374))
- Added regional historical changes diagnostics. ([#380](https://github.com/Climate-REF/climate-ref/pull/380))
- Shifts ILAMB executions to be per experiment/model/ensemble/grid as opposed to all models for a given experiment. Also dumps out series information into the database for all time traces found in the output files. ([#391](https://github.com/Climate-REF/climate-ref/pull/391))
- Made it possible to run PMP diagnostics without having conda installed. ([#392](https://github.com/Climate-REF/climate-ref/pull/392))
- Added climate drivers for fire diagnostic. ([#393](https://github.com/Climate-REF/climate-ref/pull/393))
- Added a sub-command to `ref executions` that allows you to flag an execution as dirty. ([#396](https://github.com/Climate-REF/climate-ref/pull/396))

### Improvements

- Use a self-hosted CI runner for the integration tests.
  The GitHub runner do not have enough disk space to store the required datasets,
  and these data are downloaded over the internet on each run. ([#365](https://github.com/Climate-REF/climate-ref/pull/365))
- Updated the sample data to v0.7.1. ([#385](https://github.com/Climate-REF/climate-ref/pull/385))
- Adds additional indexes for some slow queries in the API ([#395](https://github.com/Climate-REF/climate-ref/pull/395))
- Better error message when a user tries to use the HPCExecutor on Windows ([#397](https://github.com/Climate-REF/climate-ref/pull/397))

### Bug Fixes

- Fix bug in writing ESMValTool recipes that loses order of preprocessing steps.
  This bug was introduced in [#378](https://github.com/Climate-REF/climate-ref/pull/378)
  and included in the v0.6.5 release. ([#384](https://github.com/Climate-REF/climate-ref/pull/384))

## climate-ref 0.6.5 (2025-08-25)

### Features

- Add support for NCI PBS Scheduler to the HPCExecutor ([#358](https://github.com/Climate-REF/climate-ref/pull/358))
- Added sea ice sensitivity diagnostic. ([#378](https://github.com/Climate-REF/climate-ref/pull/378))
- Added data required for regional historical changes diagnostics. ([#379](https://github.com/Climate-REF/climate-ref/pull/379))

### Improvements

- Added a dataset ingestor for CMIP6 data that only uses the DRS information present in the filename. This greatly speeds up the process of ingesting a large number of files, as it does not require reading the file contents to extract the DRS information.

  This method can be opted into by specifying `cmip6_parser: "drs"` in the REF configuration file. The default parser remains the `complete` parser, which reads the file contents to all the required metadata, but this will change in a future PR. ([#366](https://github.com/Climate-REF/climate-ref/pull/366))

### Improved Documentation

- Address documentation review from the Model Benchmarking Task Team ([#371](https://github.com/Climate-REF/climate-ref/pull/371))

## climate-ref 0.6.4 (2025-08-04)

### Deprecations

- The `--package` option of the `ref celery start-worker` command has been deprecated and scheduled for removal.
  This functionality is now handled by the `--provider` option which uses entry points declared in the provider packages. ([#367](https://github.com/Climate-REF/climate-ref/pull/367))

### Features

- Use entrypoints to register provider plugins. ([#360](https://github.com/Climate-REF/climate-ref/pull/360))
- Support celery workers to consume tasks for multiple providers ([#367](https://github.com/Climate-REF/climate-ref/pull/367))

### Improvements

- Add additional dimensions to the example metric for testing purposes. ([#372](https://github.com/Climate-REF/climate-ref/pull/372))
- Added a basic script to the CMIP6 data targetted by the current set of diagnostics for the Assessment Fast Track. ([#373](https://github.com/Climate-REF/climate-ref/pull/373))

## climate-ref 0.6.3 (2025-07-17)

### Improvements

- Use a new URL for serving the reference data.
  This should now support older versions of TLS which may help some users. ([#364](https://github.com/Climate-REF/climate-ref/pull/364))

## climate-ref 0.6.2 (2025-07-09)

### Improvements

- Implemented the parsl retry function ([#341](https://github.com/Climate-REF/climate-ref/pull/341))
- Allow arbitrary environment variables to be used in paths in the configuration file. ([#349](https://github.com/Climate-REF/climate-ref/pull/349))
- No longer automatically try to create the conda environment for a provider when running diagnostics. ([#354](https://github.com/Climate-REF/climate-ref/pull/354))
- Use provider conda environments from the configured location when running tests. ([#357](https://github.com/Climate-REF/climate-ref/pull/357))
- Remove the dependency on `ruamel.yaml` in `climate-ref-core` ([#361](https://github.com/Climate-REF/climate-ref/pull/361))
- Clarify that we don't technically support Windows at the moment, but it is possible to use WSL or a VM. ([#362](https://github.com/Climate-REF/climate-ref/pull/362))

### Improved Documentation

- Add documentation for the CLI tool ([#343](https://github.com/Climate-REF/climate-ref/pull/343))

## climate-ref 0.6.1 (2025-05-28)

### Features

- Implemented a HPCExecutor.
  It could let users run REF under HPC workflows by submitting batch jobs
  and compute diagnostics on the computer nodes. Only the slurm scheduler is
  supported now. ([#305](https://github.com/Climate-REF/climate-ref/pull/305))

### Bug Fixes

- Remove keys with their value None from the output JSON for CMEC validation of PMP extratropical variability modes ([#337](https://github.com/Climate-REF/climate-ref/pull/337))

### Improved Documentation

- Add Getting Started section for ingesting and solving ([#342](https://github.com/Climate-REF/climate-ref/pull/342))

## climate-ref 0.6.0 (2025-05-27)

### Breaking Changes

- Updated the group by dimensions for the PMP diagnostics.
  This will cause duplicate runs to appear if an existing database is used.
  We recommend starting with a new database if using the next release. ([#321](https://github.com/Climate-REF/climate-ref/pull/321))

### Features

- Implemented PMP ENSO metrics ([#273](https://github.com/Climate-REF/climate-ref/pull/273))
- Added ESMValTool ENSO diagnostics. ([#320](https://github.com/Climate-REF/climate-ref/pull/320))
- Add the creation of verbose debug logs.
  These logs will be created in the `$REF_CONFIGURATION/log` directory
  (or overriden via the `config.paths.log` setting). ([#323](https://github.com/Climate-REF/climate-ref/pull/323))
- Data catalogs now only contain the latest version of a dataset.
  This will trigger new executions when a new version of a dataset is ingested.

  Some additional datasets have been added to the obs4REF dataset registry.
  These datasets should be fetched and reingested. ([#330](https://github.com/Climate-REF/climate-ref/pull/330))
- Added a comparison of `burntFractionAll` to the ILAMB list of diagnostics ([#332](https://github.com/Climate-REF/climate-ref/pull/332))
- Adds `--diagnostic` and `--provider` arguments to the `ref solve` command.
  This allows users to subset a specific diagnostic or provider that they wish to run.
  Multiple `--diagnostic` or `--provider` arguments can be used to specify multiple diagnostics or providers.
  The diagnostic or provider slug must contain one of the filter values to be included in the calculations. ([#338](https://github.com/Climate-REF/climate-ref/pull/338))

### Improvements

- Raise the ilamb3 version to 2025.5.20 and add remaining ILAMB/IOMB metrics. ([#317](https://github.com/Climate-REF/climate-ref/pull/317))
- Adds Ocean Heat Content and snow cover datasets to the ilamb/iomb registry ([#318](https://github.com/Climate-REF/climate-ref/pull/318))
- Updated the ESMValTool version to include updated recipes and diagnostics. ([#325](https://github.com/Climate-REF/climate-ref/pull/325))
- Add obs4MIPs ERA-5 ta sample data as obs4REF. ([#334](https://github.com/Climate-REF/climate-ref/pull/334))
- Enable more variables for the annual cycle diagnostics via PMP. ([#335](https://github.com/Climate-REF/climate-ref/pull/335))
- Verify the checksum of downloaded datasets by default ([#336](https://github.com/Climate-REF/climate-ref/pull/336))

### Bug Fixes

- Depth selects properly in mrsos, added regression data ([#331](https://github.com/Climate-REF/climate-ref/pull/331))

## climate-ref 0.5.5 (2025-05-21)

### Improvements

- Added additional dimensions to the ILAMB and ESMValTool metric values.
  This includes additional information about the execution group that will be useful to end-users. ([#308](https://github.com/Climate-REF/climate-ref/pull/308))
- Move the ILAMB datasets to S3 ([#309](https://github.com/Climate-REF/climate-ref/pull/309))
- Clean ECS diagnostic (remove unused keys in ESMValTool recipes and avoid "cmip6" in diagnostic name) ([#310](https://github.com/Climate-REF/climate-ref/pull/310))
- Clean TCR diagnostic (remove unused keys in ESMValTool recipes and avoid "cmip6" in diagnostic name) ([#311](https://github.com/Climate-REF/climate-ref/pull/311))

### Improved Documentation

- Updated documentation to include more information about concepts within the REF. ([#312](https://github.com/Climate-REF/climate-ref/pull/312))

## climate-ref 0.5.4 (2025-05-19)

### Bug Fixes

- Add additional dependencies to the `climate-ref-core` so that it is self-contained ([#307](https://github.com/Climate-REF/climate-ref/pull/307))

## climate-ref 0.5.3 (2025-05-19)

### Features

- Diagnostic's have been split into two phases, executing which generates any outputs and then building a result
  object from the outputs.
  This split makes it easier to make modifications to how the results are translated into the CMEC outputs. ([#303](https://github.com/Climate-REF/climate-ref/pull/303))

### Improvements

- Added automatic backup of SQLite database files before running migrations.
  Backups are stored in a `backups` directory adjacent to the database file and are named with timestamps.
  The number of backups to retain can be configured via the `db.max_backups` setting in the database configuration,
  with a default of 5 backups. ([#301](https://github.com/Climate-REF/climate-ref/pull/301))
- Update to v0.6.0 of the sample data. ([#302](https://github.com/Climate-REF/climate-ref/pull/302))
- Add a smoke test for the Celery deployment ([#304](https://github.com/Climate-REF/climate-ref/pull/304))
- Add tests that the pypi releases are installable ([#306](https://github.com/Climate-REF/climate-ref/pull/306))

### Improved Documentation

- Added page describing the required reference datasets ([#298](https://github.com/Climate-REF/climate-ref/pull/298))

## climate-ref 0.5.2 (2025-05-15)

### Bug Fixes

- Fix missing dependency in migrations ([#297](https://github.com/Climate-REF/climate-ref/pull/297))

### Improved Documentation

- Added documentation for configuration options ([#296](https://github.com/Climate-REF/climate-ref/pull/296))

## climate-ref 0.5.1 (2025-05-14)

### Features

- Added an ESMValTool metric to compute climatologies and zonal mean profiles of
  cloud radiative effects. ([#241](https://github.com/Climate-REF/climate-ref/pull/241))
- Add additional dimensions to the metrics produced by PMP.
  Added `climate_ref_core.pycmec.metric.CMECMetric.prepend_dimensions` ([#275](https://github.com/Climate-REF/climate-ref/pull/275))
- Ensure that selectors are always alphabetically sorted ([#276](https://github.com/Climate-REF/climate-ref/pull/276))
- Add data model for supporting series of metric values.
  This allows diagnostic providers to supply a collection of [climate_ref_core.metric_values.SeriesMetricValue][]
  extracted from an execution. ([#278](https://github.com/Climate-REF/climate-ref/pull/278))
- The default executor ([climate_ref.executor.LocalExecutor][]) uses a process pool to enable parallelism.
  An alternative [climate_ref.executor.SynchronousExecutor][] is available for debugging purposes,
  which runs tasks synchronously in the main thread. ([#286](https://github.com/Climate-REF/climate-ref/pull/286))

### Improvements

- Bumps the ilamb3 version to now contain all analysis modules and reformats its CMEC output bundle. ([#262](https://github.com/Climate-REF/climate-ref/pull/262))
- Adds the ability to capture the output of an execution for regression testing. ([#274](https://github.com/Climate-REF/climate-ref/pull/274))
- Update to v0.5.1 of the sample data. ([#279](https://github.com/Climate-REF/climate-ref/pull/279))
- Update to v0.5.2 of the sample data. ([#282](https://github.com/Climate-REF/climate-ref/pull/282))
- Add a CITATION.cff to the repository to make it easier to cite. ([#283](https://github.com/Climate-REF/climate-ref/pull/283))
- Added the Assessment Fast Track-related services to the `docker-compose` stack alongside improved documentation for how to use the REF via docker containers. ([#287](https://github.com/Climate-REF/climate-ref/pull/287))
- Added support for ingesting multiple directories at once.
  This is useful for ingesting large datasets that are split into multiple directories or via glob patterns.
  An example of this is importing the monthly and fx datasets from an archive of CMIP6 data:

  ```bash
  ref datasets ingest --source-type cmip6 path_to_archive/CMIP6/*/*/*/*/*/*mon path_to_archive/CMIP6/*/*/*/*/*/fx
  ``` ([#291](https://github.com/Climate-REF/climate-ref/pull/291))
- Update the default log level to INFO from WARNING.
  Added the `-q` option to decrease the log level to WARNING. ([#292](https://github.com/Climate-REF/climate-ref/pull/292))

### Bug Fixes

- Resolves an issue that was blocking some PMP executions from completing.
  Any additional dimensions are now logged and ignored rather than causing the execution to fail. ([#274](https://github.com/Climate-REF/climate-ref/pull/274))
- Sets the environment variable `FI_PROVIDER=tcp` to use the TCP provider for libfabric (part of MPICH).
  The defaults were causing MPICH errors on some systems (namely macOS).
  This also removes the PMP provider's direct dependency on the source of `pcmdi_metric`. ([#281](https://github.com/Climate-REF/climate-ref/pull/281))
- Support the use of empty metric bundles ([#284](https://github.com/Climate-REF/climate-ref/pull/284))
- Reworked the lifetimes of the database transactions during the solve process.
  This is a fix for out of process executors where the transaction was not being committed until the end of a solve. ([#288](https://github.com/Climate-REF/climate-ref/pull/288))
- Requery an Execution from the database when handling the result from the LocalExecutor.
  This ensures that the execution isn't stale and that the result is still valid. ([#293](https://github.com/Climate-REF/climate-ref/pull/293))

## climate-ref 0.5.0 (2025-05-03)

### Breaking Changes

- Renamed packages to start with `climate_ref_` and removed `metrics` from the package name to avoid confusion.
  This changes the root name of the PyPi packages from `cmip_ref` to `climate-ref` and will require updating your `requirements.txt`, `pyproject.toml`, `setup.py`, or other dependency management files to list `climate-ref` instead of `cmip_ref`. ([#270](https://github.com/Climate-REF/climate-ref/pull/270))
- Clarified the difference between a diagnostic and a metric.
  This caused significant refactoring of names of classes and functions throughout the codebase,
  as well as renaming of database tables.

  | Package                      | Old Name                   | New Name                      |
  |------------------------------|----------------------------|-------------------------------|
  | climate_ref_core.diagnostics | Metric                     | Diagnostic                    |
  | climate_ref_core.diagnostics | MetricExecutionDefinition  | ExecutionDefinition           |
  | climate_ref_core.diagnostics | MetricExecutionResult      | ExecutionResult               |
  | climate_ref.models.execution | MetricExecutionResultß     | Execution                     |
  | climate_ref.models.execution | MetricExecutionGroup       | ExecutionGroup                |
  | climate_ref.models.execution | ResultOutput               | ExecutionOutput               |
  | climate_ref_core.datasets    | MetricDataset              | ExecutionDatasetCollection    |
  | climate_ref_core.solver      | MetricSolver               | ExecutionSolver               |
  | climate_ref_core.providers   | MetricsProvider            | DiagnosticProvider            |
  | climate_ref_core.providers   | CommandLineMetricsProvider | CommandLineDiagnosticProvider |
  | climate_ref_core.providers   | CondaMetricsProvider       | CondaDiagnosticProvider       |
  | climate_ref.config           | MetricsProviderConfig      | DiagnosticProviderConfig      |

  This removes any previous database migrations and replaces them with a new clean migration.
  If you have an existing database, you will need to delete it and re-create it. ([#271](https://github.com/Climate-REF/climate-ref/pull/271))

## cmip_ref 0.4.1 (2025-05-02)

### Breaking Changes

- Removed unnecessary prefixes in the metric slugs.
  This will cause duplicate results to be generated so we recommend starting with a clean REF installation. ([#263](https://github.com/Climate-REF/climate-ref/pull/263))

### Features

- Added PMP's annual cycle metrics ([#221](https://github.com/Climate-REF/climate-ref/pull/221))
- Add a `facets` attribute to a metric.
  This attribute is used to define the facets of the values that the metric produces. ([#255](https://github.com/Climate-REF/climate-ref/pull/255))
- Added a diagnostic to calculate climate variables at global warming levels. ([#257](https://github.com/Climate-REF/climate-ref/pull/257))
- Support multiple sets of data requirements ([#266](https://github.com/Climate-REF/climate-ref/pull/266))

### Bug Fixes

- Retry downloads if they fail ([#267](https://github.com/Climate-REF/climate-ref/pull/267))
- PMP annual cycle output JSON tranformed to be more comply with CMEC ([#268](https://github.com/Climate-REF/climate-ref/pull/268))

### Improved Documentation

- Add deprecation notices to PyPi package README's ([#269](https://github.com/Climate-REF/climate-ref/pull/269))

## cmip_ref 0.4.0 (2025-04-29)

### Breaking Changes

- Removed `climate_ref.solver.MetricSolver.solve_metric_executions` in preference for a standalone function `climate_ref.solver.solve_metric_executions`
  with identical functionality. ([#229](https://github.com/Climate-REF/climate-ref/pull/229))
- Updated the algorithm to generate the unique identifier for a Metric Execution Group.
  This will cause duplicate entries in the database if the old identifier was used.
  We recommend removing your existing database and starting fresh. ([#233](https://github.com/Climate-REF/climate-ref/pull/233))
- Removed the implicit treatment of the deepest dimension. The change will cause a validation error if the deepest dimension in the `RESULTS` is a dictionary. ([#246](https://github.com/Climate-REF/climate-ref/pull/246))
- Ensure that the order of the source dataset types in the MetricExecutionGroup dataset key are stable ([#248](https://github.com/Climate-REF/climate-ref/pull/248))

### Deprecations

- Removes support for Python 3.10.
  The minimum and default supported Python version is now 3.11. ([#226](https://github.com/Climate-REF/climate-ref/pull/226))

### Features

- Add the basic framework for enforcing a controlled vocabulary
  for the results in a CMEC metrics bundle.
  This is still in the prototype stage
  and is not yet integrated into post-metric execution processing. ([#183](https://github.com/Climate-REF/climate-ref/pull/183))
- Scalar values from the metrics are now stored in the database
  if they pass validation.
  The controlled vocabulary for these metrics is still under development. ([#185](https://github.com/Climate-REF/climate-ref/pull/185))
- Added Zero Emission Commitment (ZEC) metric to the ESMValTool metrics package. ([#204](https://github.com/Climate-REF/climate-ref/pull/204))
- Added Transient Climate Response to Cumulative CO2 Emissions (TCRE) metric to the ESMValTool metrics package. ([#208](https://github.com/Climate-REF/climate-ref/pull/208))
- Add `ref datasets fetch-obs4ref-data` CLI command to fetch datasets that are in the process of being published to obs4MIPs and are appropriately licensed for use within the REF.
  The CLI command fetches the datasets and writes them to a local directory.
  These datasets can then be ingested into the REF as obs4MIPs datasets. ([#219](https://github.com/Climate-REF/climate-ref/pull/219))
- Enabled metric providers to register registries of datasets for download.
  This unifies the fetching of datasets across the REF via the `ref datasets fetch-data` CLI command.
  Added registries for the datasets that haven't been published to obs4MIPs yet (`obs4REF`) as well as PMP annual cycle datasets. ([#227](https://github.com/Climate-REF/climate-ref/pull/227))
- Capture log output for each execution and display via `ref executions inspect`. ([#232](https://github.com/Climate-REF/climate-ref/pull/232))
- Added the option to install development versions of metrics packages. ([#236](https://github.com/Climate-REF/climate-ref/pull/236))
- Added seasonal cycle and time series of sea ice area metrics. ([#239](https://github.com/Climate-REF/climate-ref/pull/239))
- The unique group identifiers for a MetricExecutionGroup are now tracked in the database. These values are used for presentation. ([#248](https://github.com/Climate-REF/climate-ref/pull/248))
- Added a new dataset source type to track PMP climatology data ([#253](https://github.com/Climate-REF/climate-ref/pull/253))

### Improvements

- Refactored `MetricSolver.solve_metric_executions` to be a standalone function for easier testing.
  Also added some additional integration tests for the Extratropical Modes of Variability metric from PMP. ([#229](https://github.com/Climate-REF/climate-ref/pull/229))
- The configuration paths are now all resolved to absolute paths ([#230](https://github.com/Climate-REF/climate-ref/pull/230))
- Verified support for PostgreSQL database backends ([#231](https://github.com/Climate-REF/climate-ref/pull/231))
- Updated the ESMValTool metric and output bundles. ([#235](https://github.com/Climate-REF/climate-ref/pull/235))
- Update to v0.5.0 of the sample data ([#264](https://github.com/Climate-REF/climate-ref/pull/264))

### Bug Fixes

- Relax some of the requirements for the availability of metadata in CMIP6 datasets. ([#245](https://github.com/Climate-REF/climate-ref/pull/245))
- Added a missing migration and tests to ensure that the migration are up to date. ([#247](https://github.com/Climate-REF/climate-ref/pull/247))
- Fixed how the path to ESMValTool outputs was determined,
  and added support for outputs in nested directories. ([#250](https://github.com/Climate-REF/climate-ref/pull/250))
- Marked failing tests as xfail as a temporary solution. ([#259](https://github.com/Climate-REF/climate-ref/pull/259))
- Fetch ESMValTool reference data in the integration test suite ([#265](https://github.com/Climate-REF/climate-ref/pull/265))

### Improved Documentation

- Now following [SPEC-0000](https://scientific-python.org/specs/spec-0000/) for dependency support windows.
  Support for Python versions will be dropped after 3 years and support for key scientific libraries will be dropped after 2 years. ([#226](https://github.com/Climate-REF/climate-ref/pull/226))
- Migrate from the use of ‘AR7 Fast Track’ to ‘Assessment Fast Track’ in response to the CMIP Panel decision to [change the name of the CMIP7 fast track](https://wcrp-cmip.org/fast-track-name-update/). ([#251](https://github.com/Climate-REF/climate-ref/pull/251))

### Trivial/Internal Changes

- [#220](https://github.com/Climate-REF/climate-ref/pull/220)

## cmip_ref 0.3.1 (2025-03-28)

### Trivial/Internal Changes

- [#218](https://github.com/Climate-REF/climate-ref/pull/218)

## cmip_ref 0.3.0 (2025-03-28)

### Breaking Changes

- We changed the `ref` Command Line Interface to make the distinction between execution
  groups and individual executions clear. A metric execution is the evaluation of
  a specific metric for a specific set of input datasets. We group together all
  executions for the same set of input datasets which are re-run because
  the metric or the input datasets were updated or because a metric execution
  failed. For showing results, it is more useful to think in terms of execution groups.
  In particular, the `ref executions list` command was re-named to
  `ref executions list-groups`. ([#165](https://github.com/Climate-REF/climate-ref/pull/165))

### Features

- Support ingesting obs4MIPs datasets into the REF ([#113](https://github.com/Climate-REF/climate-ref/pull/113))
- Add extratropical modes of variability analysis using PMP ([#115](https://github.com/Climate-REF/climate-ref/pull/115))
- Added management of conda environments for metrics package providers.

  Several new commands are available for working with providers:
  - `ref providers list` - List the available providers
  - `ref providers create-env` - Create conda environments for providers

  ([#117](https://github.com/Climate-REF/climate-ref/pull/117))
- Added [CMECMetric.create_template][climate_ref_core.pycmec.metric.CMECMetric.create_template] method to create an empty CMEC metric bundle. ([#123](https://github.com/Climate-REF/climate-ref/pull/123))
- Outputs from a metric execution and their associated metadata are now tracked in the database. This includes HTML, plots and data outputs.

  Metric providers can register outputs via the CMEC output bundle.
  These outputs are then ingested into the database if the execution was successful. ([#125](https://github.com/Climate-REF/climate-ref/pull/125))
- Build and publish container images to [Github Container Registry](https://github.com/Climate-REF/climate-ref/pkgs/container/ref) ([#156](https://github.com/Climate-REF/climate-ref/pull/156))
- Enable more variability modes for PMP modes of variability metrics ([#173](https://github.com/Climate-REF/climate-ref/pull/173))
- Add a `--timeout` option to the `solve` cli command.
  This enables the user to set a maximum time for the solver to run. ([#186](https://github.com/Climate-REF/climate-ref/pull/186))

### Improvements

- Cleanup of ilamb3 interface code, enabling IOMB comparisons. ([#124](https://github.com/Climate-REF/climate-ref/pull/124))
- Migrate the PMP provider to use a REF-managed conda environment.

  For non-MacOS users, this should be created automatically.
  MacOS users will need to create the environment using the following command:

  ```bash
  MAMBA_PLATFORM=osx-64 uv run ref providers create-env --provider pmp
  ``` ([#127](https://github.com/Climate-REF/climate-ref/pull/127))
- Fixed issue with `mypy` not being run across the celery package ([#128](https://github.com/Climate-REF/climate-ref/pull/128))

- Added the `fetch-ref-data` make command to download reference data while it's not in obs4mips, yet. ([#155](https://github.com/Climate-REF/climate-ref/pull/155))
- Improvements:
  - Drop the metric plugin version number in the environment name because the environment may not change between releases
  - Avoid calling micromamba update as this may not work for everyone
  - Print out the location of environments even when they are not installed
  - Do not mention conda as a requirement on the Hackathon page

  ([#160](https://github.com/Climate-REF/climate-ref/pull/160))
- Add activity and institute to ESMValTool recipes to allow running with models
  and experiments that are not in the CMIP6 controlled vocabulary. ([#166](https://github.com/Climate-REF/climate-ref/pull/166))
- Add an integration test for the CMIP7 AFT metric providers.
  This will be run nightly as part of the Climate-REF CI pipeline. ([#187](https://github.com/Climate-REF/climate-ref/pull/187))
- Improved the error message when running `ref datasets list` with the `--column` argument.
  If a column is specified that is not available, the error message now only mentions
  the invalid column name(s) and shows a list of available columns. ([#203](https://github.com/Climate-REF/climate-ref/pull/203))
- Do not list duplicate entries in dataframes shown from the command line. ([#210](https://github.com/Climate-REF/climate-ref/pull/210))

### Bug Fixes

- Remove example metric for ILAMB ([#121](https://github.com/Climate-REF/climate-ref/pull/121))
- Removed the "SCHEMA" attribute from the CMEC metric bundle as it is not part of the EMDS specification and unused. ([#123](https://github.com/Climate-REF/climate-ref/pull/123))
- Fixed the validation error when 'attributes' value is a dict ([#133](https://github.com/Climate-REF/climate-ref/pull/133))
- Fixed PMP's modes of variablity PDO metrics driver to use obs4MIP-complying reference dataset. Also update PMP's version to 3.9, which include turning off direct usage of conda as a part of the driver (to capture provenance info) ([#154](https://github.com/Climate-REF/climate-ref/pull/154))
- Enforce the use of relative paths when copying files after an execution. This resolves an issue where files were not being copied to the correct location causing failures in PMP. ([#170](https://github.com/Climate-REF/climate-ref/pull/170))
- If no obs4mips-compliant reference dataset is found in specified directory, give a meaningful error message. ([#174](https://github.com/Climate-REF/climate-ref/pull/174))
- Fixed the behaviour of FacetFilter with `keep=False` so all facets need to match
  before excluding a file. ([#209](https://github.com/Climate-REF/climate-ref/pull/209))

### Improved Documentation

- Renamed CMIP-REF to Climate-REF ([#119](https://github.com/Climate-REF/climate-ref/pull/119))
- Add a landing page for hackathon attendees ([#120](https://github.com/Climate-REF/climate-ref/pull/120))
- Fix the incorrect capitalisation of GitHub organisation ([#122](https://github.com/Climate-REF/climate-ref/pull/122))
- Updated the getting started documentation. ([#126](https://github.com/Climate-REF/climate-ref/pull/126))
- Update the roadmap to reflect progress as of 2025/03/10 ([#134](https://github.com/Climate-REF/climate-ref/pull/134))
- Clarified language and other small fixes in the documentation. ([#178](https://github.com/Climate-REF/climate-ref/pull/178))

### Trivial/Internal Changes

- [#161](https://github.com/Climate-REF/climate-ref/pull/161), [#182](https://github.com/Climate-REF/climate-ref/pull/182), [#184](https://github.com/Climate-REF/climate-ref/pull/184), [#207](https://github.com/Climate-REF/climate-ref/pull/207)

## cmip_ref 0.2.0 (2025-03-01)

### Breaking Changes

- Refactor `climate_ref.env` module to `climate_ref_core.env` ([#60](https://github.com/Climate-REF/climate-ref/pull/60))
- Removed `climate_ref.executor.ExecutorManager` in preference to loading an executor using a fully qualified package name.

  This allows the user to specify a custom executor as configuration
  without needing any change to the REF codebase. ([#77](https://github.com/Climate-REF/climate-ref/pull/77))
- Renamed the `$.paths.tmp` in the configuration to `$.paths.scratch` to better reflect its purpose.
  This requires a change to the configuration file if you have a custom configuration. ([#89](https://github.com/Climate-REF/climate-ref/pull/89))
- The REF now uses absolute paths throughout the application.

  This removes the need for a `config.paths.data` directory and the `config.paths.allow_out_of_tree_datasets` configuration option.
  This will enable more flexibility about where input datasets are ingested from.
  Using absolute paths everywhere does add a requirement that datasets are available via the same paths for all nodes/container that may run the REF. ([#100](https://github.com/Climate-REF/climate-ref/pull/100))
- An [Executor][climate_ref_core.executor.Executor] now supports only the asynchronous processing of tasks.
  A result is now not returned from the `run_metric` method,
  but instead optionally updated in the database.

  The `run_metric` method also now requires a `provider` argument to be passed in. ([#104](https://github.com/Climate-REF/climate-ref/pull/104))

### Features

- Adds a `cmip-ref-celery` package to the REF that provides a Celery task queue.

  Celery is a distributed task queue that allows you to run tasks asynchronously.
  This package will be used as a test bed for running the REF in a distributed environment,
  as it can be deployed locally using docker containers. ([#60](https://github.com/Climate-REF/climate-ref/pull/60))
- Add `metric_providers` and `executor` sections to the configuration which loads the metric provider and executor using a fully qualified package name. ([#77](https://github.com/Climate-REF/climate-ref/pull/77))
- Implemented Pydantic data models to validate and serialize CMEC metric and output bundles. ([#84](https://github.com/Climate-REF/climate-ref/pull/84))
- Add the `climate_ref_celery` CLI commands to the `ref` CLI tool.
  These commands should be available when the `climate_ref_celery` package is installed.
  The commands should be available in the `ref` CLI tool as `ref celery ...`. ([#86](https://github.com/Climate-REF/climate-ref/pull/86))
- Add `fetch-sample-data` to the CLI under the `datasets` command.

  ```bash
  ref datasets fetch-sample-data --version v0.3.0 --force-cleanup
  ``` ([#96](https://github.com/Climate-REF/climate-ref/pull/96))
- Add a [Celery](https://docs.celeryq.dev/en/stable/)-based executor
  to enable asynchronous processing of tasks. ([#104](https://github.com/Climate-REF/climate-ref/pull/104))
- Add `ref executions list` and `ref executions inspect` CLI commands for interacting with metric executions. ([#108](https://github.com/Climate-REF/climate-ref/pull/108))

### Improvements

- Move ILAMB/IOMB reference data initialization to a registry-dependent script. ([#83](https://github.com/Climate-REF/climate-ref/pull/83))
- ILAMB gpp metrics added with full html output and plots. ([#88](https://github.com/Climate-REF/climate-ref/pull/88))
- Saner error messages for configuration errors ([#89](https://github.com/Climate-REF/climate-ref/pull/89))
- Centralised the declaration of environment variable overrides of configuration values.

  Renamed the `REF_OUTPUT_ROOT` environment variable to `REF_RESULTS_ROOT` to better reflect its purpose.
  It was previously unused. ([#92](https://github.com/Climate-REF/climate-ref/pull/92))
- Sample data is now copied to the `test/test-data/sample-data` instead of symlinked.

  This makes it easier to use the sample data with remote executors as the data is now self-contained
  without any links to other parts of the file system. ([#96](https://github.com/Climate-REF/climate-ref/pull/96))
- Integrated the pycmec validation models into ref core and metric packages ([#99](https://github.com/Climate-REF/climate-ref/pull/99))
- Added ILAMB relationship analysis to the current metrics and flexibility to define new metrics in ILAMB via a yaml file. ([#101](https://github.com/Climate-REF/climate-ref/pull/101))
- Sped up the test suite execution ([#103](https://github.com/Climate-REF/climate-ref/pull/103))

### Improved Documentation

- Added an excerpt from the architecture design document ([#87](https://github.com/Climate-REF/climate-ref/pull/87))
- Adds a roadmap to the documentation ([#98](https://github.com/Climate-REF/climate-ref/pull/98))

### Trivial/Internal Changes

- [#97](https://github.com/Climate-REF/climate-ref/pull/97), [#102](https://github.com/Climate-REF/climate-ref/pull/102), [#116](https://github.com/Climate-REF/climate-ref/pull/116), [#118](https://github.com/Climate-REF/climate-ref/pull/118)

## cmip_ref 0.1.6 (2025-02-03)

### Features

- Added Equilibrium Climate Sensitivity (ECS) to the ESMValTool metrics package. ([#51](https://github.com/Climate-REF/climate-ref/pull/51))
- Added Transient Climate Response (TCS) to the ESMValTool metrics package. ([#62](https://github.com/Climate-REF/climate-ref/pull/62))
- Added the possibility to request datasets with complete and overlapping timeranges. ([#64](https://github.com/Climate-REF/climate-ref/pull/64))
- Added a constraint for selecting supplementary variables, e.g. cell measures or
  ancillary variables. ([#65](https://github.com/Climate-REF/climate-ref/pull/65))
- Added a sample metric to the ilamb metrics package. ([#66](https://github.com/Climate-REF/climate-ref/pull/66))
- Added a sample metric to the PMP metrics package. ([#72](https://github.com/Climate-REF/climate-ref/pull/72))
- - Added the standard ILAMB bias analysis as a metric. ([#74](https://github.com/Climate-REF/climate-ref/pull/74))

### Bug Fixes

- - Added overlooked code to fully integrate ilamb into ref. ([#73](https://github.com/Climate-REF/climate-ref/pull/73))
- Correct the expected configuration name to `ref.toml` as per the documentation. ([#82](https://github.com/Climate-REF/climate-ref/pull/82))

### Improved Documentation

- Update the package name in the changelog.

  This will simplify the release process by fixing the extraction of changelog entries. ([#61](https://github.com/Climate-REF/climate-ref/pull/61))

### Trivial/Internal Changes

- [#68](https://github.com/Climate-REF/climate-ref/pull/68)

## cmip_ref 0.1.5 (2025-01-13)

### Trivial/Internal Changes

- [#56](https://github.com/Climate-REF/climate-ref/pull/56)

## cmip_ref 0.1.4 (2025-01-13)

### Breaking Changes

- Adds an `ingest` CLI command to ingest a new set of data into the database.

  This breaks a previous migration as alembic's `render_as_batch` attribute should have been set
  to support targeting sqlite. ([#14](https://github.com/Climate-REF/climate-ref/pull/14))
- Renames `ref ingest` to `ref datasets ingest` ([#30](https://github.com/Climate-REF/climate-ref/pull/30))
- Prepend package names with `cmip_` to avoid conflicting with an existing `PyPI` package.

  This is a breaking change because it changes the package name and all imports.
  All package names will now begin with `cmip_ref`. ([#53](https://github.com/Climate-REF/climate-ref/pull/53))

### Features

- Migrate to use UV workspaces to support multiple packages in the same repository.
  Adds a `climate-ref-example` package that will be used to demonstrate the integration of a metric
  package into the REF. ([#2](https://github.com/Climate-REF/climate-ref/pull/2))
- Adds the placeholder concept of `Executor`'s which are responsible for running metrics
  in different environments. ([#4](https://github.com/Climate-REF/climate-ref/pull/4))
- Adds the concept of MetricProvider's and Metrics to the core.
  These represent the functionality that metric providers must implement in order to be part of the REF.
  The implementation is still a work in progress and will be expanding in follow-up PRs. ([#5](https://github.com/Climate-REF/climate-ref/pull/5))
- Add a collection of ESGF data that is required for test suite.

  Package developers should run `make fetch-test-data` to download the required data for the test suite. ([#6](https://github.com/Climate-REF/climate-ref/pull/6))
- Adds the `ref` package with a basic CLI interface that will allow for users to interact with the database of jobs. ([#8](https://github.com/Climate-REF/climate-ref/pull/8))
- Add `SqlAlchemy` as an ORM for the database alongside `alembic` for managing database migrations. ([#11](https://github.com/Climate-REF/climate-ref/pull/11))
- Added a `DataRequirement` class to declare the requirements for a metric.

  This provides the ability to:

  - filter a data catalog
  - group datasets together to be used in a metric calculation
  - declare constraints on the data that is required for a metric calculation

  ([#15](https://github.com/Climate-REF/climate-ref/pull/15))
- Add a placeholder iterative metric solving scheme ([#16](https://github.com/Climate-REF/climate-ref/pull/16))
- Extract a data catalog from the database to list the currently ingested datasets ([#24](https://github.com/Climate-REF/climate-ref/pull/24))
- Translated selected groups of datasets into `MetricDataset`s.
  Each `MetricDataset` contains all of the dataset's needed for a given execution of a metric.

  Added a slug to the `MetricDataset` to uniquely identify the execution
  and make it easier to identify the execution in the logs. ([#29](https://github.com/Climate-REF/climate-ref/pull/29))
- Adds `ref datasets list` command to list ingested datasets ([#30](https://github.com/Climate-REF/climate-ref/pull/30))
- Add database structures to represent a metric execution.
  We record previous executions of a metric to minimise the number of times we need to run metrics. ([#31](https://github.com/Climate-REF/climate-ref/pull/31))
- Added option to skip any datasets that fail validation and to specify the number of cores to
  use when ingesting datasets.
  This behaviour can be opted in using the `--skip-invalid` and `--n-jobs` options respectively. ([#36](https://github.com/Climate-REF/climate-ref/pull/36))
- Track datasets that were used for different metric executions ([#39](https://github.com/Climate-REF/climate-ref/pull/39))
- Added an example ESMValTool metric. ([#40](https://github.com/Climate-REF/climate-ref/pull/40))
- Support the option for different assumptions about the root paths between executors and the ref CLI.

  Where possible path fragments are stored in the database instead of complete paths.
  This allows the ability to move the data folders without needing to update the database. ([#46](https://github.com/Climate-REF/climate-ref/pull/46))

### Improvements

- Add a bump, release and deploy flow for automating the release procedures ([#20](https://github.com/Climate-REF/climate-ref/pull/20))
- Migrate test data into standalone [Climate-REF/ref-sample-data](https://github.com/Climate-REF/ref-sample-data) repository.

  The sample data will be downloaded by the test suite automatically into `tests/test-data/sample-data`,
  or manually by running `make fetch-test-data`. ([#49](https://github.com/Climate-REF/climate-ref/pull/49))

### Bug Fixes

- Adds `version` field to the `instance_id` field for CMIP6 datasets ([#35](https://github.com/Climate-REF/climate-ref/pull/35))
- Handle missing branch times.
  Fixes [#38](https://github.com/Climate-REF/climate-ref/issues/38). ([#42](https://github.com/Climate-REF/climate-ref/pull/42))
- Move alembic configuration and migrations to `cmip_ref` package so that they can be included in the distribution. ([#54](https://github.com/Climate-REF/climate-ref/pull/54))

### Improved Documentation

- Deployed documentation to <https://climate-ref.readthedocs.io/en/latest/> ([#16](https://github.com/Climate-REF/climate-ref/pull/16))
- General documentation cleanup.

  Added notebook describing the process of executing a notebook locally ([#19](https://github.com/Climate-REF/climate-ref/pull/19))
- Add Apache licence to the codebase ([#21](https://github.com/Climate-REF/climate-ref/pull/21))
- Improved developer documentation. ([#47](https://github.com/Climate-REF/climate-ref/pull/47))

### Trivial/Internal Changes

- [#41](https://github.com/Climate-REF/climate-ref/pull/41), [#44](https://github.com/Climate-REF/climate-ref/pull/44), [#48](https://github.com/Climate-REF/climate-ref/pull/48), [#52](https://github.com/Climate-REF/climate-ref/pull/52), [#55](https://github.com/Climate-REF/climate-ref/pull/55)
