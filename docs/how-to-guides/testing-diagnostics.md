# Testing Diagnostics

Each diagnostic declares its own test cases.
A test case is a small, named set of real input data for one execution of the diagnostic.
The `ref test-cases` commands fetch that data, run the diagnostic on it,
and record the results as a regression baseline that CI checks on every pull request.

Because each diagnostic owns its test data,
you only fetch data for the diagnostics you are working on,
and changing one diagnostic's test data cannot move another diagnostic's results.

## Defining test cases

Declare the test cases in the diagnostic's `test_data_spec`.

```python
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.testing import TestCase, TestDataSpecification


class MyDiagnostic(Diagnostic):
    name = "My Diagnostic"
    slug = "my-diagnostic"

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Standard test with historical data",
                requests=(
                    CMIP6Request(
                        slug="test-tas",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "tas",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                ),
            ),
        ),
    )
```

| Attribute | Meaning |
| --- | --- |
| `name` | Identifier, unique within the diagnostic, eg. `"default"` or `"cmip7"`. |
| `description` | What the test case covers. |
| `requests` | The datasets to fetch. Only these datasets are visible when the test case runs. |

Keep the data small.
A short `time_span` and a single ensemble member are usually enough.
ESGF's [Metagrid](https://esgf-node.ornl.gov/) is handy for checking which datasets a set of facets matches.

### Request types

All of these are importable from `climate_ref_core.esgf`.

| Request | Fetches |
| --- | --- |
| `CMIP6Request` | CMIP6 model output from ESGF. |
| `CMIP7Request` | CMIP7 model output published on ESGF. |
| `ConvertedCMIP7Request` | CMIP6 output converted to look like CMIP7, for models and variables not yet published as CMIP7. |
| `Obs4MIPsRequest` | obs4MIPs observations from ESGF. |
| `RegistryRequest` | Files from a REF dataset registry, such as obs4REF or the PMP climatologies. |

A diagnostic that supports both CMIP6 and CMIP7 usually has a `default` test case with a `CMIP6Request`
and a `cmip7` test case with the same data.
Use a `CMIP7Request` for the `cmip7` case when the data is published,
and a `ConvertedCMIP7Request` until then.
The example provider does this.

```python
CMIP7Request(
    slug="cmip7-tas",
    facets={
        "source_id": "UKESM1-3-LL",
        "experiment_id": "historical",
        "variable_id": "tas",
        "branding_suffix": "tavg-h2m-hxy-u",
        "frequency": "mon",
    },
    remove_ensembles=True,
)

ConvertedCMIP7Request(
    slug="cmip7-tas",
    facets={
        "source_id": "ACCESS-ESM1-5",
        "experiment_id": "historical",
        "variable_id": "tas",
        "variant_label": "r1i1p1f1",
        "table_id": "Amon",
    },
)

Obs4MIPsRequest(
    slug="obs-tas",
    facets={"source_id": "ERA5", "variable_id": "tas", "frequency": "mon"},
)

RegistryRequest(
    slug="surface-temperature-obs",
    registry_name="obs4ref",
    source_type="obs4REF",
    facets={"source_id": "HadISST-1-1", "variable_id": "ts"},
)
```

`ConvertedCMIP7Request` takes CMIP7 facet names (`variant_label` rather than `member_id`),
fetches the CMIP6 files and converts them.
The converted files are cached in `cmip7-converted/` under `$REF_DATASET_CACHE_DIR`,
or under `climate_ref/cmip7-converted/` in the user cache directory when that is unset.
To move a test case to published data, swap the request for a `CMIP7Request`,
run `ref test-cases fetch --regen` for that test case and ask for the baseline to be minted again.

CMIP7 data is only in the ESGF STAC indices (`discovery.east.esgf.io` and `discovery.west.esgf.io`).
Recent intake-esgf releases enable them by default.
An older `~/.config/intake-esgf/conf.yaml` may not list them, and a warning is shown.

## Fetching test data

```bash
ref test-cases list --provider my-provider
ref test-cases fetch --provider my-provider --diagnostic my-diagnostic
ref test-cases fetch --dry-run
```

Fetching downloads the data with [intake-esgf](https://github.com/esgf2-us/intake-esgf)
and writes a `catalog.yaml` describing the resolved datasets.
Each test case has its own directory in the provider package:

```raw
packages/climate-ref-{provider}/tests/test-data/
└── {diagnostic}/
    └── {test_case}/
        ├── catalog.yaml           # Datasets used by the test case (in git)
        ├── manifest.json          # Baseline metadata (in git)
        ├── regression/            # Committed baseline bundle (in git)
        │   ├── series.json
        │   ├── diagnostic.json
        │   └── output.json
        └── output/                # Output slots from local runs (gitignored)
```

The local file paths are kept outside the checkout, in the dataset cache.
[Test-case catalogs](../background/regression-baselines.md#test-case-catalogs) explains where.

/// admonition | Using shared ESGF data
    type: note

On HPC systems, point `esg_dataroot` at an existing CMIP6 archive so intake-esgf reads from it
instead of downloading.
New files still go to `local_cache`.

```yaml
# ~/.config/intake-esgf/conf.yaml
esg_dataroot:
  - /shared/cmip6/data
local_cache:
  - /scratch/$USER/.esgf
```

///

## Running test cases

```bash
# Every test case of a diagnostic
ref test-cases run --provider my-provider --diagnostic my-diagnostic

# One test case
ref test-cases run --provider my-provider --diagnostic my-diagnostic --test-case default

# Fetch the data first, then run
ref test-cases run --provider my-provider --diagnostic my-diagnostic --fetch
```

Each run writes what it produced to the gitignored output slot `output/latest/` in the test case directory.
This holds the native output files and the rebuilt bundle under `output/latest/regression/`.
Pass `--output-directory ./scratch` to also keep the raw execution directory, including its logs.
Add `ref --verbose` for more logging.

## Pytest

Every provider package turns its test cases into pytest tests in `tests/integration/test_diagnostics.py`:

```python
from climate_ref_example import provider

from climate_ref.testing import create_no_drift_test

test_run_test_cases = create_no_drift_test(provider)
```

This generates one test per test case, with the id `{diagnostic}/{test_case}`.
Each test runs the diagnostic on the fetched data
and checks the rebuilt bundle matches `regression/` within tolerance.
A test case without fetched data or a baseline is skipped.
The tests are marked `slow` and `test_cases`, so they need `--slow`:

```bash
uv run pytest packages/climate-ref-example/tests/integration/test_diagnostics.py --slow

uv run pytest --slow \
    "packages/climate-ref-example/tests/integration/test_diagnostics.py::test_run_test_cases[global-mean-timeseries/default]"
```

For checks the drift test does not cover, run a test case directly with `TestCaseRunner`.
The `config` fixture comes from the `climate_ref` pytest plugin,
which is registered when `climate-ref` is installed.

```python
import pytest
from my_provider import provider

from climate_ref.testing import TestCaseRunner
from climate_ref_core.testing import TestCasePaths, load_datasets_from_yaml


@pytest.mark.slow
def test_my_diagnostic(config, tmp_path):
    diagnostic = provider.get("my-diagnostic")
    diagnostic.provider.configure(config)

    paths = TestCasePaths.from_diagnostic(diagnostic, "default")
    if paths is None or not paths.catalog.exists():
        pytest.skip("Test data not fetched, run `ref test-cases fetch` first")

    runner = TestCaseRunner(config=config, datasets=load_datasets_from_yaml(paths.catalog, paths.catalog_paths))
    result = runner.run(diagnostic, "default", output_dir=tmp_path / "output")

    assert result.successful
```

## Regression baselines

Diagnostics are slow, so CI does not re-run them on every pull request.
Each test case is instead pinned to a regression baseline:
a recorded set of results that a pull request must still reproduce.
See [Regression baselines](../background/regression-baselines.md) for how CI checks this.

### Terms

| Term | Meaning |
| --- | --- |
| Committed bundle | The small JSON results in `regression/` (`series.json`, `diagnostic.json`, `output.json`). These are in git, so reviewers see your numbers change in the diff. |
| Native files | The heavy outputs (`.nc`, `.png`, ...) that the JSON refers to. These are kept in shared storage, not in git. |
| `manifest.json` | A record of which committed and native files belong to the baseline, plus two version numbers. It is in git. |
| Mint | Upload the native files and record them in `manifest.json`. Only a maintainer can mint. |
| Replay | Rebuild the committed bundle from the stored native files, without re-running the diagnostic, and compare. |
| Output slot | A gitignored folder, `output/latest/` by default, where every run puts what it produced so you can inspect it. |

### Adding a test case

1. Define the test case and fetch its data (see [Fetching test data](#fetching-test-data)).
2. Run it:

   ```bash
   ref test-cases run --provider my-provider --diagnostic my-diagnostic --test-case my-case
   ```

   The first run creates `regression/` and `manifest.json`.
3. Check the numbers in `regression/` look right.
   CI cannot check them yet, so you and the reviewer are the only check.
4. Commit `catalog.yaml`, `manifest.json` and `regression/`, then open a pull request.
   The PR gate shows `skip` for the new case.
5. Ask a maintainer in the pull request to mint the baseline.
   Afterwards the CI checks if the baseline is available.

### Changing a diagnostic's results

1. Bump `version` on your diagnostic class.
   It is an integer that starts at `1`.
   Bump it for algorithm changes, bug fixes or data requirement changes that alter the results.
   A new `version` will trigger all existing executions to be recalculated.

   ```python
   class MyDiagnostic(Diagnostic):
       version = 2
   ```

2. Check the new results locally with `ref test-cases run`,
   then compare the output slot with the old baseline:

   ```bash
   cd packages/climate-ref-my-provider/tests/test-data/my-diagnostic/default
   git diff --no-index regression/ output/latest/regression/
   ```

3. Commit your code change, but not changes to `regression/` or `manifest.json`.
   The PR gate fails at this point, because the baseline is older than your `version`.
   This is expected.
4. Ask a maintainer in the pull request to mint.
   The mint reruns the diagnostic, commits the new `regression/` and `manifest.json` to the branch,
   and posts a report comparing the old and new outputs.
5. Push any follow-up commit so the PR gate runs again and replays the new baseline.

Do not commit a baseline from `run --force-regen` when you are going to mint.
`mint` bumps `test_case_version` only when its bundle differs from the committed one,
so a pre-committed bundle leaves the version unbumped and the gate fails.

`run` warns about any `regression/` file over `--size-threshold` (default 1.0 MB).
Keep committed files small, because large outputs belong in the native files.

Maintainers mint with the workflow described in
[Gated mint](../background/regression-baselines.md#gated-mint).

### What the PR gate says

| Outcome | Meaning | What to do |
| --- | --- | --- |
| **skip** | Nothing relevant changed, or there are no native files to replay yet. | Nothing. For a new or changed baseline, check the diff yourself. |
| **replay** | CI rebuilt the committed bundle from the stored native files and it matched. | Nothing. If it does not match, your change moved the results, so follow [Changing a diagnostic's results](#changing-a-diagnostics-results). |
| **fail** | The baseline does not match the code, eg. `version` was bumped but nothing was minted, or `regression/` was edited without a version bump. | Ask a maintainer to mint, or revert the change to `regression/`. |

Run the same gate locally with `make regression-gate`, which compares against `origin/main`.

### When a test case drifts

A drift test or the PR gate reports that the rebuilt bundle no longer matches `regression/`.

- Re-run the test case with `ref test-cases run` and diff the output slot against `regression/`, as above.
  The drift assertion also prints which values moved beyond tolerance.
- Run `ref test-cases replay --provider my-provider`.
  It rebuilds the bundle from the stored native files without running the diagnostic.
  If replay passes but `run` drifts, your change moved the results.
  If replay also fails, the baseline itself is inconsistent and needs minting again.
- Compare two runs with named slots.
  `latest` is overwritten on every run, but `--label before` and `--label after` are kept.
- Use pytest when you need a debugger at the point of failure:

  ```bash
  uv run pytest --slow --pdb \
      "packages/climate-ref-my-provider/tests/integration/test_diagnostics.py::test_run_test_cases[my-diagnostic/default]"
  ```

  The run happens in the test's `tmp_path`, under `/tmp/pytest-of-<user>/`.
  The execution directory is `exec/` and the rebuilt bundle is `slot/regression/`.

## Common errors

| Error | Cause | Solution |
| --- | --- | --- |
| `NoTestDataSpecError` | Diagnostic has no `test_data_spec` | Add a `test_data_spec` attribute |
| `TestCaseNotFoundError` | Invalid test case name | Check the names with `ref test-cases list` |
| `DatasetResolutionError` | Missing test data | Run `ref test-cases fetch` |
| `No datasets found` | The ESGF query returned nothing | Check the request facets |

## Other useful links

- [Adding a diagnostic](adding_custom_diagnostics.md)
- [Running Diagnostics Locally](running-diagnostics-locally.py)
- [Dataset Selection](dataset-selection.py)
