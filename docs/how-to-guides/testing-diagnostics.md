# Testing Diagnostics

A test case is a small set of real input data for one run of a diagnostic.
Each diagnostic declares its own.
CI checks every test case on every pull request,
against a recorded set of results called a regression baseline.

## Defining test cases

Add a `test_data_spec` to the diagnostic:

```python
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.testing import TestCase, TestDataSpecification


class MyDiagnostic(Diagnostic):
    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Historical tas from ACCESS-ESM1-5",
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

Keep the data small: a short `time_span` and one ensemble member.
Only the requested datasets are visible when the test case runs.

The request types are in `climate_ref_core.esgf`:

- `CMIP6Request` and `CMIP7Request` fetch model output from ESGF.
- `ConvertedCMIP7Request` fetches CMIP6 output and converts it to look like CMIP7.
  Use it for data not yet published as CMIP7.
- `Obs4MIPsRequest` fetches observations from ESGF.
- `RegistryRequest` fetches files from a REF dataset registry, such as obs4REF.

A diagnostic that takes CMIP6 and CMIP7 usually has a `default` case and a `cmip7` case.
The example provider shows both.

## Fetching test data

```bash
ref test-cases list --provider my-provider
ref test-cases fetch --provider my-provider --diagnostic my-diagnostic
```

Fetching downloads the data and writes `catalog.yaml`.
Each test case has a directory in the provider package:

```raw
packages/climate-ref-{provider}/tests/test-data/{diagnostic}/{test_case}/
├── catalog.yaml     # The datasets (in git)
├── manifest.json    # Baseline metadata (in git)
├── regression/      # The baseline results (in git)
└── output/          # Local runs (gitignored)
```

On HPC, point `esg_dataroot` in `~/.config/intake-esgf/conf.yaml` at an existing archive
to read from it instead of downloading.

## Running test cases

```bash
ref test-cases run --provider my-provider --diagnostic my-diagnostic
ref test-cases run --provider my-provider --diagnostic my-diagnostic --test-case default
```

Without `--test-case`, every test case runs.
Results go to `output/latest/` in the test case directory.

Every provider also runs its test cases through pytest,
from `tests/integration/test_diagnostics.py`:

```python
from climate_ref.testing import create_no_drift_test

test_run_test_cases = create_no_drift_test(provider)
```

These tests are slow, so they need `--slow`:

```bash
uv run pytest packages/climate-ref-example/tests/integration/test_diagnostics.py --slow
```

## Regression baselines

A baseline has two parts:

- `regression/` holds small JSON results in git, so reviewers see the numbers change in the diff.
- The large files they refer to, such as netCDF and plots, are kept in shared storage.

Only a maintainer can upload a baseline. This is called minting.
[Regression baselines](../background/regression-baselines.md) explains how CI uses them.

### Adding a test case

1. Define the test case and fetch its data.
2. Run it with `ref test-cases run`.
   The first run creates `regression/` and `manifest.json`.
3. Check the numbers look right.
   Nothing else checks them yet.
4. Commit `catalog.yaml`, `manifest.json` and `regression/`, and open a pull request.
5. Ask a maintainer in the pull request to mint the baseline.

### Changing a diagnostic's results

1. Bump `version` on the diagnostic.
2. Run the test case, then compare with the old results:

   ```bash
   cd packages/climate-ref-my-provider/tests/test-data/my-diagnostic/default
   git diff --no-index regression/ output/latest/regression/
   ```

3. Commit the code, but not `regression/` or `manifest.json`.
   The PR gate fails until the baseline is minted. This is expected.
4. Ask a maintainer to mint.
   The mint commits the new baseline to your branch.

Do not commit a baseline made with `run --force-regen` before a mint,
because the gate then fails.

### What the PR gate says

| Outcome | Meaning |
| --- | --- |
| **skip** | Nothing relevant changed, or there is no baseline to check yet. |
| **replay** | CI rebuilt the results from the stored files, and they matched. |
| **fail** | The baseline does not match the code. Usually `version` was bumped and nothing was minted yet. |

Run the same gate locally with `make regression-gate`.

### When a test case drifts

Drift means the results no longer match `regression/`.

- Run `ref test-cases run` and diff `output/latest/regression/` against `regression/`.
- Run `ref test-cases replay --provider my-provider`.
  If replay passes but `run` drifts, your change moved the results.
  If replay fails too, the baseline needs minting again.
- Debug with pytest, using `--slow --pdb` and the test id `{diagnostic}/{test_case}`.
