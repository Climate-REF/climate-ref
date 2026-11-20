# Test data

This directory contains data used by the test suite (and documentation).
Not all test data are checked into this repository due to the size of the files.

The test suite will automatically download the required data when it is run.
Alternatively, these data can be downloaded manually by running:

```
make fetch-test-data
```

This also fetches the test data neeed by the ILAMB metric provider.
These data are stored in a pooch cache outside of the repository.

## Sample Data

A consistent set of [sample data](https://github.com/Climate-REF/ref-sample-data)
is used by the REF test suite.
This ensures that the tests are reproducible and that the test data are versioned.

These data are used in the REF tests and are not intended to be used for any other purpose
as they are decimated to reduce the size of the test dataset.

These data are fetched using pooch and then copied into the `sample-data` directory,
during the `make fetch-test-data` process.
These sample data are also used to generate the documentation.

## CMIP7

The `cmip7` directory holds decimated copies of published CMIP7 files in their DRS directories
(CanESM5-1 `esm-flat10` `tas` and `areacella`).
They keep every attribute of the published files, so the dataset adapters are tested against real CMIP7 headers,
but only a few grid cells and the first two years of data.
They are checked in, and regenerated from the published files with `scripts/decimate-cmip7-test-data.py`.

## CMEC Output

Reference output from the CMEC model is stored in the `cmec-output` directory.
