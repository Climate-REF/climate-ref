# Ingest Datasets

Ingestion extracts metadata from your locally downloaded datasets and stores it in a local catalog for easy querying and filtering.
This makes subsequesnt operations, such as running diagnostics, more efficient as the system can quickly access the necessary metadata without needing to reprocess the files.

Before you begin, ensure you have:

- Fetched your reference data (see [Download Required Datasets](02-download-datasets.md)).
- CMOR-compliant files accessible either locally or on a mounted filesystem.

## 1. Ingest reference datasets

The `obs4REF` collection uses the `obs4mips` source type in Climate-REF:

```bash
ref datasets ingest --source-type obs4mips $REF_CONFIGURATION/datasets/obs4ref
```

Replace `$REF_CONFIGURATION/datasets/obs4ref` with the directory used when [fetched the obs4REF data](02-download-datasets.md#fetch-obs4ref-datasets).

## 2. Ingest PMP Climatology data

Use the `pmp-climatology` source type:

```bash
ref datasets ingest --source-type pmp-climatology /path/to/datasets/pmp-climatology
```

This registry contains pre-computed climatology fields used by the PMP diagnostics.

## 3. Ingest CMIP6 data

To ingest CMIP6 files, point the CLI at a directory of netCDF files and set `cmip6` as the source type:

```bash
ref datasets ingest --source-type cmip6 /path/to/cmip6/data
```


Globbed-style paths can be used to specify multiple directories or file patterns.
For example, if you have CMIP6 data organized by the CMIP6 DRS,
you can use the following command to ingest all monthly and ancillary variables:

```bash
ref datasets ingest --source-type cmip6 /path/to/cmip6/data/CMIP6/*/*/*/*/*/*mon /path/to/cmip6/data/CMIP6/*/*/*/*/*/*fx --n-jobs 64
```

/// admonition | Tip

As part of the Climate-REF test suite,
we provide a sample set of CMIP6 (and obs4REF) data that can be used for testing and development purposes.
These datasets have been decimated to reduce their size.
These datasets should not be used for production runs, but they are useful for testing the ingestion and diagnostic processes.

To fetch and ingest the sample CMIP6 data, run the following commands:

```bash
ref datasets fetch-data --registry sample-data --output-directory $REF_CONFIGURATION/datasets/sample-data
ref datasets ingest --source-type cmip6 $REF_CONFIGURATION/datasets/sample-data/CMIP6
```

///

## 4. Query your catalog

After ingestion, list the datasets to verify:

```bash
ref datasets list
```

You can also filter by column:

```bash
ref datasets list --column instance_id --column variable_id
```


[//]: # (TODO: Add links to CLI reference once available)
[//]: # (For a complete list of flags, see the [Datasets CLI reference]&#40;../how-to-guides/ingest-datasets.md&#41;.)

## Next steps

With your data cataloged, you’re ready to run diagnostics. Proceed to the [Solve tutorial](04-solve.md).
