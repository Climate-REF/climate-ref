# Five-minute Quick Start

This quick start gets you from a fresh install to your first diagnostic results in **under five minutes**,
without downloading the full reference archive or building any provider environments.

It uses the **example provider**,
a small, pure-Python provider you install alongside Climate-REF,
so there is no conda/mamba environment to create.
You will fetch a small, curated set of sample data,
run two diagnostics,
and inspect the results.

When you are ready for a full assessment with the complete providers
(ESMValTool, PMP, ILAMB) and the full reference data,
continue with the [Configure](01-configure.md) tutorial.

/// admonition | What you get
    type: tip

- A model-only diagnostic: **Global Mean Timeseries** (annual global-mean of a CMIP variable).
- A model-vs-observation diagnostic: **Global Mean Surface Temperature Bias**,
  which compares modelled global-mean sea surface temperature (`tos`, area-weighted by `areacello`)
  against the HadISST-1-1 observational record, reports the root-mean-square error and mean bias,
  and produces two figures (the model and reference series together, and the bias over time).
///

## Prerequisites

- Python 3.12 or newer on Linux or macOS.
- Roughly 200 MB of free disk space for the sample data and results.

## 1. Create an environment and install (~1 min)

Install Climate-REF into a **virtual environment** rather than your global Python.
The REF is under active development, so we recommend an isolated environment
that you can update or remove without affecting the rest of your system.

Using the Python standard library's `venv`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install climate-ref climate-ref-example
```

If you already use [uv](https://docs.astral.sh/uv/), the equivalent is:

```bash
uv venv
source .venv/bin/activate
uv pip install climate-ref climate-ref-example
```

This installs the core `climate-ref` package together with the small, pure-Python
`climate-ref-example` provider used by this guide.
Both are published on PyPI, so no conda/mamba environment is required.

Verify the CLI is available:

```bash
ref --version
```

## 2. Configure (~10 s)

Choose a directory for this quick start and point Climate-REF at it,
then write a configuration file with [ref config init](../cli.md#init):

```bash
export REF_CONFIGURATION=$(pwd)/ref-quickstart
ref config init
```

`ref config init` writes a `ref.toml` template into `$REF_CONFIGURATION` with sensible defaults:
a local SQLite database and the **local executor**, so no external services are required.
You can confirm the configuration is valid at any time:

```bash
ref config validate
```

Check that the example provider was discovered.
It is registered automatically because `climate-ref-example` is installed:

```bash
ref providers list
```

You should see `example` in the list.
If you do not, make sure `climate-ref-example` was installed in the active environment (step 1).

## 3. Get a small set of data (~1-2 min)

Fetch the curated CMIP6 sample data (a decimated subset, not for production use)
and the single surface-temperature observation used by the model-vs-obs diagnostic.

The observation comes from the dedicated **`quickstart`** registry,
which holds just one HadISST-1-1 file rather than the full obs4REF collection (~30 GB):

```bash
ref datasets fetch-data --registry sample-data --output-directory $REF_CONFIGURATION/datasets/sample-data
ref datasets fetch-data --registry quickstart --output-directory $REF_CONFIGURATION/datasets/quickstart
```

## 4. Ingest (~20 s)

Extract metadata from the downloaded files into the local catalog.
The model data is CMIP6; the observation uses the `obs4mips` source type:

```bash
ref datasets ingest --source-type cmip6 $REF_CONFIGURATION/datasets/sample-data/CMIP6
ref datasets ingest --source-type obs4mips $REF_CONFIGURATION/datasets/quickstart/obs4REF
```

Check the catalog:

```bash
ref datasets list
```

## 5. Run the diagnostics (~1 min)

Solve and execute, restricting the run to the example provider.
Scoping to `--provider example` keeps the run fast and avoids touching any of the
heavier providers:

```bash
ref solve --provider example
```

To run only the model-vs-observation diagnostic, add a diagnostic filter:

```bash
ref solve --provider example --diagnostic global-mean-surface-temperature-bias
```

## 6. Observe the results (~20 s)

List the execution groups and inspect one to see its metric values and output files:

```bash
ref executions list-groups
ref executions inspect <group_id>
```

Replace `<group_id>` with an id from the previous command.
For the surface-temperature diagnostic you will see the `rmse` and `mean-bias`
of the modelled global-mean surface temperature against the HadISST-1-1 observations.

The output files are written under your results directory,
organised by provider, diagnostic, execution group, and dataset hash:

```bash
ls -d $REF_CONFIGURATION/results/*/*/*/
```

Each result directory contains a [CMEC](https://pcmdi.github.io/CMEC/) output bundle (`output.json`),
a CMEC metric bundle (`diagnostic.json`) with the scalar metrics,
and the diagnostic's outputs.
The surface-temperature diagnostic writes a NetCDF exposing the **model** series, the **reference** (observation) series,
and the **bias** between them, plus two figures:
`surface_temperature_timeseries.png` (model and reference together) and `surface_temperature_bias.png` (the bias over time).

## Control and configuration options

The quick start is deliberately minimal.
The following options let you tune it:

| Goal | How |
| --- | --- |
| Keep this run isolated from any existing install | Set `REF_CONFIGURATION` to a fresh directory (as above). |
| Run a single diagnostic only | `ref solve --provider example --diagnostic <slug>`. |
| Run a single provider only | `ref solve --provider example`. Without a filter, `ref solve` schedules every available provider. |
| Reduce log output | Use the global `--quiet` flag, e.g. `ref --quiet solve --provider example`. |
| See what would run without executing | `ref solve --provider example --dry-run`. |
| Save disk by not copying the cache | Add `--symlink` to the `fetch-data` commands. |
| Move the download cache | `export REF_DATASET_CACHE_DIR=/path/to/cache` before fetching. |
| Inspect or change configuration | `ref config get <key>`, `ref config set <key> <value>`, `ref config validate`. |

The executor, database, and data paths all live in `$REF_CONFIGURATION/ref.toml`.
See the [Configuration documentation](../configuration.md) for the full set of options.

## Where to next?

You have run Climate-REF end-to-end on a small example.
For a full assessment with the complete diagnostic providers and reference data:

- Start the full [Getting Started](01-configure.md) tutorial, beginning with configuration.
- Review the [Download Required Datasets](02-download-datasets.md) guide for the complete obs4REF and CMIP6 data.
- Browse the available [Diagnostics](../diagnostics/index.md) and providers.
