# Download Required Datasets

This tutorial covers how to fetch all reference datasets needed to run Climate-REF diagnostics. You may see references to *fetch*, *download*, and *retrieve* all of which refer to the process of copying data from one computer system to another. [Ingesting](../concepts.md) these datasets is covered in the next tutorial.

These commands should be rerun after new releases of Climate-REF to ensure you have the latest datasets.

## Input datasets

The Climate-REF requires local input datasets from CMIP6/CMIP6plus to evaluate. Depending on where you are running the REF, a local archive of CMIP6 datasets may be available already, if not the target datasets can be fetched from [ESGF](https://esgf-node.ornl.gov/search) directly. We have provided a script in [./scripts/fetch-esgf.py](https://github.com/Climate-REF/climate-ref/blob/main/scripts/fetch-esgf.py) for fetching the datasets that can be evaluated by the REF. This involves a moderate volume of data, requireing more than 4TB of storage when assessing a single ensemble member per model.

Note that not all of these datasets are required. The Climate-REF will determine which diagnostics can be evaluated according the datasets that are available.

The data used by the Climate-REF do not necessarily need to have been previously published to ESGF. As long as the datasets match the data requirements of the diagnostics and they conform with the CMIP6 era cmorisation process they can be evaluated via the REF.

If you are preparing data for a modelling centre,
start with the [modelling centre onboarding guide](modelling-centres.md)
to capture any additional requirements before ingesting local output.

## Reference dataset requirements

Climate-REF uses public, open-license reference data.
Where possible, datasets from [obs4MIPs](https://pcmdi.github.io/obs4MIPs/) are recommended—they are [CMOR](https://github.com/PCMDI/cmor)-compliant, openly licensed, and archived on [ESGF](https://esgf-node.ornl.gov/search).

During development, additional datasets have been identified for inclusion in obs4MIPs and will be added as they become available.
This collection of datasets is referred to as `obs4REF` in the Climate-REF documentation.

The required datasets are listed in the [obs4REF registry](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref/src/climate_ref/dataset_registry/obs4ref_reference.txt).

/// admonition | Note

By default, downloaded data is stored in a cache directory which is in your user directory.

You can override this location by setting the `REF_DATASET_CACHE_DIR` environment variable:

```bash
export REF_DATASET_CACHE_DIR=/path/to/cache
```

This can use up a large amount of disk space, so it is important to choose a location with sufficient storage.
///

[](){#fetch-obs4ref-datasets}

## 1. Fetching obs4REF datasets

Use the [ref datasets fetch-data](../cli.md#fetch-data) command to retrieve each registry.
Replace example paths with your desired output directories.

These are hosted temporarily in one location until they become available on ESGF.
This archive is ~30 GB in size, so ensure you have sufficient disk space available.
In the future, these datasets will be available on ESGF and can be fetched directly from there:

```bash
ref datasets fetch-data --registry obs4ref --output-directory $REF_CONFIGURATION/datasets/obs4ref
```

[](){#fetch-obs4mips-datasets}

## 2. Fetching obs4MIPs datasets from ESGF

The obs4REF registry does not cover every reference dataset the diagnostics need.
The following are already published to obs4MIPs on ESGF and must be fetched from there:

| `source_id` | Variables | Required by |
| --- | --- | --- |
| `20CR-V2` | `psl` | `pmp/extratropical-modes-of-variability-{nam,nao,npo,pna,sam}` |
| `C3S-GTO-ECV-9-0` | `toz` | `esmvaltool/ozone-{annual-cycle,lat-time,nh-mar,sh-oct}` |
| `CERES-EBAF-4-2-1` | `rlut`, `rlutcs`, `rsut`, `rsutcs` | `esmvaltool/cloud-radiative-effects` |
| `ERA-5` | `psl`, `ta`, `tas`, `ua` | `esmvaltool/cloud-scatterplots-reference`, `esmvaltool/regional-historical-{annual-cycle,timeseries,trend}` |
| `NOAA-NCEI-LAI-AVHRR-5-0` | `lai` | `ilamb/lai-avh15c1` |

A diagnostic whose reference data is missing simply plans no executions,
so an incomplete fetch shows up as a diagnostic that never runs rather than as an error.

The same [./scripts/fetch-esgf.py](https://github.com/Climate-REF/climate-ref/blob/main/scripts/fetch-esgf.py)
script used for the CMIP6 input data can fetch these,
and `--kind obs4mips` restricts it to the reference data:

```bash
python scripts/fetch-esgf.py --kind obs4mips
```

This is a much smaller download than the CMIP6 input data (a few GB).
Files land in the [intake-esgf `local_cache`](https://intake-esgf.readthedocs.io/en/latest/configure.html),
and are ingested with the `obs4mips` source type, the same as the obs4REF collection.

/// admonition | Note

The script also fetches `CERES-EBAF-4-2`, `GPCP-Monthly-3-2`, `HadISST-1-1` and `TropFlux-1-0`,
which the obs4REF registry ships as well.
These are the ESGF-published copies of datasets that were curated for the REF before publication,
so if you have already fetched the obs4REF registry you do not need them.

Fetching both is safe. Where the two copies carry the same version they share an `instance_id`
and ingest as a single dataset. Where the published copy is newer
(for example `CERES-EBAF-4-2` `v20240513` on ESGF against `v20230209` in obs4REF)
both are ingested and the catalog uses the later version.
///

/// admonition | Why one request per `source_id`?

An ESGF search intersects its facets,
so a single request naming every `source_id` and every `variable_id` would ask each source
for variables it does not have and return nothing.
The requests in `scripts/fetch-esgf.py` are therefore grouped by `source_id`.
///

### Future work

The Climate-REF team is working on providing a more integrated way to fetch and manage these datasets from the Next Generation ESGF infrastructure that in the process of being deployed.
This should minimise the need to manually fetch datasets and ensure that all required datasets are available for diagnostics.

## Next steps

After fetching your data, proceed to the [Ingest datasets](03-ingest.md) tutorial to load them into Climate-REF.
