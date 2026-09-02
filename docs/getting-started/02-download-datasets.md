# Download Required Datasets

This tutorial covers how to fetch all reference datasets needed to run Climate-REF diagnostics.
You may see references to *fetch*, *download*, and *retrieve* all of which refer to the process of copying data from one computer system to another. [Ingesting](../concepts.md) these datasets is covered in the next tutorial.

These commands should be rerun after new releases of Climate-REF to ensure you have the latest datasets.

## Input datasets

The Climate-REF requires local input datasets from CMIP6/CMIP6plus/CMIP7 to evaluate.
Depending on where you are running the REF, a local archive of CMIP6 datasets may be available already,
if not the target datasets can be fetched from [ESGF](https://esgf-node.ornl.gov/search) directly.
We have provided a script in [./scripts/fetch-esgf.py](https://github.com/Climate-REF/climate-ref/blob/main/scripts/fetch-esgf.py) for fetching the datasets that can be evaluated by the REF.
This involves a moderate volume of data, requiring more than 4TB of storage when assessing a single ensemble member per model.

Note that not all of these datasets are required.
The Climate-REF will determine which diagnostics can be evaluated according to the datasets that are available.

The data used by the Climate-REF do not necessarily need to have been previously published to ESGF.
As long as the datasets match the data requirements of the diagnostics and they conform with the CMIP6 era cmorisation process, they can be evaluated via the REF.

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

The command fetches up to four files concurrently by default.
This can be overridden by setting the `REF_DATASET_FETCH_WORKERS` environment variable.

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
script used for the CMIP6 input data fetches these.
If you fetched the obs4REF registry in the previous step, ask for exactly these five:

```bash
python scripts/fetch-esgf.py --request-id pmp-modes-20cr-obs4mips
python scripts/fetch-esgf.py --request-id esmvaltool-ozone-obs4mips
python scripts/fetch-esgf.py --request-id esmvaltool-cloud-radiative-effects-obs4mips
python scripts/fetch-esgf.py --request-id esmvaltool-cloud-scatterplots-obs4mips
python scripts/fetch-esgf.py --request-id esmvaltool-historical-obs4mips
python scripts/fetch-esgf.py --request-id ilamb-lai-obs4mips
```

(`ERA-5` is split across two requests, hence six commands for five datasets.)

`--kind obs4mips` fetches all of the reference data in one go,
but it also re-fetches four datasets the obs4REF registry already provides,
so only use it if you are **not** using that registry — see the warning below.

Files land in the [intake-esgf `local_cache`](https://intake-esgf.readthedocs.io/en/latest/configure.html),
and are ingested with the `obs4mips` source type.
The obs4REF collection is ingested with `obs4ref` instead, so the two are never confused.

/// admonition | Fetching these twice
    type: note

The script also fetches `CERES-EBAF-4-2`, `GPCP-Monthly-3-2`, `HadISST-1-1` and `TropFlux-1-0`.
These are the ESGF-published copies of datasets that were curated for the REF before publication,
so the obs4REF registry ships them as well.

If you fetch these from ESGF as well as from the obs4REF registry, the ESGF copy is the one used.
obs4MIPs is the official home of the reference data, and the registry only fills in what is not published yet.
`ref doctor` lists the registry copies that have been superseded this way.
///

### Future work

The Climate-REF team is working on providing a more integrated way to fetch and manage these datasets from the Next Generation ESGF infrastructure that in the process of being deployed.
This should minimise the need to manually fetch datasets and ensure that all required datasets are available for diagnostics.

## Next steps

After fetching your data, proceed to the [Ingest datasets](03-ingest.md) tutorial to load them into Climate-REF.
