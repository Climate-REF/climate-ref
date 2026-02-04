# Download Required Datasets

This tutorial covers how to fetch all reference datasets needed to run Climate-REF diagnostics. You may see references to *fetch*, *download*, and *retrieve* all of which refer to the process of copying data from one computer system to another. [Ingesting](../nutshell.md) these datasets is covered in the next tutorial.

These commands should be rerun after new releases of Climate-REF to ensure you have the latest datasets.

## Input datasets

The Climate-REF requires local input datasets from CMIP6/CMIP6plus to evaluate. Depending on where you are running the REF, a local archive of CMIP6 datasets may be available already, if not the target datasets can be fetched from [ESGF](https://esgf-node.ornl.gov/search) directly. We have provided a script in [./scripts/fetch-esfgf.py](https://github.com/Climate-REF/climate-ref/blob/main/scripts/fetch-esfgf.py) for fetching the datasets that can be evaluated by the REF. This involves a moderate volume of data, requireing more than 4TB of storage when assessing a single ensemble member per model.

Note that not all of these datasets are required. The Climate-REF will determine which diagnostics can be evaluated according the datasets that are available.

The data used by the Climate-REF do not necessarily need to have been previously published to ESGF. As long as the datasets match the data requirements of the diagnostics and they conform with the CMIP6 era cmorisation process they can be evaluated via the REF.

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

### Future work

The Climate-REF team is working on providing a more integrated way to fetch and manage these datasets from the Next Generation ESGF infrastructure that in the process of being deployed.
This should minimise the need to manually fetch datasets and ensure that all required datasets are available for diagnostics.

## Next steps

After fetching your data, proceed to the [Ingest datasets](03-ingest.md) tutorial to load them into Climate-REF.
