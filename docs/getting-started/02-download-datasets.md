# Download Required Datasets

This tutorial covers how to fetch all reference datasets needed to run Climate-REF diagnostics. You may see references to *fetch*, *download*, and *retrieve* all of which refer to the process of copying data from one computer system to another. [Ingesting](../nutshell.md) these datasets is covered in the next tutorial.

These commands should be rerun after new releases of Climate-REF to ensure you have the latest datasets.

## Reference dataset requirements

Climate-REF uses public, open-license reference data.
Where possible, datasets from [obs4MIPs](https://pcmdi.github.io/obs4MIPs/) are recommended—they are [CMOR](https://github.com/PCMDI/cmor)-compliant, openly licensed, and archived on [ESGF](https://esgf-node.ornl.gov/search).

During development, additional datasets have been identified for inclusion in obs4MIPs and will be added as they become available.
This collection of datasets is referred to as `obs4REF` in the Climate-REF documentation.

The required datasets are listed in the [obs4REF registry](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref/src/climate_ref/dataset_registry/obs4ref_reference.txt).


/// admonition | Note

By default, downloaded data is stored in a cache directory which is in your user directory by default.

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

[](){#fetch-pmp-climatology-datasets}
## 2. PMP Climatology datasets

PMP has generated a set of climatology datasets based on obs4MIPs data.
These datasets are used for the PMP diagnostics and are not part of the obs4REF collection.
These datasets will later be [ingested](../nutshell.md) into the REF and used in diagnostic data requirements.

```bash
ref datasets fetch-data --registry pmp-climatology --output-directory $REF_CONFIGURATION/datasets/pmp-climatology
```

## 3. Provider-specific datasets

Some diagnostics require additional datasets that are not ingested into the REF and must be fetched separately.
These datasets will eventually be integrated into the REF, but for now, they can be fetched using the following commands:


```bash
ref datasets fetch-data --registry ilamb
ref datasets fetch-data --registry iomb
ref datasets fetch-data --registry esmvaltool
```

### Future work

The Climate-REF team is working on providing a more integrated way to fetch and manage these datasets from the Next Generation ESGF infrastructure that in the process of being deployed.
This should minimise the need to manually fetch datasets and ensure that all required datasets are available for diagnostics.


## Next steps

After fetching your data, proceed to the [Ingest datasets](03-ingest.md) tutorial to load them into Climate-REF.
