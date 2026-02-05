# Ingest Datasets

This guide will walk you through the process of ingesting local datasets into the REF. In REF, to *ingest* means that we record local datasets in the REF database, letting REF know where they exist and to what format they conform. Ingesting datasets is the first step in the REF workflow.

The REF supports the following dataset formats:

* CMIP6
* Obs4MIPs

Downloading the input data is out of scope for this guide,
but we recommend using the [intake-esgf](https://github.com/esgf2-us/intake-esgf/) to download CMIP6 data.
If you have access to a high-performance computing (HPC) system,
you may have a local archive of CMIP6 data already available.


## What is Ingestion?

When processing diagnostics, the REF needs to know the location of the datasets and various metadata.
Ingestion is the process of extracting metadata from datasets and storing it in a local database.
This makes it easier to query and filter datasets for further processing.

The collection of metadata, also known as a data catalog, is stored in a local SQLite database.
This database is used to query and filter datasets for further processing.

For CMIP6 datasets, the default DRS parser extracts metadata from file paths and directory names
without opening each file. This makes ingestion very fast, even for archives with tens of thousands of files.
Any remaining metadata (such as exact time ranges) is extracted automatically at solve time
for datasets that match a diagnostic's data requirements.
If you need all metadata upfront, you can set `cmip6_parser: "complete"` in your configuration file,
though this will be significantly slower for large archives.

## Ingesting Datasets

To ingest datasets, use the `ref datasets ingest` command.
This command takes a path to a directory containing datasets as an argument
and the type of the dataset being ingested (only cmip6 is currently supported).

This will walk through the provided directory looking for datasets to ingest.
Metadata will be extracted from each dataset and stored in the database.

```
>>> ref --log-level INFO datasets ingest --source-type cmip6 /path/to/cmip6
2024-12-05 12:00:05.979 | INFO     | climate_ref.database:__init__:77 - Connecting to database at sqlite:///.climate_ref/db/climate_ref.db
2024-12-05 12:00:05.987 | INFO     | alembic.runtime.migration:__init__:215 - Context impl SQLiteImpl.
2024-12-05 12:00:05.987 | INFO     | alembic.runtime.migration:__init__:218 - Will assume non-transactional DDL.
2024-12-05 12:00:05.989 | INFO     | alembic.runtime.migration:run_migrations:623 - Running upgrade  -> ea2aa1134cb3, dataset-rework
2024-12-05 12:00:05.995 | INFO     | climate_ref.cli.datasets:ingest:115 - ingesting /path/to/cmip6
2024-12-05 12:00:06.401 | INFO     | climate_ref.cli.datasets:ingest:127 - Found 9 files for 5 datasets

  activity_id   institution_id   source_id       experiment_id   member_id   table_id   variable_id   grid_label   version
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rlut          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rlut          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rsdt          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rsdt          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rsut          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       rsut          gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       tas           gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    Amon       tas           gn           v20210318
  ScenarioMIP   CSIRO            ACCESS-ESM1-5   ssp126          r1i1p1f1    fx         areacella     gn           v20210318

2024-12-05 12:00:06.409 | INFO     | climate_ref.cli.datasets:ingest:131 - Processing dataset CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rlut.gn
2024-12-05 12:00:06.431 | INFO     | climate_ref.cli.datasets:ingest:131 - Processing dataset CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsdt.gn
2024-12-05 12:00:06.441 | INFO     | climate_ref.cli.datasets:ingest:131 - Processing dataset CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsut.gn
2024-12-05 12:00:06.449 | INFO     | climate_ref.cli.datasets:ingest:131 - Processing dataset CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn
2024-12-05 12:00:06.459 | INFO     | climate_ref.cli.datasets:ingest:131 - Processing dataset CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn
```


### Querying ingested datasets

You can query the ingested datasets using the `ref datasets list` command.
This will display a list of datasets and their associated metadata.
The `--column` flag allows you to specify which columns to display (defaults to all columns).
See `ref datasets list-columns` for a list of available columns.

```
>>> ref datasets list --column instance_id --column variable_id

  instance_id                                                             variable_id
 ─────────────────────────────────────────────────────────────────────────────────────
  CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rlut.gn      rlut
  CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsdt.gn      rsdt
  CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsut.gn      rsut
  CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn       tas
  CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn   areacella
```
