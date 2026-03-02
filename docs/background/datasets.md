# Datasets in the REF

The Reference Evaluation Framework (REF) supports multiple types of datasets, each with specific metadata requirements and use cases.
Understanding these dataset types is essential for working with the REF data catalog and ingestion workflows.

## Dataset Types

### Target Datasets

Target datasets are the datasets that diagnostics are designed to evaluate.
The REF currently supports CMIP6 and CMIP7 datasets,
but has been designed to be extensible to support other datasets in future.

#### CMIP6 Datasets

- **Description:** Climate model output from the Coupled Model Intercomparison Project Phase 6 (CMIP6).
- **Metadata:** Includes model and experiment information such as `activity_id`, `institution_id`, `source_id`, `experiment_id`, `member_id`, `table_id`, `variable_id`, `grid_label`, and `version`.
  The `complete` parser additionally extracts parent experiment details, grid information, variable metadata (`standard_name`, `long_name`, `units`), and other attributes from file headers.
- **Unique Identifier:** `instance_id`, constructed as `CMIP6.<activity_id>.<institution_id>.<source_id>.<experiment_id>.<member_id>.<table_id>.<variable_id>.<grid_label>.<version>`.
- **Usage:** Used as model output for benchmarking and evaluation against reference datasets.

#### CMIP7 Datasets

- **Description:** Climate model output from the Coupled Model Intercomparison Project Phase 7 (CMIP7), following the [CMIP7 Global Attributes v1.0](https://doi.org/10.5281/zenodo.17250297) specification.
- **Metadata:** Core DRS attributes include `activity_id`, `institution_id`, `source_id`, `experiment_id`, `variant_label`, `variable_id`, `grid_label`, `frequency`, `region`, `branding_suffix`, and `version`.
  CMIP7 introduces a `branding_suffix` that, combined with `variable_id`, forms a `branded_variable` (e.g. `tas_tavg-h2m-hxy-u`).
  The complete parser additionally extracts `mip_era`, `realm`, `nominal_resolution`, `license_id`, parent experiment details, and variable metadata.
- **Unique Identifier:** `instance_id`, constructed as `CMIP7.<activity_id>.<institution_id>.<source_id>.<experiment_id>.<variant_label>.<region>.<frequency>.<variable_id>.<branding_suffix>.<grid_label>.<version>`.
- **Usage:** Used as model output for benchmarking and evaluation against reference datasets.

### Reference Datasets

The REF requires reference datasets for diagnostics, which are used to compare against target datasets. These datasets are typically observational or post-processed climatology datasets that provide a baseline for model evaluation and benchmarking.

These datasets can be downloaded manually or automatically via the `ref datasets fetch-data` CLI command.

#### obs4MIPs Datasets

- **Description:** Observational datasets formatted to be compatible with CMIP model output conventions, facilitating direct comparison.
- **Metadata:** Includes fields such as `activity_id`, `institution_id`, `source_id`, `variable_id`, `grid_label`, `source_version_number`, and variable-specific metadata like `long_name`, `units`, and `vertical_levels`.
- **Unique Identifier:** `instance_id` (constructed from key metadata fields and version).
- **Usage:** Used as observational reference data for model evaluation.

#### PMP Climatology Datasets

- **Description:** Post-processed climatology datasets, often derived from obs4MIPs or other sources, typically used in the PCMDI Metrics Package (PMP).
- **Metadata:** Similar to obs4MIPs, with fields for `activity_id`, `institution_id`, `source_id`, `variable_id`, `grid_label`, `source_version_number`, and climatology-specific metadata.
- **Unique Identifier:** `instance_id`.
- **Usage:** Used for climatological benchmarking and diagnostics.

#### Additional Reference Datasets

- **Description:** Other reference datasets not yet included in obs4MIPs or PMP, often managed via the REF dataset registry.
- **Metadata:** Varies by dataset; managed using a registry file with checksums and metadata.
- **Usage:** Used to supplement the core reference datasets, especially for new or experimental data.

## Metadata Parsing

For CMIP6 and CMIP7 target datasets, the REF supports two methods for extracting metadata during ingestion.
The choice of parser controls the trade-off between initial ingestion speed and metadata completeness.
For a large archive, detailed metadata about datasets that aren't covered by REF diagnostics aren't needed
so we should avoid reading them.

### Complete Parser (default)

The complete parser opens every netCDF file and reads its global attributes and variable metadata.
This provides all available metadata at ingestion time but is slower,
as every file must be opened and read.

This is the current default parser for both CMIP6 and CMIP7 datasets.

### DRS Parser

The DRS parser extracts metadata entirely from file paths and directory names,
following the [CMIP6 Data Reference Syntax](https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit?tab=t.0)
or the CMIP7 equivalent.
Because it never opens netCDF files, ingestion is dramatically faster --
making it the recommended choice for large archives with tens of thousands of files,
such as those found on HPC systems or shared CMIP data pools.

The DRS parser extracts the following metadata from the file path and name:

| Field                                    | Source                                    |
| ---------------------------------------- | ----------------------------------------- |
| `activity_id`                            | Directory path                            |
| `institution_id`                         | Directory path                            |
| `source_id`                              | Filename                                  |
| `experiment_id`                          | Filename                                  |
| `member_id` / `variant_label`            | Filename                                  |
| `table_id` (CMIP6) / `frequency` (CMIP7) | Filename                                  |
| `variable_id`                            | Filename                                  |
| `grid_label`                             | Filename                                  |
| `version`                                | Directory path (e.g. `v20210318`)         |
| `start_time`, `end_time`                 | Filename time range (approximate)         |
| `frequency` (CMIP6 only)                 | Inferred from `table_id` via lookup table |

The following metadata fields are **not** available from the DRS parser
and are left unpopulated until finalisation:

- Variable metadata: `standard_name`, `long_name`, `units`
- Parent experiment details: `branch_time_in_child`, `branch_time_in_parent`, `parent_experiment_id`, `parent_source_id`, `parent_activity_id`, `parent_variant_label`, `parent_time_units`
- Additional attributes: `realm`, `nominal_resolution`, `grid`, `source_type`, `product`, `experiment`, `vertical_levels`
- CMIP6 only: `branch_method`, `sub_experiment`, `sub_experiment_id`
- CMIP7 only: `license_id`, `external_variables`, `parent_mip_era`

### Selecting a parser

You can select the parser in your `ref.toml` configuration file:

```toml
# For CMIP6 datasets
cmip6_parser = "drs"       # fast, path-based parsing
# cmip6_parser = "complete"  # slow, opens every file (default)

# For CMIP7 datasets
cmip7_parser = "drs"       # fast, path-based parsing
# cmip7_parser = "complete"  # slow, opens every file (default)
```

Or via environment variables:

```bash
export REF_CMIP6_PARSER=drs
export REF_CMIP7_PARSER=drs
```

## Two-Phase Ingestion (Lazy Finalisation)

When using the DRS parser, the REF uses a two-phase workflow
to keep ingestion fast while still providing complete metadata for diagnostics.

### Phase 1: Fast Ingest

Running `ref datasets ingest` with the DRS parser scans file paths in parallel
and stores metadata in the database with `finalised=False`.
No netCDF files are opened, so this phase completes quickly as file listings are very fast even for parallel file systems.

### Phase 2: Lazy Finalisation at Solve Time

When `ref solve` runs, the solver identifies which datasets match each diagnostic's data requirements.
For any matched datasets that are still unfinalised (`finalised=False`),
the REF opens only those specific files to extract the remaining metadata.
The updated metadata is persisted back to the database with `finalised=True`,
so subsequent solves do not need to re-read the same files.

This means that for an archive with hundreds of thousands of files,
only the subset that is actually needed by a diagnostic is ever opened --
not the entire archive.

### Re-ingestion Behaviour

If a dataset already has `finalised=True` in the database and is encountered again during a DRS ingest,
the existing metadata is preserved and only newly discovered files are appended.
This makes re-ingestion safe and incremental.

## Dataset Metadata and Cataloging

Each dataset type has a corresponding adapter and model in the REF codebase, ensuring that metadata is consistently extracted, validated, and stored.
The unique identifier (`instance_id`) is used to group files belonging to the same dataset and track versions.

When a dataset is ingested into the REF, its metadata is stored in the database.
This allows users to find datasets matching specific criteria for use in diagnostics and to track which datasets were used to produce a given diagnostic execution.

For more details on the metadata fields for each dataset type, see the code in `climate_ref/models/dataset.py` and the dataset adapters in `climate_ref/datasets/`.

## Dataset Selection for Diagnostics

Diagnostics specify their data requirements through the `data_requirements` attribute, which defines:

1. **Source type**: Which dataset collection to use (CMIP6, CMIP7, obs4MIPs, etc.)
2. **Filters**: Conditions that datasets must meet to be included
3. **Grouping**: How to organise the datasets for separate diagnostic executions
4. **Constraints**: Additional validation rules for dataset groups

For a detailed guide on selecting datasets for diagnostics, see the [dataset selection how-to guide](../how-to-guides/dataset-selection.py).
