# Datasets in the REF

The Reference Evaluation Framework (REF) supports multiple types of datasets, each with specific metadata requirements and use cases.
Understanding these dataset types is essential for working with the REF data catalog and ingestion workflows.

## Dataset Types

### Target Datasets

These datasets are the datasets that diagnostics are designed to evaluate. Currently, the REF only supports CMIP6 datasets,
but CMIP7 and other future datasets will be added in the future.

#### CMIP6 Datasets

- **Description:** Climate model output from the Coupled Model Intercomparison Project Phase 6 (CMIP6).
- **Metadata:** Includes detailed model and experiment information such as `activity_id`, `institution_id`, `source_id`, `experiment_id`, `member_id`, `table_id`, `variable_id`, `grid_label`, and `version`.
  Additional fields may include parent experiment details, grid information, and variable metadata.
- **Unique Identifier:** `instance_id` (constructed from key metadata fields and version).
- **Usage:** Used as model reference data for benchmarking and evaluation.
- **Metadata Parsing:**
  The REF supports two methods for parsing CMIP6 metadata:
  - **DRS Parser (default)**: Extracts metadata from file paths and names according to the [Data Reference Syntax (DRS)](https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit?tab=t.0). This approach enables fast loading of metadata without opening each file.
  - **Complete Parser**: Opens each file and extracts all available metadata from the file's attributes. This provides more comprehensive metadata but is significantly slower.

You can select the parser by setting `cmip6_parser: "drs"` or `cmip6_parser: "complete"` in your REF configuration file.

### Reference Datasets

The REF requires reference datasets for diagnostics, which are used to compare against target datasets. These datasets are typically observational or post-processed climatology datasets that provide a baseline for model evaluation and benchmarking.

These datasets can be downloaded manually or automatically via the `ref datasets fetch-data` cli command.

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

## Dataset Metadata and Cataloging

Each dataset type has a corresponding adapter and model in the REF codebase, ensuring that metadata is consistently extracted, validated, and stored. The unique identifier (`instance_id`) is used to group files belonging to the same dataset and track versions.

When a dataset is ingested into the REF, its metadata is stored in the database. This allows users to find datasets matching specific criteria for use in diagnostics and to track which datasets where used to produce a given diagnostic execution.

For more details on the metadata fields for each dataset type, see the code in `climate_ref/models/dataset.py` and the dataset adapters in `climate_ref/datasets/`.

## Dataset Selection for Diagnostics

Diagnostics specify their data requirements through the `data_requirements` attribute, which defines:

1. **Source type**: Which dataset collection to use (CMIP6, obs4MIPs, etc.)
2. **Filters**: Conditions that datasets must meet to be included
3. **Grouping**: How to organize the datasets for separate diagnostic executions
4. **Constraints**: Additional validation rules for dataset groups

For a detailed guide on selecting datasets for diagnostics, see the [dataset selection how-to guide](../how-to-guides/dataset-selection.py).
