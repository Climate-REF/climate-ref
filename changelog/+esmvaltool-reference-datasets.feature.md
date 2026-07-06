ESMValTool reference (observational/reanalysis) datasets are now ingested into the database under a dedicated `esmvaltool-reference` dataset type during `ref providers setup`, so the data each diagnostic uses can be recorded for provenance and surfaced in the frontend.
Because this data is not CMOR/obs4MIPs compliant, its metadata is parsed from the ESMValTool `OBS`/`OBS6`, `native6` and `obs4MIPs` layout conventions rather than from file global attributes.
ESMValTool diagnostics now declare the reference datasets they use via a `reference_datasets` specification instead of hardcoding them inside recipe construction, giving a single source of truth for provenance.
Each execution now records the reference datasets the diagnostic compared against, so they appear alongside the model datasets in the execution's dataset list.
The near-identical obs4MIPs and PMP climatology dataset models were also consolidated onto a shared mixin while remaining distinct dataset types.
