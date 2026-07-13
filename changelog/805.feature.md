Added two new source dataset types, `obs4REF` and `ESMValToolReference`, together with their database tables.

`obs4REF` is REF-curated observational data that follows the obs4MIPs metadata conventions that has not yet been published to the obs4MIPs ESGF archive.
 `ESMValToolReference` is ESMValTool's own reference data, which is not CMOR compliant and so carries a smaller set of metadata.
 The shared obs4MIPs-style column block used by `obs4MIPs`, `PMPClimatology` and now `obs4REF` was extracted into a `ReferenceDatasetMixin`.
