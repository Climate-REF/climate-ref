Added a `--chunk-size` option to `ref datasets ingest` (CMIP6 only) that streams
the catalog in directory-aligned batches instead of loading the whole archive
into memory at once. Peak memory is now bounded by `chunk_size` rather than by
the total number of files in the input tree.
