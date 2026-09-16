"""
Render an HTML report of the regression baselines changed on a branch.

A mint rewrites each test case's ``manifest.json`` and uploads the curated native outputs to the
content-addressed object store.
The manifest diff therefore names every native file that changed, and both the old and the new blob
remain fetchable by digest, so the change can be reviewed without checking anything out locally.

The pipeline runs in three stages:

- :mod:`~climate_ref.baseline_report.collect` reads the repository either side of the base ref,
  both the manifests and the committed bundle they track.
- :mod:`~climate_ref.baseline_report.analyse` fetches native blobs and builds the diffs.
- :mod:`~climate_ref.baseline_report.render` writes the static site.

Python decides and templates place, so no module here builds HTML.
"""
