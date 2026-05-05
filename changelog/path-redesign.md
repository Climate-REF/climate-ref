Reworked the layout of execution output directories.
New executions now write to ``<provider>/<diagnostic>/<group_short>/<execution_id>/``
instead of ``<provider>/<diagnostic>/<dataset_hash>/``,
so reruns of the same diagnostic group no longer overwrite earlier outputs.
Existing rows on disk continue to resolve through their stored ``Execution.output_fragment``.
