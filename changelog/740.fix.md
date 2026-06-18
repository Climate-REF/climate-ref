Fixed a bug where a diagnostic whose results failed to ingest was incorrectly recorded as successful with no metric values; such executions are now marked failed and retried on the next solve.
