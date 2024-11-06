# ref-celery

This package provides celery task generation from Provider and Metric definitions.

## CLI tool

The `ref-celery` package provides a CLI tool to start a worker instance from a REF metrics provider.

### Usage

For example, to start a worker instance for the `ref-metrics-example` package:

```bash
ref-celery --package ref-metrics-example
```

This requires the `ref-metrics-example` package to be installed in the current virtual environment.
