# Configuration

## Environment Variables

Environment variables are used to control some aspects of the model.
The default values for these environment variables are generally suitable,
but if you require updating these values we recommend the use of a `.env` file
to make the changes easier to reproduce in future.

### `REF_EXECUTOR`

Executor to use for running the metrics.

Defaults to use the local executor ("local").

### `REF_ESGF_ROOT`

Path to the root of the ESGF output data.

### `REF_OUTPUT_ROOT`

Path to the root directory where data should be stored.
This has to be shared between any workers and the parent
process.
