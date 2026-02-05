# Solve Diagnostics

With your datasets ingested and cataloged, you can now [solve and execute](../nutshell.md) diagnostics using the `ref solve` command.

## 1. Run all diagnostics (default)

By default, [ref solve](../cli.md#solve) will discover and schedule _all_ available diagnostics across all providers. The default executor is the **local executor**, which runs diagnostics in parallel using a process pool:

```bash
ref solve --timeout 3600
```

This will:

- Query the catalog of ingested datasets (observations and model-output), completing any missing metadata as needed
- Determine which diagnostics are applicable and how many different executions are needed
- Execute each diagnostic in parallel on your machine
- Use a timeout of 3600 seconds (1 hour) to complete the runs

Note: it is normal for some executions to fail (e.g., due to missing data or configuration).
You can re-run or inspect failures as needed.

/// admonition | Tip

To target a specific provider or diagnostic, use the `--provider` and `--diagnostic` flags:

```bash
# Run only PMP diagnostics
ref solve --provider pmp

# Run only diagnostics containing "enso" in their slug
ref solve --diagnostic enso
```

Replace `pmp` or `enso` with any provider or diagnostic slug listed in your installation.
///

## 2. Monitor execution status

You can view the status of execution groups with:

```bash
ref executions list-groups
```

Each group corresponds to a set of related executions (e.g., all runs of a diagnostic for one model).
To see details for a specific group, use:

```bash
ref executions inspect <group_id>
```

This will show the status (pending, running, succeeded, failed) of each execution in the group and any error messages.
This log output is very useful to include if you need to [report an issue or seek help](https://github.com/Climate-REF/climate-ref/issues).

## Next steps

Once diagnostics have completed, visualise the results in the [Visualise tutorial](05-visualise.md).
