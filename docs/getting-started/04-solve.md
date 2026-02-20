# Solve Diagnostics

With your datasets ingested and cataloged, you can now [solve and execute](../nutshell.md) diagnostics using the `ref solve` command.

## 1. Run all diagnostics (default)

By default, [ref solve](../cli.md#solve) will discover and schedule _all_ available diagnostics across all providers. The default executor is the **local executor**, which runs diagnostics in parallel using a process pool:

```bash
ref solve --timeout 3600
```

This will:

- Query the catalog of ingested datasets (observations and model-output)
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

## 3. Re-execution and the dirty flag

Each execution group tracks whether it is **dirty** -- meaning it needs to be rerun.
The solver uses this flag, along with a hash of the input datasets,
to decide which executions to schedule on each solve.

An execution group is automatically marked dirty when:

- It is **first created** (no executions have been run yet)
- **New data is ingested** that changes the set of input datasets

An execution group is marked **not dirty** when an execution **completes**,
regardless of whether it succeeded or failed.
This prevents failed executions from being retried indefinitely with the same data.

### Retrying failed executions

If a diagnostic fails, it will not be retried on subsequent solves unless you take action.
There are several ways to retry:

**Retry specific groups** using `flag-dirty`:

```bash
# Find failed execution groups
ref executions list-groups --not-successful

# Flag a specific group for retry
ref executions flag-dirty <group_id>

# Re-solve to pick up the flagged groups
ref solve
```

**Retry all failed executions** using `--rerun-failed`:

```bash
ref solve --rerun-failed
```

This is useful after fixing a bug in a diagnostic provider or resolving
an environment issue that caused widespread failures.

**Retry stuck executions** using `fail-running`:

If executions appear stuck (e.g., due to a worker crash or out-of-memory error),
you can mark them as failed and flag their groups for retry:

```bash
# Fail all stuck executions
ref executions fail-running --force

# Fail only executions stuck for more than 2 hours
ref executions fail-running --older-than 2 --force

# Re-solve to retry the flagged groups
ref solve
```

## Next steps

Once diagnostics have completed, visualize the results in the [Visualise tutorial](05-visualise.md).
