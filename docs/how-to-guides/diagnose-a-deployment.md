# Diagnose a deployment

`ref doctor` looks for the problems that a solve hides rather than reports:reference data that is missing,
so its diagnostics quietly plan no executions;
data ingested under a source type no diagnostic requires, so nothing selects it;
and datasets whose files cover the same period twice, so a diagnostic reads that period more than once.

```bash
ref doctor
```

Findings are grouped by the check that produced them,
under the remedy they have in common,
so a deployment missing twenty reference datasets reads as one instruction and twenty names:

```text
3 findings from 4 checks: 3 warnings

missing-reference-data 3 warnings
  Fetch these, then ingest the directory they land in.
  ref datasets fetch-data --registry obs4ref --output-directory <dir>

  WOA-23 (obs4mips) is not ingested, so 2 diagnostics will not run
    Needed for so, thetao by ilamb/so-woa2023-surface, ilamb/thetao-woa2023-surface.
  ...
```

The command exits non-zero when it finds an error,
or when it finds a warning and `--strict` is used, so it can gate a run:

```bash
ref doctor --strict && ref solve
```

To see which checks would run, and where each came from:

```bash
ref doctor --list
```

## Reporting a problem

`--format markdown` produces a report that can be pasted into an issue.
Alongside the findings it describes the environment:
package versions, platform, configuration, paths, enabled providers, what is ingested, and the `REF_*`, `DASK_*` and `ESMVALTOOL_*` environment variables that are set.
This environment excluded with the `--no-environment` option.

```bash
ref doctor --format markdown
```

`--format json` produces the same content for scripting.

## Adding a check

A check is a function that takes a `DoctorContext` and returns a list of `Finding`s,
declared with `climate_ref.doctor.check`.
The context loads providers and catalogs lazily, so a check pays only for what it reads,
and a check that raises becomes a finding rather than stopping the rest of the run.

```python
import os

from climate_ref.doctor import DoctorContext, Finding, Severity, check


@check("scratch-writable", "The scratch directory can be written to")
def check_scratch_writable(context: DoctorContext) -> list[Finding]:
    if context.config is None or os.access(context.config.paths.scratch, os.W_OK):
        return []
    return [
        Finding(
            severity=Severity.ERROR,
            summary=f"The scratch directory {context.config.paths.scratch} is not writable",
            detail="Executions write their working files here, so every execution will fail.",
            remedy="Grant the user running the REF write access to it.",
        )
    ]
```

`detail` explains one finding, and `remedy` says what to do about it.
Findings sharing a remedy are reported under it once,
so keep that wording free of anything specific to a single finding.
A `command` that carries out the remedy goes in its own field, where it is printed unwrapped and stays pasteable.

The check does not name itself in its findings: the runner stamps the slug from the registration,
so the two cannot drift apart.
A check must tolerate a context that has nothing ingested for a given source type,
and one built without a database (`DoctorContext.from_catalogs`).

Checks that ship with the REF live in `climate_ref.doctor.checks`.
A package outside `climate_ref` contributes its own by pointing an entry point at the module that declares them:

```toml
[project.entry-points."climate-ref.doctor-checks"]
my_provider = "my_package.doctor_checks"
```

`ref doctor` imports that module for its `@check` declarations.
A module that cannot be imported is reported as an error finding rather than taking the command down,
because a check that never ran must not look like a check that passed.
