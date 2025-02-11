# Getting started

The REF is designed to be run in a variety of environments, including local execution, cloud-based execution, and execution on HPC systems.
The REF can be run as a standalone application, as a set of services, or as a set of docker containers.

After installing the REF, you can start using the REF via the CLI tool:

```bash
$ uv run ref

 Usage: ref [OPTIONS] COMMAND [ARGS]...

 cmip_ref: A CLI for the CMIP Rapid Evaluation Framework

╭─ Options ─────────────────────────────────────────────────────────────────────────╮
│ --configuration-directory          PATH                  Configuration directory  │
│                                                          [default: None]          │
│ --verbose                  -v                                                     │
│ --log-level                        [WARNING|DEBUG|INFO]  [default: WARNING]       │
│ --version                                                                         │
│ --install-completion                                     Install completion for   │
│                                                          the current shell.       │
│ --show-completion                                        Show completion for the  │
│                                                          current shell, to copy   │
│                                                          it or customize the      │
│                                                          installation.            │
│ --help                                                   Show this message and    │
│                                                          exit.                    │
╰───────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────╮
│ solve      Solve for metrics that require recalculation                           │
│ config     View and update the REF configuration                                  │
│ datasets   View and ingest input datasets                                         │
│ celery     Managing remote execution workers                                      │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

This provides the ability to:

* **Ingest** new input datasetes
* **Solve** for the unique metrics executions that are required
* **Execute** the metrics either locally or remotely
