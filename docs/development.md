[](){#development-reference}

# Development

Notes for developers. If you want to get involved, please do!
We welcome all kinds of contributions, for example:

- docs fixes/clarifications
- bug reports
- bug fixes
- feature requests
- pull requests
- tutorials

## Development Installation

For development, we rely on [uv](https://docs.astral.sh/uv) for all our
dependency management. To get started, you will need to make sure that uv
is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/)).

We use our `Makefile` to provide an easy way to run common developer commands.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.

The following steps are required to set up a development environment.
This will install the required dependencies and fetch some test data,
as well as set up the configuration for the REF.

```bash
# Create a virtual environment containing the REF and its dependencies.
make virtual-environment

# Configure the REF.
mkdir $PWD/.ref
uv run ref config list > $PWD/.ref/ref.toml
export REF_CONFIGURATION=$PWD/.ref
```

`uv` will create a virtual Python environment in the directory `.venv` containing
the REF and its (development) dependencies.
To use the software installed in this environment without starting every command
with `uv run`, activate it by calling `. .venv/bin/activate`.
It can be deactivated with the command `deactivate`.

The local `ref.toml` configuration file will make it easier to play around with settings.
By default, the database will be stored in your home directory,
this can be modified by changing the `db.database_url` setting in the `ref.toml` file.

If there are any issues, the messages from the `Makefile` should guide you
through. If not, please raise an issue in the
[issue tracker](https://github.com/Climate-REF/climate-ref/issues).

### Ingesting datasets

The REF requires datasets, both reference and model, to be ingested into the database.
These ingested datasets are then used to solve for what executions are available and require running.

We have a consistent set of decimated sample data that is used for testing.
These can be ingested using the following command:

```bash
make fetch-test-data
uv run ref datasets ingest --source-type cmip6 $PWD/tests/test-data/sample-data/CMIP6/
uv run ref datasets ingest --source-type obs4mips $PWD/tests/test-data/sample-data/obs4REF/
```

Additional reference datasets can be fetched by following the instructions [here](getting-started/02-download-datasets.md).
The Obs4REF step is not required as we have already ingested these datasets above.

### Setting up providers

The REF uses a number of different providers to run the diagnostics.
Providers may require conda environments and reference data to be set up before running.

```bash
uv run ref providers setup
```

This command creates conda environments and fetches any required reference data.
All operations are idempotent, so it's safe to run multiple times.

The installed providers and their status can be viewed using:

```bash
uv run ref providers list
```

### Running your first `solve`

Once you have ingested some sample data and created any required environments,
you can run your first `solve` command.

A `solve` will take the ingested datasets and the providers declared in the configuration, and determine
which new executions are required.

Note that this will take a while to run.

```shell
uv run ref solve
```

Afterwards, you can check the output of `uv run ref executions list-groups` to see if metrics
were evaluated successfully, and if they were, you find the results in the
`$PWD/.ref/results` folder.
Don't worry too much if some executions are failing for you,
things are still in active development at the moment.

### Pip editable installation

If you would like to install the REF into an existing (conda) environment
without using `uv`, run the command

```bash
for package in packages/climate-ref*; do
     pip install -e $package;
done
```

### Advanced provider setup

/// admonition | Windows support
   type: warning

Window's doesn't support some of the packages required by the diagnostic providers,
so we only support MacOS and Linux.
Windows users are recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
or a Linux VM if they wish to use the REF.

///

/// admonition | Note

The PMP conda environment is not yet available for arm-based MacOS users,
so the automatic installation process will fail.

Arm-based MacOS users can use the following command to set up the PMP provider manually:

```bash
MAMBA_PLATFORM=osx-64 uv run ref providers setup --provider pmp
```

///

#### Updating conda-lock files

To update a conda-lock file for a provider, run for example:

```bash
uvx conda-lock -p linux-64 -p osx-64 -p osx-arm64 -f packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/requirements/environment.yml
mv conda-lock.yml packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/requirements/conda-lock.yml
```

## Tests and code quality

The test suite can then be run using `make test`.
This will run the test suites for each package and finally the integration test suite.

We make use of [`ruff`](https://docs.astral.sh/ruff/) (code formatting and
linting) and [`mypy`](https://mypy.readthedocs.io/en/stable/) (type checking)
and [`pre-commit`](https://pre-commit.com/) (checks before committing) to
maintain good code quality.

These tools can be run as usual after activating the virtual environment or
using the makefile:

```bash
make pre-commit
make mypy
make test
```

If you require executing a specific diagnostic, you manually invoke `pytest` as follows for `ilamb's`gpp-fluxnet2015` diagnostic:

```bash
pytest --slow -k gpp-fluxnet2015
```

Some diagnostics may require additional filtering to limit the pytest's scope to a directory or test name.
Adding `--collect-only` to this will describe which tests will be executed which is useful as some of these test may take 30s to minutes to run.

When adding a new diagnostic to a provider, you should run the above command with `--force-regen` attribute to capture the output from the execution.

### Sample data

We use sample data  from [ref-sample-data](https://github.com/Climate-REF/ref-sample-data)
to provide a consistent set of data for testing.
These data are fetched automatically by the test suite.

As we support more metrics,
we should expand the sample data to include additional datasets to be able to adequately test the REF.
If you wish to use a particular dataset for testing,
please open a pull request to add it to the sample data repository.

The sample data is versioned and periodically we need to update the targeted version in the REF.
Updating the sample data can be done by running the following command:

```bash
# Fetch the latest registry from the sample data repository
make update-sample-data-registry

# Manually edit the `SAMPLE_VERSION` in `packages/climate-ref/src/climate_ref/testing.py`

# Regenerate any failing regression tests that depend on the sample data catalog
export PYTEST_ADDOPTS="--force-regen"
make test
```

Some other manual tweaks may be required to get the test suite to pass,
but we should try and write tests that don't change when new data becomes available,
or to use [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/api.html) to be able to
regenerate the expected output files.

### ESGF parquet catalogs

Pre-generated parquet catalogs in `tests/test-data/esgf-catalog/` contain dataset metadata
from a real ESGF archive without requiring actual data files on disk.
These are used by tests that only need DataFrames for solver logic,
avoiding the need to download sample data.

```
tests/test-data/esgf-catalog/
├── cmip6_catalog.parquet             # ~96k CMIP6 entries
├── obs4mips_catalog.parquet          # ~176 obs4MIPs entries
└── pmp_climatology_catalog.parquet   # ~25 PMP climatology entries
```

Two test fixtures are available for tests that need data catalogs:

| Fixture             | Source                    | Requires sample data download? |
| ------------------- | ------------------------- | ------------------------------ |
| `data_catalog`      | Scans sample data on disk | Yes                            |
| `esgf_data_catalog` | Reads parquet catalogs    | No                             |

Use `esgf_data_catalog` for tests that only need DataFrames (solver logic, constraint testing, formatting).
Use `data_catalog` for tests that need actual files on disk (integration tests, diagnostic execution).

To regenerate or extend the catalogs from a local ESGF archive:

```bash
python scripts/generate_esgf_catalog.py \
    --cmip6-dir /path/to/CMIP6 \
    --obs4mips-dir /path/to/obs4MIPs \
    --pmp-climatology-dir /path/to/PMP_obs4MIPsClims \
    --output-dir tests/test-data/esgf-catalog/ \
    --strip-prefix /path/to/data/root
```

### Solver regression tests

Solver regression tests in `packages/climate-ref/tests/unit/test_solve_regression.py`
run the solver against the ESGF parquet catalogs for each provider
and compare the output against YAML baselines.
Any change to solver logic, constraints, or data requirements
will produce a diff in these baseline files.

To regenerate baselines after an intentional change:

```bash
uv run pytest packages/climate-ref/tests/unit/test_solve_regression.py --force-regen
```

## Documentation

Our documentation is written in Markdown and built using
[`mkdocs`](https://www.mkdocs.org/).
It can be viewed while editing by running `make docs-serve`.

It is hosted by
[Read the Docs (RtD)](https://www.readthedocs.org/),
a service for which we are very grateful.
The RtD configuration can be found in the `.readthedocs.yaml` file
in the root of this repository.
The docs are automatically deployed at
[climate-ref.readthedocs.io](https://climate-ref.readthedocs.io/en/latest/).

## Workflows

We don't mind whether you use a branching or forking workflow.
However, please only push to your own branches,
pushing to other people's branches is often a recipe for disaster,
is never required in our experience
so is best avoided.

Try and keep your pull requests as small as possible
(focus on one thing if you can).
This makes life much easier for reviewers
which allows contributions to be accepted at a faster rate.

## Language

We use British English for our development.
We do this for consistency with the broader work context of our lead developers.

## Versioning

This package follows the version format
described in [PEP440](https://peps.python.org/pep-0440/)
and [Semantic Versioning](https://semver.org/)
to describe how the version should change
depending on the updates to the code base.

Our changelog entries and compiled [changelog](./changelog.md)
allow us to identify where key changes were made.

## Changelog

We use [towncrier](https://towncrier.readthedocs.io/en/stable/)
to manage our changelog which involves writing a news fragment
for each Merge Request that will be added to the [changelog](./changelog.md) on the next release.
See the [changelog](https://github.com/Climate-REF/climate-ref/tree/main/changelog) directory
for more information about the format of the changelog entries.

## Dependency management

We manage our dependencies using [uv](https://docs.astral.sh/uv/).
This allows the ability to author multiple packages in a single repository,
and provides a consistent way to manage dependencies across all of our packages.
This mono-repo approach might change once the packages become more mature,
but since we are in the early stages of development,
there will be a lot of refactoring of the interfaces to find the best approach.

We follow [SPEC-0000](https://scientific-python.org/specs/spec-0000/)
which defines a 2-year support window for key scientific-computing libraries
and 3-year window for Python versions.
The aim of this specification is to reduce the maintenance burden of older packages and Python versions.

Our test suite does not currently test the oldest versions of these dependencies
([#205](https://github.com/Climate-REF/climate-ref/issues/205)).
Please raise an issue if you find that the REF doesn't comply with SPEC-0000.

## Database management

The REF uses a local Sqlite database to store state information.
We use [alembic](https://alembic.sqlalchemy.org/en/latest/) to manage our database migrations
as the schema of this database changes.

When making changes to the database models (`climate_ref.models`),
a migration must also be added (see below).

The migration definitions (and the alembic configuration file)
are included in the `climate_ref` package (`packages/climate-ref/src/climate_ref/migrations`)
to enable users to apply these migrations transparently.
Any new migrations are performed automatically when using the `ref` command line tool.

/// admonition | Note

We support both PostgreSQL and SQLite as target databases.
This leads to some subtle differences in the migrations to provide support for both database dialects.

The most notable difference occurs when modifying an existing table,
which isn't a concept that SQLite supports.
The alembic docs on [batch mode](https://alembic.sqlalchemy.org/en/latest/batch.html) is useful reading if you encounter any errors.

It may be useful to add postgres/sqlite specific blocks to the autogenerated revisions.
``
if batch_op.get_context().dialect.name == "postgresql":

# Postgresql specific code

``

A postgreSQL-specific test is part of the integration test suite.
This test checks that any migrations work using postgreSQL.

///

### Adding a database migration

If you have made changes to the database models,
you will need to create a new migration to apply these changes.
Alembic can autogenerate these migrations for you,
but they will need to be reviewed to ensure they are correct.

```
uv run alembic -c packages/climate-ref/src/climate_ref/alembic.ini \
   revision --autogenerate --message "your_migration_message"
```

[](){releasing-reference}

## Releasing

Releasing is semi-automated via a CI job.
The CI job requires the type of version bump
that will be performed to be manually specified.
The supported bump types are:

- `major`
- `minor`
- `patch`

We don't yet support pre-release versions,
but this is something that we will consider in the future.

### Standard process

The steps required are the following:

1. Bump the version: manually trigger the "bump" workflow from the main branch
   (see here: [bump workflow](https://github.com/Climate-REF/climate-ref/actions/workflows/bump.yaml)).
   A valid "bump_rule" will need to be specified.
   This will then trigger a draft release.

1. Edit the draft release which has been created
   (see here: [project releases](https://github.com/Climate-REF/climate-ref/releases)).
   Once you are happy with the release (removed placeholders, added key
   announcements etc.) then hit 'Publish release'.
   This triggers the [deploy workflow](https://github.com/Climate-REF/climate-ref/actions/workflows/deploy.yaml).
   This workflow deploys the built wheels and source distributions from the release to PyPI.

1. That's it, release done, make noise on social media of choice, do whatever
   else

1. Enjoy the newly available version
