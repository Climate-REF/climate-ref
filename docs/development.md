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

# Download some test data and ingest the sample datasets.
make fetch-test-data
uv run ref datasets ingest --source-type cmip6 $PWD/tests/test-data/sample-data/CMIP6/
uv run ref datasets ingest --source-type obs4mips $PWD/tests/test-data/sample-data/obs4MIPs/
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

### Running your first `solve`

If you want to run the sample data through the whole pipeline, you need to download
reference data, but note that the reference data is severable Gigabytes in size.

```shell
# Download reference data which is not (yet) included in obs4mips
make fetch-ref-data
```

After that, you can let the REF calculate all included metrics for the sample data.
Note that this will take a while to run.

```shell
uv run ref solve
```

Afterwards, you can check the output of `uv run ref executions list` to see if metrics
were evaluated successfully, and if they were, you find the results in the
`.ref/results` folder.

### Pip editable installation

If you would like to install the REF into an existing (conda) environment
without using `uv`, run the command

```bash
for package in packages/cmip_ref_core packages/cmip_ref packages/cmip_ref_metrics-*; do
     pip install -e $package;
done
```

### Installing metric provider dependencies

/// admonition | Windows support
   type: warning

Window's doesn't support some of the packages required by the metrics providers,
so we only support MacOS and Linux.
Windows users are recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
or a Linux VM if they wish to use the REF.

///

Some metric providers can use their own conda environments.
The REF can manage these for you,
using a bundled version of [micromamba](https://github.com/mamba-org/micromamba-releases).

A new environment for a metric provider can be created with the following command:

```bash
ref --log-level=info providers create-env --provider esmvaltool
```

A new environment will be created automatically for each conda-based metric provider when it is first used,
if one does not already exist.

/// admonition | Note

The PMP conda environment is not yet available for arm-based MacOS users,
so the automatic installation process will fail.

Arm-based MacOS users can use the following command to create the conda environment manually:

```bash
MAMBA_PLATFORM=osx-64 uv run ref providers create-env --provider pmp
```

///


To update a conda-lock file, run for example:

```bash
uvx uvx conda-lock -p linux-64 -p osx-64 -p osx-arm64 -f packages/ref-metrics-esmvaltool/src/cmip_ref_metrics_esmvaltool/requirements/environment.yml
mv conda-lock.yml packages/ref-metrics-esmvaltool/src/cmip_ref_metrics_esmvaltool/requirements/conda-lock.yml
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

# Manually edit the `SAMPLE_VERSION` in `packages/ref/src/cmip_ref/testing.py`

# Regenerate any failing regression tests that depend on the sample data catalog
export PYTEST_ADDOPTS="--force-regen"
make test
```

Some other manual tweaks may be required to get the test suite to pass,
but we should try and write tests that don't change when new data becomes available,
or to use [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/api.html) to be able to
regenerate the expected output files.

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
[cmip-ref.readthedocs.io](https://cmip-ref.readthedocs.io/en/latest/).

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

## Database management

The REF uses a local Sqlite database to store state information.
We use [alembic](https://alembic.sqlalchemy.org/en/latest/) to manage our database migrations
as the schema of this database changes.

When making changes to the database models (`cmip_ref.models`),
a migration must also be added (see below).

The migration definitions (and the alembic configuration file)
are included in the `cmip_ref` package (`packages/ref/src/cmip_ref/migrations`)
to enable users to apply these migrations transparently.
Any new migrations are performed automatically when using the `ref` command line tool.

### Adding a database migration

If you have made changes to the database models,
you will need to create a new migration to apply these changes.
Alembic can autogenerate these migrations for you,
but they will need to be reviewed to ensure they are correct.

```
uv run alembic -c packages/ref/src/cmip_ref/alembic.ini \
   revision --autogenerate --message "your_migration_message"
```

[](){releasing-reference}
## Releasing

Releasing is semi-automated via a CI job.
The CI job requires the type of version bump
that will be performed to be manually specified.
The supported bump types are:

* `major`
* `minor`
* `patch`

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
