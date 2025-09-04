# Installation

There are several ways to install and use the `climate-ref` package and associated CLI tool (ref). If you find the various options to be confusing, we recommend reading this [overview](packaging_confusion.md) about the state of python package installation.

The REF itself is a pure Python package and does not require any conda-based dependencies.
This was a deliberate decision to make it easy to make the framework easy to install in a range of different environments.
Some of the [diagnostic providers](nutshell.md) require additional dependencies in order to run an execution.
For these providers, the REF can automatically create a new Conda environment and install the required dependencies in this standalone environment.
Each of these provider-specific environments are decoupled to allow for potentially incompatible dependencies.
This uses a bundled version of [micromamba](https://github.com/mamba-org/micromamba-releases)
to create and manage the environments so no additional dependencies are required.

/// admonition | HPC Users
    type: warning

HPC users are recommended to install the REF inside a conda environment to ensure that key system dependencies are available.

This can be achieved using the following commands:

```bash
conda create -n climate-ref python=3.11
conda activate climate-ref
pip install climate-ref[aft-providers]
```

///

## Installing `climate-ref`
### PyPI

You can install `climate-ref` using `pip`:

```bash
pip install climate-ref
```

The PyPI package contains some extras that bundle some additional dependencies.
Depending on your use case you may want to install:

* `celery` - for running the REF in a distributed manner
* `aft-providers` - diagnostic providers for Assessment Fast Track
* `postgres` - additional dependencies required for using a PostgresSQL database

Extras can be installed by including the name inside square brackets after the package name:

```bash
pip install climate-ref[celery,aft-providers]
```

/// admonition | Note

`pip install climate-ref[aft-providers]` additionally installs the [diagnostic
providers](nutshell.md) that are used for the Assessment Fast Track. These
diagnostic providers declare the available diagnostics and the rules for when
they should be run. The actual execution occurs within a provider-specific conda
environment.
///

### Conda

/// admonition | conda-forge
    type: warning

The conda-forge packages are a work in progress and are not yet available.
See [#80](https://github.com/Climate-REF/climate-ref/issues/80) for more information.
///

We intend on providing a recipe on conda-forge. Once complete you can install `climate-ref` using `mamba` or `conda`:

```bash
mamba install -c conda-forge climate-ref
conda install -c conda-forge climate-ref
```

A modern alternative to using `conda` as package a manager is [Pixi](https://pixi.sh/dev/).
Pixi uses the same packages as `conda` but is faster and can create reproducible environments out of the box.


### Docker

For production use, we recommend using the `climate-ref` Docker image to provide a consistent environment for running the REF.
Not all users may support Docker directly, instead requiring the use of [Apptainer](https://apptainer.org/docs/user/latest/) to provide some additional isolation.

```
docker pull ghcr.io/climate-ref/climate-ref:latest
```

or

```bash
apptainer pull docker://ghcr.io/climate-ref/climate-ref:latest
```

If you want to use the latest development version, you can build the Docker image from the source code:

```bash
git clone https://github.com/Climate-REF/climate-ref.git
cd climate-ref
docker build
```

If you require the full-stack of services recommended for a production deployment, you can use the `docker-compose` file to start the services.
This requires an initialisation step to fetch some required datasets and environments before starting the services.

```bash
bash scripts/initialise-docker.sh
docker-compose up
```

### From Source

To install `climate-ref` from the source code, clone the repository and install it:

```bash
git clone https://github.com/Climate-REF/climate-ref.git
cd climate-ref
make virtual-environment
```

See the [development documentation](development.md) for more information on how to contribute to the project.

[](){#provider-dependencies}
## Installing metric provider dependencies

/// admonition | Windows support
    type: warning

Windows doesn't support some of the packages required by the [diagnostic providers](nutshell.md),
so we only support MacOS and Linux.
Windows users are recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
or a Linux VM if they wish to use the REF.

///

Some metric providers can use their own conda environments.
The REF can manage these for you,
using a bundled version of [micromamba](https://github.com/mamba-org/micromamba-releases).

The conda environments for the registered providers can be created with the following command:

```bash
ref providers create-env
```

A new environment will be created automatically for each conda-based metric provider when it is first used,
if one does not already exist.
This can cause issues if the environment is created on a node that doesn't have internet access,
or if a race condition occurs when multiple processes try to create the environment at the same time.


/// admonition | Note

The PMP conda environment is not yet available for arm-based MacOS users,
so the automatic installation process will fail.

Arm-based MacOS users can use the following command to create the conda environment manually:

```bash
MAMBA_PLATFORM=osx-64 ref providers create-env --provider pmp
```

///

The created environments and their locations can be viewed using the command:

```bash
ref providers list
```
