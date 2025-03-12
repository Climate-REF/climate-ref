# Build the container for the the REF compute engine
# This docker container packages up the REF cli tool with the default set of diagnostic providers
# used as part of the CMIP7 FastTrack process.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /app

FROM base AS build
# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-editable --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app

# Sync the project as non-editable installs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev

# Runtime container
FROM base AS runtime

LABEL maintainer="Jared Lewis <jared.lewis@climate-resource.com>"
LABEL description="Base Docker image for the REF compute engine"

# Create a non-root user
RUN useradd -m -u 1000 app

# Allow celery to run as root
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV=/app/.venv

# Copy the installed packages from the build stage
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/scripts /app/scripts

# Location of the REF configuration files
ENV REF_CONFIGURATION=/ref

# Set matplotlib config directory to a location with write permissions
ENV MPLCONFIGDIR=$REF_CONFIGURATION/software/matplotlib

# Set micromamba root to locations with write permissions
ENV MAMBA_ROOT_PREFIX=$REF_CONFIGURATION/software/conda
ENV CONDA_PKGS_DIRS=$REF_CONFIGURATION/software/conda/pkgs
ENV XDG_CACHE_HOME=$REF_CONFIGURATION/cache

# Create necessary directories with proper permissions
RUN install --owner=app --group=app -d /ref /app/cache && \
    install --mode=555 -d /home/app/.conda && \
    touch /home/app/.conda/environments.txt  # conda tries to write to ~/.conda/environments.txt despite its root being set elsewhere. See: https://github.com/conda/conda/issues/8804

# Switch to non-root user -- use numeric ID for k8s systems that enforce runAsUser
USER 1000

# Pre-cache matplotlib fonts and ilamb3 config
RUN /app/.venv/bin/python -c "import matplotlib; import ilamb3"

# Run the REF CLI tool by default
ENTRYPOINT ["/app/.venv/bin/ref"]
