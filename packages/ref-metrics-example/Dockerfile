# Build the container for the ref-metrics-package
#
# ref-metrics-example is a relatively simple package so can be directly installed via pip.
# More complex packages may require a multi-stage build to build the package before installing it.

# TODO: Remove build stage when the package is published
# Build wheels for the core packages
# This is temporary until the package is published on PyPI
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

COPY --from=root . /build

RUN uv build --package ref-core -o /wheels
RUN uv build --package ref-celery -o /wheels

# Build the container for the ref-metrics-package
FROM python:3.12-slim-bookworm

LABEL maintainer="Jared Lewis <jared.lewis@climate-resource.com>"
LABEL description="Docker image with the execution environment for the ref-metrics-example package"

ENV C_FORCE_ROOT=false

WORKDIR /app
RUN useradd -ms /bin/bash celery

COPY pyproject.toml README.md /app/

# TODO: Remove this section when the package is published
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

# TODO: Uncomment this line when the package is published
# ref-celery is required enable remote execution of the registered metrics
#RUN pip install ref-celery

COPY . /app
RUN chown -R celery:celery /app
RUN pip install -e .

USER celery
