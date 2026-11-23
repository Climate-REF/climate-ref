r"""
Generate the CMIP7 parquet catalog from CMIP7 data published on ESGF.

ESGF's STAC index lists the datasets and their files.
Each file's header is then read over HTTP with a few small range requests
instead of a download of the whole file,
and parsed by the same function the complete CMIP7 parser uses.
So the rows are what ``ref datasets ingest`` with the complete parser would give for the same files.

Selected are the latest versions of the datasets whose branded variable is in the bundled
Data Request mapping (which covers every CMIP7 requirement of the providers),
of at most ``--max-variants`` ensemble members per model and experiment (the lowest ones),
plus the members that the selected datasets name as their parents.

Header reads are appended to ``--cache`` as they complete, so an interrupted run resumes where it stopped.
Files that cannot be read leave their whole dataset out of the catalog, so no dataset is incomplete.

Usage::

    uv run --with h5netcdf --with h5py --with aiohttp python scripts/generate_cmip7_catalog.py
"""

from __future__ import annotations

import json
import multiprocessing
import re
import signal
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import typer
from loguru import logger

from climate_ref.datasets import CMIP7DatasetAdapter
from climate_ref.datasets.cmip7_parsers import parse_cmip7_dataset
from climate_ref.solve_helpers import write_catalog_parquet
from climate_ref_core.data_request import dreq_entries

STAC_INDICES = ("discovery.east.esgf.io", "discovery.west.esgf.io")
PAGE_SIZE = 200
READ_BLOCK_SIZE = 2**16
"""Bytes fetched per range request; small blocks are fastest as distant data nodes are slow to send."""

DatasetKey = tuple[str, str, str]
"""(source_id, experiment_id, variant_label)"""


def _retry(func: Any, attempts: int = 4, delay: float = 5.0) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.debug(f"Attempt {attempt} failed ({exc}), retrying")
            time.sleep(delay * attempt)


def search_index(index: str, variable_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Return the STAC items of the latest CMIP7 datasets of the given variables."""
    session = requests.Session()
    body: dict[str, Any] = {
        "collections": ["CMIP7"],
        "limit": PAGE_SIZE,
        "filter-lang": "cql2-json",
        "filter": {
            "op": "and",
            "args": [
                {"op": "in", "args": [{"property": "cmip7:variable_id"}, sorted(variable_ids)]},
                {"op": "=", "args": [{"property": "latest"}, True]},
            ],
        },
    }

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        response = session.post(f"https://{index}/search", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    page = _retry(lambda: post(body))
    logger.info(f"{index} matches {page.get('numberMatched')} datasets")
    items: list[dict[str, Any]] = []
    while True:
        items.extend(page["features"])
        next_links = [link for link in page.get("links", []) if link.get("rel") == "next"]
        if not next_links:
            return items
        link = next_links[0]
        if link.get("method", "GET") == "POST":
            page = _retry(lambda link=link: post(link["body"]))
        else:
            page = _retry(lambda link=link: session.get(link["href"], timeout=120).json())


def _variant_order(variant_label: str) -> tuple[int, ...]:
    return tuple(int(number) for number in re.findall(r"\d+", variant_label))


def _key(properties: dict[str, Any], prefix: str = "") -> DatasetKey:
    return (
        properties.get(f"cmip7:{prefix}source_id", ""),
        properties.get(f"cmip7:{prefix}experiment_id", ""),
        properties.get(f"cmip7:{prefix}variant_label", ""),
    )


def select_datasets(
    items: list[dict[str, Any]],
    wanted: set[tuple[str, str, str]],
    max_variants: int,
) -> list[dict[str, Any]]:
    """
    Select the datasets the catalog holds.

    Parameters
    ----------
    items
        STAC items of the candidate datasets
    wanted
        The ``(branded_variable, frequency, region)`` combinations to keep
    max_variants
        How many ensemble members to keep per model and experiment, lowest first

    Returns
    -------
    :
        The selected items, which also include the ensemble members the selected datasets
        name as their parents
    """
    candidates = [item for item in items if _is_wanted(item["properties"], wanted)]
    selected = _with_parents(candidates, _lowest_variants(candidates, max_variants))
    return [item for item in candidates if _key(item["properties"]) in selected]


def _is_wanted(properties: dict[str, Any], wanted: set[tuple[str, str, str]]) -> bool:
    variable = (
        properties.get("cmip7:variable_branded_name"),
        properties.get("cmip7:frequency"),
        properties.get("cmip7:region"),
    )
    return variable in wanted and not properties.get("retracted", False)


def _lowest_variants(candidates: list[dict[str, Any]], max_variants: int) -> set[DatasetKey]:
    """Keep the lowest ``max_variants`` ensemble members of each model and experiment."""
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in candidates:
        source_id, experiment_id, variant_label = _key(item["properties"])
        variants[(source_id, experiment_id)].add(variant_label)
    return {
        (source_id, experiment_id, variant_label)
        for (source_id, experiment_id), labels in variants.items()
        for variant_label in sorted(labels, key=_variant_order)[:max_variants]
    }


def _with_parents(candidates: list[dict[str, Any]], selected: set[DatasetKey]) -> set[DatasetKey]:
    """Add the parents of the selected members until every parent is selected as well."""
    available = {_key(item["properties"]) for item in candidates}
    while True:
        parents = {
            _key(item["properties"], prefix="parent_")
            for item in candidates
            if _key(item["properties"]) in selected
        }
        added = (parents & available) - selected
        if not added:
            return selected
        selected = selected | added


def data_files(item: dict[str, Any]) -> list[tuple[str, str]]:
    """
    List the files of a dataset that can be read over HTTP.

    Returns
    -------
    :
        ``(path, url)`` pairs, where ``path`` is the file's path in the DRS from ``MIP-DRS7`` onwards
    """
    files = []
    for asset in item["assets"].values():
        url = asset.get("href", "")
        # Globus links point at a web app, not at the file
        is_file = url.startswith(("https://", "http://")) and urlparse(url).path.endswith(".nc")
        if "data" not in asset.get("roles", []) or not is_file:
            continue
        path = asset.get("file:local_path") or url[url.index("MIP-DRS7/") :]
        files.append((path, url))
    return sorted(files)


def _plain(value: Any) -> Any:
    """Turn the numpy and bytes values a header read returns into JSON-serialisable ones."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    return value


def read_header(url: str, path: str) -> dict[str, Any]:
    """Parse a remote file's metadata the way the complete parser parses a local one."""
    import aiohttp  # noqa: PLC0415
    import fsspec  # noqa: PLC0415
    import h5netcdf.legacyapi  # noqa: PLC0415

    # Without a read timeout, a data node that stops sending leaves the read waiting forever
    timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, sock_read=60)
    with fsspec.open(
        url, block_size=READ_BLOCK_SIZE, cache_type="blockcache", client_kwargs={"timeout": timeout}
    ) as remote:
        with h5netcdf.legacyapi.Dataset(remote, "r") as ds:
            row = parse_cmip7_dataset(ds, f"{{data_dir}}/{path}")
    return {key: _plain(value) for key, value in row.items()}


def _load_cache(cache: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[record["path"]] = record
    return rows


READ_TIMEOUT = 300
"""Seconds one header read may take; a data node can send a response too slowly for aiohttp to time it out."""


def _read_with_timeout(url: str, path: str) -> dict[str, Any]:
    def give_up(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Reading the header took longer than {READ_TIMEOUT} s")

    # Each read runs in the main thread of its own worker process, so an alarm can interrupt it
    previous = signal.signal(signal.SIGALRM, give_up)
    signal.alarm(READ_TIMEOUT)
    try:
        return read_header(url, path)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _read_with_retries(url: str, path: str) -> dict[str, Any]:
    return _retry(lambda: _read_with_timeout(url, path), attempts=3)  # type: ignore[no-any-return]


def read_headers(
    files: list[tuple[str, str]],
    cache: Path,
    workers: int,
    per_host: int,
) -> dict[str, dict[str, Any]]:
    """
    Read the headers of the given files, reusing and extending ``cache``.

    The reads run in separate processes: h5py holds a global lock while it reads,
    so threads would wait for each other's network round trips.

    Returns
    -------
    :
        The parsed rows, keyed by path. Files that could not be read are left out.
    """
    rows = _load_cache(cache)
    todo = [(path, url) for path, url in files if f"{{data_dir}}/{path}" not in rows]
    logger.info(f"{len(files) - len(todo)} headers cached, {len(todo)} to read")

    queues: dict[str, deque[tuple[str, str]]] = defaultdict(deque)
    for path, url in todo:
        queues[urlparse(url).netloc].append((path, url))
    in_flight: Counter[str] = Counter()
    pending: dict[Future[dict[str, Any]], tuple[str, str]] = {}

    failed = done = 0
    context = multiprocessing.get_context("fork")
    with cache.open("a") as out, ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:

        def fill() -> None:
            # Take from every data node in turn, so a slow one does not hold up the others
            progressed = True
            while progressed and len(pending) < workers:
                progressed = False
                for host, queue in queues.items():
                    if queue and in_flight[host] < per_host and len(pending) < workers:
                        path, url = queue.popleft()
                        pending[pool.submit(_read_with_retries, url, path)] = (host, url)
                        in_flight[host] += 1
                        progressed = True

        fill()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                host, url = pending.pop(future)
                in_flight[host] -= 1
                done += 1
                try:
                    row = future.result()
                except Exception as exc:
                    failed += 1
                    logger.warning(f"Could not read {url}: {exc}")
                    continue
                rows[row["path"]] = row
                out.write(json.dumps(row) + "\n")
                out.flush()
                if done % 500 == 0:
                    logger.info(f"Read {done}/{len(todo)} headers ({failed} failed)")
            fill()

    if failed:
        logger.warning(f"{failed} headers could not be read; rerun to retry them")
    return rows


def build_catalog(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Turn parsed rows into the catalog the CMIP7 adapter would build from the same files."""
    adapter = CMIP7DatasetAdapter()
    catalog = adapter._enrich_parsed_catalog(pd.DataFrame(list(rows)))
    columns = [
        *adapter.dataset_specific_metadata,
        "branded_variable",
        *adapter.file_specific_metadata,
    ]
    return catalog[columns].sort_values(["instance_id", "path"]).reset_index(drop=True)


def _search_any_index(variable_ids: set[str]) -> list[dict[str, Any]]:
    """Search the first CMIP7 index that answers."""
    for index in STAC_INDICES:
        try:
            return search_index(index, variable_ids)
        except Exception as exc:
            logger.warning(f"Searching {index} failed: {exc}")
    return []


def complete_datasets(
    files_by_dataset: dict[str, list[tuple[str, str]]], rows: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the rows of the datasets whose every file was read, so no dataset is incomplete."""
    complete: list[dict[str, Any]] = []
    for dataset, dataset_files in files_by_dataset.items():
        dataset_rows = [rows.get(f"{{data_dir}}/{path}") for path, _ in dataset_files]
        if not dataset_files or any(row is None for row in dataset_rows):
            logger.warning(f"Leaving out {dataset}: not every file could be read")
            continue
        complete.extend(row for row in dataset_rows if row is not None)
    return complete


app = typer.Typer()


@app.command()
def main(
    output_path: Path = typer.Option(
        Path("tests/test-data/esgf-catalog/cmip7_catalog.parquet"), "--output", help="The catalog to write"
    ),
    cache: Path = typer.Option(
        Path("cmip7-catalog-headers.jsonl"), help="Where header reads are kept between runs"
    ),
    max_variants: int = typer.Option(2, help="Ensemble members to keep per model and experiment"),
    workers: int = typer.Option(32, help="Headers read at the same time, each in its own process"),
    per_host: int = typer.Option(12, help="Headers read at the same time from one data node"),
) -> None:
    """Generate the CMIP7 parquet catalog from CMIP7 data published on ESGF."""
    mappings = dreq_entries()
    wanted = {(entry.branded_variable, entry.frequency, entry.region) for entry in mappings}
    variable_ids = {entry.variable_id for entry in mappings}

    items = _search_any_index(variable_ids)
    if not items:
        raise typer.Exit(1)

    selected = select_datasets(items, wanted, max_variants)
    files_by_dataset = {item["id"]: data_files(item) for item in selected}
    files = [file for dataset_files in files_by_dataset.values() for file in dataset_files]
    logger.info(f"Selected {len(selected)} datasets with {len(files)} files")

    rows = read_headers(files, cache, workers=workers, per_host=per_host)

    catalog = build_catalog(complete_datasets(files_by_dataset, rows))
    write_catalog_parquet(catalog, output_path)
    logger.info(f"Wrote {len(catalog)} files of {catalog['instance_id'].nunique()} datasets to {output_path}")


if __name__ == "__main__":
    app()
