"""
Benchmark peak memory and runtime for whole-tree vs chunked ingest.

Creates a synthetic CMIP6 DRS tree of empty ``.nc`` files (the DRS parser
only reads filenames, so empty files suffice). Runs the ingest pipeline in
two modes — the legacy whole-tree mode and the streaming chunked mode added
in this PR — under ``tracemalloc`` and reports the peak resident allocation
and elapsed time for each.

Run example::

    uv run python scripts/benchmark_ingest_memory.py \\
        --num-files 50000 --num-datasets 200 --chunk-size 10000

The script is intentionally a stand-alone diagnostic — not wired into CI —
so contributors can sanity-check that streaming actually bounds peak memory.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

# Suppress loguru chatter for cleaner benchmark output.
os.environ.setdefault("LOGURU_LEVEL", "WARNING")


def _synth_filename(
    variable: str,
    table: str,
    source: str,
    experiment: str,
    member: str,
    grid: str,
    daterange: str,
) -> str:
    return f"{variable}_{table}_{source}_{experiment}_{member}_{grid}_{daterange}.nc"


def build_synthetic_archive(
    root: Path, num_files: int, num_datasets: int, files_per_dataset: int | None = None
) -> tuple[int, int]:
    """
    Create a synthetic CMIP6 DRS tree of empty ``.nc`` files.

    Parameters
    ----------
    root
        Directory under which the synthetic archive is created.
    num_files
        Total target number of files. Combined with ``num_datasets`` to derive
        ``files_per_dataset``.
    num_datasets
        Number of distinct (instance_id) datasets to create.
    files_per_dataset
        Override how many files belong to each dataset; otherwise computed as
        ``ceil(num_files / num_datasets)``.

    Returns
    -------
    :
        (actual_num_files, actual_num_datasets) — may differ slightly from the
        requested values due to rounding.
    """
    if files_per_dataset is None:
        files_per_dataset = max(1, num_files // num_datasets)

    actual_files = 0
    actual_datasets = 0

    table_id = "Amon"
    variable_id = "tas"
    activity_id = "CMIP"
    grid_label = "gn"
    institution_id = "TEST-INST"
    experiment_id = "historical"

    for ds_idx in range(num_datasets):
        if actual_files >= num_files:
            break
        source_id = f"SRC-{ds_idx:04d}"
        member_id = "r1i1p1f1"
        version = "v20240101"

        # CMIP6 DRS path:
        # CMIP6/<activity>/<institution>/<source>/<experiment>/<member>/<table>/<variable>/<grid>/<version>/
        dataset_dir = (
            root
            / "CMIP6"
            / activity_id
            / institution_id
            / source_id
            / experiment_id
            / member_id
            / table_id
            / variable_id
            / grid_label
            / version
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for fi in range(files_per_dataset):
            if actual_files >= num_files:
                break
            year_start = 1850 + fi * 10
            year_end = year_start + 9
            daterange = f"{year_start}01-{year_end}12"
            filename = _synth_filename(
                variable_id, table_id, source_id, experiment_id, member_id, grid_label, daterange
            )
            (dataset_dir / filename).touch()
            actual_files += 1
        actual_datasets += 1

    return actual_files, actual_datasets


def _make_isolated_config(scratch: Path):
    """
    Build a fresh climate_ref ``Config`` rooted inside ``scratch``.

    Imports happen lazily so ``REF_CONFIGURATION`` is set before climate_ref
    reads any defaults. The CMIP6 parser is forced to ``drs`` because our
    synthetic archive uses empty ``.touch()`` files that cannot be opened by
    netCDF4.
    """
    os.environ["REF_CONFIGURATION"] = str(scratch / "ref-config")
    (scratch / "ref-config").mkdir(parents=True, exist_ok=True)

    from climate_ref.config import Config

    cfg = Config.default()
    cfg.cmip6_parser = "drs"
    cfg.save()
    return cfg


def _run(
    label: str,
    archive: Path,
    fn: Callable[[], None],
) -> dict[str, float | str]:
    """Run ``fn`` once under tracemalloc and return timing/memory stats."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "archive": str(archive),
        "elapsed_s": round(elapsed, 3),
        "peak_mib": round(peak / (1024 * 1024), 2),
        "current_mib": round(current / (1024 * 1024), 2),
    }


def _fresh_db(cfg):
    """Build a fresh database with migrations applied."""
    from climate_ref.database import Database

    return Database.from_config(cfg, run_migrations=True)


def benchmark(num_files: int, num_datasets: int, chunk_size: int) -> None:
    """Run baseline vs streaming on the same synthetic tree and print results."""
    with tempfile.TemporaryDirectory(prefix="ref-ingest-bench-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "archive"
        archive.mkdir()

        print(f"Building synthetic archive at {archive}...")
        actual_files, actual_datasets = build_synthetic_archive(archive, num_files, num_datasets)
        print(f"  -> {actual_files} files across {actual_datasets} datasets\n")

        # Defer imports until env is set so Config.default() lands in our scratch dir.
        cfg_baseline_scratch = tmp_path / "baseline"
        cfg_baseline_scratch.mkdir()
        cfg = _make_isolated_config(cfg_baseline_scratch)

        from climate_ref.datasets import get_dataset_adapter, ingest_datasets

        adapter = get_dataset_adapter("cmip6", config=cfg)

        # Baseline: whole-tree ingest.
        db = _fresh_db(cfg)
        baseline_stats = _run(
            "baseline-whole-tree",
            archive,
            lambda: ingest_datasets(adapter, archive, db, skip_invalid=True).log_summary("baseline:"),
        )
        db.close()

        # Streaming: chunked ingest with a brand-new DB to keep DB-row count comparable.
        cfg_stream_scratch = tmp_path / "streaming"
        cfg_stream_scratch.mkdir()
        cfg2 = _make_isolated_config(cfg_stream_scratch)
        db2 = _fresh_db(cfg2)
        streaming_stats = _run(
            "streaming",
            archive,
            lambda: ingest_datasets(
                adapter, archive, db2, skip_invalid=True, chunk_size=chunk_size
            ).log_summary("streaming:"),
        )
        db2.close()

        print("\n=== Results ===")
        print(f"files: {actual_files}, datasets: {actual_datasets}, chunk_size: {chunk_size}")
        for stats in (baseline_stats, streaming_stats):
            print(
                f"  [{stats['label']:<22}] "
                f"peak={stats['peak_mib']:>8} MiB  "
                f"final={stats['current_mib']:>8} MiB  "
                f"elapsed={stats['elapsed_s']:>6} s"
            )

        baseline_peak = float(baseline_stats["peak_mib"])
        streaming_peak = float(streaming_stats["peak_mib"])
        if streaming_peak > 0:
            ratio = baseline_peak / streaming_peak
            print(f"\nBaseline peak / streaming peak: {ratio:.2f}x")
        if streaming_peak >= baseline_peak:
            print(
                "  WARNING: streaming did not reduce peak memory. This is expected for "
                "very small archives where overhead dominates."
            )

        shutil.rmtree(archive, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-files",
        type=int,
        default=50_000,
        help="Total number of synthetic .nc files to create (default: 50000).",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=500,
        help="Number of distinct datasets to spread the files across (default: 500).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Chunk size for the streaming ingest run (default: 10000).",
    )
    args = parser.parse_args()
    benchmark(args.num_files, args.num_datasets, args.chunk_size)


if __name__ == "__main__":
    main()
