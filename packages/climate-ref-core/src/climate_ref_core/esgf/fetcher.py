"""
ESGF dataset fetcher for downloading test data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from loguru import logger

from climate_ref_core.esgf.base import ESGFRequest

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import AbstractDiagnostic as Diagnostic


class ESGFFetcher:
    """
    Fetches and manages ESGF datasets for testing.

    This class handles downloading datasets from ESGF based on request
    specifications and organizing them in a local directory structure.
    """

    def __init__(self, output_dir: Path, cache_dir: Path | None = None):
        """
        Initialize the ESGF fetcher.

        Parameters
        ----------
        output_dir
            Directory where datasets will be stored
        cache_dir
            Optional cache directory for raw downloads.
            If None, uses the default pooch cache.
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_request(
        self,
        request: ESGFRequest,
        symlink: bool = False,
    ) -> list[Path]:
        """
        Fetch datasets for a single ESGF request.

        Parameters
        ----------
        request
            The ESGF request specifying what to fetch
        symlink
            If True, create symlinks to cached files instead of copying

        Returns
        -------
        list[Path]
            List of paths to the fetched dataset files
        """
        logger.info(f"Fetching datasets for request: {request.slug}")

        # Search ESGF for matching datasets
        datasets_df = request.fetch_datasets()

        if datasets_df.empty:
            logger.warning(f"No datasets found for request: {request.slug}")
            return []

        logger.info(f"Found {len(datasets_df)} datasets for request: {request.slug}")

        fetched_paths: list[Path] = []

        for _, row in datasets_df.iterrows():
            files = row["files"]
            if not files:
                logger.warning(f"No files for dataset: {row.get('key', 'unknown')}")
                continue

            for _file_path in files:
                file_path = Path(_file_path)

                if not file_path.exists():
                    logger.warning(f"File not found (may need to download from ESGF): {file_path}")
                    continue

                # Load dataset to generate output path
                try:
                    ds = xr.open_dataset(file_path)
                    output_path = request.generate_output_path(row, ds, file_path)
                    ds.close()
                except Exception as e:
                    logger.error(f"Error loading dataset {file_path}: {e}")
                    continue

                # Create output directory and copy/symlink file
                full_output_path = self.output_dir / output_path
                full_output_path.parent.mkdir(parents=True, exist_ok=True)

                if full_output_path.exists():
                    logger.debug(f"File already exists: {full_output_path}")
                    fetched_paths.append(full_output_path)
                    continue

                if symlink:
                    full_output_path.symlink_to(file_path)
                    logger.debug(f"Symlinked {file_path} -> {full_output_path}")
                else:
                    shutil.copy2(file_path, full_output_path)
                    logger.debug(f"Copied {file_path} -> {full_output_path}")

                fetched_paths.append(full_output_path)

        logger.info(f"Fetched {len(fetched_paths)} files for request: {request.slug}")
        return fetched_paths

    def fetch_for_diagnostic(
        self,
        diagnostic: Diagnostic,
        test_case_name: str = "default",
        symlink: bool = False,
    ) -> dict[str, list[Path]]:
        """
        Fetch all test data for a diagnostic's test case.

        Parameters
        ----------
        diagnostic
            The diagnostic to fetch data for
        test_case_name
            Name of the test case to fetch data for
        symlink
            If True, create symlinks instead of copying

        Returns
        -------
        dict[str, list[Path]]
            Mapping from request ID to list of fetched file paths
        """
        if diagnostic.test_data_spec is None:
            logger.warning(f"Diagnostic {diagnostic.slug} has no test_data_spec")
            return {}

        test_case = diagnostic.test_data_spec.get_case(test_case_name)

        if test_case.requests is None:
            logger.warning(f"Test case {test_case_name} has no ESGF requests")
            return {}

        results: dict[str, list[Path]] = {}

        for request in test_case.requests:
            paths = self.fetch_request(request, symlink=symlink)
            results[request.slug] = paths

        return results

    def list_requests_for_diagnostic(self, diagnostic: Diagnostic) -> list[tuple[str, ESGFRequest]]:
        """
        List all ESGF requests for a diagnostic across all test cases.

        Parameters
        ----------
        diagnostic
            The diagnostic to list requests for

        Returns
        -------
        list[tuple[str, ESGFRequest]]
            List of (test_case_name, request) tuples
        """
        if diagnostic.test_data_spec is None:
            return []

        results: list[tuple[str, ESGFRequest]] = []

        for test_case in diagnostic.test_data_spec.test_cases:
            if test_case.requests:
                for request in test_case.requests:
                    results.append((test_case.name, request))

        return results
