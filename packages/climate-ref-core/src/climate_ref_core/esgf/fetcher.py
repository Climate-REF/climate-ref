"""
ESGF dataset fetcher for downloading test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from climate_ref_core.esgf.base import ESGFRequest

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import AbstractDiagnostic as Diagnostic


class ESGFFetcher:
    """
    Fetches datasets from ESGF and returns metadata with file paths.

    Uses intake-esgf to search and download datasets.
    Files that cannot be found locally are stored in intake-esgf's cache directory.
    """

    def fetch_request(self, request: ESGFRequest) -> pd.DataFrame:
        """
        Fetch datasets for a single ESGF request.

        Parameters
        ----------
        request
            The ESGF request specifying what to fetch

        Returns
        -------
        pd.DataFrame
            DataFrame containing dataset metadata and file paths.
            Each row represents one file, with a 'path' column pointing
            to the file (either in intake-esgf's cache or one of the root data locations).

            This format is not identical to the DataCatalog, but it is broadly compatible.
        """
        logger.info(f"Fetching datasets for request: {request.slug}")

        # Search ESGF for matching datasets
        datasets_df = request.fetch_datasets()

        if datasets_df.empty:
            logger.warning(f"No datasets found for request: {request.slug}")
            return pd.DataFrame()

        logger.info(f"Found {len(datasets_df)} datasets for request: {request.slug}")

        # Expand files column - each file becomes a row with a 'path' column
        rows = []
        for _, row in datasets_df.iterrows():
            files = row.get("files", [])
            if not files:
                logger.warning(f"No files for dataset: {row.get('key', 'unknown')}")
                continue

            for file_path in files:
                if not Path(file_path).exists():
                    logger.warning(f"File not found (may need to download from ESGF): {file_path}")
                    continue

                row_copy = row.to_dict()
                row_copy["path"] = str(file_path)
                rows.append(row_copy)

        if not rows:
            logger.warning(f"No files found for request: {request.slug}")
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        result["source_type"] = request.source_type

        logger.info(f"Fetched {len(result)} files for request: {request.slug}")
        return result

    def fetch_for_test_case(
        self,
        requests: tuple[ESGFRequest, ...] | None,
    ) -> pd.DataFrame:
        """
        Fetch all data for a test case's requests.

        Parameters
        ----------
        requests
            The ESGF requests from the test case

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all datasets, grouped by source_type
        """
        if not requests:
            return pd.DataFrame()

        dfs = []
        for request in requests:
            df = self.fetch_request(request)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

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
