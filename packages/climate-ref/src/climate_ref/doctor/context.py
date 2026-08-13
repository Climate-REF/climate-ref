"""
The deployment a check runs against.
"""

from collections.abc import Iterable

import pandas as pd
from attrs import define, field

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.datasets import get_dataset_adapter
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.source_types import SourceDatasetType

EMPTY_CATALOG = pd.DataFrame()
"""Stands in for a source type with nothing ingested."""


@define
class DoctorContext:
    """
    The deployment being checked.

    Providers and catalogs are loaded lazily so a check that does not need them does not
    pay for them, and so a failure to load one provider does not stop the other checks.
    """

    config: Config | None
    database: Database | None
    _providers: list[DiagnosticProvider] | None = field(default=None, alias="_providers")
    _catalogs: dict[SourceDatasetType, pd.DataFrame] = field(factory=dict, alias="_catalogs")

    @classmethod
    def from_catalogs(
        cls,
        catalogs: dict[SourceDatasetType, pd.DataFrame],
        providers: Iterable[DiagnosticProvider],
    ) -> "DoctorContext":
        """
        Build a context from catalogs already in hand, with no database behind it.

        Source types absent from ``catalogs`` are treated as having nothing ingested, so
        every check can run without reaching for a database that is not there.

        Parameters
        ----------
        catalogs
            The catalogs to check, keyed by source type.
        providers
            The providers to treat as enabled.

        Returns
        -------
        :
            A context backed by nothing but the supplied catalogs and providers.
        """
        complete = {
            source_type: catalogs.get(source_type, EMPTY_CATALOG) for source_type in SourceDatasetType
        }
        return cls(config=None, database=None, _providers=list(providers), _catalogs=complete)

    @property
    def providers(self) -> list[DiagnosticProvider]:
        """The diagnostic providers this deployment has enabled."""
        if self._providers is None:
            from climate_ref.provider_registry import ProviderRegistry  # noqa: PLC0415

            if self.config is None or self.database is None:
                raise ValueError("This context has no configuration to load providers from")
            # Doctor only reads a provider's metadata, so it takes neither of the side effects
            # `build_from_config` offers: `configure` bootstraps conda, and `register` writes to
            # the database being inspected.
            registry = ProviderRegistry.build_from_config(
                self.config, self.database, configure=False, register=False
            )
            self._providers = list(registry.providers)
        return self._providers

    def catalog(self, source_type: SourceDatasetType) -> pd.DataFrame:
        """
        Load the ingested catalog for a source type, one row per file.

        Parameters
        ----------
        source_type
            The source type to load.

        Returns
        -------
        :
            The catalog, or an empty frame when nothing of that type has been ingested.
        """
        if source_type not in self._catalogs:
            if self.database is None:
                raise ValueError("This context has no database to load a catalog from")
            adapter = get_dataset_adapter(source_type.value)
            self._catalogs[source_type] = adapter.load_catalog(self.database)
        return self._catalogs[source_type]
