"""
Registry of the currently active providers in the REF

This module provides a registry for the currently active providers.
Often, we can't directly import a provider and it's diagnostics
as each provider maintains its own virtual environment to avoid dependency conflicts.

For remote providers, a proxy is used to access the metadata associated with the diagnostics.
These diagnostics cannot be run locally, but can be executed using other executors.
"""

from attrs import field, frozen
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.providers import DiagnosticProvider, import_provider


def _register_provider(db: Database, provider: DiagnosticProvider) -> None:
    """
    Register a provider with the database

    This is temporary until we have a proper flow for registering providers

    Parameters
    ----------
    provider
        DiagnosticProvider instance
    """
    from climate_ref.models import Diagnostic, Provider

    provider_model, created = db.create_or_update(
        Provider,
        slug=provider.slug,
        values={
            "name": provider.name,
            # TODO: Handle if this changes
            "version": provider.version,
        },
    )
    if created:
        logger.info(f"Created provider {provider.slug}")
        db.session.flush()

    for diagnostic in provider.diagnostics():
        diagnostic_model, created = db.create_or_update(
            Diagnostic,
            slug=diagnostic.slug,
            provider_id=provider_model.id,
            values={
                "name": diagnostic.name,
            },
        )
        if created:
            db.session.flush()
            logger.info(f"Created diagnostic {diagnostic_model.full_slug()}")


@frozen
class ProviderRegistry:
    """
    Registry for the currently active providers

    In some cases we can't directly import a provider and it's diagnostics,
    in this case we need to proxy the diagnostics.
    """

    providers: list[DiagnosticProvider] = field(factory=list)

    def get(self, slug: str) -> DiagnosticProvider:
        """
        Retrieve a provider by name

        Parameters
        ----------
        slug
            Slug of the provider of interest

        Raises
        ------
        KeyError
            A provider with the matching slug has not been registered

        Returns
        -------
            The requested provider
        """
        for p in self.providers:
            if p.slug == slug:
                return p

        raise KeyError(f"No provider with slug matching: {slug}")

    def get_metric(self, provider_slug: str, diagnostic_slug: str) -> "Diagnostic":
        """
        Retrieve a diagnostic by name

        This is a convenience method to retrieve a diagnostic from a provider

        Parameters
        ----------
        provider_slug :
            Slug of the provider of interest
        diagnostic_slug
            Slug of the diagnostic of interest

        Raises
        ------
        KeyError
            If the provider/diagnostic with the given slugs is not found.

        Returns
        -------
            The requested diagnostic.
        """
        return self.get(provider_slug).get(diagnostic_slug)

    @staticmethod
    def build_from_config(config: Config, db: Database) -> "ProviderRegistry":
        """
        Create a ProviderRegistry instance using information from the database

        Parameters
        ----------
        config
            Configuration object
        db
            Database instance

        Returns
        -------
        :
            A new ProviderRegistry instance
        """
        providers = []
        for provider_info in config.diagnostic_providers:
            provider = import_provider(provider_info.provider)
            provider.configure(config)
            providers.append(provider)

        with db.session.begin():
            for provider in providers:
                _register_provider(db, provider)

        return ProviderRegistry(providers=providers)
