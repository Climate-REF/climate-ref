"""
Queue routing for Celery task submission

A deployment may wish to map diagnostics to different queues depending on their size.
The routing table maps diagnostics to queues,
so that each execution lands on a size-specific queue such as ``esmvaltool-large``.
Differently sized worker pools can then consume the queues independently.

The table is a TOML file whose path is given by the ``REF_CELERY_ROUTES`` environment variable.
When the variable is unset, no table is loaded and every execution uses the bare provider queue.

Example:

.. code-block:: toml

    default = "medium"

    [esmvaltool]
    default = "medium"
    rules = [
      { match = "portrait-*", size = "large" },
      { match = "sea-ice-basic", size = "small" },
    ]

    [ilamb]
    default = "small"

Rules are matched against the diagnostic slug in order, first match wins.
Patterns use :func:`fnmatch.fnmatchcase` semantics, so exact strings and glob wildcards both work.
A provider ``default`` applies when no rule matches.
The top-level ``default`` applies when the provider has no entry.
With no default and no match, the queue is the bare provider slug.
"""

import fnmatch
import os
import tomllib
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from attrs import field, frozen
from loguru import logger

ROUTES_ENV_VAR = "REF_CELERY_ROUTES"
"""Environment variable holding the path to the routing table file"""


class RoutingTableError(ValueError):
    """
    Raised when a routing table file is malformed

    A malformed table fails hard rather than falling back to default routing,
    because silent fallback would misplace large jobs onto small workers.
    """


def _require_string(value: Any, path: Path, entry: str) -> str:
    if not isinstance(value, str):
        raise RoutingTableError(f"Routing table {path}: {entry} must be a string, got {value!r}")
    return value


@frozen
class RoutingRule:
    """
    A single pattern to size-class rule
    """

    match: str
    """Pattern matched against the diagnostic slug, with ``fnmatch`` semantics"""

    size: str
    """Size class assigned when the pattern matches"""


@frozen
class ProviderRoutes:
    """
    The ordered rules and optional default size class for one provider
    """

    rules: tuple[RoutingRule, ...] = ()
    default: str | None = None

    def size_for(self, diagnostic_slug: str) -> str | None:
        """
        Resolve the size class for a diagnostic, or the provider default if no rule matches
        """
        for rule in self.rules:
            if fnmatch.fnmatchcase(diagnostic_slug, rule.match):
                return rule.size
        return self.default


@frozen
class RoutingTable:
    """
    Deployment-supplied mapping of diagnostics to size classes

    An empty table routes everything to the bare provider queue,
    which matches the behaviour when no table is configured.
    """

    providers: Mapping[str, ProviderRoutes] = field(factory=dict)
    default: str | None = None

    def size_for(self, provider_slug: str, diagnostic_slug: str) -> str | None:
        """
        Resolve the size class for an execution

        Returns
        -------
        :
            The size class, or ``None`` when neither a rule nor a default applies
        """
        provider = self.providers.get(provider_slug)
        if provider is None:
            return self.default
        return provider.size_for(diagnostic_slug)

    def queue_for(self, provider_slug: str, diagnostic_slug: str) -> str:
        """
        Compute the queue name for an execution

        Returns
        -------
        :
            ``{provider_slug}-{size}`` when a size class applies, otherwise the bare provider slug
        """
        size = self.size_for(provider_slug, diagnostic_slug)
        if size is None:
            return provider_slug
        return f"{provider_slug}-{size}"

    @classmethod
    def from_file(cls, path: Path, known_providers: Collection[str] | None = None) -> "RoutingTable":
        """
        Load and validate a routing table from a TOML file

        Parameters
        ----------
        path
            Path to the TOML file
        known_providers
            Slugs of the currently registered providers.
            An entry for a provider not in this collection logs a warning, not an error,
            because deployments may share one table across environments with different provider sets.
            ``None`` skips the check.

        Raises
        ------
        RoutingTableError
            The file is missing, is not valid TOML, or contains a malformed entry
        """
        try:
            raw = tomllib.loads(path.read_text())
        except OSError as exc:
            raise RoutingTableError(f"Routing table {path}: cannot read file ({exc})") from exc
        except tomllib.TOMLDecodeError as exc:
            raise RoutingTableError(f"Routing table {path}: invalid TOML ({exc})") from exc

        default: str | None = None
        providers: dict[str, ProviderRoutes] = {}

        for key, value in raw.items():
            if key == "default":
                default = _require_string(value, path, "top-level 'default'")
                continue
            if not isinstance(value, dict):
                raise RoutingTableError(
                    f"Routing table {path}: '{key}' must be a provider table, got {value!r}"
                )
            providers[key] = cls._parse_provider(value, path, key)

        if known_providers is not None:
            for slug in providers.keys() - set(known_providers):
                logger.warning(f"Routing table {path}: provider '{slug}' is not currently registered")

        return cls(providers=providers, default=default)

    @classmethod
    def _parse_provider(cls, raw: dict[str, Any], path: Path, provider: str) -> ProviderRoutes:
        unknown = raw.keys() - {"default", "rules"}
        if unknown:
            raise RoutingTableError(f"Routing table {path}: [{provider}] has unknown keys {sorted(unknown)}")

        default: str | None = None
        if "default" in raw:
            default = _require_string(raw["default"], path, f"[{provider}] 'default'")

        raw_rules = raw.get("rules", [])
        if not isinstance(raw_rules, list):
            raise RoutingTableError(f"Routing table {path}: [{provider}] 'rules' must be a list")

        rules = []
        for index, raw_rule in enumerate(raw_rules):
            entry = f"[{provider}] rule {index}"
            if not isinstance(raw_rule, dict) or raw_rule.keys() != {"match", "size"}:
                raise RoutingTableError(
                    f"Routing table {path}: {entry} must have exactly 'match' and 'size' keys, "
                    f"got {raw_rule!r}"
                )
            rules.append(
                RoutingRule(
                    match=_require_string(raw_rule["match"], path, f"{entry} 'match'"),
                    size=_require_string(raw_rule["size"], path, f"{entry} 'size'"),
                )
            )

        patterns = [rule.match for rule in rules]
        for pattern in sorted({p for p in patterns if patterns.count(p) > 1}):
            logger.warning(f"Routing table {path}: [{provider}] has duplicate pattern '{pattern}'")

        return ProviderRoutes(rules=tuple(rules), default=default)


def load_routing_table(known_providers: Collection[str] | None = None) -> RoutingTable:
    """
    Load the routing table named by ``REF_CELERY_ROUTES``, or an empty table if unset

    Parameters
    ----------
    known_providers
        Slugs of the currently registered providers, used to warn about stale entries.
        ``None`` skips the check.

    Raises
    ------
    RoutingTableError
        The variable is set but the file is missing or malformed
    """
    path = os.environ.get(ROUTES_ENV_VAR)
    if not path:
        return RoutingTable()
    return RoutingTable.from_file(Path(path), known_providers=known_providers)
