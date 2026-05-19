"""
Helpers for allocating non-colliding output fragment paths.
"""

import datetime
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from climate_ref.models.execution import Execution

_TOKEN_RE = re.compile(r"[^A-Za-z0-9_-]+")
_DEFAULT_TOKEN_LIMIT = 64
_GROUP_SHORT_MAX = 96

PLACEHOLDER_FRAGMENT = "_pending"
"""Output-fragment placeholder used until ``Execution.id`` is known."""

_DEFAULT_DIAGNOSTIC_VERSION = 1
"""
Default integer version baked into the output-fragment hash and suffix.

Diagnostics do not currently expose a version attribute; this constant is the
value used for every diagnostic until that is introduced.
"""


def allocate_output_fragment(base_fragment: str, results_dir: Path) -> str:
    """
    Return a unique output fragment by appending a UTC timestamp.

    The returned fragment is ``{base_fragment}_{YYYYMMDDTHHMMSSffffff}``, which is
    practically unique without needing any database or filesystem lookups.
    Microsecond resolution avoids collisions from rapid successive calls.

    Parameters
    ----------
    base_fragment
        The natural fragment, e.g. ``provider/diagnostic/dataset_hash``
    results_dir
        The results root directory. Used to verify the allocated fragment
        does not already exist on disk.

    Returns
    -------
    :
        A new fragment with a timestamp suffix

    Raises
    ------
    FileExistsError
        If the computed output directory already exists under *results_dir*
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    fragment = f"{base_fragment}_{now.strftime('%Y%m%dT%H%M%S%f')}"

    target = results_dir / fragment
    if target.exists():
        raise FileExistsError(
            f"Output directory already exists: {target}. Cannot allocate fragment '{fragment}'."
        )

    return fragment


_UNSAFE_SEGMENT_RE = re.compile(r"[/\\\x00]")


def _validate_path_segment(value: str, *, label: str) -> str:
    r"""
    Ensure *value* is a safe single path segment.

    Rejects empty strings, path separators (``/``, ``\``), NUL bytes, and dot-segments
    (``.``, ``..``) so that downstream ``Path`` joins cannot escape the intended
    results/scratch base directories.
    """
    if not value or _UNSAFE_SEGMENT_RE.search(value) or value in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r} is not a safe single path segment")
    return value


def _sanitize_token(value: str) -> str:
    """
    Sanitize a single selector value to ``[A-Za-z0-9_-]+``.

    Non-conforming characters are collapsed to a single underscore.
    Leading/trailing underscores are stripped.
    """
    cleaned = _TOKEN_RE.sub("_", value).strip("_")
    return cleaned


def _truncate_at_boundary(text: str, limit: int) -> str:
    """
    Truncate *text* at *limit* characters, preferring an underscore boundary.

    If *text* is at or below *limit*, return it unchanged.
    Otherwise truncate to the rightmost ``_`` at or before *limit*; fall back to a
    hard cut if no boundary exists.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    boundary = head.rfind("_")
    if boundary > 0:
        return head[:boundary]
    return head


def compute_group_short(
    selectors: Mapping[str, Iterable[tuple[str, str]]],
    group_id: int,
    diagnostic_version: int,
    *,
    token_limit: int = _DEFAULT_TOKEN_LIMIT,
) -> str:
    """
    Compute a short, deterministic, human-readable path segment for an execution group.

    Human-readable hint for operators browsing the filesystem.
    Not unique -- ``execution_id`` is the uniqueness guarantee.

    Selector values across all source types are sorted (first by source-type key,
    then by facet key),
    sanitized to ``[A-Za-z0-9_-]``,
    joined by ``_``,
    and truncated to *token_limit* characters at an underscore boundary.
    A suffix ``_g{group_id}_v{diagnostic_version}_{digest}`` is appended,
    where ``digest`` is an 8-character BLAKE2s hash of the canonical
    ``group_id|diagnostic_version|sorted_selectors`` representation.

    The returned string is ASCII, capped at roughly 96 characters,
    and deterministic for fixed inputs.

    Parameters
    ----------
    selectors
        Mapping from source-type key (e.g. ``"cmip6"``) to an iterable of
        ``(facet_key, facet_value)`` tuples.
    group_id
        The ``ExecutionGroup.id`` this fragment belongs to.
    diagnostic_version
        The integer ``Diagnostic.version`` at solve time.
    token_limit
        Maximum length of the sanitized selector portion before the suffix.

    Returns
    -------
    :
        A short, sanitized, deterministic identifier suitable for use as a
        filesystem path segment.
    """
    # Build a canonical representation: sort source types, then facet pairs.
    canonical_pairs: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for source_type in sorted(selectors.keys()):
        pairs = tuple(sorted((str(k), str(v)) for k, v in selectors[source_type]))
        canonical_pairs.append((source_type, pairs))

    # Build the human-readable token portion from the values only.
    raw_tokens: list[str] = []
    for _, pairs in canonical_pairs:
        for _, value in pairs:
            token = _sanitize_token(value)
            if token:
                raw_tokens.append(token)

    token_part = _truncate_at_boundary("_".join(raw_tokens), token_limit)

    # Stable hash input: group_id, version, and the canonical selector pairs.
    # BLAKE2s with a 4-byte digest emits an 8-char hex string without truncation;
    # it is non-cryptographic for our purposes but avoids the deprecated-hash linter.
    hash_payload = repr((group_id, diagnostic_version, canonical_pairs)).encode("utf-8")
    digest = hashlib.blake2s(hash_payload, digest_size=4).hexdigest()

    suffix = f"_g{group_id}_v{diagnostic_version}_{digest}"

    if token_part:
        result = f"{token_part}{suffix}"
    else:
        # Strip leading underscore so we don't start with one.
        result = suffix.lstrip("_")

    if len(result) > _GROUP_SHORT_MAX:
        # Trim the token portion further so the suffix is preserved.
        overflow = len(result) - _GROUP_SHORT_MAX
        trimmed_token = _truncate_at_boundary(token_part, max(0, len(token_part) - overflow))
        result = f"{trimmed_token}{suffix}" if trimmed_token else suffix.lstrip("_")

    # Hard cap: boundary-aware trimming above may still leave overflow when no
    # underscore boundary exists close enough to the limit.
    if len(result) > _GROUP_SHORT_MAX:
        result = result[:_GROUP_SHORT_MAX]

    return result


def assign_execution_fragment(  # noqa: PLR0913
    session: "Session",
    execution: "Execution",
    *,
    provider_slug: str,
    diagnostic_slug: str,
    selectors: Mapping[str, Iterable[tuple[str, str]]],
    group_id: int,
    diagnostic_version: int = _DEFAULT_DIAGNOSTIC_VERSION,
) -> str:
    """Flush *execution* to materialise its id, then assign the canonical output fragment.

    Returns the assigned fragment string.
    """
    _validate_path_segment(provider_slug, label="provider slug")
    _validate_path_segment(diagnostic_slug, label="diagnostic slug")
    session.flush()
    group_short = compute_group_short(selectors, group_id=group_id, diagnostic_version=diagnostic_version)
    fragment = str(Path(provider_slug) / diagnostic_slug / group_short / str(execution.id))
    execution.output_fragment = fragment
    session.flush()
    return fragment
