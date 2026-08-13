"""
Helpers for the text the REF shows a person.
"""


def pluralise(count: int, singular: str, plural: str | None = None) -> str:
    """
    Render a count and its noun, e.g. ``1 diagnostic`` or ``5 diagnostics``.

    Parameters
    ----------
    count
        How many there are.
    singular
        The noun in its singular form.
    plural
        The plural form, when appending an "s" does not give it.

    Returns
    -------
    :
        The count followed by the noun in the matching form.
    """
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"
