"""A module for defining and fetching ILAMB datasets"""

from pathlib import Path

import pandas as pd
import pooch

from climate_ref_core.datasets import DatasetCollection


def registry_to_collection(registry: pooch.Pooch) -> DatasetCollection:
    """
    Convert an ILAMB registry to a DatasetCollection for use in REF.

    Parameters
    ----------
    registry : pooch.Pooch
        The pooch registry.

    Returns
    -------
    DatasetCollection
        The converted collection.
    """
    df = pd.DataFrame(
        [
            {
                "key": key,
                "path": registry.abspath / Path(key),  # type: ignore
            }
            for key in registry.registry.keys()
        ]
    )
    collection = DatasetCollection(df, "key")
    return collection
