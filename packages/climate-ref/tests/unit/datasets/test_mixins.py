"""Unit tests for the finalisation helpers that are not adapter specific."""

import pandas as pd

from climate_ref.datasets.mixins import _chunk_by_dataset


def _frame(slugs: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {"instance_id": slugs, "path": [f"/{i}.nc" for i in range(len(slugs))]},
        index=pd.RangeIndex(len(slugs)),
    )


class TestChunkByDataset:
    def test_keeps_a_dataset_in_one_chunk(self):
        """A chunk may exceed the target rather than split a dataset."""
        chunks = list(_chunk_by_dataset(_frame(["a", "a", "a", "b"]), "instance_id", 1))

        assert [len(c) for c in chunks] == [3, 1]

    def test_fills_chunks_to_the_target(self):
        chunks = list(_chunk_by_dataset(_frame(["a", "b", "c", "d"]), "instance_id", 2))

        assert [len(c) for c in chunks] == [2, 2]

    def test_covers_every_row(self):
        """The caller rebuilds the catalog from the chunks, so none may be dropped."""
        frame = _frame(["a", "b", "a"])

        labels = [label for chunk in _chunk_by_dataset(frame, "instance_id", 2) for label in chunk]

        assert sorted(labels) == list(frame.index)

    def test_covers_rows_with_a_missing_slug(self):
        frame = _frame(["a", None, "b"])

        labels = [label for chunk in _chunk_by_dataset(frame, "instance_id", 1) for label in chunk]

        assert sorted(labels) == list(frame.index)
