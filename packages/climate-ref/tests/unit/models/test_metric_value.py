import re

import pytest

from climate_ref.models import ScalarMetricValue, SeriesIndex, SeriesMetricValue


class TestScalarMetricValue:
    @pytest.mark.parametrize(
        "attributes",
        (
            {
                "key": "value",
            },
            None,
        ),
    )
    def test_build(self, db_seeded, attributes):
        item_orig = ScalarMetricValue.build(
            execution_id=1,
            value=1.0,
            attributes=attributes,
            dimensions={"source_id": "test"},
        )
        db_seeded.session.add(item_orig)
        db_seeded.session.commit()

        item = db_seeded.session.get(ScalarMetricValue, item_orig.id)
        assert item.attributes == attributes
        assert item.value == 1.0

        assert item.dimensions == {"source_id": "test"}

    def test_invalid_dimension(self, db_seeded):
        exp_msg = "Unknown dimension column 'not_a_dimension'"
        with pytest.raises(KeyError, match=exp_msg):
            ScalarMetricValue.build(
                execution_id=1,
                value=1.0,
                attributes=None,
                dimensions={"not_a_dimension": "test"},
            )

    def test_register_dimensions(self, cmip7_aft_cv):
        metric_value_kwargs = dict(
            execution_id=1, value=1.0, attributes=None, dimensions={"source_id": "test"}
        )
        ScalarMetricValue._reset_cv_dimensions()
        assert ScalarMetricValue._cv_dimensions == []

        with pytest.raises(KeyError):
            ScalarMetricValue.build(**metric_value_kwargs)

        ScalarMetricValue.register_cv_dimensions(cmip7_aft_cv)
        assert ScalarMetricValue._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]

        # Should work now that the dimension has been registered
        item = ScalarMetricValue.build(**metric_value_kwargs)

        for k in ScalarMetricValue._cv_dimensions:
            assert hasattr(item, k)

    def test_register_dimensions_multiple_times(self, cmip7_aft_cv):
        ScalarMetricValue._reset_cv_dimensions()
        assert ScalarMetricValue._cv_dimensions == []

        ScalarMetricValue.register_cv_dimensions(cmip7_aft_cv)
        assert ScalarMetricValue._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]

        ScalarMetricValue.register_cv_dimensions(cmip7_aft_cv)
        assert ScalarMetricValue._cv_dimensions == [d.name for d in cmip7_aft_cv.dimensions]


class TestSeriesMetricValue:
    def test_build_with_matching_lengths(self, db_seeded):
        """Test that building a SeriesMetricValue with matching lengths works"""
        axis = SeriesIndex.get_or_create(db_seeded.session, "time", [0, 1, 2])
        item = SeriesMetricValue.build(
            execution_id=1,
            values=[1.0, 2.0, 3.0],
            index_axis=axis,
            dimensions={"source_id": "test"},
            attributes={"attr": "value"},
        )
        db_seeded.session.add(item)
        db_seeded.session.commit()

        item = db_seeded.session.get(SeriesMetricValue, item.id)
        assert item.values == [1.0, 2.0, 3.0]
        assert item.index == [0, 1, 2]
        assert item.index_name == "time"
        assert item.dimensions == {"source_id": "test"}
        assert item.attributes == {"attr": "value"}

    def test_build_with_mismatched_lengths(self, db_seeded):
        """Test that building a SeriesMetricValue with mismatched lengths raises an error"""
        axis = SeriesIndex.get_or_create(db_seeded.session, "time", [0, 1])
        with pytest.raises(ValueError, match=re.escape(r"Index length (2) must match values length (3)")):
            SeriesMetricValue.build(
                execution_id=1,
                values=[1.0, 2.0, 3.0],
                index_axis=axis,
                dimensions={"source_id": "test"},
                attributes=None,
            )

    def test_update_with_mismatched_lengths(self, db_seeded):
        """Test that updating a SeriesMetricValue with mismatched lengths raises an error"""
        axis = SeriesIndex.get_or_create(db_seeded.session, "time", [0, 1])
        item = SeriesMetricValue.build(
            execution_id=1,
            values=[1.0, 2.0],
            index_axis=axis,
            dimensions={"source_id": "test"},
            attributes=None,
        )
        db_seeded.session.add(item)
        db_seeded.session.commit()

        item = db_seeded.session.get(SeriesMetricValue, item.id)
        item.values = [1.0, 2.0, 3.0]  # Make lengths mismatch

        with pytest.raises(ValueError, match=re.escape("Index length (2) must match values length (3)")):
            db_seeded.session.commit()
