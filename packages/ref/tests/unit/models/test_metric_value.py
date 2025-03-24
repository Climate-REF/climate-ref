import pytest

from cmip_ref.models import MetricValue


class TestMetricValue:
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
        item_orig = MetricValue.build(
            metric_execution_result_id=1, value=1.0, attributes=attributes, dimensions={"model": "test"}
        )
        db_seeded.session.add(item_orig)
        db_seeded.session.commit()

        item = db_seeded.session.query(MetricValue).get(item_orig.id)
        assert item.attributes == attributes
        assert item.value == 1.0

        assert item.dimensions == {"model": "test"}

    def test_invalid_dimension(self, db_seeded):
        exp_msg = "Unknown dimension column 'not_a_dimension'"
        with pytest.raises(KeyError, match=exp_msg):
            MetricValue.build(
                metric_execution_result_id=1,
                value=1.0,
                attributes=None,
                dimensions={"not_a_dimension": "test"},
            )

    def test_register_dimensions(self, ar7_ft_cv):
        metric_value_kwargs = dict(
            metric_execution_result_id=1, value=1.0, attributes=None, dimensions={"model": "test"}
        )
        MetricValue._reset_cv_dimensions()
        assert MetricValue._cv_dimensions == []

        with pytest.raises(KeyError):
            MetricValue.build(**metric_value_kwargs)

        MetricValue.register_cv_dimensions(ar7_ft_cv)
        assert MetricValue._cv_dimensions == [d.name for d in ar7_ft_cv.dimensions]

        # Should work now that the dimension has been registered
        item = MetricValue.build(**metric_value_kwargs)

        for k in MetricValue._cv_dimensions:
            assert hasattr(item, k)

    def test_register_dimensions_multiple_times(self, ar7_ft_cv):
        MetricValue._reset_cv_dimensions()
        assert MetricValue._cv_dimensions == []

        MetricValue.register_cv_dimensions(ar7_ft_cv)
        assert MetricValue._cv_dimensions == [d.name for d in ar7_ft_cv.dimensions]

        MetricValue.register_cv_dimensions(ar7_ft_cv)
        assert MetricValue._cv_dimensions == [d.name for d in ar7_ft_cv.dimensions]
