from ref_core.metrics import Configuration, MetricResult


class TestMetricResult:
    def test_build(self, tmp_path):
        config = Configuration(output_fragment=tmp_path)
        result = MetricResult.build(config, {"data": "value"})

        assert result.successful
        assert result.output_bundle.exists()
        assert result.output_bundle.is_file()
        with open(result.output_bundle) as f:
            assert f.read() == '{"data": "value"}'

        assert result.output_bundle.is_relative_to(tmp_path)
