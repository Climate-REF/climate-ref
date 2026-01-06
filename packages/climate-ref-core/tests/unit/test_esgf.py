"""
These tests verify the ESGF request classes and protocol without
requiring actual ESGF connections.
"""

from unittest.mock import MagicMock

from climate_ref_core.esgf import CMIP6Request, ESGFFetcher, ESGFRequest, Obs4MIPsRequest


class TestCMIP6Request:
    """Tests for CMIP6Request class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        assert request.slug == "test-request"
        assert request.facets == {"source_id": "ACCESS-ESM1-5", "variable_id": "tas"}
        assert request.remove_ensembles is False
        assert request.time_span is None
        assert request.source_type == "CMIP6"

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            time_span=("2000-01", "2010-12"),
        )
        assert request.time_span == ("2000-01", "2010-12")

    def test_init_with_remove_ensembles(self):
        """Test initialization with remove_ensembles flag."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
        )
        assert request.remove_ensembles is True

    def test_repr(self):
        """Test string representation."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        repr_str = repr(request)
        assert "CMIP6Request" in repr_str
        assert "test-request" in repr_str
        assert "ACCESS-ESM1-5" in repr_str

    def test_protocol_compliance(self):
        """Test that CMIP6Request satisfies ESGFRequest protocol."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        assert isinstance(request, ESGFRequest)

    def test_cmip6_path_items_valid(self):
        """Test that all path items are in available facets."""
        for item in CMIP6Request.cmip6_path_items:
            assert item in CMIP6Request.available_facets

    def test_cmip6_filename_paths_valid(self):
        """Test that all filename path items are in available facets."""
        for item in CMIP6Request.cmip6_filename_paths:
            assert item in CMIP6Request.available_facets


class TestObs4MIPsRequest:
    """Tests for Obs4MIPsRequest class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG", "variable_id": "pr"},
        )
        assert request.slug == "test-obs"
        assert request.facets == {"source_id": "GPCP-SG", "variable_id": "pr"}
        assert request.remove_ensembles is False
        assert request.time_span is None
        assert request.source_type == "obs4MIPs"

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
            time_span=("1990-01", "2020-12"),
        )
        assert request.time_span == ("1990-01", "2020-12")

    def test_repr(self):
        """Test string representation."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
        )
        repr_str = repr(request)
        assert "Obs4MIPsRequest" in repr_str
        assert "test-obs" in repr_str
        assert "GPCP-SG" in repr_str

    def test_protocol_compliance(self):
        """Test that Obs4MIPsRequest satisfies ESGFRequest protocol."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
        )
        assert isinstance(request, ESGFRequest)

    def test_obs4mips_path_items_valid(self):
        """Test that all path items are in available facets."""
        for item in Obs4MIPsRequest.obs4mips_path_items:
            assert item in Obs4MIPsRequest.avail_facets

    def test_obs4mips_filename_paths_valid(self):
        """Test that all filename path items are in available facets."""
        for item in Obs4MIPsRequest.obs4mips_filename_paths:
            assert item in Obs4MIPsRequest.avail_facets


class TestESGFRequestProtocol:
    """Tests for the ESGFRequest protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol can be used for isinstance checks."""
        request = CMIP6Request(slug="test", facets={})
        assert isinstance(request, ESGFRequest)

    def test_non_compliant_object(self):
        """Test that non-compliant objects don't satisfy the protocol."""

        class NotARequest:
            pass

        obj = NotARequest()
        assert not isinstance(obj, ESGFRequest)

    def test_required_attributes(self):
        """Test that protocol requires expected attributes."""
        request = CMIP6Request(slug="test", facets={})

        # These should all exist
        assert hasattr(request, "slug")
        assert hasattr(request, "source_type")
        assert hasattr(request, "time_span")
        assert hasattr(request, "fetch_datasets")
        assert hasattr(request, "generate_output_path")


class TestESGFFetcher:
    """Tests for ESGFFetcher class."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        output_dir = tmp_path / "output"
        fetcher = ESGFFetcher(output_dir=output_dir)
        assert fetcher.output_dir == output_dir
        assert fetcher.cache_dir is None
        assert output_dir.exists()

    def test_init_with_cache(self, tmp_path):
        """Test initialization with cache directory."""
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"
        fetcher = ESGFFetcher(output_dir=output_dir, cache_dir=cache_dir)
        assert fetcher.output_dir == output_dir
        assert fetcher.cache_dir == cache_dir

    def test_init_creates_output_dir(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "output"
        assert not output_dir.exists()
        ESGFFetcher(output_dir=output_dir)
        assert output_dir.exists()

    def test_list_requests_for_diagnostic_no_spec(self, tmp_path):
        """Test listing requests when diagnostic has no test_data_spec."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []

    def test_list_requests_for_diagnostic_with_requests(self, tmp_path):
        """Test listing requests when diagnostic has test cases with requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        # Create mock requests
        mock_request1 = MagicMock()
        mock_request2 = MagicMock()

        # Create mock test cases
        mock_test_case1 = MagicMock()
        mock_test_case1.name = "default"
        mock_test_case1.requests = (mock_request1,)

        mock_test_case2 = MagicMock()
        mock_test_case2.name = "edge-case"
        mock_test_case2.requests = (mock_request2,)

        # Create mock test_data_spec
        mock_spec = MagicMock()
        mock_spec.test_cases = (mock_test_case1, mock_test_case2)

        # Create mock diagnostic
        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert len(result) == 2
        assert result[0] == ("default", mock_request1)
        assert result[1] == ("edge-case", mock_request2)

    def test_list_requests_for_diagnostic_no_requests(self, tmp_path):
        """Test listing requests when test case has no requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_test_case = MagicMock()
        mock_test_case.name = "default"
        mock_test_case.requests = None

        mock_spec = MagicMock()
        mock_spec.test_cases = (mock_test_case,)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []
