"""Tests for downloading published CMIP7 data with `CMIP7Request`."""

from unittest.mock import MagicMock, patch

import intake_esgf
import pandas as pd
import pytest

from climate_ref_core.esgf import CMIP7Request, ESGFRequest
from climate_ref_core.esgf.cmip7 import STAC_INDICES
from climate_ref_core.exceptions import DatasetResolutionError

BASE_ID = "CMIP7.CMIP.UKNCSP.UKESM1-3-LL.historical.r1i1p1f2.glb.mon.tas.tavg-h2m-hxy-u.g110"
INSTANCE_ID = f"{BASE_ID}.v20261004"
OTHER_ID = "CMIP7.CMIP.UKNCSP.UKESM1-3-LL.piControl.r1i1p1f2.glb.mon.tas.tavg-h2m-hxy-u.g110.v20261001"


def _mock_catalog(*esgf_ids):
    cat = MagicMock()
    cat.df = pd.DataFrame({"id": [[f"MIP-DRS7.{esgf_id}|discovery.east.esgf.io"] for esgf_id in esgf_ids]})
    cat.df["key"] = [f"ds{i}" for i in range(len(esgf_ids))]
    cat.to_path_dict.side_effect = lambda **_: {key: [f"/{key}.nc"] for key in cat.df["key"]}
    return cat


@pytest.fixture
def stac_enabled(monkeypatch):
    monkeypatch.setitem(intake_esgf.conf, "stac_indices", dict.fromkeys(STAC_INDICES, True))


class TestCMIP7Request:
    def test_satisfies_protocol(self):
        request = CMIP7Request(slug="test", facets={"variable_id": "tas"})
        assert isinstance(request, ESGFRequest)
        assert request.source_type == "CMIP7"
        assert request.project == "CMIP7"

    @pytest.mark.usefixtures("stac_enabled")
    def test_fetch(self):
        request = CMIP7Request(slug="test", facets={"variable_id": "tas"})
        cat = _mock_catalog(INSTANCE_ID)

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=cat):
            result = request.fetch_datasets()

        cat.search.assert_called_once_with(project="CMIP7", variable_id="tas")
        assert result["files"].tolist() == [["/ds0.nc"]]

    @pytest.mark.usefixtures("stac_enabled")
    def test_maps_branding_suffix_to_esgf_name(self):
        request = CMIP7Request(slug="test", facets={"branding_suffix": "tavg-h2m-hxy-u"})
        cat = _mock_catalog(INSTANCE_ID)

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=cat):
            request.fetch_datasets()

        cat.search.assert_called_once_with(project="CMIP7", variable_branding_suffix="tavg-h2m-hxy-u")

    @pytest.mark.usefixtures("stac_enabled")
    def test_pinned_fetch_searches_by_base_id_and_version(self):
        request = CMIP7Request(slug="test", facets={"variable_id": "tas"}, remove_ensembles=True)
        # A dataset is recorded once per file it holds
        pinned = request.pin_to_datasets(
            [{"instance_id": INSTANCE_ID}, {"instance_id": INSTANCE_ID}, {"instance_id": OTHER_ID}]
        )
        cat = _mock_catalog(INSTANCE_ID, OTHER_ID)

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=cat):
            result = pinned.fetch_datasets()

        cat.search.assert_called_once_with(
            project="CMIP7",
            base_id=[f"MIP-DRS7.{BASE_ID}", f"MIP-DRS7.{OTHER_ID.rsplit('.', 1)[0]}"],
            version=["20261001", "20261004"],
            latest=[True, False],
        )
        cat.remove_ensembles.assert_not_called()
        assert sorted(result["key"]) == ["ds0", "ds1"]

    @pytest.mark.usefixtures("stac_enabled")
    def test_pinned_fetch_raises_when_a_pin_no_longer_resolves(self):
        pinned = CMIP7Request(slug="test", facets={}).pin_to_datasets(
            [{"instance_id": INSTANCE_ID}, {"instance_id": OTHER_ID}]
        )
        # Searching every base_id and version together can match a dataset that was not pinned
        unpinned = f"{OTHER_ID.rsplit('.', 1)[0]}.v20261004"
        cat = _mock_catalog(INSTANCE_ID, unpinned)

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=cat):
            with pytest.raises(DatasetResolutionError, match="no longer resolves 1 pinned"):
                pinned.fetch_datasets()

    @pytest.mark.usefixtures("stac_enabled")
    def test_pinned_fetch_raises_when_a_pin_is_not_retrieved(self):
        pinned = CMIP7Request(slug="test", facets={}).pin_to_datasets(
            [{"instance_id": INSTANCE_ID}, {"instance_id": OTHER_ID}]
        )
        cat = _mock_catalog(INSTANCE_ID, OTHER_ID)
        # intake-esgf leaves out a dataset it failed to download when break_on_error is off
        cat.to_path_dict.side_effect = lambda **_: {"ds0": ["/ds0.nc"]}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=cat):
            with pytest.raises(DatasetResolutionError, match=r"Could not retrieve pinned datasets.*ds1"):
                pinned.fetch_datasets()

    def test_warns_when_no_cmip7_index_is_enabled(self, monkeypatch):
        monkeypatch.setitem(intake_esgf.conf, "stac_indices", {"api.stac.ceda.ac.uk": False})
        request = CMIP7Request(slug="test", facets={"variable_id": "tas"})
        messages = []

        with (
            patch("climate_ref_core.esgf.cmip7.logger.warning", side_effect=messages.append),
            patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=_mock_catalog(INSTANCE_ID)),
        ):
            request.fetch_datasets()

        assert len(messages) == 1
        assert all(index in messages[0] for index in STAC_INDICES)
        # The user's configuration is left alone
        assert intake_esgf.conf["stac_indices"] == {"api.stac.ceda.ac.uk": False}
