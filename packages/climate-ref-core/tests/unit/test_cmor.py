import pytest

from climate_ref_core.cmor import frequency_from_mip_table


class TestFrequencyFromMipTable:
    """``Amon`` -> ``mon``."""

    @pytest.mark.parametrize(
        "mip_table, expected",
        [
            # The common monthly tables across realms.
            ("Amon", "mon"),
            ("Omon", "mon"),
            ("Lmon", "mon"),
            ("LImon", "mon"),
            ("SImon", "mon"),
            ("AERmon", "mon"),
            ("Emon", "mon"),
            # Daily.
            ("day", "day"),
            ("Oday", "day"),
            ("SIday", "day"),
            ("CFday", "day"),
            # CMIP5-era sea-ice tables, which reference data still uses.
            ("OImon", "mon"),
            ("OIday", "day"),
            # Sub-daily, including the point-sampled variants.
            ("3hr", "3hr"),
            ("E3hrPt", "3hrPt"),
            ("6hrLev", "6hr"),
            ("6hrPlevPt", "6hrPt"),
            ("AERhr", "1hr"),
            # Fixed fields and yearly.
            ("fx", "fx"),
            ("Ofx", "fx"),
            ("Oyr", "yr"),
            ("IyrGre", "yr"),
            # The irregular ones the enumeration exists for.
            ("Oclim", "monC"),
            ("E1hrClimMon", "1hrCM"),
            ("Esubhr", "subhrPt"),
            # Zonal-mean tables keep the frequency of their non-zonal counterpart.
            ("AERmonZ", "mon"),
            ("EdayZ", "day"),
            ("E6hrZ", "6hr"),
        ],
    )
    def test_maps_mip_table_to_frequency(self, mip_table, expected):
        assert frequency_from_mip_table(mip_table) == expected

    @pytest.mark.parametrize("frequency", ["mon", "day", "fx", "yr", "3hr", "subhrPt"])
    def test_existing_frequency_passes_through(self, frequency):
        """Some layouts already carry a frequency, so the same call site handles them."""
        assert frequency_from_mip_table(frequency) == frequency

    @pytest.mark.parametrize("value", ["day", "fx"])
    def test_names_that_are_both_table_and_frequency_are_idempotent(self, value):
        assert frequency_from_mip_table(value) == value
        assert frequency_from_mip_table(frequency_from_mip_table(value)) == value

    @pytest.mark.parametrize("value", ["Amonthly", "", "monthly", "Xmon", "AMON"])
    def test_unknown_value_raises(self, value):
        """Failing loudly beats defaulting: a silent fallback would collapse two datasets that
        differ only by frequency onto one ``instance_id``."""
        with pytest.raises(ValueError, match="Unknown MIP table or frequency"):
            frequency_from_mip_table(value)
