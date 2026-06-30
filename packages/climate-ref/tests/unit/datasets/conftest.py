import pandas as pd
import pytest


@pytest.fixture
def catalog_regression(data_regression, sample_data_dir):
    def check(df: pd.DataFrame, basename: str):
        # Strip the path to make the test more robust
        df["path"] = df["path"].str.replace(str(sample_data_dir), "{esgf_data_dir}")

        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        data_regression.check(records, basename=basename)

    return check
