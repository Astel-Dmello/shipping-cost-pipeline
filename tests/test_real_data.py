from pathlib import Path

import pytest

from real_data import load_real_data, real_continuous_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SCMS_Delivery_History_Dataset.csv"
pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="real dataset not present")


def test_loads_and_cleans():
    df = load_real_data(str(DATA_PATH))
    assert len(df) > 0
    assert "shipping_cost" in df.columns
    assert (df["shipping_cost"] > 0).all()


def test_target_has_no_missing():
    df = load_real_data(str(DATA_PATH))
    assert df["shipping_cost"].isna().sum() == 0


def test_high_cardinality_columns_capped():
    df = load_real_data(str(DATA_PATH))
    assert df["vendor"].nunique() <= 16   # top 15 + "Other"
    assert df["manufacturing_site"].nunique() <= 16
    assert df["country"].nunique() <= 21


def test_continuous_columns_match_schema():
    df = load_real_data(str(DATA_PATH))
    cols = real_continuous_columns()
    assert all(c in df.columns for c in cols)
