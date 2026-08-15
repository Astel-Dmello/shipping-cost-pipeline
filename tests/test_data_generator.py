import numpy as np
import pytest

from config import DataConfig
from data_generator import generate


@pytest.fixture(scope="module")
def data():
    return generate(DataConfig(n_rows=500, seed=1))


def test_shape(data):
    assert len(data) == 500
    assert "shipping_cost" in data.columns


def test_target_positive(data):
    assert (data["shipping_cost"] > 0).all()


def test_linear_distance_is_top_correlate(data):
    """Sanity check against the paper's Table 2 finding: linear_distance should have
    among the strongest correlations with shipping_cost."""
    numeric_candidates = [
        "linear_distance", "actual_distance", "freight_weight",
        "vehicle_tonnage", "standard_fare",
    ]
    corr = data[numeric_candidates + ["shipping_cost"]].corr()["shipping_cost"].drop("shipping_cost")
    top = corr.abs().idxmax()
    assert top in ("linear_distance", "standard_fare", "actual_distance"), (
        f"expected a distance/fare variable to dominate, got {top}"
    )


def test_missingness_injected(data):
    assert data[["freight_weight", "standard_fare", "actual_distance"]].isna().any().any()


def test_reproducible_with_seed():
    a = generate(DataConfig(n_rows=200, seed=7))
    b = generate(DataConfig(n_rows=200, seed=7))
    pd_equal = a.drop(columns=["loading_datetime", "unloading_datetime"]).equals(
        b.drop(columns=["loading_datetime", "unloading_datetime"])
    )
    assert pd_equal
