from config import DataConfig
from data_generator import generate
from preprocessing import build_variants, listwise_deletion, mean_imputation, remove_outliers_iqr


def _raw():
    return generate(DataConfig(n_rows=1000, seed=2))


def test_outlier_removal_shrinks_or_keeps_rows():
    raw = _raw()
    cleaned = remove_outliers_iqr(raw)
    assert len(cleaned) <= len(raw)


def test_listwise_deletion_has_no_nulls():
    raw = _raw()
    result = listwise_deletion(raw)
    assert result.isna().sum().sum() == 0


def test_mean_imputation_fills_numeric_nulls():
    raw = _raw()
    result = mean_imputation(raw)
    numeric_cols = result.select_dtypes(include="number").columns
    assert result[numeric_cols].isna().sum().sum() == 0


def test_variants_differ_in_row_count():
    """Mean imputation should retain at least as many rows as listwise deletion,
    since it doesn't drop rows for missingness."""
    raw = _raw()
    variants = build_variants(raw)
    assert variants["mean_imputation"].shape[0] >= variants["listwise_deletion"].shape[0]
