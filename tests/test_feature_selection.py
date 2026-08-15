from config import DataConfig
from data_generator import generate
from feature_selection import (
    build_feature_branches, correlation_selection, numeric_frame, vif_filter,
)
from preprocessing import build_variants


def _variants():
    raw = generate(DataConfig(n_rows=1500, seed=3))
    return build_variants(raw)


def test_correlation_selection_respects_threshold():
    variants = _variants()
    cols, values = correlation_selection(variants["listwise_deletion"], threshold=0.1)
    assert all(abs(v) >= 0.1 for v in values)
    assert "linear_distance" in cols  # should always clear a 0.1 threshold given the target formula


def test_vif_filter_reduces_collinearity():
    variants = _variants()
    cols, _ = correlation_selection(variants["listwise_deletion"], threshold=0.1)
    reduced = vif_filter(numeric_frame(variants["listwise_deletion"]), cols, threshold=10.0)
    assert len(reduced) <= len(cols)


def test_branches_give_mlr_a_different_smaller_or_equal_set():
    """Regression test for the VIF-scoping bug: MLR's feature set should be VIF-filtered
    (<= the shared set), while the shared 'features' set used by trees/DNN should NOT
    be VIF-filtered."""
    variants = _variants()
    branches = build_feature_branches(variants)
    assert branches, "expected at least one branch"
    for key, branch in branches.items():
        assert len(branch["mlr_features"]) <= len(branch["features"]), (
            f"MLR features should be a VIF-reduced subset for branch {key}"
        )
        # the non-MLR feature set should generally retain more signal (or be equal at minimum)
        assert set(branch["mlr_features"]).issubset(set(branch["features"]))
