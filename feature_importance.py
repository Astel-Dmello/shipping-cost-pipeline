"""
SHAP-based feature importance, applied to LightGBM (the paper's best-performing model,
Sec 5.7) on the stepwise-selected variable set (the paper's highest-R^2 branch). Prints
the ranked mean |SHAP value| per feature, comparable to the paper's Figure 3 finding:
linear_distance dominates (>50% contribution), followed by actual_distance, freight_weight,
vehicle_tonnage.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap

from feature_selection import numeric_frame, TARGET
from logging_config import get_logger
from models import build_lightgbm

logger = get_logger(__name__)


def shap_importance(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    """Fits a LightGBM model on the given features and returns (mean |SHAP|, % share)."""
    work = numeric_frame(df)[feature_cols + [TARGET]].dropna()
    X, y = work[feature_cols], work[TARGET]

    model = build_lightgbm()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
    share = importance / importance.sum() * 100
    return importance, share


if __name__ == "__main__":
    from data_generator import generate
    from preprocessing import build_variants
    from feature_selection import build_feature_branches

    raw = generate()
    variants = build_variants(raw)
    branches = build_feature_branches(variants)

    branch = branches[("stepwise", "listwise_deletion")]
    importance, share = shap_importance(variants["listwise_deletion"], branch["features"])
    logger.info("Feature importance (mean |SHAP|), %% of total:\n%s",
                "\n".join(f"  {f:20s} {share[f]:5.1f}%" for f in importance.index))
