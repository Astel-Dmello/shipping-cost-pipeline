"""
Variable selection matching the paper's two methods (Sec 4):
  1. Pearson correlation filter, |r| >= 0.1 against the target.
  2. Stepwise selection (approximated with forward SequentialFeatureSelector, since the
     paper doesn't specify AIC/BIC/p-value as its exact stepwise criterion).

Correctness note (fixed after initial testing): the paper's VIF-based multicollinearity
removal (Sec 5.2) is scoped to the MLR model only -- it's there because OLS assumes
non-collinear inputs, not because the other models need it. XGBoost, LightGBM, and the
DNN are trained on the correlation/stepwise-selected variables directly, WITHOUT VIF
filtering, matching what the paper actually does. An earlier version of this pipeline
applied VIF filtering to every model's inputs, which starved the boosting models of
features and made MLR look artificially competitive -- the opposite of the paper's
finding. See config.py's FeatureSelectionConfig.vif_only_for.
"""
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import FeatureSelectionConfig
from logging_config import get_logger

logger = get_logger(__name__)

TARGET = "shipping_cost"
DROP_ALWAYS = ["primary_key", "loading_datetime", "unloading_datetime",
               "shipper_number", "registrant_key"]


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categoricals and drops ID/datetime columns that carry no
    generalizable signal, returning an all-numeric frame ready for modeling.
    Column names are sanitized (special JSON/regex characters stripped) because
    LightGBM rejects them outright -- real category values like "Côte d'Ivoire" or
    "N/A" survive into one-hot column names and would otherwise crash training."""
    work = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], errors="ignore")
    cat_cols = work.select_dtypes(include=["object", "category", "str"]).columns
    work = pd.get_dummies(work, columns=list(cat_cols), drop_first=True)
    work.columns = [
        re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in work.columns
    ]
    return work


def correlation_selection(
    df: pd.DataFrame, target: str = TARGET, threshold: float = 0.1
) -> Tuple[List[str], pd.Series]:
    """Pearson correlation filter (paper Sec 4.1)."""
    work = numeric_frame(df)
    corr = work.corr()[target].drop(target)
    selected = corr[corr.abs() >= threshold].sort_values(key=abs, ascending=False)
    return list(selected.index), selected


def stepwise_selection(
    df: pd.DataFrame, target: str = TARGET, n_features: int = 15
) -> List[str]:
    """
    Approximate stepwise selection: forward sequential feature selection with a linear
    model, scored by cross-validated R^2. The paper's exact stepwise criterion
    (AIC/BIC/p-value) isn't stated, so this is a documented, reasonable stand-in.
    """
    work = numeric_frame(df).dropna()
    X = work.drop(columns=[target])
    y = work[target]
    n_features = min(n_features, X.shape[1])
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select=n_features, direction="forward", cv=3
    )
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])


def vif_filter(df: pd.DataFrame, columns: List[str], threshold: float = 10.0) -> List[str]:
    """
    Removes variables with VIF >= threshold (paper Sec 5.2), one at a time (highest first),
    recomputing VIF after each removal. Used ONLY for the MLR model's inputs -- see the
    module docstring for why this isn't applied to the other models.
    """
    work = df[list(columns)].dropna().astype(float)
    if work.shape[1] < 2:
        return list(work.columns)
    while work.shape[1] > 1:
        vifs = [variance_inflation_factor(work.values, i) for i in range(work.shape[1])]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        drop_idx = int(np.argmax(vifs))
        work = work.drop(columns=[work.columns[drop_idx]])
    return list(work.columns)


def build_feature_branches(
    variants: dict, config: FeatureSelectionConfig = FeatureSelectionConfig()
) -> dict:
    """
    Builds the paper's 2x2 branch structure (correlation-analysis / stepwise) x
    (listwise_deletion / mean_imputation), returning, for each branch, the shared
    variable set used by tree/DNN models and the VIF-reduced set used by MLR only.

    Returns:
        {
          (selection_method, preprocessing_variant): {
              "features": [...],           # used by XGBoost, LightGBM, DNN
              "mlr_features": [...],       # VIF-reduced, used by MLR only
          },
          ...
        }
    """
    branches = {}
    for variant_name, df in variants.items():
        corr_cols, _ = correlation_selection(df, threshold=config.correlation_threshold)
        step_cols = stepwise_selection(df, n_features=config.stepwise_n_features)

        for method_name, cols in [("correlation", corr_cols), ("stepwise", step_cols)]:
            if not cols:
                logger.warning("No features selected for %s / %s, skipping",
                                method_name, variant_name)
                continue
            mlr_cols = vif_filter(numeric_frame(df), cols, threshold=config.vif_threshold)
            branches[(method_name, variant_name)] = {
                "features": cols,
                "mlr_features": mlr_cols,
            }
            logger.info(
                "%s + %s: %d features selected, %d retained for MLR after VIF filter",
                method_name, variant_name, len(cols), len(mlr_cols),
            )
    return branches


if __name__ == "__main__":
    from data_generator import generate
    from preprocessing import build_variants

    raw = generate()
    variants = build_variants(raw)
    build_feature_branches(variants)
