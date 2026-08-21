"""
Variable selection for the shipping-cost pipeline.

Paper-compatible methods:
  1. Pearson correlation filter, |r| >= configured threshold.
  2. Forward sequential selection using LinearRegression.

Real-data support:
  - Categorical variables are one-hot encoded.
  - Missing categorical values are represented as "Unknown".
  - Constant features are removed.
  - Correlation selection can retain more features for the richer SCMS dataset.

VIF filtering remains scoped to MLR only. Tree models and the DNN use the selected
feature set directly.
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

DROP_ALWAYS = [
    "primary_key",
    "loading_datetime",
    "unloading_datetime",
    "shipper_number",
    "registrant_key",
]


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the dataframe into an all-numeric modeling frame.

    Categorical columns are one-hot encoded. Missing categories are explicitly
    represented as "Unknown" so useful rows are not lost during later dropna calls.
    Constant columns are removed because they contain no predictive information.
    """
    work = df.copy()

    work = work.drop(
        columns=[c for c in DROP_ALWAYS if c in work.columns],
        errors="ignore",
    )

    cat_cols = list(
        work.select_dtypes(
            include=["object", "category", "str"]
        ).columns
    )

    if cat_cols:
        for col in cat_cols:
            work[col] = work[col].fillna("Unknown").astype(str)

        work = pd.get_dummies(
            work,
            columns=cat_cols,
            drop_first=False,
            dtype=float,
        )

    # Remove constant columns, except for the target.
    constant_cols = [
        col for col in work.columns
        if col != TARGET and work[col].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        work = work.drop(columns=constant_cols)

    # Sanitize names because LightGBM rejects some characters.
    work.columns = [
        re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
        for c in work.columns
    ]

    # Avoid duplicate names after sanitization.
    work = work.loc[:, ~work.columns.duplicated()].copy()

    return work


def correlation_selection(
    df: pd.DataFrame,
    target: str = TARGET,
    threshold: float = 0.05,
    max_features: int = 60,
) -> Tuple[List[str], pd.Series]:
    """
    Pearson correlation selection.

    The paper-compatible threshold can still be supplied through config.py.
    max_features prevents a very large one-hot encoded real dataset from producing
    an unnecessarily huge feature set.
    """
    work = numeric_frame(df)

    if target not in work.columns:
        raise ValueError(f"Target column '{target}' not found")

    numeric = work.select_dtypes(include=[np.number])

    corr = numeric.corr()[target].drop(target)

    selected = corr[
        corr.abs() >= threshold
    ].sort_values(
        key=abs,
        ascending=False,
    )

    if max_features is not None:
        selected = selected.head(max_features)

    return list(selected.index), selected


def stepwise_selection(
    df: pd.DataFrame,
    target: str = TARGET,
    n_features: int = 25,
) -> List[str]:
    """
    Forward sequential feature selection.

    Uses linear regression as the documented approximation of the paper's unspecified
    stepwise criterion. For the richer real SCMS dataset, more than 15 features can
    be retained.
    """
    work = numeric_frame(df)

    if target not in work.columns:
        raise ValueError(f"Target column '{target}' not found")

    # SequentialFeatureSelector cannot handle missing values.
    # Median-impute numeric feature columns locally for selection.
    X = work.drop(columns=[target]).copy()
    y = work[target].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].isna().any():
            median = X[col].median()
            X[col] = X[col].fillna(
                median if pd.notna(median) else 0.0
            )

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.shape[1] == 0:
        return []

    n_features = min(n_features, X.shape[1])

    if n_features == X.shape[1]:
        return list(X.columns)

    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features,
        direction="forward",
        cv=3,
        n_jobs=-1,
    )

    sfs.fit(X, y)

    return list(X.columns[sfs.get_support()])


def vif_filter(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 10.0,
) -> List[str]:
    """
    Iteratively remove variables with VIF >= threshold.

    Used ONLY for MLR.
    """
    work = df[list(columns)].copy()

    work = work.replace([np.inf, -np.inf], np.nan)

    for col in work.columns:
        if work[col].isna().any():
            median = work[col].median()
            work[col] = work[col].fillna(
                median if pd.notna(median) else 0.0
            )

    work = work.astype(float)

    # Remove zero-variance columns before calculating VIF.
    zero_variance = [
        col for col in work.columns
        if work[col].nunique() <= 1
    ]

    if zero_variance:
        work = work.drop(columns=zero_variance)

    if work.shape[1] < 2:
        return list(work.columns)

    while work.shape[1] > 1:
        try:
            vifs = [
                variance_inflation_factor(
                    work.values,
                    i,
                )
                for i in range(work.shape[1])
            ]
        except Exception as exc:
            logger.warning(
                "VIF calculation stopped: %s",
                exc,
            )
            break

        vifs = np.asarray(vifs, dtype=float)

        # Treat non-finite VIF as a reason to remove the feature.
        if not np.isfinite(vifs).all():
            drop_idx = int(
                np.where(~np.isfinite(vifs))[0][0]
            )
            work = work.drop(
                columns=[work.columns[drop_idx]]
            )
            continue

        max_vif = float(vifs.max())

        if max_vif < threshold:
            break

        drop_idx = int(np.argmax(vifs))

        work = work.drop(
            columns=[work.columns[drop_idx]]
        )

    return list(work.columns)


def build_feature_branches(
    variants: dict,
    config: FeatureSelectionConfig = FeatureSelectionConfig(),
) -> dict:
    """
    Build feature branches for both selection methods.

    Tree/DNN models use the complete selected feature set.
    MLR alone receives the VIF-reduced feature set.
    """
    branches = {}

    for variant_name, df in variants.items():

        # Preserve the configured paper threshold, but allow the correlation branch
        # to retain a manageable number of useful real-data features.
        corr_cols, _ = correlation_selection(
            df,
            threshold=config.correlation_threshold,
            max_features=60,
        )

        # Retain at least the configured number. The real dataset benefits from
        # a somewhat richer set than the original small synthetic setup.
        step_feature_count = max(
            config.stepwise_n_features,
            25,
        )

        step_cols = stepwise_selection(
            df,
            n_features=step_feature_count,
        )

        for method_name, cols in [
            ("correlation", corr_cols),
            ("stepwise", step_cols),
        ]:
            if not cols:
                logger.warning(
                    "No features selected for %s / %s, skipping",
                    method_name,
                    variant_name,
                )
                continue

            numeric = numeric_frame(df)

            # Ensure every selected column is actually available.
            cols = [
                col for col in cols
                if col in numeric.columns
            ]

            mlr_cols = vif_filter(
                numeric,
                cols,
                threshold=config.vif_threshold,
            )

            branches[
                (method_name, variant_name)
            ] = {
                "features": cols,
                "mlr_features": mlr_cols,
            }

            logger.info(
                "%s + %s: %d features selected, %d retained for MLR after VIF filter",
                method_name,
                variant_name,
                len(cols),
                len(mlr_cols),
            )

    return branches


if __name__ == "__main__":
    from data_generator import generate
    from preprocessing import build_variants

    raw = generate()
    variants = build_variants(raw)
    build_feature_branches(variants)