"""
Preprocessing pipeline matching the paper's steps:
  1. Outlier removal via IQR, applied only to continuous variables.
  2. Missing-data handling: TWO variants compared, as the paper does --
     listwise deletion and mean imputation.
Both variants are exposed as separate functions so evaluate.py can loop over them,
mirroring the paper's side-by-side comparison.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from config import PreprocessConfig
from logging_config import get_logger

logger = get_logger(__name__)

CONTINUOUS_COLUMNS = [
    "linear_distance", "actual_distance", "freight_weight", "vehicle_tonnage",
    "standard_fare", "loading_time", "unloading_time", "precipitation",
]


def remove_outliers_iqr(
    df: pd.DataFrame, columns: List[str] = CONTINUOUS_COLUMNS, k: float = 1.5
) -> pd.DataFrame:
    """
    IQR-based outlier removal, applied only to continuous variables (paper Sec 3.2.2).
    The paper doesn't state its IQR multiplier; k=1.5 is the standard default, used here
    as a documented assumption rather than an invented paper-stated number.
    """
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    for col in columns:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        col_mask = out[col].between(lower, upper) | out[col].isna()
        mask &= col_mask
    result = out[mask].reset_index(drop=True)
    logger.info("Outlier removal: %d -> %d rows", len(df), len(result))
    return result


def listwise_deletion(df: pd.DataFrame) -> pd.DataFrame:
    """Removes all rows with any missing values (paper Sec 3.2.3)."""
    return df.dropna().reset_index(drop=True)


def mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces missing values with each column's mean (numeric only) (paper Sec 3.2.3)."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].mean())
    return out


def min_max_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Min-max scale continuous columns to [0, 1] (paper Sec 5.1)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        c_min, c_max = out[col].min(), out[col].max()
        out[col] = (out[col] - c_min) / (c_max - c_min) if c_max > c_min else 0.0
    return out


def build_variants(
    raw_df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig(),
    continuous_columns: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Produces the two preprocessing variants the paper compares:
    outlier-removed data with (a) listwise deletion, (b) mean imputation.
    continuous_columns overrides which columns get IQR outlier treatment -- needed
    when running on a dataset with a different schema than the synthetic generator's
    (see real_data.py's real_continuous_columns()).
    """
    cols = continuous_columns if continuous_columns is not None else CONTINUOUS_COLUMNS
    cleaned = remove_outliers_iqr(raw_df, columns=cols, k=config.iqr_multiplier)
    variants = {
        "listwise_deletion": listwise_deletion(cleaned),
        "mean_imputation": mean_imputation(cleaned),
    }
    for name, d in variants.items():
        logger.info("%s: %d rows", name, len(d))
    return variants


if __name__ == "__main__":
    from data_generator import generate

    raw = generate()
    build_variants(raw)
