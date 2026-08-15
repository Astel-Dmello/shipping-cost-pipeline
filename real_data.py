"""
Loader/adapter for a REAL public logistics dataset: the Supply Chain Management System
(SCMS) Delivery History Dataset (USAID-administered health-commodity shipments,
~10,324 rows, 2006-2015). Source: https://www.usaid.gov/opengov/developer/datasets/
SCMS_Delivery_History_Dataset_20150929.csv (mirrored on GitHub, see download_real_data.sh).

This is NOT the paper's own dataset (which is private) -- it's a different real freight/
logistics pricing dataset used to validate that the pipeline's MECHANICS (messy-data
handling, preprocessing, feature selection, model training, evaluation, SHAP) hold up
on genuine real-world data, not just the paper-matched synthetic generator. Column
semantics differ from the paper (this is international health-commodity shipping, not
Korean domestic freight brokerage), so results here are a pipeline stress-test, not a
second reproduction of the paper.

Real-world messiness handled here (absent from the synthetic generator, since that's
built to be clean by design):
  - Freight Cost (USD) and Weight (Kilograms) are stored as text columns containing
    non-numeric placeholder values ("Freight Included in Commodity Cost", "Weight
    Captured Separately", "See ASN-93 (ID#:1281)", "Invoiced Separately") mixed in with
    real numbers -- ~40% of rows in each column. These are coerced to NaN, which then
    flow through the SAME listwise-deletion / mean-imputation preprocessing variants
    used on synthetic data.
  - High-cardinality categoricals (Vendor: 73 levels, Manufacturing Site: 88 levels)
    are capped to the top N levels + "Other" bucket, to keep one-hot dimensionality and
    VIF computation sane -- an engineering call a real project would also have to make.
"""
from typing import List
import os
import urllib.request

import pandas as pd

from feature_selection import TARGET
from logging_config import get_logger

logger = get_logger(__name__)

REAL_DATA_URL = (
    "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/"
    "master/SCMS_Delivery_History_Dataset.csv"
)

RAW_COLUMN_MAP = {
    "Freight Cost (USD)": TARGET,
    "Weight (Kilograms)": "freight_weight",
    "Pack Price": "pack_price",
    "Unit Price": "unit_price",
    "Line Item Quantity": "line_item_quantity",
    "Line Item Value": "line_item_value",
    "Line Item Insurance (USD)": "line_item_insurance",
    "Country": "country",
    "Shipment Mode": "shipment_mode",
    "Product Group": "product_group",
    "Vendor": "vendor",
    "Manufacturing Site": "manufacturing_site",
    "Unit of Measure (Per Pack)": "unit_of_measure",
}

NUMERIC_COLUMNS = [
    TARGET, "freight_weight", "pack_price", "unit_price",
    "line_item_quantity", "line_item_value", "line_item_insurance", "unit_of_measure",
]

HIGH_CARDINALITY_CAP = {"vendor": 15, "manufacturing_site": 15, "country": 20}


def _cap_categories(series: pd.Series, top_n: int) -> pd.Series:
    """Keeps the top_n most frequent categories, buckets the rest as 'Other'."""
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other="Other")


def _ensure_downloaded(csv_path: str) -> None:
    """Downloads the real dataset to csv_path if it isn't already present. Lets the
    pipeline work out of the box both locally and on a fresh deploy (e.g. Streamlit
    Community Cloud) where the CSV isn't committed to git (see .gitignore) but the
    runtime does have outbound internet access."""
    if os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    logger.info("Real dataset not found at %s, downloading from %s", csv_path, REAL_DATA_URL)
    urllib.request.urlretrieve(REAL_DATA_URL, csv_path)
    logger.info("Downloaded real dataset to %s", csv_path)


def load_real_data(csv_path: str) -> pd.DataFrame:
    """Loads and cleans the SCMS dataset into the same shape the rest of the pipeline
    expects: a numeric target column named per feature_selection.TARGET, numeric
    feature columns coerced from placeholder-laden text, and capped categoricals.
    Auto-downloads csv_path if it doesn't exist locally (see _ensure_downloaded)."""
    _ensure_downloaded(csv_path)
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info("Loaded real dataset: %d rows, %d columns", *raw.shape)

    df = raw.rename(columns=RAW_COLUMN_MAP)
    df = df[[c for c in RAW_COLUMN_MAP.values() if c in df.columns]].copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        before_numeric = pd.to_numeric(df[col], errors="coerce")
        non_numeric_rate = before_numeric.isna().mean() - df[col].isna().mean()
        if non_numeric_rate > 0:
            logger.info("Coercing '%s' to numeric: %.1f%% of rows had non-numeric "
                        "placeholder text, now NaN", col, non_numeric_rate * 100)
        df[col] = before_numeric

    for col, cap in HIGH_CARDINALITY_CAP.items():
        if col in df.columns:
            n_before = df[col].nunique()
            df[col] = _cap_categories(df[col], cap)
            logger.info("Capped '%s': %d levels -> top %d + Other", col, n_before, cap)

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    logger.info("After dropping rows with missing/invalid target: %d rows remain", len(df))

    return df.reset_index(drop=True)


def real_continuous_columns() -> List[str]:
    """Continuous columns for this dataset, for preprocessing.remove_outliers_iqr."""
    return ["freight_weight", "pack_price", "unit_price", "line_item_quantity",
            "line_item_value", "line_item_insurance", "unit_of_measure"]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/SCMS_Delivery_History_Dataset.csv"
    data = load_real_data(path)
    print(data.describe(include="all").T)
